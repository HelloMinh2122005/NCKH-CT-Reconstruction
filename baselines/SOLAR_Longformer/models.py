"""
================================================================================
KIẾN TRÚC MÔ HÌNH: SOLAR_Longformer (LIMITED-ANGLE CT RECONSTRUCTION)
Second-Order Dual-Branch Newton-CG Unrolling with Longformer Attention
Dự án: Nghiên cứu Tái tạo Ảnh Cắt lớp CT Góc Giới hạn (Limited-Angle CT Reconstruction)
Tác giả: MinhPD — Nhóm Nghiên cứu Tái tạo Ảnh Y tế (VNU-HCM UIT)
================================================================================
"""

from typing import Optional, Tuple
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
import odl
from odl.contrib import torch as odl_torch
from transformers.models.longformer.modeling_longformer import (
    LongformerSelfAttention,
    LongformerConfig,
)
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)


# =============================================================================
# 1. BỘ GIẢI CONJUGATE GRADIENT TỐI ƯU HÓA BẬC 2 (SAFE MATRIX-FREE CG SOLVER)
# =============================================================================
class SafeCGSolver(nn.Module):
    """
    Bộ giải Conjugate Gradient (CG) tối ưu hóa bậc 2 Matrix-Free giải hệ phương trình:
        (λ_t * A_FBP^† A + μ_t * I) x_{t+1} = b_t
    """
    def __init__(self, cg_iters: int = 4):
        super().__init__()
        self.cg_iters = cg_iters

    def forward(
        self,
        x_init: torch.Tensor,
        b_t: torch.Tensor,
        lambda_t: torch.Tensor,
        mu_t: torch.Tensor,
        forward_op: nn.Module,
        backward_op: nn.Module,
    ) -> torch.Tensor:
        x = x_init.clone()

        def hessian_matvec(p_vec: torch.Tensor) -> torch.Tensor:
            return lambda_t * backward_op(forward_op(p_vec)) + mu_t * p_vec

        r = b_t - hessian_matvec(x)
        p = r.clone()
        rs_old = torch.sum(r * r, dim=(1, 2, 3), keepdim=True)

        for _ in range(self.cg_iters):
            Ap = hessian_matvec(p)
            pAp = torch.sum(p * Ap, dim=(1, 2, 3), keepdim=True)
            alpha = rs_old / (pAp + 1e-7)

            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = torch.sum(r * r, dim=(1, 2, 3), keepdim=True)

            if torch.max(rs_new) < 1e-6:
                break

            beta = rs_new / (rs_old + 1e-7)
            p = r + beta * p
            rs_old = rs_new

        return x


# =============================================================================
# 2. KHỐI ĐIỀU HÒA KÉP SONG SONG (DUAL-BRANCH REGULARIZER WITH LONGFORMER)
# =============================================================================
class DualBranchRegularizer(nn.Module):
    """
    Khối Tiên Nghiệm Học Sâu Phân Nhánh Kép:
    - Nhánh Local: Multi-Scale Res-CNN (Kernel 3x3, 5x5).
    - Nhánh Global: Longformer Sliding-Chunks Self-Attention (6 heads, window=256).
    - Fusion: Cat -> Conv3x3 -> ReLU -> Conv3x3.
    """
    def __init__(
        self,
        in_channels: int = 1,
        channels: int = 48,
        window_size: int = 2,
        image_size: int = 256,
        num_layers: int = 12,
        num_global: int = 50,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.window_size = window_size
        self.image_size = image_size
        self.token_dim = channels * (window_size ** 2) # 48 * 4 = 192
        self.num_global = num_global

        # --- Nhánh 1: Cục bộ (Local Multi-Scale Res-CNN) ---
        self.loc_conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        nn.init.normal_(self.loc_conv1.weight, mean=0.0, std=0.01)

        self.loc_conv2 = nn.Conv2d(channels, channels, kernel_size=5, padding=2)
        nn.init.normal_(self.loc_conv2.weight, mean=0.0, std=0.01)

        self.loc_conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        nn.init.normal_(self.loc_conv3.weight, mean=0.0, std=0.01)

        # --- Nhánh 2: Toàn cục (Global Longformer Attention) ---
        self.glob_proj = nn.Conv2d(in_channels, channels, kernel_size=1)
        nn.init.normal_(self.glob_proj.weight, mean=0.0, std=0.01)

        self.longformer_config = LongformerConfig(
            num_attention_heads=6,
            num_layers=num_layers,
            hidden_size=self.token_dim,
            attention_window=[256] * num_layers,
            attention_dilation=[1] * num_layers,
            attention_mode="sliding_chunks",
            autoregressive=False,
        )
        self.longformer_attention = LongformerSelfAttention(
            config=self.longformer_config,
            layer_id=0,
        )

        # --- Hợp nhất hai luồng ---
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, in_channels, kernel_size=3, padding=1),
        )
        nn.init.normal_(self.fusion[0].weight, mean=0.0, std=0.01)
        nn.init.normal_(self.fusion[2].weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # 1. Trích xuất đặc trưng Cục bộ (Local Res-CNN)
        f_loc = F.relu(self.loc_conv1(x))
        f_loc = F.relu(self.loc_conv2(f_loc)) + f_loc
        f_loc = self.loc_conv3(f_loc)

        # 2. Trích xuất đặc trưng Toàn cục qua Token 2x2 (Global Longformer)
        f_in = self.glob_proj(x)
        tokens = (
            f_in.unfold(2, self.window_size, self.window_size)
            .unfold(3, self.window_size, self.window_size)
            .contiguous()
            .view(B, self.channels, -1, self.window_size ** 2)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(B, -1, self.token_dim)
        )
        N = tokens.shape[1]

        # Tạo mask attention cho Longformer
        attention_mask = torch.zeros(B, N, dtype=torch.long, device=tokens.device)
        is_index_masked = attention_mask != 0

        # Chọn đều đặn các vị trí Global Attention dọc theo chuỗi token
        global_positions = torch.linspace(
            0, N - 1, steps=self.num_global, device=tokens.device
        ).long()
        is_index_global_attn = torch.zeros(B, N, dtype=torch.bool, device=tokens.device)
        is_index_global_attn[:, global_positions] = True

        # Gọi LongformerSelfAttention
        attn_out = self.longformer_attention(
            tokens,
            attention_mask=attention_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=True,
            output_attentions=False,
        )[0]

        # Unpatchify về 2D (B, 48, 256, 256)
        new_spatial = H // self.window_size
        f_glob = (
            attn_out.view(B, new_spatial, new_spatial, self.channels, self.window_size, self.window_size)
            .permute(0, 3, 1, 4, 2, 5)
            .contiguous()
            .view(B, self.channels, H, W)
        )

        # 3. Hợp nhất hai luồng
        fused = torch.cat([f_loc, f_glob], dim=1)
        grad_R = self.fusion(fused)
        return grad_R


# =============================================================================
# 3. MÔ HÌNH TOÀN CỤC SOLAR_Longformer_LA (PYTORCH LIGHTNING MODULE)
# =============================================================================
class SOLAR_Longformer_LA(pl.LightningModule):
    """
    Mô hình Kiến trúc Đề Xuất SOLAR_Longformer hoàn chỉnh cho Limited-Angle CT.
    """
    def __init__(
        self,
        n_iterations: int = 8,
        cg_iters: int = 4,
        num_view: int = 64,
        num_detectors: int = 512,
        start_ang: float = -np.pi / 3,
        end_ang: float = np.pi / 3,
        input_size: int = 256,
        window_size: int = 2,
        num_global: int = 50,
        initial_lr: float = 1e-4,
        final_lr: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.n_iterations = n_iterations
        self.cg_iters = cg_iters
        self.num_view = num_view
        self.num_detectors = num_detectors
        self.start_ang = start_ang
        self.end_ang = end_ang
        self.input_size = input_size
        self.initial_lr = initial_lr
        self.final_lr = final_lr

        self.cg_solver = SafeCGSolver(cg_iters=cg_iters)

        self.regularizer = DualBranchRegularizer(
            in_channels=1,
            channels=48,
            window_size=window_size,
            image_size=input_size,
            num_global=num_global,
        )

        self.raw_lambda = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.54)) for _ in range(n_iterations)]
        )
        self.raw_mu = nn.ParameterList(
            [nn.Parameter(torch.tensor(-2.25)) for _ in range(n_iterations)]
        )

        radon_curr, fbp_curr = self.radon_transform(
            num_view=num_view,
            start_ang=start_ang,
            end_ang=end_ang,
            num_detectors=num_detectors,
            input_size=input_size,
        )
        self.forward_module = radon_curr
        self.backward_module = fbp_curr

        self.grid: Optional[torch.Tensor] = None

    def forward(self, x_0: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_t = x_0
        bp_y = self.backward_module(y)

        for i in range(self.n_iterations):
            lambda_t = F.softplus(self.raw_lambda[i]) + 1e-4
            mu_t = F.softplus(self.raw_mu[i]) + 1e-4

            grad_R = self.regularizer(x_t)
            b_t = lambda_t * bp_y + mu_t * x_t - grad_R

            x_t = self.cg_solver(
                x_init=x_t,
                b_t=b_t,
                lambda_t=lambda_t,
                mu_t=mu_t,
                forward_op=self.forward_module,
                backward_op=self.backward_module,
            )

        return x_t

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.initial_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=5,
            eta_min=self.final_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
            },
        }

    def _get_batch_data_range(self, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_min = target.amin()
        batch_max = target.amax()
        if torch.isclose(batch_max, batch_min):
            batch_max = batch_min + torch.tensor(
                1e-8, device=target.device, dtype=target.dtype
            )
        return (batch_min, batch_max)

    def rmse(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.mean((y_true - y_pred) ** 2))

    def training_step(self, train_batch, batch_idx):
        phantom, fbp_u, sino_noisy = train_batch
        x_reconstructed = self.forward(fbp_u, sino_noisy)
        loss = F.mse_loss(phantom, x_reconstructed)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        phantom, fbp_u, sino_noisy = val_batch
        x_reconstructed = self.forward(fbp_u, sino_noisy)
        loss = F.mse_loss(phantom, x_reconstructed)

        data_range = self._get_batch_data_range(phantom)
        ssim_p = structural_similarity_index_measure(
            x_reconstructed, phantom, data_range=data_range
        )
        psnr_p = peak_signal_noise_ratio(
            x_reconstructed, phantom, data_range=data_range
        )
        rmse_p = self.rmse(phantom, x_reconstructed)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_ssim", ssim_p, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_psnr", psnr_p, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_rmse", rmse_p, on_step=False, on_epoch=True, prog_bar=False)

        self.grid = torchvision.utils.make_grid(x_reconstructed.detach().clamp(min=0.0))
        return {"val_loss": loss, "val_ssim": ssim_p, "val_psnr": psnr_p, "val_rmse": rmse_p}

    def test_step(self, batch, batch_idx):
        phantom, fbp_u, sino_noisy = batch
        x_reconstructed = self.forward(fbp_u, sino_noisy)
        loss = F.mse_loss(phantom, x_reconstructed)

        data_range = self._get_batch_data_range(phantom)
        ssim_p = structural_similarity_index_measure(
            x_reconstructed, phantom, data_range=data_range
        )
        psnr_p = peak_signal_noise_ratio(
            x_reconstructed, phantom, data_range=data_range
        )
        rmse_p = self.rmse(phantom, x_reconstructed)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_ssim", ssim_p, on_step=False, on_epoch=True)
        self.log("test_psnr", psnr_p, on_step=False, on_epoch=True)
        self.log("test_rmse", rmse_p, on_step=False, on_epoch=True)
        return {"SSIM": ssim_p, "PSNR": psnr_p, "RMSE": rmse_p}

    def on_validation_epoch_end(self):
        if self.grid is not None and self.logger is not None:
            tag = f"generated_images_epoch_{self.current_epoch}"
            self.logger.experiment.add_image(tag, self.grid, self.current_epoch)

    def radon_transform(
        self,
        num_view: int = 64,
        start_ang: float = -np.pi / 3,
        end_ang: float = np.pi / 3,
        num_detectors: int = 512,
        input_size: int = 256,
    ):
        xx = 200
        space = odl.uniform_discr(
            [-xx, -xx],
            [xx, xx],
            [input_size, input_size],
            dtype="float32",
        )

        angles = int(num_view)
        angle_partition = odl.uniform_partition(start_ang, end_ang, angles)
        detector_partition = odl.uniform_partition(-480, 480, num_detectors)

        geometry = odl.tomo.FanBeamGeometry(
            angle_partition,
            detector_partition,
            src_radius=600,
            det_radius=290,
        )

        impl = "astra_cuda" if torch.cuda.is_available() else "astra_cpu"
        operator = odl.tomo.RayTransform(space, geometry, impl=impl)
        op_layer = odl_torch.operator.OperatorModule(operator)

        fbp = odl.tomo.fbp_op(
            operator,
            filter_type="Ram-Lak",
            frequency_scaling=0.9,
        ) * np.sqrt(2)
        op_layer_fbp = odl_torch.operator.OperatorModule(fbp)

        return op_layer, op_layer_fbp
