"""
================================================================================
KIẾN TRÚC MÔ HÌNH: SOLAR_LongNet (LIMITED-ANGLE CT RECONSTRUCTION)
Second-Order Dual-Branch Newton-CG Unrolling with LongNet Attention
Dự án: Nghiên cứu Tái tạo Ảnh Cắt lớp CT Góc Giới hạn (Limited-Angle CT Reconstruction)
Tác giả: MinhPD — Nhóm Nghiên cứu Tái tạo Ảnh Y tế (VNU-HCM UIT)
Mô tả:
    - Khắc phục hiện tượng ma trận Hessian A^T A suy biến trên Limited-Angle CT.
    - Động cơ tối ưu hóa bậc 2 Matrix-Free qua Bộ giải Safe Conjugate Gradient (SafeCGSolver).
    - Khối điều hòa kép song song (DualBranchRegularizer):
        + Nhánh Cục bộ: Multi-Scale Res-CNN (Kernel 3x3, 5x5) bảo vệ mô mềm.
        + Nhánh Toàn cục: Token siêu mịn 2x2 Multi-Scale Dilated Attention (LongNet, N=16384) bù đắp nêm khuyết.
    - Cơ chế chia sẻ trọng số lặp (Recurrent Weight Sharing) kiểm soát dung lượng mô hình ~0.27M params.
    - Ràng buộc tham số xác định dương (Strict SPD) qua hàm Softplus, triệt tiêu nguy cơ NaN gradient.
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
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)

# Import DilatedAttention
try:
    from long_net import DilatedAttention
except ImportError:
    try:
        from baselines.SOLAR_LongNet.long_net import DilatedAttention
    except ImportError:
        from .long_net import DilatedAttention


# =============================================================================
# 1. BỘ GIẢI CONJUGATE GRADIENT TỐI ƯU HÓA BẬC 2 (SAFE MATRIX-FREE CG SOLVER)
# =============================================================================
class SafeCGSolver(nn.Module):
    """
    Bộ giải Conjugate Gradient (CG) tối ưu hóa bậc 2 Matrix-Free giải hệ phương trình:
        (λ_t * A_FBP^† A + μ_t * I) x_{t+1} = b_t
        
    Đặc điểm kỹ thuật:
    1. Matrix-Free: Hoàn toàn không bao giờ phân bổ hay lưu trữ ma trận Hessian 274 GB vào VRAM.
    2. Strict SPD: Được bảo vệ nghiêm ngặt tính đối xứng xác định dương nhờ μ_t > 0.
    3. Numerical Stability: Có hệ số epsilon chống chia cho 0 và điều kiện dừng sớm khi residual < 1e-6.
    
    Tham số khởi tạo:
        cg_iters (int): Số bước lặp CG nội bộ trong mỗi stage unrolling (mặc định = 4).
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
        """
        Luồng tính toán Forward của Bộ giải CG:
        Đầu vào:
            x_init (torch.Tensor): Nghiệm khởi tạo tại bước lặp (shape: B, 1, H, W).
            b_t (torch.Tensor): Vector vế phải của hệ phương trình bậc 2 (shape: B, 1, H, W).
            lambda_t (torch.Tensor): Trọng số bước đo đạc vật lý (vô hướng > 0).
            mu_t (torch.Tensor): Hệ số cản dịu Tikhonov (vô hướng > 0).
            forward_op (nn.Module): Toán tử chiếu thuận Radon A.
            backward_op (nn.Module): Toán tử chiếu ngược có lọc FBP A_FBP^†.
        Đầu ra:
            x (torch.Tensor): Nghiệm x_{t+1} sau K_CG bước cập nhật Newton-CG.
        """
        x = x_init.clone()

        # Định nghĩa toán tử nhân ma trận Hessian xấp xỉ với vector p:
        # H_t(p) = λ_t * A_FBP^†( A(p) ) + μ_t * p
        def hessian_matvec(p_vec: torch.Tensor) -> torch.Tensor:
            return lambda_t * backward_op(forward_op(p_vec)) + mu_t * p_vec

        # Khởi tạo residual ban đầu r_0 = b_t - H_t(x_0) và hướng tìm kiếm p_0 = r_0
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
# 2. KHỐI ĐIỀU HÒA KÉP SONG SONG (DUAL-BRANCH REGULARIZER WITH LONGNET)
# =============================================================================
class DualBranchRegularizer(nn.Module):
    """
    Khối Tiên Nghiệm Học Sâu Phân Nhánh Kép (Dual-Branch Learned Regularizer R_theta):
    
    Cấu trúc:
    1. Nhánh Cục bộ (Local Multi-Scale Res-CNN):
       - Conv 3x3 (1 -> 48) -> ReLU -> Conv 5x5 (48 -> 48) + Skip -> Conv 3x3 (48 -> 48).
       - Khử nhiễu vi mô và bảo toàn ranh giới mô mềm.
    2. Nhánh Toàn cục (Global Long-Sequence Dilated Attention):
       - Chiếu đặc trưng 1x1 (1 -> 48).
       - Patchify 2x2 thành N = 16,384 tokens (D = 192).
       - Dilated Attention đa tỷ lệ (LongNet, heads=4, segment_size=16, dilation_rate=1).
       - Unpatchify khôi phục tensor ảnh 2D (48, 256, 256).
    3. Hợp nhất Đặc trưng (Feature Fusion):
       - Ghép kênh Cat(F_loc, F_glob) -> Conv 3x3 (96 -> 48) -> ReLU -> Conv 3x3 (48 -> 1).
    """
    def __init__(
        self,
        in_channels: int = 1,
        channels: int = 48,
        window_size: int = 2,
        image_size: int = 256,
        segment_size: int = 16,
        dilation_rate: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.window_size = window_size
        self.image_size = image_size
        self.token_dim = channels * (window_size ** 2) # 48 * 4 = 192

        # --- Nhánh 1: Cục bộ (Local Multi-Scale Res-CNN) ---
        self.loc_conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        nn.init.normal_(self.loc_conv1.weight, mean=0.0, std=0.01)

        self.loc_conv2 = nn.Conv2d(channels, channels, kernel_size=5, padding=2)
        nn.init.normal_(self.loc_conv2.weight, mean=0.0, std=0.01)

        self.loc_conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        nn.init.normal_(self.loc_conv3.weight, mean=0.0, std=0.01)

        # --- Nhánh 2: Toàn cục (Global LongNet Dilated Attention) ---
        self.glob_proj = nn.Conv2d(in_channels, channels, kernel_size=1)
        nn.init.normal_(self.glob_proj.weight, mean=0.0, std=0.01)

        self.longnet_attn = DilatedAttention(
            dim=self.token_dim,
            heads=4,
            dilation_rate=dilation_rate,
            segment_size=segment_size,
            dropout=dropout,
            qk_norm=True,
        )

        # --- Hợp nhất hai luồng (Feature Fusion & Gradient Output) ---
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

        # 2. Trích xuất đặc trưng Toàn cục qua Token 2x2 (Global LongNet)
        f_in = self.glob_proj(x)
        
        # Patchify 2x2 -> (B, 16384, 192)
        tokens = (
            f_in.unfold(2, self.window_size, self.window_size)
            .unfold(3, self.window_size, self.window_size)
            .contiguous()
            .view(B, self.channels, -1, self.window_size ** 2)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(B, -1, self.token_dim)
        )

        # Dilated Attention
        tokens_attn = self.longnet_attn(tokens)

        # Unpatchify về (B, 48, 256, 256)
        new_spatial = H // self.window_size
        f_glob = (
            tokens_attn.view(B, new_spatial, new_spatial, self.channels, self.window_size, self.window_size)
            .permute(0, 3, 1, 4, 2, 5)
            .contiguous()
            .view(B, self.channels, H, W)
        )

        # 3. Hợp nhất hai luồng
        fused = torch.cat([f_loc, f_glob], dim=1)
        grad_R = self.fusion(fused)
        return grad_R


# =============================================================================
# 3. MÔ HÌNH TOÀN CỤC SOLAR_LongNet_LA (PYTORCH LIGHTNING MODULE)
# =============================================================================
class SOLAR_LongNet_LA(pl.LightningModule):
    """
    Mô hình Kiến trúc Đề Xuất SOLAR_LongNet hoàn chỉnh cho Limited-Angle CT.
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
        segment_size: int = 16,
        dilation_rate: int = 1,
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

        # Khởi tạo Bộ giải Safe CG
        self.cg_solver = SafeCGSolver(cg_iters=cg_iters)

        # Khởi tạo Khối Điều Hòa Kép (Recurrent Weight Sharing)
        self.regularizer = DualBranchRegularizer(
            in_channels=1,
            channels=48,
            window_size=window_size,
            image_size=input_size,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
        )

        # Các tham số học được riêng từng stage (bảo vệ SPD qua softplus)
        self.raw_lambda = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.54)) for _ in range(n_iterations)] # softplus(0.54) ≈ 1.0
        )
        self.raw_mu = nn.ParameterList(
            [nn.Parameter(torch.tensor(-2.25)) for _ in range(n_iterations)] # softplus(-2.25) ≈ 0.1
        )

        # Khởi tạo toán tử Radon & FBP
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
