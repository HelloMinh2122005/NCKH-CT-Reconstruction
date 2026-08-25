from typing import Optional, List
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
import odl
from odl.contrib import torch as odl_torch
from transformers import LongformerSelfAttention, LongformerConfig
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)


class LongformerAttentionBlock(nn.Module):
    """
    Khối Attention Longformer cho bài toán Tái tạo ảnh CT.
    
    Nguyên lý hoạt động:
    1. Chia ảnh đặc trưng (B, 48, 256, 256) thành 16.384 tokens kích thước 2x2 pixel (mỗi token có dim = 192).
    2. Áp dụng cơ chế Sliding Window Local Attention (cửa sổ trượt w = 256 tokens) kết hợp $50$ Global Attention Tokens 
       được phân bố đều đặn dọc theo chuỗi.
    3. Cơ chế kết hợp Local + Global cho phép mô hình thu nhận thông tin liên kết toàn cục từ khắp các góc ảnh mà vẫn
       duy trì độ phức tạp tính toán O(N * w) tuyến tính.
    
    Tham số khởi tạo:
    - window_size (int): Kích thước mỗi patch token (mặc định = 2, tức 2x2 pixel).
    - patch_channels (int): Số kênh đặc trưng của tensor (mặc định = 48).
    - image_size (int): Kích thước không gian của ảnh (mặc định = 256).
    - num_layers (int): Số lớp transformer bên trong (mặc định = 12).
    - num_global (int): Số lượng vị trí token toàn cục Global Attention được chọn đều (mặc định = 50).
    """
    def __init__(
        self,
        window_size: int = 2,
        patch_channels: int = 48,
        image_size: int = 256,
        num_layers: int = 12,
        num_global: int = 50,
    ):
        super().__init__()
        self.window_size = window_size
        self.patch_channels = patch_channels
        self.image_size = image_size
        self.new_spatial = image_size // window_size  # 256 // 2 = 128
        self.token_dim = patch_channels * (window_size ** 2)  # 48 * 4 = 192
        self.num_global = num_global

        # Cấu hình Longformer Attention của HuggingFace
        self.longformer_config = LongformerConfig(
            num_attention_heads=6,                     # 6 attention heads (mỗi head có dim = 192 / 6 = 32)
            num_layers=num_layers,                     # 12 layers
            hidden_size=self.token_dim,                # 192
            attention_window=[256] * num_layers,       # Cửa sổ attention cục bộ w = 256
            attention_dilation=[1] * num_layers,       # Dilation rate = 1
            attention_mode="sliding_chunks",           # Chế độ sliding window hiệu năng cao
            autoregressive=False,
        )
        # Khởi tạo lớp LongformerSelfAttention (layer_id = 0)
        self.longformer_attention = LongformerSelfAttention(
            config=self.longformer_config,
            layer_id=0,
        )

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Luồng tính toán Forward của Longformer Attention:
        Đầu vào: x có shape (Batch_size, Channels=48, Height=256, Width=256)
        Đầu ra: Tensor có shape (Batch_size, 48, 256, 256)
        """
        B, C, H, W = x.shape

        # Bước 1: Tokenize các ô 2x2 thành chuỗi token N = 16384, D = 192
        tokens = (
            x.unfold(2, self.window_size, self.window_size)
            .unfold(3, self.window_size, self.window_size)
            .contiguous()
            .view(B, C, -1, self.window_size * self.window_size)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(B, -1, self.token_dim)
        )
        N = tokens.shape[1]  # N = 16384

        # Bước 2: Tạo attention_mask (0 = local attention thông thường)
        if attention_mask is None:
            attention_mask = torch.zeros(B, N, dtype=torch.long, device=tokens.device)

        is_index_masked = attention_mask != 0

        # Bước 3: Chọn đều đặn 50 vị trí Global Attention dọc theo chuỗi token
        global_positions = torch.linspace(
            0, N - 1, steps=self.num_global, device=tokens.device
        ).long()
        is_index_global_attn = torch.zeros(B, N, dtype=torch.bool, device=tokens.device)
        is_index_global_attn[:, global_positions] = True

        # Bước 4: Gọi module LongformerSelfAttention
        attention_output = self.longformer_attention(
            tokens,
            attention_mask=attention_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=True,
            output_attentions=False,
        )

        # Bước 5: Untokenize — Tái cấu trúc chuỗi 1D về lại tensor ảnh 2D (B, 48, 256, 256)
        out = (
            attention_output[0]
            .view(
                B,
                self.new_spatial,
                self.new_spatial,
                self.patch_channels,
                self.window_size,
                self.window_size,
            )
            .permute(0, 3, 1, 4, 2, 5)
            .contiguous()
            .view(
                B,
                self.patch_channels,
                self.new_spatial * self.window_size,
                self.new_spatial * self.window_size,
            )
        )
        return out


class RegularizationBlock(nn.Module):
    """
    Khối Điều Hòa Tiên Nghiệm (Learned Regularizer R_theta) dựa trên Longformer.
    
    Cấu trúc:
    Conv1 (1 -> 48, kernel 5x5) -> ReLU -> LongformerAttentionBlock -> Conv2 (48 -> 48, kernel 5x5) -> ReLU -> Conv3 (48 -> 1, kernel 5x5)
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 5,
        patch_channels: int = 48,
        image_size: int = 256,
    ):
        super().__init__()
        padding_value = kernel_size // 2

        # Tầng tích chập 1
        self.conv1 = nn.Conv2d(
            in_channels,
            patch_channels,
            kernel_size=kernel_size,
            padding=padding_value,
        )
        nn.init.normal_(self.conv1.weight, mean=0.0, std=0.01)

        # Khối Longformer Attention
        self.longformer_attn = LongformerAttentionBlock(
            window_size=2,
            patch_channels=patch_channels,
            image_size=image_size,
        )

        # Tầng tích chập 2
        self.conv2 = nn.Conv2d(
            patch_channels,
            patch_channels,
            kernel_size=kernel_size,
            padding=padding_value,
        )
        nn.init.normal_(self.conv2.weight, mean=0.0, std=0.01)

        # Tầng tích chập 3
        self.conv3 = nn.Conv2d(
            patch_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding_value,
        )
        nn.init.normal_(self.conv3.weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.longformer_attn(x)
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class GradientFunction(nn.Module):
    """
    Một giai đoạn (Stage) Unrolling của thuật toán LEARN sử dụng Longformer.
    Công thức: g_t = alpha_t * A^T(A x_t - y) + R_{theta_t}(x_t)
    """
    def __init__(self, image_size: int = 256):
        super().__init__()
        self.regularitation_term = RegularizationBlock(image_size=image_size)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        x_t: torch.Tensor,
        y: torch.Tensor,
        forward_module: nn.Module,
        backward_module: nn.Module,
    ) -> torch.Tensor:
        data_fidelity_term = forward_module(x_t) - y
        bp_data_fidelity = backward_module(data_fidelity_term)
        reg_value = self.regularitation_term(x_t)
        gradient = self.alpha * bp_data_fidelity + reg_value
        return gradient


class LEARN_Longformer_LA(pl.LightningModule):
    """
    Mô hình LEARN_Longformer hoàn chỉnh cho bài toán Limited-Angle CT (LA-CT).
    """
    def __init__(
        self,
        n_iterations: int = 14,
        num_view: int = 64,
        num_detectors: int = 512,
        start_ang: float = -np.pi / 3,
        end_ang: float = np.pi / 3,
        input_size: int = 256,
        initial_lr: float = 1e-4,
        final_lr: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.n_iterations = n_iterations
        self.num_view = num_view
        self.num_detectors = num_detectors
        self.start_ang = start_ang
        self.end_ang = end_ang
        self.input_size = input_size
        self.initial_lr = initial_lr
        self.final_lr = final_lr

        self.gradient_list = nn.ModuleList(
            [GradientFunction(image_size=input_size) for _ in range(n_iterations)]
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

    def forward(self, x_t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Thực thi vòng lặp unrolling 14 giai đoạn"""
        for i in range(self.n_iterations):
            x_t = x_t - self.gradient_list[i](
                x_t, y, self.forward_module, self.backward_module
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

    def _get_batch_data_range(self, target: torch.Tensor):
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
