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
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)

# Thử import lớp DilatedAttention từ các nguồn khác nhau
try:
    from long_net import DilatedAttention
except ImportError:
    try:
        from baselines.LEARN_LongNet.long_net import DilatedAttention
    except ImportError:
        try:
            from .long_net import DilatedAttention
        except ImportError:
            DilatedAttention = None


class LongNetAttentionBlock(nn.Module):
    """
    Khối LongNet Attention Block cho bài toán Tái tạo ảnh CT.
    
    Nguyên lý hoạt động:
    1. Chia tensor đặc trưng (B, 48, 256, 256) thành chuỗi các token kích thước window_size x window_size.
    2. Áp dụng cơ chế Dilated Attention đa tỷ lệ (Multi-Scale Dilated Attention) để tiếp nhận chuỗi dài O(N).
    3. Tái cấu trúc chuỗi 1D về không gian 2D ban đầu.
    
    Tham số khởi tạo:
    - window_size (int): Kích thước mỗi patch token (mặc định = 2, hoặc 16 tùy cấu hình).
    - patch_channels (int): Số kênh đặc trưng của tensor (mặc định = 48).
    - image_size (int): Kích thước không gian của ảnh (mặc định = 256).
    """
    def __init__(
        self,
        window_size: int = 2,
        patch_channels: int = 48,
        image_size: int = 256,
    ):
        super().__init__()
        self.window_size = window_size
        self.patch_channels = patch_channels
        self.image_size = image_size
        self.new_spatial = image_size // window_size
        self.token_dim = patch_channels * (window_size ** 2)

        # Khởi tạo lớp Dilated Attention
        if DilatedAttention is not None:
            self.longnet_attention = DilatedAttention(
                dim=self.token_dim,
                heads=4,
                dilation_rate=1,
                segment_size=16,
                dropout=0.1,
                qk_norm=True,
            )
        else:
            self.longnet_attention = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Luồng tính toán Forward của LongNet Attention Block:
        Đầu vào: x có shape (Batch_size, Channels=48, Height=256, Width=256)
        Đầu ra: Tensor có shape (Batch_size, 48, 256, 256)
        """
        batch_size, channels, height, width = x.shape

        # Bước 1: Tokenize — Gom các patch thành chuỗi token
        tokens = (
            x.unfold(2, self.window_size, self.window_size)
            .unfold(3, self.window_size, self.window_size)
            .contiguous()
            .view(batch_size, channels, -1, self.window_size * self.window_size)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, -1, self.token_dim)
        )

        # Bước 2: Thực thi Dilated Attention
        attention_output = self.longnet_attention(tokens)

        # Bước 3: Untokenize — Tái cấu trúc chuỗi 1D về lại tensor ảnh 2D
        tokens_reshaped = attention_output.view(
            batch_size,
            self.new_spatial,
            self.new_spatial,
            self.patch_channels,
            self.window_size,
            self.window_size,
        )
        tokens_reshaped = tokens_reshaped.permute(0, 3, 1, 4, 2, 5).contiguous()
        reconstructed = tokens_reshaped.view(
            batch_size,
            self.patch_channels,
            self.new_spatial * self.window_size,
            self.new_spatial * self.window_size,
        )
        return reconstructed


class RegularizationBlock(nn.Module):
    """
    Khối Điều Hòa Tiên Nghiệm (Learned Regularizer R_theta) dựa trên LongNet.
    
    Cấu trúc:
    Conv1 (1 -> 48, kernel 5x5) -> ReLU -> LongNetAttentionBlock -> Conv2 (48 -> 48, kernel 5x5) -> ReLU -> Conv3 (48 -> 1, kernel 5x5)
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 5,
        patch_channels: int = 48,
        image_size: int = 256,
        window_size: int = 2,
    ):
        super().__init__()
        padding_value = kernel_size // 2

        self.conv1 = nn.Conv2d(
            in_channels,
            patch_channels,
            kernel_size=kernel_size,
            padding=padding_value,
        )
        nn.init.normal_(self.conv1.weight, mean=0.0, std=0.01)

        self.longnet_attn = LongNetAttentionBlock(
            window_size=window_size,
            patch_channels=patch_channels,
            image_size=image_size,
        )

        self.conv2 = nn.Conv2d(
            patch_channels,
            patch_channels,
            kernel_size=kernel_size,
            padding=padding_value,
        )
        nn.init.normal_(self.conv2.weight, mean=0.0, std=0.01)

        self.conv3 = nn.Conv2d(
            patch_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding_value,
        )
        nn.init.normal_(self.conv3.weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.longnet_attn(x)
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class GradientFunction(nn.Module):
    """
    Một giai đoạn (Stage) Unrolling của thuật toán LEARN sử dụng LongNet.
    Công thức: g_t = alpha_t * A^T(A x_t - y) + R_{theta_t}(x_t)
    """
    def __init__(self, image_size: int = 256, window_size: int = 2):
        super().__init__()
        self.regularitation_term = RegularizationBlock(
            image_size=image_size,
            window_size=window_size,
        )
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


class LEARN_LongNet_LA(pl.LightningModule):
    """
    Mô hình LEARN_LongNet hoàn chỉnh cho bài toán Limited-Angle CT (LA-CT).
    """
    def __init__(
        self,
        n_iterations: int = 14,
        num_view: int = 64,
        num_detectors: int = 512,
        start_ang: float = -np.pi / 3,
        end_ang: float = np.pi / 3,
        input_size: int = 256,
        window_size: int = 2,
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
        self.window_size = window_size
        self.initial_lr = initial_lr
        self.final_lr = final_lr

        self.gradient_list = nn.ModuleList(
            [
                GradientFunction(image_size=input_size, window_size=window_size)
                for _ in range(n_iterations)
            ]
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
