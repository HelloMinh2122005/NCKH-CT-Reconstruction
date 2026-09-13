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


class WindowAttention(nn.Module):
    """
    Cơ chế Tự Chú Ý Dựa Trên Cửa Sổ (Window-based Multi-Head Self-Attention - W-MSA).
    
    Nguyên lý hoạt động toán học:
    Thay vì tính ma trận tương quan chú ý trên toàn bộ N = H x W = 256 x 256 = 65.536 điểm ảnh (gây bùng nổ O(N^2) bộ nhớ),
    ảnh đặc trưng được chia thành các cửa sổ con kích thước M x M (ví dụ M = 8, tức mỗi cửa sổ có 64 điểm).
    Self-Attention chỉ được tính toán cục bộ bên trong từng cửa sổ:
        Attention(Q, K, V) = Softmax(Q K^T / sqrt(d) + B_rel) V
    trong đó B_rel là ma trận vị trí tương đối (Relative Position Bias) mã hóa hình học không gian.
    
    Kích thước Tensor:
    - Đầu vào x: (num_windows * B, N_win=M*M, C)
    - Q, K, V: (num_windows * B, num_heads, N_win, C // num_heads)
    - Đầu ra: (num_windows * B, N_win, C)
    """
    def __init__(self, dim: int, window_size: int = 8, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Bảng mã hóa vị trí tương đối Relative Position Bias Table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Tạo chỉ số vị trí tương đối cho mọi cặp điểm trong cửa sổ MxM
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # (2, M, M)
        coords_flatten = torch.flatten(coords, 1)  # (2, M*M)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, M*M, M*M)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (M*M, M*M, 2)
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # (M*M, M*M)
        self.register_buffer("relative_position_index", relative_position_index)

        # Tầng biến đổi tuyến tính chiếu Q, K, V
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Mỗi tensor có shape: (B_, num_heads, N, head_dim)

        # Tính điểm tương đồng Scaled Dot-Product Attention
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B_, num_heads, N, N)

        # Cộng Relative Position Bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (num_heads, N, N)
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        return out


class SwinTransformerBlock(nn.Module):
    """
    Một khối Swin Transformer cho nhánh Non-local của RegFormer.
    Bao gồm: LayerNorm -> WindowAttention (W-MSA) -> Residual -> LayerNorm -> MLP (Mạng nơ-ron đa tầng) -> Residual
    """
    def __init__(self, dim: int = 48, window_size: int = 8, num_heads: int = 4, mlp_ratio: float = 2.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim=dim, window_size=window_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Đầu vào: x có shape (B, C=48, H=256, W=256)
        Đầu ra: Tensor có cùng shape (B, C=48, H=256, W=256)
        """
        B, C, H, W = x.shape
        shortcut = x
        
        # Chuyển kênh về cuối để chuẩn hóa LayerNorm: (B, H, W, C)
        x_perm = x.permute(0, 2, 3, 1).contiguous()
        x_norm = self.norm1(x_perm)

        # Phân chia thành các cửa sổ không gian không chồng lấn:
        # Số cửa sổ theo H = H // window_size = 256 // 8 = 32
        # Số cửa sổ theo W = W // window_size = 256 // 8 = 32 -> Tổng cộng 1.024 cửa sổ
        ws = self.window_size
        x_windows = x_norm.view(B, H // ws, ws, W // ws, ws, C)
        x_windows = x_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws, C)  # (B*num_windows, ws*ws, C)

        # Tính Window Attention
        attn_windows = self.attn(x_windows)  # (B*num_windows, ws*ws, C)

        # Ghép nối các cửa sổ trở lại ảnh ban đầu (Merge Windows)
        attn_windows = attn_windows.view(B, H // ws, W // ws, ws, ws, C)
        x_merge = attn_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)

        # Kết nối tắt 1
        x_perm = x_perm + x_merge

        # Khối MLP với chuẩn hóa LayerNorm 2
        x_mlp = self.mlp(self.norm2(x_perm))
        x_perm = x_perm + x_mlp

        # Chuyển kênh về dạng chuẩn PyTorch NCHW: (B, C, H, W)
        out = x_perm.permute(0, 3, 1, 2).contiguous()
        return out


class LocalCNNBranch(nn.Module):
    """
    Nhánh Trích Xuất Đặc Trưng Cục Bộ (Local Branch) trong khối Điều Hòa của RegFormer.
    
    Cấu trúc:
    - Sử dụng các lớp tích chập sâu Residual Convolution 3x3 với hàm kích hoạt LeakyReLU.
    - Duy trì các đặc trưng tần số cao, các đường viền giải phẫu, các vách mô hạch và cấu trúc xương nhỏ mịn.
    """
    def __init__(self, in_channels: int = 48, out_channels: int = 48):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class RegFormerRegularizationBlock(nn.Module):
    """
    Khối Điều Hòa Tiên Nghiệm Kết Hợp Cục Bộ - Toàn Cục (Local-Nonlocal Regularizer) của RegFormer (Xia et al. 2023).
    
    Kiến trúc hoạt động:
    1. Shallow Feature Extraction: Conv 3x3 đưa ảnh 1 kênh lên không gian đặc trưng C=48 kênh.
    2. Local Branch: Khối LocalCNNBranch trích xuất độ sắc nét viền cạnh cục bộ.
    3. Non-Local Branch: Khối SwinTransformerBlock tính toán tương quan toàn cục đa cửa sổ, triệt tiêu artifact vệt sọc sải dài.
    4. Feature Fusion: Ghép nối (Concatenation) hai luồng đặc trưng qua Conv 1x1 để điều phối tỷ trọng thông tin.
    5. Reconstruction Projection: Conv 3x3 chiếu về 1 kênh ảnh hiệu chỉnh gradient.
    
    Tổng số tham số toàn mạng unrolling 14 stages đạt xấp xỉ ~2.7M tham số (khớp chuẩn bài báo MVA của Thành).
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, feature_dim: int = 48, window_size: int = 8):
        super().__init__()
        self.shallow_conv = nn.Conv2d(in_channels, feature_dim, kernel_size=3, padding=1)
        nn.init.normal_(self.shallow_conv.weight, mean=0.0, std=0.01)

        # Nhánh Cục Bộ (Local Branch)
        self.local_branch = LocalCNNBranch(in_channels=feature_dim, out_channels=feature_dim)

        # Nhánh Toàn Cục (Non-Local Branch)
        self.nonlocal_branch = SwinTransformerBlock(dim=feature_dim, window_size=window_size, num_heads=4)

        # Tầng dung hợp đặc trưng Fusion Layer
        self.fusion = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(feature_dim, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_shallow = F.leaky_relu(self.shallow_conv(x), negative_slope=0.2)
        f_local = self.local_branch(f_shallow)
        f_nonlocal = self.nonlocal_branch(f_shallow)

        f_cat = torch.cat([f_local, f_nonlocal], dim=1)  # (B, 96, 256, 256)
        out = self.fusion(f_cat)  # (B, 1, 256, 256)
        return out


class RegFormerGradientFunction(nn.Module):
    """
    Một giai đoạn mở cuộn (Stage) của mạng RegFormer:
    g_t = alpha_t * A^T(A x_t - y) + RegFormerRegularizer(x_t)
    """
    def __init__(self, feature_dim: int = 48, window_size: int = 8):
        super().__init__()
        self.regularizer = RegFormerRegularizationBlock(feature_dim=feature_dim, window_size=window_size)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        x_t: torch.Tensor,
        y: torch.Tensor,
        forward_module: nn.Module,
        backward_module: nn.Module,
    ) -> torch.Tensor:
        data_fidelity = forward_module(x_t) - y
        bp_fidelity = backward_module(data_fidelity)
        reg_val = self.regularizer(x_t)
        return self.alpha * bp_fidelity + reg_val


class RegFormer_LA(pl.LightningModule):
    """
    Mô hình Baseline RegFormer hoàn chỉnh cho bài toán Limited-Angle CT (LA-CT).
    
    Đặc điểm:
    1. Kiến trúc mở cuộn Unrolling K = 14 stages.
    2. Mỗi stage sở hữu khối Local-Nonlocal Regularizer (Swin Transformer + CNN) độc lập.
    3. Tích hợp toán tử Fan-Beam ODL/ASTRA cho góc quét 120° ([-pi/3, +pi/3], 64 views, 512 detectors).
    4. Giám sát PSNR, SSIM, RMSE qua từng epoch.
    """
    def __init__(
        self,
        n_iterations: int = 14,
        num_view: int = 64,
        num_detectors: int = 512,
        start_ang: float = -np.pi / 3,
        end_ang: float = np.pi / 3,
        input_size: int = 256,
        feature_dim: int = 48,
        window_size: int = 8,
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
            [RegFormerGradientFunction(feature_dim=feature_dim, window_size=window_size) for _ in range(n_iterations)]
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
            batch_max = batch_min + torch.tensor(1e-8, device=target.device, dtype=target.dtype)
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
        ssim_val = structural_similarity_index_measure(x_reconstructed, phantom, data_range=data_range)
        psnr_val = peak_signal_noise_ratio(x_reconstructed, phantom, data_range=data_range)
        rmse_val = self.rmse(phantom, x_reconstructed)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_ssim", ssim_val, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_psnr", psnr_val, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_rmse", rmse_val, on_step=False, on_epoch=True, prog_bar=False)

        self.grid = torchvision.utils.make_grid(x_reconstructed.detach().clamp(min=0.0))
        return {
            "val_loss": loss,
            "val_ssim": ssim_val,
            "val_psnr": psnr_val,
            "val_rmse": rmse_val,
        }

    def test_step(self, batch, batch_idx):
        phantom, fbp_u, sino_noisy = batch
        x_reconstructed = self.forward(fbp_u, sino_noisy)
        loss = F.mse_loss(phantom, x_reconstructed)

        data_range = self._get_batch_data_range(phantom)
        ssim_val = structural_similarity_index_measure(x_reconstructed, phantom, data_range=data_range)
        psnr_val = peak_signal_noise_ratio(x_reconstructed, phantom, data_range=data_range)
        rmse_val = self.rmse(phantom, x_reconstructed)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_ssim", ssim_val, on_step=False, on_epoch=True)
        self.log("test_psnr", psnr_val, on_step=False, on_epoch=True)
        self.log("test_rmse", rmse_val, on_step=False, on_epoch=True)

        return {
            "SSIM": ssim_val,
            "PSNR": psnr_val,
            "RMSE": rmse_val,
        }

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
