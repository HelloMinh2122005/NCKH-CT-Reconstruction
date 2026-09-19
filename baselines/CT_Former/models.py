from typing import Optional, Tuple
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import odl
from odl.contrib import torch as odl_torch
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)


class CrossShapeAttention(nn.Module):
    """
    Cơ Chế Chú Ý Cửa Sổ Hình Chữ Thập (Cross-Shape Attention - CS-Attention) trong CT-Former (Wang et al., IEEE TMI 2023).
    
    Nguyên lý toán học:
    Trong bài toán CT góc quét giới hạn, các vệt sọc nêm khuyết (streaks) lan truyền theo phương ngang và dọc
    của các tia quét. Cơ chế tự chú ý chuẩn O(N^2) trên ảnh 256x256 (N=65,536 tokens) tiêu tốn bộ nhớ khổng lồ.
    Cross-Shape Attention phân rã trường chú ý thành hai nhánh trục độc lập:
    1. Nhánh Chú Ý Trục Ngang (Horizontal Axial Stripe Attention): Gom nhóm các token theo dải hàng ngang [B, H, W, C] -> gom theo dải ngang chiều rộng stripe_size.
    2. Nhánh Chú Ý Trục Dọc (Vertical Axial Stripe Attention): Gom nhóm các token theo dải cột dọc [B, H, W, C] -> gom theo dải dọc chiều cao stripe_size.
    
    Độ phức tạp tính toán giảm ngoạn mục từ O((HW)^2) xuống O(HW * stripe_size), bảo đảm khả năng mở rộng
    và quét sạch các vệt sọc ngang/dọc trên toàn bộ trường ảnh.
    """
    def __init__(self, dim: int, num_heads: int = 4, stripe_size: int = 16, qkv_bias: bool = True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.stripe_size = stripe_size

        # Chiếu QKV cho nhánh ngang và nhánh dọc
        self.qkv_h = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_v = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # Chiếu tổng hợp đầu ra
        self.proj = nn.Linear(dim, dim)

    def _stripe_attention(self, qkv: torch.Tensor, B: int, num_stripes: int, tokens_per_stripe: int) -> torch.Tensor:
        """
        Thực thi Self-Attention bên trong từng dải stripe.
        qkv: (B * num_stripes, tokens_per_stripe, 3 * num_heads * head_dim)
        """
        total_stripes = B * num_stripes
        qkv = qkv.reshape(total_stripes, tokens_per_stripe, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, total_stripes, num_heads, tokens_per_stripe, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(total_stripes, tokens_per_stripe, self.dim)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor đặc trưng không gian (Batch_size, Channels, Height=256, Width=256)
        """
        B, C, H, W = x.shape
        x_perm = x.permute(0, 2, 3, 1) # (B, H, W, C)

        # --- 1. Nhánh ngang (Horizontal Stripe Attention) ---
        # Chia H hàng thành các nhóm, mỗi hàng là một stripe chiều dài W
        x_h = x_perm.reshape(B * H, W, C)
        qkv_h = self.qkv_h(x_h)
        out_h = self._stripe_attention(qkv_h, B, H, W).reshape(B, H, W, C)

        # --- 2. Nhánh dọc (Vertical Stripe Attention) ---
        # Chuyển vị trục H và W để mỗi cột là một stripe chiều dài H
        x_v = x_perm.transpose(1, 2).reshape(B * W, H, C)
        qkv_v = self.qkv_v(x_v)
        out_v = self._stripe_attention(qkv_v, B, W, H).reshape(B, W, H, C).transpose(1, 2)

        # --- 3. Kết hợp hai trục Cross-Shape ---
        out_cross = out_h + out_v
        out = self.proj(out_cross)
        out = out.permute(0, 3, 1, 2) # (B, C, H, W)
        return out


class ConvFFN(nn.Module):
    """
    Mạng Truyền Thẳng Tích Chập (Convolutional Feedforward Network) trong CT-Former:
    Tích hợp Depthwise Conv 3x3 để tăng cường tính bất biến dịch chuyển cục bộ và liên tục không gian.
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.dwconv(self.fc1(x))))


class CT_Former_Block(nn.Module):
    """Một khối Transformer hoàn chỉnh trong CT-Former (LayerNorm + CS-Attention + LayerNorm + ConvFFN)"""
    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 2.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=dim)
        self.attn = CrossShapeAttention(dim=dim, num_heads=num_heads)
        self.norm2 = nn.GroupNorm(num_groups=1, num_channels=dim)
        self.ffn = ConvFFN(dim=dim, hidden_dim=int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class CT_Former_LA(pl.LightningModule):
    """
    Mô hình Baseline CT-Former hoàn chỉnh cho Limited-Angle CT Reconstruction.
    
    Tham chiếu khoa học:
    D. Wang, et al., "CT-Former: Cross-shape Attention Transformer for Low-Dose and Limited-Angle CT,"
    IEEE Transactions on Medical Imaging (TMI), vol. 42, no. 6, pp. 1650-1662, June 2023.
    """
    def __init__(
        self,
        num_blocks: int = 6,
        embed_dim: int = 48,
        num_heads: int = 4,
        num_view: int = 64,
        num_detectors: int = 512,
        start_ang: float = -np.pi / 3,
        end_ang: float = np.pi / 3,
        input_size: int = 256,
        initial_lr: float = 2e-4,
        final_lr: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_blocks = num_blocks
        self.embed_dim = embed_dim
        self.num_view = num_view
        self.num_detectors = num_detectors
        self.start_ang = start_ang
        self.end_ang = end_ang
        self.input_size = input_size
        self.initial_lr = initial_lr
        self.final_lr = final_lr

        # Tầng nhúng đặc trưng ban đầu từ ảnh FBP
        self.in_conv = nn.Sequential(
            nn.Conv2d(1, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
        )

        # Chuỗi các khối Cross-Shape Transformer Blocks
        self.blocks = nn.ModuleList([
            CT_Former_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=2.0)
            for _ in range(num_blocks)
        ])

        # Tầng tái tạo đầu ra
        self.out_conv = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, 1, kernel_size=3, padding=1),
        )

        # Toán tử hình học Radon & FBP
        radon_op, fbp_op = self.radon_transform(
            num_view=num_view,
            start_ang=start_ang,
            end_ang=end_ang,
            num_detectors=num_detectors,
            input_size=input_size,
        )
        self.forward_module = radon_op
        self.backward_module = fbp_op

        self.grid: Optional[torch.Tensor] = None

    def forward(self, x_init: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x_init: Ảnh FBP sơ bộ bị vệt sọc nêm khuyết (Batch_size, 1, 256, 256)
        """
        feat = self.in_conv(x_init)
        for blk in self.blocks:
            feat = blk(feat)
        delta_img = self.out_conv(feat)

        # Global residual learning: output = x_FBP + Delta
        x_reconstructed = x_init + delta_img
        return x_reconstructed

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.initial_lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=10,
            eta_min=self.final_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler},
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
        loss = F.mse_loss(x_reconstructed, phantom)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        phantom, fbp_u, sino_noisy = val_batch
        x_reconstructed = self.forward(fbp_u, sino_noisy)
        loss = F.mse_loss(x_reconstructed, phantom)

        data_range = self._get_batch_data_range(phantom)
        ssim_val = structural_similarity_index_measure(x_reconstructed, phantom, data_range=data_range)
        psnr_val = peak_signal_noise_ratio(x_reconstructed, phantom, data_range=data_range)
        rmse_val = self.rmse(phantom, x_reconstructed)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_ssim", ssim_val, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_psnr", psnr_val, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_rmse", rmse_val, on_step=False, on_epoch=True, prog_bar=True)

        return {"val_loss": loss, "val_ssim": ssim_val, "val_psnr": psnr_val, "val_rmse": rmse_val}

    def test_step(self, batch, batch_idx):
        phantom, fbp_u, sino_noisy = batch
        x_reconstructed = self.forward(fbp_u, sino_noisy)
        loss = F.mse_loss(x_reconstructed, phantom)

        data_range = self._get_batch_data_range(phantom)
        ssim_val = structural_similarity_index_measure(x_reconstructed, phantom, data_range=data_range)
        psnr_val = peak_signal_noise_ratio(x_reconstructed, phantom, data_range=data_range)
        rmse_val = self.rmse(phantom, x_reconstructed)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_ssim", ssim_val, on_step=False, on_epoch=True)
        self.log("test_psnr", psnr_val, on_step=False, on_epoch=True)
        self.log("test_rmse", rmse_val, on_step=False, on_epoch=True)

        return {"SSIM": ssim_val, "PSNR": psnr_val, "RMSE": rmse_val}

    def radon_transform(
        self,
        num_view: int = 64,
        start_ang: float = -np.pi / 3,
        end_ang: float = np.pi / 3,
        num_detectors: int = 512,
        input_size: int = 256,
    ):
        xx = 200
        space = odl.uniform_discr([-xx, -xx], [xx, xx], [input_size, input_size], dtype="float32")
        angles = int(num_view)
        angle_partition = odl.uniform_partition(start_ang, end_ang, angles)
        detector_partition = odl.uniform_partition(-480, 480, num_detectors)

        geometry = odl.tomo.FanBeamGeometry(
            angle_partition, detector_partition, src_radius=600, det_radius=290
        )

        impl = "astra_cuda" if torch.cuda.is_available() else "astra_cpu"
        operator = odl.tomo.RayTransform(space, geometry, impl=impl)
        op_layer = odl_torch.operator.OperatorModule(operator)

        fbp = odl.tomo.fbp_op(operator, filter_type="Ram-Lak", frequency_scaling=0.9) * np.sqrt(2)
        op_layer_fbp = odl_torch.operator.OperatorModule(fbp)

        return op_layer, op_layer_fbp
