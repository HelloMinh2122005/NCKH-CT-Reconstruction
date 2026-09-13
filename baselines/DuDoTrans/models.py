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


class SinogramTransformerBlock(nn.Module):
    """
    Khối Transformer Xử Lý Trên Miền Chiếu (Sinogram Domain Transformer Block).
    
    Nguyên lý hoạt động toán học:
    Trong bài toán Limited-Angle CT, sinogram bị khuyết mất một dải góc lớn (Missing Wedge),
    khiến cho các đường sin parabol biểu diễn quỹ đạo chùm tia bị đứt đoạn nghiêm trọng.
    Khối Sinogram Transformer Block áp dụng cơ chế Multi-Head Self-Attention theo chiều góc chiếu (view dimension)
    và theo chiều cảm biến (detector dimension) để suy luận và phục hồi cấu trúc liên tục của sinogram:
        Attention(Q, K, V) = Softmax( (Q K^T) / sqrt(d_k) ) V
    
    Kích thước Tensor:
    - Đầu vào x: (Batch_size, Channels=48, Views=64, Detectors=512)
    - Reshape sang chuỗi token: (Batch_size, Views=64, Channels*Detectors hoặc Patch_dim)
    - Đầu ra: Tensor phục hồi (Batch_size, Channels=48, Views=64, Detectors=512)
    """
    def __init__(self, in_channels: int = 48, num_heads: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads

        # Tầng tích chập 1D theo chiều detector để giảm chiều đặc trưng trước khi tự chú ý
        self.detector_embed = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Multi-Head Self-Attention trên miền góc chiếu (View Attention)
        self.norm1 = nn.GroupNorm(num_groups=4, num_channels=in_channels)
        self.mha = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True)

        # Feed-Forward Network (FFN)
        self.norm2 = nn.GroupNorm(num_groups=4, num_channels=in_channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, V, D = x.shape  # (B, 48, 64, 512)
        shortcut = x

        # Bước 1: Trích xuất đặc trưng cục bộ trên các kênh detector
        feat = self.detector_embed(x)

        # Bước 2: Chuẩn bị token cho View Attention
        # Pooling đặc trưng detector để thu nhỏ kích thước tính toán attention trên 64 views:
        feat_norm = self.norm1(feat)
        # Lấy giá trị đại diện trung bình của các detector: (B, C, V, 1) -> permute thành (B, V, C)
        view_tokens = feat_norm.mean(dim=-1).permute(0, 2, 1)  # (B, V=64, C=48)

        attn_out, _ = self.mha(view_tokens, view_tokens, view_tokens)  # (B, V, C)
        attn_weight = attn_out.permute(0, 2, 1).unsqueeze(-1)  # (B, C, V, 1)

        # Điều chế lại đặc trưng sinogram ban đầu bằng trọng số attention
        feat = feat + feat * torch.sigmoid(attn_weight)

        # Bước 3: FFN và Residual Connection
        feat = feat + self.ffn(self.norm2(feat))
        out = shortcut + feat
        return out


class SinogramRestorationNet(nn.Module):
    """
    Mạng Khôi Phục Sinogram (Sinogram Inpainting & Restoration Network).
    
    Đầu vào: Sinogram góc giới hạn y (B, 1, 64, 512).
    Đầu ra: Sinogram đã được lọc và bù đắp thông tin (B, 1, 64, 512).
    """
    def __init__(self, in_channels: int = 1, feature_dim: int = 48, num_blocks: int = 3):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.blocks = nn.ModuleList([
            SinogramTransformerBlock(in_channels=feature_dim, num_heads=4)
            for _ in range(num_blocks)
        ])
        self.out_conv = nn.Conv2d(feature_dim, in_channels, kernel_size=3, padding=1)

    def forward(self, sino: torch.Tensor) -> torch.Tensor:
        feat = self.in_conv(sino)
        for block in self.blocks:
            feat = block(feat)
        delta_sino = self.out_conv(feat)
        # Phục hồi phần dư (Residual sinogram restoration): y_hat = y + delta_y
        restored_sino = sino + delta_sino
        return restored_sino


class ImageRefinementTransformerBlock(nn.Module):
    """
    Khối Transformer Tinh Chỉnh Trên Miền Ảnh (Image Domain Refinement Transformer Block).
    
    Nguyên lý:
    Sử dụng cơ chế Depthwise-Separable Convolution kết hợp Multi-Head Self-Attention cục bộ
    để khử triệt để các vệt sọc hình nón (Streaking Artifacts) do góc quét giới hạn sinh ra.
    """
    def __init__(self, dim: int = 48, num_heads: int = 4):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=4, num_channels=dim)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),  # Depthwise
            nn.Conv2d(dim, dim, kernel_size=1),                         # Pointwise
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.norm2 = nn.GroupNorm(num_groups=4, num_channels=dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x_norm = self.norm1(x)
        x_conv = self.spatial_conv(x_norm)
        x = shortcut + x_conv

        x_mlp = self.mlp(self.norm2(x))
        out = x + x_mlp
        return out


class ImageDomainNet(nn.Module):
    """
    Mạng Tinh Chỉnh Miền Ảnh (Image Domain Reconstruction Network).
    
    Đầu vào: Ghép nối giữa ảnh FBP từ sinogram phục hồi và ảnh FBP khởi tạo (B, 2, 256, 256).
    Đầu ra: Ảnh CT giải phẫu hoàn thiện không còn artifact (B, 1, 256, 256).
    """
    def __init__(self, in_channels: int = 2, feature_dim: int = 48, num_blocks: int = 4):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.blocks = nn.ModuleList([
            ImageRefinementTransformerBlock(dim=feature_dim, num_heads=4)
            for _ in range(num_blocks)
        ])
        self.out_conv = nn.Conv2d(feature_dim, 1, kernel_size=3, padding=1)

    def forward(self, img_input: torch.Tensor) -> torch.Tensor:
        feat = self.in_conv(img_input)
        for block in self.blocks:
            feat = block(feat)
        out = self.out_conv(feat)
        return out


class DuDoTrans_LA(pl.LightningModule):
    """
    Mô hình Baseline DuDoTrans hoàn chỉnh (Dual-Domain Transformer) cho Limited-Angle CT (Wang et al. 2021).
    
    Kiến trúc gồm 3 module liên kết chặt chẽ:
    1. Sinogram Domain Network (S-Net): Học phục hồi sinogram góc giới hạn (B, 1, 64, 512).
    2. Differentiable Radon Inversion Bridge (FBP Operator): Áp dụng toán tử FBP vi phân qua ODL/ASTRA
       để đưa sinogram phục hồi sang không gian ảnh sơ bộ x_bridge = FBP(y_restored).
    3. Image Domain Network (I-Net): Nhận cả x_init (FBP ban đầu) và x_bridge (FBP từ sinogram phục hồi),
       tinh chỉnh và khử các vệt artifact thiếu góc để tái tạo ảnh giải phẫu x_final.
    4. Hàm mục tiêu tối ưu kép (Dual-Domain Loss):
       L_total = MSE(x_final, x_gt) + lambda_sino * MSE(A(x_final), y)
    """
    def __init__(
        self,
        num_view: int = 64,
        num_detectors: int = 512,
        start_ang: float = -np.pi / 3,
        end_ang: float = np.pi / 3,
        input_size: int = 256,
        feature_dim: int = 48,
        lambda_sino: float = 0.1,
        initial_lr: float = 1e-4,
        final_lr: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_view = num_view
        self.num_detectors = num_detectors
        self.start_ang = start_ang
        self.end_ang = end_ang
        self.input_size = input_size
        self.lambda_sino = lambda_sino
        self.initial_lr = initial_lr
        self.final_lr = final_lr

        # Module 1: Sinogram Domain Transformer
        self.sino_net = SinogramRestorationNet(in_channels=1, feature_dim=feature_dim, num_blocks=3)

        # Module 2: Image Domain Refinement Network
        self.image_net = ImageDomainNet(in_channels=2, feature_dim=feature_dim, num_blocks=4)

        # Cầu nối biến đổi hình học chiếu (Radon Transform & FBP Operators)
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

    def forward(self, fbp_u: torch.Tensor, sino: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Luồng tính toán đa miền Dual-Domain:
        1. sino -> sino_net -> sino_restored
        2. sino_restored -> FBP -> x_bridge
        3. concat(fbp_u, x_bridge) -> image_net -> delta_img
        4. x_final = x_bridge + delta_img
        """
        # Bước 1: Phục hồi trên miền Sinogram
        sino_restored = self.sino_net(sino)

        # Bước 2: Cầu nối Radon nghịch đảo đưa sang miền ảnh
        x_bridge = self.backward_module(sino_restored)

        # Bước 3: Ghép nối đặc trưng hai miền tại không gian ảnh
        img_input = torch.cat([fbp_u, x_bridge], dim=1)  # (B, 2, 256, 256)

        # Bước 4: Tinh chỉnh miền ảnh triệt tiêu vệt sọc
        delta_img = self.image_net(img_input)
        x_final = x_bridge + delta_img

        return x_final, sino_restored

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

        x_final, sino_restored = self.forward(fbp_u, sino_noisy)

        # Mất mát trên miền ảnh: MSE giữa ảnh tái tạo x_final và ảnh gốc phantom
        loss_img = F.mse_loss(phantom, x_final)

        # Ràng buộc độ khớp dữ liệu chiếu: Chiếu thẳng x_final và so sánh với sinogram đo đạc sino_noisy
        sino_proj = self.forward_module(x_final)
        loss_sino = F.mse_loss(sino_noisy, sino_proj)

        # Tổng hàm mất mát đa miền
        total_loss = loss_img + self.lambda_sino * loss_sino

        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss_img", loss_img, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_loss_sino", loss_sino, on_step=False, on_epoch=True, prog_bar=False)
        return total_loss

    def validation_step(self, val_batch, batch_idx):
        phantom, fbp_u, sino_noisy = val_batch
        x_final, sino_restored = self.forward(fbp_u, sino_noisy)
        loss = F.mse_loss(phantom, x_final)

        data_range = self._get_batch_data_range(phantom)
        ssim_val = structural_similarity_index_measure(x_final, phantom, data_range=data_range)
        psnr_val = peak_signal_noise_ratio(x_final, phantom, data_range=data_range)
        rmse_val = self.rmse(phantom, x_final)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_ssim", ssim_val, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_psnr", psnr_val, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_rmse", rmse_val, on_step=False, on_epoch=True, prog_bar=False)

        self.grid = torchvision.utils.make_grid(x_final.detach().clamp(min=0.0))
        return {
            "val_loss": loss,
            "val_ssim": ssim_val,
            "val_psnr": psnr_val,
            "val_rmse": rmse_val,
        }

    def test_step(self, batch, batch_idx):
        phantom, fbp_u, sino_noisy = batch
        x_final, sino_restored = self.forward(fbp_u, sino_noisy)
        loss = F.mse_loss(phantom, x_final)

        data_range = self._get_batch_data_range(phantom)
        ssim_val = structural_similarity_index_measure(x_final, phantom, data_range=data_range)
        psnr_val = peak_signal_noise_ratio(x_final, phantom, data_range=data_range)
        rmse_val = self.rmse(phantom, x_final)

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
