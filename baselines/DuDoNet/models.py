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


class SinogramInpaintingNet(nn.Module):
    """
    Mạng Khôi Phục Miền Sinogram (Sinogram Inpainting Network) trong DuDoNet (Lin et al., CVPR 2019).
    
    Nguyên lý hoạt động:
    Dữ liệu chiếu sinogram y có kích thước (Batch_size, 1, num_view=64, num_detectors=512).
    Do góc quét bị giới hạn (LA-CT), sinogram thiếu hụt trầm trọng thông tin hình học.
    Mạng sử dụng kiến trúc U-Net 2D đa tầng với dilated convolution để học cách nội suy mượt mà
    các góc chiếu bị khuyết dựa trên tính liên tục của đường sinogram (sinusoidal curves):
        hat_y = y + M_sino(y)
    """
    def __init__(self, in_channels: int = 1, base_channels: int = 32):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1)

        # Bottleneck với Dilated Conv mở rộng trường tiếp nhận theo trục góc chiếu
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y: (B, 1, 64, 512)
        e1 = self.enc1(y)                  # (B, 32, 64, 512)
        d1 = self.down1(e1)                # (B, 64, 32, 256)

        e2 = self.enc2(d1)                 # (B, 64, 32, 256)
        d2 = self.down2(e2)                # (B, 128, 16, 128)

        b = self.bottleneck(d2)            # (B, 128, 16, 128)

        u2 = self.up2(b)                   # (B, 64, 32, 256)
        c2 = self.dec2(torch.cat([u2, e2], dim=1)) # (B, 64, 32, 256)

        u1 = self.up1(c2)                  # (B, 32, 64, 512)
        delta_y = self.dec1(torch.cat([u1, e1], dim=1)) # (B, 1, 64, 512)

        # Residual connection
        return y + delta_y


class ImageRefinementNet(nn.Module):
    """
    Mạng Tinh Chỉnh Miền Ảnh (Image Refinement Network) trong DuDoNet.
    
    Sau khi ảnh CT sơ bộ x_0 được tái tạo từ sinogram đã nội suy thông qua toán tử FBP,
    mạng này tiếp tục loại bỏ các vệt sọc nêm khuyết (missing wedge streaks) còn sót lại
    và bảo toàn chi tiết ranh giới giải phẫu mô mềm:
        x_final = x_0 + M_img(x_0)
    """
    def __init__(self, in_channels: int = 1, base_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class DuDoNet_LA(pl.LightningModule):
    """
    Mô hình Baseline DuDoNet (Dual-Domain Network) cho Limited-Angle CT.
    
    Tham chiếu khoa học:
    W. Lin, et al., "DuDoNet: Dual Domain Network for CT Metal Artifact Reduction and Reconstruction,"
    IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.
    
    Đặc điểm:
    1. Cơ chế khôi phục đa miền: Kết hợp đồng thời mạng phục hồi Sinogram và mạng tinh chỉnh Ảnh.
    2. Cầu nối vật lý: Toán tử FBP khả vi chuyển đổi mượt mà giữa 2 miền dữ liệu.
    3. Hàm mất mát đa miền: L = L_img + gamma * L_sino (gamma = 0.1).
    """
    def __init__(
        self,
        num_view: int = 64,
        num_detectors: int = 512,
        start_ang: float = -np.pi / 3,
        end_ang: float = np.pi / 3,
        input_size: int = 256,
        sino_loss_weight: float = 0.1,
        initial_lr: float = 2e-4,
        final_lr: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_view = num_view
        self.num_detectors = num_detectors
        self.start_ang = start_ang
        self.end_ang = end_ang
        self.input_size = input_size
        self.sino_loss_weight = sino_loss_weight
        self.initial_lr = initial_lr
        self.final_lr = final_lr

        # Khối miền Sinogram
        self.sino_net = SinogramInpaintingNet(in_channels=1, base_channels=32)

        # Khối miền Ảnh
        self.img_net = ImageRefinementNet(in_channels=1, base_channels=32)

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

    def forward(self, x_init: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Luồng truyền xuôi của DuDoNet:
        1. Khôi phục sinogram: y_hat = SinoNet(y)
        2. Chuyển đổi sang miền ảnh: x_sino = FBP(y_hat)
        3. Tinh chỉnh miền ảnh: x_final = ImgNet(x_sino)
        """
        y_restored = self.sino_net(y)
        x_from_sino = self.backward_module(y_restored)
        x_final = self.img_net(x_from_sino)
        return x_final, y_restored

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.initial_lr)
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
        x_final, y_restored = self.forward(fbp_u, sino_noisy)

        # Mất mát miền ảnh
        loss_img = F.mse_loss(x_final, phantom)
        # Mất mát miền sinogram (chiếu phantom sang miền sinogram làm GT)
        with torch.no_grad():
            sino_gt = self.forward_module(phantom)
        loss_sino = F.mse_loss(y_restored, sino_gt)

        total_loss = loss_img + self.sino_loss_weight * loss_sino

        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss_img", loss_img, on_step=False, on_epoch=True)
        self.log("train_loss_sino", loss_sino, on_step=False, on_epoch=True)
        return total_loss

    def validation_step(self, val_batch, batch_idx):
        phantom, fbp_u, sino_noisy = val_batch
        x_final, _ = self.forward(fbp_u, sino_noisy)
        loss = F.mse_loss(x_final, phantom)

        data_range = self._get_batch_data_range(phantom)
        ssim_val = structural_similarity_index_measure(x_final, phantom, data_range=data_range)
        psnr_val = peak_signal_noise_ratio(x_final, phantom, data_range=data_range)
        rmse_val = self.rmse(phantom, x_final)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_ssim", ssim_val, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_psnr", psnr_val, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_rmse", rmse_val, on_step=False, on_epoch=True, prog_bar=True)

        return {"val_loss": loss, "val_ssim": ssim_val, "val_psnr": psnr_val, "val_rmse": rmse_val}

    def test_step(self, batch, batch_idx):
        phantom, fbp_u, sino_noisy = batch
        x_final, _ = self.forward(fbp_u, sino_noisy)
        loss = F.mse_loss(x_final, phantom)

        data_range = self._get_batch_data_range(phantom)
        ssim_val = structural_similarity_index_measure(x_final, phantom, data_range=data_range)
        psnr_val = peak_signal_noise_ratio(x_final, phantom, data_range=data_range)
        rmse_val = self.rmse(phantom, x_final)

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
