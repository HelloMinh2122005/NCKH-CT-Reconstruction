from typing import Optional, Tuple, List
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


class LearnableSoftThresholding(nn.Module):
    """
    Toán tử Ngưỡng Co Rút Mềm Học Được (Learnable Soft-Thresholding Operator) trong FISTA-Net:
        S_tau(z) = sgn(z) * max(|z| - tau, 0)
    trong đó tau > 0 là ngưỡng co rút được tham số hóa qua hàm Softplus để bảo đảm tính không âm nghiêm ngặt.
    """
    def __init__(self, num_features: int, init_tau: float = 0.01):
        super().__init__()
        # Tham số ngưỡng tau riêng biệt cho từng kênh đặc trưng
        self.raw_tau = nn.Parameter(torch.ones(1, num_features, 1, 1) * init_tau)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        tau = F.softplus(self.raw_tau)
        return torch.sign(z) * F.relu(torch.abs(z) - tau)


class FISTA_StageBlock(nn.Module):
    """
    Một giai đoạn mở cuộn (Stage / Iteration) của thuật toán FISTA-Net (Xiang et al., IEEE TMI 2021).
    
    Quy trình tính toán biến phân tại stage k:
    1. Bước quán tính Nesterov (Momentum):
       v_k = x_{k-1} + beta_k * (x_{k-1} - x_{k-2})
    2. Bước Gradient Descent độ khớp dữ liệu (Data Fidelity Gradient):
       u_k = v_k - alpha_k * A^T(A v_k - y)
    3. Bước Chiếu Tiên Nghiệm qua Toán Tử Ngưỡng Co Rút (Proximal Mapping with Thresholding):
       feat = Conv_in(u_k)
       feat_thresh = SoftThreshold(feat, tau_k)
       x_k = u_k + Conv_out(feat_thresh)
    
    Kích thước Tensor:
    - x_{k-1}, x_{k-2}: (Batch_size, 1, 256, 256)
    - y: (Batch_size, 1, 64, 512)
    - feat: (Batch_size, 32, 256, 256)
    - x_k: (Batch_size, 1, 256, 256)
    """
    def __init__(self, in_channels: int = 1, hidden_channels: int = 32):
        super().__init__()
        # Bước nhảy gradient alpha học được
        self.alpha = nn.Parameter(torch.tensor(0.1))
        
        # Hệ số động lượng Nesterov beta học được (khởi tạo ~0.5)
        self.beta = nn.Parameter(torch.tensor(0.5))

        # Tầng biến đổi xuôi vào không gian đặc trưng thưa (Analysis Transform)
        self.conv_in1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=True)
        self.conv_in2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(hidden_channels)

        # Toán tử ngưỡng mềm
        self.threshold = LearnableSoftThresholding(num_features=hidden_channels, init_tau=0.01)

        # Tầng biến đổi ngược về không gian ảnh (Synthesis Transform)
        self.conv_out1 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=True)
        self.conv_out2 = nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1, bias=True)
        self.bn_out = nn.BatchNorm2d(hidden_channels)

    def forward(
        self,
        x_prev: torch.Tensor,
        x_prev2: torch.Tensor,
        y: torch.Tensor,
        forward_op: nn.Module,
        backward_op: nn.Module,
    ) -> torch.Tensor:
        # 1. Bước quán tính Nesterov
        v_k = x_prev + self.beta * (x_prev - x_prev2)

        # 2. Bước gradient dữ liệu: u_k = v_k - alpha * A^T(A v_k - y)
        res_sino = forward_op(v_k) - y
        grad_fidelity = backward_op(res_sino)
        u_k = v_k - self.alpha * grad_fidelity

        # 3. Bước biến đổi phân tích & ngưỡng co rút
        feat = F.relu(self.conv_in1(u_k))
        feat = self.bn_in(self.conv_in2(feat))
        feat_sparse = self.threshold(feat)

        # 4. Bước tổng hợp tái tạo ảnh
        feat_rec = F.relu(self.conv_out1(feat_sparse))
        feat_rec = self.bn_out(feat_rec)
        delta_x = self.conv_out2(feat_rec)

        # Kết nối tắt residual
        x_k = u_k + delta_x
        return x_k


class FISTA_Net_LA(pl.LightningModule):
    """
    Mô hình Baseline FISTA-Net hoàn chỉnh cho bài toán Limited-Angle CT Reconstruction.
    
    Tham chiếu khoa học:
    J. Xiang, Y. Dong, Y. Yang, "FISTA-Net: Deep Unfolding Network With Convergence Analysis for Inverse Problems,"
    IEEE Transactions on Medical Imaging (TMI), vol. 40, no. 5, pp. 1329-1339, May 2021.
    """
    def __init__(
        self,
        n_iterations: int = 10,
        hidden_channels: int = 32,
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

        self.n_iterations = n_iterations
        self.hidden_channels = hidden_channels
        self.num_view = num_view
        self.num_detectors = num_detectors
        self.start_ang = start_ang
        self.end_ang = end_ang
        self.input_size = input_size
        self.initial_lr = initial_lr
        self.final_lr = final_lr

        # Danh sách n_iterations stages FISTA mở cuộn độc lập
        self.stages = nn.ModuleList([
            FISTA_StageBlock(in_channels=1, hidden_channels=hidden_channels)
            for _ in range(n_iterations)
        ])

        # Khởi tạo toán tử Radon xuôi và ngược
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

    def forward(self, x_init: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Vòng lặp mở cuộn unrolling của FISTA-Net qua n_iterations stages:
        - x_{-1} = x_0 = x_init (FBP sơ bộ)
        """
        x_prev2 = x_init
        x_prev = x_init

        for k in range(self.n_iterations):
            x_curr = self.stages[k](
                x_prev=x_prev,
                x_prev2=x_prev2,
                y=y,
                forward_op=self.forward_module,
                backward_op=self.backward_module,
            )
            x_prev2 = x_prev
            x_prev = x_curr

        return x_curr

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
