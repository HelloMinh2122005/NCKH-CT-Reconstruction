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


class MoDL_CNN_Regularizer(nn.Module):
    """
    Khối Điều Hòa Học Sâu (Learned CNN Denoiser / Regularizer D_w) của mạng MoDL (Aggarwal et al., IEEE TMI 2019).
    
    Nguyên lý toán học:
    Trong bài toán biến phân tối ưu: min_x ||A x - y||_2^2 + lambda * ||D_w(x)||_2^2,
    khối D_w đóng vai trò là một toán tử tiên nghiệm lọc bỏ artifact và khử nhiễu (Denoising Prior).
    Mô hình sử dụng mạng tích chập 5 tầng (5-layer CNN) với kết nối tắt (residual skip connection):
        D_w(x) = x + ResBlock(x)
    
    Cấu trúc mạng:
    - Conv1: 1 kênh đầu vào -> 64 kênh ẩn, kernel 3x3, padding 1 + ReLU.
    - Conv2, Conv3, Conv4: 64 kênh -> 64 kênh, kernel 3x3, padding 1 + Batch Normalization + ReLU.
    - Conv5: 64 kênh -> 1 kênh đầu ra, kernel 3x3, padding 1.
    - Kết nối tắt: output = x + Conv5(feat)
    
    Kích thước Tensor:
    - Đầu vào x: (Batch_size, 1, 256, 256)
    - Đầu ra D_w(x): (Batch_size, 1, 256, 256)
    """
    def __init__(self, in_channels: int = 1, hidden_channels: int = 64, num_layers: int = 5):
        super().__init__()
        layers = []
        # Tầng đầu vào
        layers.append(nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        
        # Các tầng ẩn trung gian
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_channels))
            layers.append(nn.ReLU(inplace=True))
            
        # Tầng đầu ra
        layers.append(nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1, bias=True))
        self.net = nn.Sequential(*layers)

        # Khởi tạo trọng số He/Kaiming
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass với residual learning: D_w(x) = x + Net(x)"""
        residual = self.net(x)
        return x + residual


class MoDL_CGSolver(nn.Module):
    """
    Bộ giải lặp Conjugate Gradient (CG) cho bước nhất quán dữ liệu (Data Consistency) trong MoDL.
    
    Hệ phương trình tuyến tính cần giải tại mỗi giai đoạn k:
        (A^T A + lambda * I) x_k = A^T y + lambda * z_k
    trong đó:
        - A: Toán tử chiếu thẳng Radon (Forward Projection).
        - A^T: Toán tử chiếu ngược liên hợp chuẩn (Adjoint RayTransform).
               Lưu ý: Bắt buộc dùng toán tử liên hợp A^T (operator.adjoint), tuyệt đối không dùng
               FBP vì FBP * A không đối xứng xác định dương (not SPD), làm CG phân kỳ.
        - y: Sinogram đo đạc thực tế từ máy quét góc giới hạn (LA-Sinogram).
        - z_k = D_w(x_{k-1}): Ảnh đã qua khối điều hòa CNN.
        - lambda: Trọng số cân bằng độ khớp dữ liệu và điều hòa tiên nghiệm (Learnable parameter lambda > 0).
    
    Phương pháp: Matrix-Free Conjugate Gradient chạy lặp n_cg_iters bước, bảo đảm hội tụ nghiệm tối ưu.
    """
    def __init__(self, cg_iters: int = 6):
        super().__init__()
        self.cg_iters = cg_iters

    def forward(
        self,
        x_init: torch.Tensor,
        rhs: torch.Tensor,
        lambda_val: torch.Tensor,
        forward_op: nn.Module,
        backward_op: nn.Module,
    ) -> torch.Tensor:
        """
        x_init: Nghiệm khởi tạo cho CG (B, 1, 256, 256)
        rhs: Vế phải của hệ phương trình b = A^T y + lambda * z (B, 1, 256, 256)
        lambda_val: Trọng số chính quy hóa lambda
        forward_op: Toán tử chiếu thẳng A
        backward_op: Toán tử chiếu ngược liên hợp A^T (Adjoint)
        """
        x = x_init.clone()

        def matvec(p: torch.Tensor) -> torch.Tensor:
            # Tính (A^T A + lambda * I) p với backward_op là toán tử liên hợp chuẩn A^T
            return backward_op(forward_op(p)) + lambda_val * p

        r = rhs - matvec(x)
        p = r.clone()
        rs_old = torch.sum(r * r, dim=(1, 2, 3), keepdim=True)

        for _ in range(self.cg_iters):
            Ap = matvec(p)
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


class MoDL_LA(pl.LightningModule):
    """
    Mô hình Baseline MoDL (Model-Based Deep Learning) hoàn chỉnh cho bài toán Limited-Angle CT.
    
    Tham chiếu khoa học:
    H. K. Aggarwal, M. P. Mani, M. Jacob, "MoDL: Model-Based Deep Learning Architecture for Inverse Problems,"
    IEEE Transactions on Medical Imaging (TMI), vol. 38, no. 2, pp. 394-405, Feb. 2019.
    
    Đặc điểm kiến trúc:
    1. Unrolling K stages: Mở cuộn qua n_iterations giai đoạn lặp (mặc định = 10 stages).
    2. Chia sẻ trọng số (Weight Sharing) hoặc không chia sẻ: Khối CNN D_w được chia sẻ giữa các stage
       đúng theo thiết kế nguyên bản của Aggarwal et al. để tiết kiệm tham số (~0.15M params) và tăng tính ổn định.
    3. Bước nhất quán dữ liệu vật lý (Data Consistency) giải trực tiếp qua Conjugate Gradient Solver.
    """
    def __init__(
        self,
        n_iterations: int = 10,
        cg_iters: int = 6,
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
        self.cg_iters = cg_iters
        self.num_view = num_view
        self.num_detectors = num_detectors
        self.start_ang = start_ang
        self.end_ang = end_ang
        self.input_size = input_size
        self.initial_lr = initial_lr
        self.final_lr = final_lr

        # Khối CNN điều hòa tiên nghiệm (Denoising Prior) - chia sẻ trọng số qua các stages
        self.regularizer = MoDL_CNN_Regularizer(in_channels=1, hidden_channels=64, num_layers=5)

        # Bộ giải Conjugate Gradient Matrix-Free cho bước Data Consistency
        self.cg_solver = MoDL_CGSolver(cg_iters=cg_iters)

        # Tham số học được lambda (hệ số chính quy hóa) cho từng giai đoạn, ràng buộc dương qua Softplus
        self.raw_lambda = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5)) for _ in range(n_iterations)
        ])

        # Khởi tạo toán tử Radon xuôi (RayTransform), liên hợp ngược A^T (Adjoint) và FBP
        radon_op, adjoint_op, fbp_op = self.radon_transform(
            num_view=num_view,
            start_ang=start_ang,
            end_ang=end_ang,
            num_detectors=num_detectors,
            input_size=input_size,
        )
        self.forward_module = radon_op
        self.adjoint_module = adjoint_op
        self.backward_module = adjoint_op  # Giữ backward_module là adjoint A^T để bảo đảm tính tương thích
        self.fbp_module = fbp_op

        self.grid: Optional[torch.Tensor] = None

    def forward(self, x_init: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Vòng lặp mở cuộn unrolling của MoDL:
        - Đầu vào:
          + x_init: Ảnh FBP khởi tạo (Batch_size, 1, 256, 256)
          + y: Sinogram đo đạc góc giới hạn (Batch_size, 1, 64, 512)
        - Tại mỗi giai đoạn k = 0, ..., K-1:
          1. z_k = D_w(x_k)                 (Khử nhiễu / điều hòa miền ảnh)
          2. b_k = A^T y + lambda_k * z_k    (Tính vế phải bằng toán tử liên hợp chuẩn Adjoint)
          3. x_{k+1} = (A^T A + lambda_k I)^(-1) b_k  (Giải CG Data Consistency đối xứng xác định dương)
        """
        x_curr = x_init
        # Tính trước A^T y một lần duy nhất bằng toán tử liên hợp chuẩn Adjoint A^T để tối ưu tốc độ tính toán
        at_y = self.adjoint_module(y)

        for k in range(self.n_iterations):
            # 1. Bước Điều Hòa Tiên Nghiệm (CNN Denoiser)
            z_k = self.regularizer(x_curr)

            # 2. Lấy trọng số lambda_k dương qua softplus
            lambda_k = F.softplus(self.raw_lambda[k]) + 1e-4

            # 3. Tính vế phải rhs = A^T y + lambda * z_k
            rhs = at_y + lambda_k * z_k

            # 4. Bước Nhất Quán Dữ Liệu (CG Solver): giải hệ (A^T A + lambda_k I) x = rhs
            x_curr = self.cg_solver(
                x_init=x_curr,
                rhs=rhs,
                lambda_val=lambda_k,
                forward_op=self.forward_module,
                backward_op=self.adjoint_module,
            )

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

        # Toán tử chiếu ngược liên hợp thuần túy A^T (Adjoint RayTransform)
        # Bắt buộc dùng cho bước giải lặp Conjugate Gradient (CG) để bảo đảm ma trận (A^T A + lambda I)
        # đối xứng xác định dương (Symmetric Positive Definite - SPD), tránh phân kỳ nghiệm.
        adjoint_operator = operator.adjoint
        op_layer_adjoint = odl_torch.operator.OperatorModule(adjoint_operator)

        # Toán tử FBP (Filtered Backprojection) dùng bộ lọc Ram-Lak phục vụ khởi tạo sơ bộ
        fbp = odl.tomo.fbp_op(operator, filter_type="Ram-Lak", frequency_scaling=0.9) * np.sqrt(2)
        op_layer_fbp = odl_torch.operator.OperatorModule(fbp)

        return op_layer, op_layer_adjoint, op_layer_fbp
