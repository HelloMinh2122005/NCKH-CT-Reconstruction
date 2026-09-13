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


class RegularizationBlock(nn.Module):
    """
    Khối Điều Hòa Tiên Nghiệm (Learned Regularizer R_theta) nguyên bản của mạng LEARN (Chen et al., IEEE TMI 2018).
    
    Nguyên lý hoạt động toán học:
    Trong bài toán tối ưu hóa biến phân CT: min_x (1/2)||A x - y||_2^2 + lambda * R(x),
    gradient của hàm điều hòa grad_R(x) được xấp xỉ bằng một mạng tích chập sâu có thể học được.
    Ở mô hình LEARN nguyên bản, khối này bao gồm 3 tầng tích chập Conv2D thuần túy với hàm kích hoạt phi tuyến ReLU:
    1. Conv1: Tăng số kênh đặc trưng từ ảnh gốc (1 kênh) lên không gian biểu diễn ẩn đa chiều (48 kênh) với kernel 5x5.
    2. Conv2: Trích xuất và biểu diễn các tương quan không gian cục bộ tần số cao (48 kênh -> 48 kênh) với kernel 5x5.
    3. Conv3: Chiếu ngược lại không gian ảnh gốc (48 kênh -> 1 kênh) với kernel 5x5 để sinh ra vector gradient hiệu chỉnh.
    
    Kích thước Tensor:
    - Đầu vào x: (Batch_size, Channels=1, Height=256, Width=256)
    - Sau Conv1 + ReLU: (Batch_size, Channels=48, Height=256, Width=256)
    - Sau Conv2 + ReLU: (Batch_size, Channels=48, Height=256, Width=256)
    - Sau Conv3 (Đầu ra): (Batch_size, Channels=1, Height=256, Width=256)
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 5,
        patch_channels: int = 48,
    ):
        super().__init__()
        # Để giữ nguyên kích thước không gian ảnh 256x256, với kernel_size=5 thì padding = 5 // 2 = 2
        padding_value = kernel_size // 2

        # Tầng tích chập 1: 1 kênh đầu vào -> 48 kênh đặc trưng
        self.conv1 = nn.Conv2d(
            in_channels,
            patch_channels,
            kernel_size=kernel_size,
            padding=padding_value,
        )
        # Khởi tạo trọng số phân phối chuẩn mean=0, std=0.01 theo thiết lập của tác giả Chen et al. và Thành (Thanhld)
        nn.init.normal_(self.conv1.weight, mean=0.0, std=0.01)

        # Tầng tích chập 2: 48 kênh đặc trưng -> 48 kênh đặc trưng
        self.conv2 = nn.Conv2d(
            patch_channels,
            patch_channels,
            kernel_size=kernel_size,
            padding=padding_value,
        )
        nn.init.normal_(self.conv2.weight, mean=0.0, std=0.01)

        # Tầng tích chập 3: 48 kênh đặc trưng -> 1 kênh ảnh tái tạo hiệu chỉnh
        self.conv3 = nn.Conv2d(
            patch_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding_value,
        )
        nn.init.normal_(self.conv3.weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Luồng truyền xuôi (Forward pass):
        x -> Conv1 -> ReLU -> Conv2 -> ReLU -> Conv3 -> output
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class GradientFunction(nn.Module):
    """
    Một giai đoạn mở cuộn (Stage / Iteration) của thuật toán tối ưu Gradient Descent trong LEARN.
    
    Công thức cập nhật biến phân tại giai đoạn thứ t:
        x_{t} = x_{t-1} - [ alpha_t * A^T(A x_{t-1} - y) + R_{theta_t}(x_{t-1}) ]
    
    Thành phần:
    - Term 1: alpha_t * A^T(A x_{t-1} - y) là đạo hàm của hàm độ khớp dữ liệu (Data Fidelity Term).
      Trong đó:
        + A: Toán tử chiếu thẳng Radon (Forward Projection).
        + A^T: Toán tử chiếu ngược FBP (Filtered Backprojection Adjoint).
        + y: Sinogram đo đạc thực tế từ máy quét góc giới hạn (LA Sinogram).
        + alpha_t: Bước nhảy gradient có thể học được (Learnable step size), khởi tạo bằng 0.1.
    - Term 2: R_{theta_t}(x_{t-1}) là đạo hàm của hàm điều hòa tiên nghiệm (Regularization Term),
      được tham số hóa bởi mạng tích chập 3 tầng RegularizationBlock riêng biệt cho từng stage.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, kernel_size: int = 5, patch_channels: int = 48):
        super().__init__()
        self.regularitation_term = RegularizationBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            patch_channels=patch_channels,
        )
        # Hệ số bước nhảy gradient alpha có thể học qua quá trình backpropagation
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        x_t: torch.Tensor,
        y: torch.Tensor,
        forward_module: nn.Module,
        backward_module: nn.Module,
    ) -> torch.Tensor:
        """
        Tính toán vector gradient g_t để cập nhật nghiệm ảnh tại bước t.
        Đầu vào:
        - x_t: Ảnh ước lượng hiện tại (Batch_size, 1, 256, 256)
        - y: Sinogram đo đạc góc giới hạn (Batch_size, 1, num_view=64, num_detectors=512)
        - forward_module: Lớp toán tử Radon A
        - backward_module: Lớp toán tử chiếu ngược A^T (FBP)
        Đầu ra:
        - gradient: Tensor (Batch_size, 1, 256, 256) đại diện cho hướng điều chỉnh ảnh
        """
        # Bước 1: Tính sai lệch trên miền chiếu sinogram e = A(x_t) - y
        data_fidelity_term = forward_module(x_t) - y
        
        # Bước 2: Chiếu ngược sai lệch về miền ảnh e_img = A^T(e)
        bp_data_fidelity = backward_module(data_fidelity_term)
        
        # Bước 3: Tính toán thành phần điều hòa tiên nghiệm từ miền ảnh
        reg_value = self.regularitation_term(x_t)
        
        # Bước 4: Tổng hợp vector gradient hoàn chỉnh
        gradient = self.alpha * bp_data_fidelity + reg_value
        return gradient


class LEARN_LA(pl.LightningModule):
    """
    Mô hình Baseline LEARN hoàn chỉnh (Original LEARN) cho bài toán Tái tạo ảnh CT Góc Giới Hạn (Limited-Angle CT).
    
    Đặc điểm kỹ thuật:
    1. Kế thừa nguyên bản thuật toán LEARN từ Hu Chen et al. (IEEE TMI 2018) và triển khai của Thành (`Thanhld`).
    2. Thích ứng hoàn hảo với bài toán Limited-Angle CT (LA-CT 120°, 64 góc chiếu, 512 cảm biến detector).
    3. Mở cuộn qua n_iterations giai đoạn (mặc định = 14 stages để đảm bảo so sánh công bằng với các baseline khác).
    4. Tích hợp toán tử chiếu Fan-Beam ASTRA GPU qua ODL Module.
    5. Tự động theo dõi các chỉ số đánh giá y tế: PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity), RMSE.
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

        # Danh sách n_iterations module GradientFunction mở cuộn độc lập
        self.gradient_list = nn.ModuleList(
            [GradientFunction() for _ in range(n_iterations)]
        )

        # Xây dựng toán tử Radon xuôi (Forward Projection) và ngược (FBP) cho hình học góc giới hạn
        radon_curr, fbp_curr = self.radon_transform(
            num_view=num_view,
            start_ang=start_ang,
            end_ang=end_ang,
            num_detectors=num_detectors,
            input_size=input_size,
        )
        self.forward_module = radon_curr
        self.backward_module = fbp_curr

        # Bộ đệm lưu lưới ảnh trực quan hóa lên TensorBoard
        self.grid: Optional[torch.Tensor] = None

    def forward(self, x_t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Vòng lặp mở cuộn unrolling qua n_iterations giai đoạn:
        Đầu vào:
        - x_t: Ảnh tái tạo FBP ban đầu bị nhiễu do thiếu góc (Batch_size, 1, 256, 256)
        - y: Sinogram đo đạc góc giới hạn (Batch_size, 1, 64, 512)
        Đầu ra:
        - Ảnh CT đã được khử triệt để vệt sọc missing wedge (Batch_size, 1, 256, 256)
        """
        for i in range(self.n_iterations):
            x_t = x_t - self.gradient_list[i](
                x_t, y, self.forward_module, self.backward_module
            )
        return x_t

    def configure_optimizers(self):
        """
        Cấu hình bộ tối ưu hóa Adam kết hợp lịch hạ tốc độ học Cosine Annealing.
        """
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
        """Xác định dải động của ảnh phục vụ tính PSNR và SSIM chuẩn xác"""
        batch_min = target.amin()
        batch_max = target.amax()
        if torch.isclose(batch_max, batch_min):
            batch_max = batch_min + torch.tensor(
                1e-8, device=target.device, dtype=target.dtype
            )
        return (batch_min, batch_max)

    def rmse(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Tính sai số toàn phương căn bậc hai Root Mean Square Error"""
        return torch.sqrt(torch.mean((y_true - y_pred) ** 2))

    def training_step(self, train_batch, batch_idx):
        """
        Một bước huấn luyện (Training Step):
        - Đầu vào batch chứa: (phantom: Ảnh gốc Ground Truth, fbp_u: Ảnh FBP khởi tạo góc thiếu, sino_noisy: Sinogram)
        - Hàm mục tiêu tối ưu: Sai số toàn phương trung bình MSE Loss = ||x_hat - phantom||_2^2
        """
        phantom, fbp_u, sino_noisy = train_batch
        x_t = fbp_u
        y = sino_noisy

        # Tái tạo ảnh qua mạng mở cuộn LEARN
        x_reconstructed = self.forward(x_t, y)
        loss = F.mse_loss(phantom, x_reconstructed)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        """
        Một bước kiểm định (Validation Step):
        - Đo lường định lượng các chỉ số chất lượng ảnh y tế: PSNR, SSIM, RMSE, Validation MSE Loss.
        - Lưu lưới ảnh tái tạo để hiển thị tiến trình khử sọc missing wedge trên TensorBoard.
        """
        phantom, fbp_u, sino_noisy = val_batch
        x_t = fbp_u
        y = sino_noisy

        x_reconstructed = self.forward(x_t, y)
        loss = F.mse_loss(phantom, x_reconstructed)

        data_range = self._get_batch_data_range(phantom)

        ssim_val = structural_similarity_index_measure(
            x_reconstructed,
            phantom,
            data_range=data_range,
        )
        psnr_val = peak_signal_noise_ratio(
            x_reconstructed,
            phantom,
            data_range=data_range,
        )
        rmse_val = self.rmse(phantom, x_reconstructed)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_ssim", ssim_val, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_psnr", psnr_val, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_rmse", rmse_val, on_step=False, on_epoch=True, prog_bar=False)

        # Lưu ảnh lát cắt đầu tiên để vẽ lên TensorBoard
        self.grid = torchvision.utils.make_grid(x_reconstructed.detach().clamp(min=0.0))

        return {
            "val_loss": loss,
            "val_ssim": ssim_val,
            "val_psnr": psnr_val,
            "val_rmse": rmse_val,
        }

    def test_step(self, batch, batch_idx):
        """
        Bước đánh giá trên tập kiểm thử (Test Step):
        """
        phantom, fbp_u, sino_noisy = batch
        x_t = fbp_u
        y = sino_noisy

        x_reconstructed = self.forward(x_t, y)
        loss = F.mse_loss(phantom, x_reconstructed)

        data_range = self._get_batch_data_range(phantom)

        ssim_val = structural_similarity_index_measure(
            x_reconstructed,
            phantom,
            data_range=data_range,
        )
        psnr_val = peak_signal_noise_ratio(
            x_reconstructed,
            phantom,
            data_range=data_range,
        )
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
        """Ghi nhận ảnh tái tạo trực quan vào TensorBoard sau mỗi epoch kiểm định"""
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
        """
        Xây dựng toán tử hình học chiếu Fan-Beam CT góc giới hạn (Limited-Angle FanBeam Geometry) sử dụng ODL và ASTRA-Toolbox:
        - Không gian ảnh tái tạo: Lưới giải phẫu 256x256 pixel trong phạm vi tọa độ [-200mm, 200mm].
        - Dải góc quét giới hạn: start_ang = -pi/3 (-60 độ) đến end_ang = +pi/3 (+60 độ) (cung quét 120 độ).
        - Phân chia góc chiếu: num_view = 64 góc.
        - Detector: 512 phần tử cảm biến cong, khoảng cách nguồn tia X - tâm quay = 600mm, tâm quay - đầu dò = 290mm.
        - Toán tử FBP sử dụng bộ lọc Ram-Lak với tần số tỷ lệ 0.9.
        """
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

        # Toán tử chiếu ngược FBP chuẩn hóa
        fbp = odl.tomo.fbp_op(
            operator,
            filter_type="Ram-Lak",
            frequency_scaling=0.9,
        ) * np.sqrt(2)
        op_layer_fbp = odl_torch.operator.OperatorModule(fbp)

        return op_layer, op_layer_fbp
