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

# Thử import hàm quét chọn lọc (selective_scan_fn) từ thư viện mamba_ssm
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None


class SelectiveSSMBlock(nn.Module):
    """
    Khối Selective State Space Model (SSM) cho bài toán Tái tạo ảnh CT.
    
    Nguyên lý hoạt động:
    1. Chuyển đổi tensor đặc trưng 2D thành chuỗi các token siêu mịn 2x2 pixel (N = 16,384 tokens).
    2. Áp dụng cơ chế quét có chọn lọc (Selective Scan) tuyến tính O(N) với các tham số B, C, Delta 
       phụ thuộc động vào dữ liệu đầu vào.
    3. Tái cấu trúc chuỗi 1D ngược về không gian 2D (B, C, H, W).
    
    Tham số khởi tạo:
    - window_size (int): Kích thước mỗi patch token (mặc định = 2, tức 2x2 pixel).
    - patch_channels (int): Số kênh đặc trưng của tensor ảnh (mặc định = 48).
    - image_size (int): Kích thước không gian của ảnh (mặc định = 256).
    - num_heads (int): Số lượng attention heads / state heads (mặc định = 4).
    - A_init_range (Tuple[float, float]): Khoảng giá trị khởi tạo đều cho ma trận hệ thống A (1 đến 16).
    """
    def __init__(
        self,
        window_size: int = 2,
        patch_channels: int = 48,
        image_size: int = 256,
        num_heads: int = 4,
        A_init_range: Tuple[float, float] = (1, 16),
        device=None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.window_size = window_size
        self.patch_channels = patch_channels
        self.image_size = image_size
        self.new_spatial = image_size // window_size  # 256 // 2 = 128
        self.token_dim = patch_channels * (window_size ** 2)  # 48 * 4 = 192
        self.num_heads = num_heads

        D = self.token_dim  # Chiều của mỗi token = 192

        # 1. Khởi tạo ma trận chuyển trạng thái hệ thống A (kích thước D x D = 192 x 192)
        # Ma trận A được lưu dưới dạng logarit để đảm bảo tính ổn định và tính âm khi lũy thừa: A_thực = -exp(A_log)
        A_init = torch.empty(D, D, device=device, dtype=dtype)
        nn.init.uniform_(A_init, *A_init_range)
        self.A_log = nn.Parameter(torch.log(A_init))

        # 2. Tham số đường truyền trực tiếp D (Skip Connection parameter)
        self.D = nn.Parameter(torch.ones(D, device=device, dtype=dtype))

        # 3. Các lớp chiếu tuyến tính học tham số động B (Input Matrix), C (Output Matrix), và Delta (Step Size)
        self.proj_B = nn.Linear(D, D, bias=True, device=device, dtype=dtype)
        self.proj_C = nn.Linear(D, D, bias=True, device=device, dtype=dtype)
        self.dt_proj = nn.Linear(D, D, bias=True, device=device, dtype=dtype)
        self.dt_proj.bias._no_weight_decay = True  # Không áp dụng weight decay cho bias của bước thời gian dt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Luồng tính toán Forward của Selective SSM:
        Đầu vào: x có shape (Batch_size, Channels=48, Height=256, Width=256)
        Đầu ra: Tensor có cùng kích thước (Batch_size, 48, 256, 256)
        """
        B, C, H, W = x.shape

        # Bước 1: Tokenize — Gom các ô nhỏ 2x2 thành các token
        # x.unfold(2, 2, 2).unfold(3, 2, 2) tạo ra tensor (B, 48, 128, 128, 2, 2)
        tokens = (
            x.unfold(2, self.window_size, self.window_size)
            .unfold(3, self.window_size, self.window_size)
            .reshape(B, C, -1, self.window_size * self.window_size)  # (B, 48, 16384, 4)
            .permute(0, 2, 1, 3)                                    # (B, 16384, 48, 4)
            .reshape(B, -1, self.token_dim)                          # (B, 16384, 192)
        )
        _, N, D = tokens.shape  # N = 16384 tokens, D = 192 dimensions

        # Bước 2: Tính toán các ma trận chọn lọc động theo từng token x_t
        B_mat = self.proj_B(tokens)       # Ma trận nạp trạng thái B_t: (B, N, D)
        C_mat = self.proj_C(tokens)       # Ma trận trích xuất C_t: (B, N, D)
        raw_dt = self.dt_proj(tokens)     # Bước thời gian thô
        Delta = F.softplus(raw_dt)        # Hàm Softplus ép bước thời gian dt luôn dương

        # Chuyển đổi định dạng tensor phù hợp với hàm CUDA selective_scan_fn
        u = tokens.permute(0, 2, 1).contiguous()  # (B, D, N)

        def to_state4(t3: torch.Tensor) -> torch.Tensor:
            """Hàm phụ trợ mở rộng chiều cho multi-head SSM: (B, num_heads, D, N)"""
            t = t3.permute(0, 2, 1).unsqueeze(1)
            return t.expand(-1, self.num_heads, -1, -1).contiguous()

        Bp = to_state4(B_mat)
        Cp = to_state4(C_mat)

        delta = Delta.permute(0, 2, 1).contiguous()
        delta_bias = self.dt_proj.bias.contiguous()
        z = torch.ones_like(delta)
        A = -torch.exp(self.A_log)  # Ma trận chuyển đổi ổn định thực tế A < 0

        # Bước 3: Thực thi Hardware-Aware Parallel Scan trong SRAM GPU
        if selective_scan_fn is not None and x.is_cuda:
            out = selective_scan_fn(
                u,
                delta,
                A,
                Bp,
                Cp,
                self.D,
                z,
                delta_bias,
                True,
            )
        else:
            # Fallback đơn giản cho môi trường CPU/Debug nếu không có GPU CUDA kernel
            out = u * self.D.unsqueeze(0).unsqueeze(-1)

        # Bước 4: Untokenize — Tái cấu trúc chuỗi 1D (B, D, N) ngược về ảnh 2D (B, 48, 256, 256)
        out = (
            out.view(
                B,
                self.new_spatial,
                self.new_spatial,
                self.patch_channels,
                self.window_size,
                self.window_size,
            )
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(
                B,
                self.patch_channels,
                self.new_spatial * self.window_size,
                self.new_spatial * self.window_size,
            )
        )
        return out


class RegularizationBlock(nn.Module):
    """
    Khối Điều Hòa Tiên Nghiệm (Learned Regularizer R_theta).
    
    Cấu trúc:
    Conv1 (1 -> 48, kernel 5x5) -> ReLU -> SelectiveSSMBlock -> Conv2 (48 -> 48, kernel 5x5) -> ReLU -> Conv3 (48 -> 1, kernel 5x5)
    
    Vai trò:
    - Conv1 & Conv2: Học các đặc trưng không gian cục bộ (mép xương, ranh giới mô mềm).
    - SelectiveSSMBlock: Nhận diện cấu trúc liên kết toàn cục tầm xa, xóa vệt sọc nêm khuyết (Missing Wedge Artifacts).
    - Conv3: Ánh xạ đặc trưng sâu về lại miền gradient ảnh 1 kênh để cập nhật bước lặp tiếp theo.
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
        padding_value = kernel_size // 2  # kernel 5 -> padding 2 (giữ nguyên độ phân giải không gian)

        # Tầng tích chập 1: Mở rộng từ 1 kênh ảnh lên 48 kênh đặc trưng
        self.conv1 = nn.Conv2d(
            in_channels,
            patch_channels,
            kernel_size=kernel_size,
            padding=padding_value,
        )
        nn.init.normal_(self.conv1.weight, mean=0.0, std=0.01)

        # Khối Mamba Selective SSM chuỗi dài
        self.selective_ssm = SelectiveSSMBlock(
            window_size=2,
            patch_channels=patch_channels,
            image_size=image_size,
            A_init_range=(1, 16),
        )

        # Tầng tích chập 2: Tinh chỉnh đặc trưng sau SSM
        self.conv2 = nn.Conv2d(
            patch_channels,
            patch_channels,
            kernel_size=kernel_size,
            padding=padding_value,
        )
        nn.init.normal_(self.conv2.weight, mean=0.0, std=0.01)

        # Tầng tích chập 3: Chiếu đặc trưng về lại 1 kênh ảnh đạo hàm
        self.conv3 = nn.Conv2d(
            patch_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding_value,
        )
        nn.init.normal_(self.conv3.weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.selective_ssm(x)
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class GradientFunction(nn.Module):
    """
    Một giai đoạn (Stage) Unrolling của thuật toán LEARN.
    
    Công thức cập nhật gradient tại mỗi bước lặp t:
        g_t = alpha_t * A^T(A x_t - y) + R_{theta_t}(x_t)
        
    Trong đó:
    - A: Toán tử chiếu thuận Radon (Mô phỏng máy CT).
    - A^T: Toán tử chiếu ngược Backprojection (Phân bổ sai số đo đạc về ảnh).
    - alpha_t: Hệ số bước nhảy học được (khởi tạo = 0.1).
    - R_{theta_t}: Mạng điều hòa học sâu (CNN + SSM).
    """
    def __init__(self, image_size: int = 256):
        super().__init__()
        self.regularitation_term = RegularizationBlock(image_size=image_size)
        self.alpha = nn.Parameter(torch.tensor(0.1))  # Tham số bước nhảy vật lý học được

    def forward(
        self,
        x_t: torch.Tensor,
        y: torch.Tensor,
        forward_module: nn.Module,
        backward_module: nn.Module,
    ) -> torch.Tensor:
        # 1. Nhánh nhất quán dữ liệu vật lý (Data Fidelity Term)
        data_fidelity_term = forward_module(x_t) - y          # Vector sai số sinogram: delta_y = A*x_t - y
        bp_data_fidelity = backward_module(data_fidelity_term)# Chiếu ngược sai số về ảnh: A^T(delta_y)
        
        # 2. Nhánh tiên nghiệm sâu (Learned Regularization Term)
        reg_value = self.regularitation_term(x_t)              # R_theta(x_t)
        
        # 3. Tổng hợp gradient
        gradient = self.alpha * bp_data_fidelity + reg_value
        return gradient


class LEARN_Mamba_LA(pl.LightningModule):
    """
    Mô hình LEARN_Mamba hoàn chỉnh cho bài toán Limited-Angle CT (LA-CT).
    
    Kế thừa kiến trúc từ paper của Thành, được chuyển giao và tối ưu hóa toán tử
    cho dải góc quét giới hạn Fan-Beam (ví dụ: [-60 độ, +60 độ], 64 views, 512 detectors).
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

        # Khởi tạo danh sách n_iterations giai đoạn unrolling độc lập
        self.gradient_list = nn.ModuleList(
            [GradientFunction(image_size=input_size) for _ in range(n_iterations)]
        )

        # Xây dựng toán tử Radon và FBP góc giới hạn
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
        """
        Vòng lặp Unrolling 14 giai đoạn:
        x_{t+1} = x_t - GradientFunction_t(x_t, y, A, A^T)
        """
        for i in range(self.n_iterations):
            x_t = x_t - self.gradient_list[i](
                x_t, y, self.forward_module, self.backward_module
            )
        return x_t

    def configure_optimizers(self):
        """Thiết lập bộ tối ưu Adam và lịch hạ tốc độ học Cosine Annealing"""
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
        """Tính toán dải động dữ liệu động (Data Range) phục vụ đo lường PSNR/SSIM chuẩn xác"""
        batch_min = target.amin()
        batch_max = target.amax()
        if torch.isclose(batch_max, batch_min):
            batch_max = batch_min + torch.tensor(
                1e-8, device=target.device, dtype=target.dtype
            )
        return (batch_min, batch_max)

    def rmse(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Tính chỉ số sai số căn bậc hai trung bình (Root Mean Squared Error)"""
        return torch.sqrt(torch.mean((y_true - y_pred) ** 2))

    def training_step(self, train_batch, batch_idx):
        """Bước huấn luyện (Training Step): Tối ưu hóa hàm mất mát Mean Squared Error (MSE)"""
        phantom, fbp_u, sino_noisy = train_batch
        x_reconstructed = self.forward(fbp_u, sino_noisy)
        loss = F.mse_loss(phantom, x_reconstructed)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        """Bước kiểm định (Validation Step): Tính toán chi tiết các chỉ số PSNR, SSIM, RMSE"""
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

        # Lưu ảnh lưới tái tạo để xuất lên TensorBoard
        self.grid = torchvision.utils.make_grid(x_reconstructed.detach().clamp(min=0.0))
        return {"val_loss": loss, "val_ssim": ssim_p, "val_psnr": psnr_p, "val_rmse": rmse_p}

    def test_step(self, batch, batch_idx):
        """Bước kiểm thử (Test Step): Đánh giá trên tập bệnh nhân kiểm thử độc lập"""
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
        """Đẩy hình ảnh tái tạo lên TensorBoard sau mỗi epoch để theo dõi trực quan chất lượng ảnh"""
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
        Khởi tạo toán tử chiếu Radon và FBP cho bài toán Fan-Beam Limited-Angle CT.
        Tự động chọn 'astra_cuda' khi chạy trên GPU cluster hoặc fallback 'astra_cpu' khi ở Headnode.
        """
        xx = 200  # Miền vật lý [-200mm, 200mm]
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
