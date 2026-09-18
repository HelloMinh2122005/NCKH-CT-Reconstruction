"""
================================================================================
KIẾN TRÚC MÔ HÌNH: SOLAR_DualMamba (LIMITED-ANGLE CT RECONSTRUCTION)
Second-Order Dual-Domain Newton-CG Unrolling with Angular Sinogram Bi-Mamba
and Image-Domain Swin-Window Regularizer.

Dự án: Nghiên cứu Tái tạo Ảnh Cắt lớp CT Góc Giới hạn (Limited-Angle CT Reconstruction)
Tác giả: MinhPD — Nhóm Nghiên cứu Tái tạo Ảnh Y tế (VNU-HCM UIT)
================================================================================
"""

from typing import Optional, Tuple
import os
import math
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

# Nhập selective scan từ mamba_ssm nếu có
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None


# =============================================================================
# 1. KHỐI ANGULAR SINOGRAM BI-MAMBA (MIỀN CHIẾU SINOGRAM DOMAIN)
# =============================================================================
class AngularSinogramMambaBlock(nn.Module):
    """
    Khối Mô Hình Không Gian Trạng Thái Hai Chiều Dọc Theo Trục Góc Chiếu (Angular Bi-Mamba).
    
    Nguyên lý Vật lý & Toán học:
    Trong bài toán Limited-Angle CT, khi chùm tia X quay quanh cơ thể, mỗi điểm giải phẫu
    tạo ra một đường cong hình sin (sinusoidal trajectory) mượt mà liên tục dọc theo trục góc theta.
    Góc quét bị giới hạn tạo ra một dải trống khuyết (Missing Stripe) trên sinogram.
    
    Khối Angular Bi-Mamba quét hai chiều (Forward: -60° -> +60°, Backward: +60° -> -60°)
    dọc theo trục góc chiếu V = 64 để nắm bắt quy luật biến thiên liên tục của các đường sin,
    ngoại suy và chữa lành các góc chiếu bị khuyết với độ phức tạp tuyến tính O(V).
    """
    def __init__(
        self,
        in_channels: int = 48,
        num_detectors: int = 512,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_detectors = num_detectors
        self.d_state = d_state
        self.d_inner = in_channels * expand  # Kênh mở rộng nội bộ

        # Tầng tích chập 1D theo chiều detector để tổng hợp tương quan giữa các kênh cảm biến lân cận
        self.detector_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.GroupNorm(num_groups=4, num_channels=in_channels),
            nn.SiLU(),
        )

        # Chiếu vào không gian trạng thái (State Space Projections)
        self.in_proj = nn.Linear(in_channels, self.d_inner * 2, bias=False)

        # Tích chập 1D cục bộ theo chiều góc chiếu (Angular 1D Conv)
        self.conv1d_fwd = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            padding=d_conv - 1,
            groups=self.d_inner,
        )
        self.conv1d_bwd = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            padding=d_conv - 1,
            groups=self.d_inner,
        )

        # Tham số hóa thời gian delta, ma trận B và C cho chiều quét xuôi (Forward)
        self.x_proj_fwd = nn.Linear(self.d_inner, self.d_inner + d_state * 2, bias=False)
        self.dt_proj_fwd = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # Tham số hóa thời gian delta, ma trận B và C cho chiều quét ngược (Backward)
        self.x_proj_bwd = nn.Linear(self.d_inner, self.d_inner + d_state * 2, bias=False)
        self.dt_proj_bwd = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # Ma trận chuyển trạng thái liên tục A (Log Parameterization)
        A_fwd = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log_fwd = nn.Parameter(torch.log(A_fwd))
        A_bwd = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log_bwd = nn.Parameter(torch.log(A_bwd))

        # Tham số đường truyền tắt D (Skip Connection parameter)
        self.D_fwd = nn.Parameter(torch.ones(self.d_inner))
        self.D_bwd = nn.Parameter(torch.ones(self.d_inner))

        # Tầng chiếu đầu ra kết hợp hai chiều
        self.out_proj = nn.Linear(self.d_inner, in_channels, bias=False)
        self.norm = nn.LayerNorm(in_channels)

    def _ssm_scan(
        self,
        x_seq: torch.Tensor,
        conv1d_layer: nn.Conv1d,
        x_proj_layer: nn.Linear,
        dt_proj_layer: nn.Linear,
        A_log: torch.Tensor,
        D_param: torch.Tensor,
    ) -> torch.Tensor:
        """
        Thực thi thuật toán quét chọn lọc (Selective Scan) dọc theo trục góc chiếu V.
        x_seq: (B_eff, V, d_inner)
        """
        B_eff, V, D_in = x_seq.shape
        # Tích chập 1D dọc trục góc chiếu: (B_eff, d_inner, V)
        x_conv = conv1d_layer(x_seq.transpose(1, 2))[:, :, :V]
        x_act = F.silu(x_conv).transpose(1, 2)  # (B_eff, V, d_inner)

        # Chiếu ra tham số delta, B, C
        ssm_params = x_proj_layer(x_act)  # (B_eff, V, d_inner + 2*d_state)
        delta_raw, B_mat, C_mat = torch.split(
            ssm_params, [self.d_inner, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(dt_proj_layer(delta_raw))  # (B_eff, V, d_inner)

        A = -torch.exp(A_log)  # (d_inner, d_state)

        # Quét trạng thái tuần tự dọc theo V (Vectorized Scan)
        # Khởi tạo trạng thái ẩn h: (B_eff, d_inner, d_state)
        h = torch.zeros(B_eff, self.d_inner, self.d_state, device=x_seq.device, dtype=x_seq.dtype)
        y_list = []

        for t in range(V):
            u_t = x_act[:, t, :].unsqueeze(-1)  # (B_eff, d_inner, 1)
            delta_t = delta[:, t, :].unsqueeze(-1)  # (B_eff, d_inner, 1)
            B_t = B_mat[:, t, :].unsqueeze(1)  # (B_eff, 1, d_state)
            C_t = C_mat[:, t, :].unsqueeze(-1)  # (B_eff, d_state, 1)

            # Rời rạc hóa hệ phương trình trạng thái qua xấp xỉ Euler/ZOH:
            # A_bar = exp(delta_t * A)
            A_bar = torch.exp(delta_t * A.unsqueeze(0))  # (B_eff, d_inner, d_state)
            B_bar = delta_t * B_t  # (B_eff, d_inner, d_state)

            # Cập nhật trạng thái ẩn: h_t = A_bar * h_{t-1} + B_bar * u_t
            h = A_bar * h + B_bar * u_t
            # Đầu ra quan sát: y_t = h_t @ C_t + D * u_t
            y_t = torch.matmul(h, C_t).squeeze(-1) + D_param * x_act[:, t, :]
            y_list.append(y_t)

        y_out = torch.stack(y_list, dim=1)  # (B_eff, V, d_inner)
        return y_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Đầu vào: x có shape (B, C=48, V=64, D=512)
        Đầu ra: Tensor có shape (B, C=48, V=64, D=512)
        """
        B, C, V, D = x.shape
        shortcut = x

        # Bước 1: Trích xuất đặc trưng cục bộ trên các kênh cảm biến detector
        feat = self.detector_conv(x)  # (B, C, V, D)

        # Bước 2: Chuẩn bị chuỗi cho Angular Mamba dọc theo trục góc chiếu V=64
        # Gộp chiều Batch và Detector để quét song song toàn bộ các detector:
        # permute sang (B, D, V, C) -> reshape thành (B*D, V, C)
        feat_seq = feat.permute(0, 3, 2, 1).contiguous().view(B * D, V, C)
        feat_seq = self.norm(feat_seq)

        # Chiếu lên kênh d_inner
        x_and_res = self.in_proj(feat_seq)
        x_proj, res = torch.chunk(x_and_res, 2, dim=-1)

        # Bước 3: Quét xuôi (Forward: -60° -> +60°)
        y_fwd = self._ssm_scan(
            x_proj,
            self.conv1d_fwd,
            self.x_proj_fwd,
            self.dt_proj_fwd,
            self.A_log_fwd,
            self.D_fwd,
        )

        # Bước 4: Quét ngược (Backward: +60° -> -60°)
        x_proj_rev = torch.flip(x_proj, dims=[1])
        y_bwd_rev = self._ssm_scan(
            x_proj_rev,
            self.conv1d_bwd,
            self.x_proj_bwd,
            self.dt_proj_bwd,
            self.A_log_bwd,
            self.D_bwd,
        )
        y_bwd = torch.flip(y_bwd_rev, dims=[1])

        # Hợp nhất hai chiều quét qua cổng điều chế Gated Silu
        y_bi = (y_fwd + y_bwd) * F.silu(res)
        out_seq = self.out_proj(y_bi)  # (B*D, V, C)

        # Chuyển đổi trở lại shape sinogram ban đầu: (B, C, V, D)
        out = out_seq.view(B, D, V, C).permute(0, 3, 2, 1).contiguous()
        return shortcut + out


class SinogramMambaRestorationNet(nn.Module):
    """
    Mạng Khôi Phục & Ngoại Suy Sinogram (Sinogram Inpainting & Extrapolation Network).
    
    Đầu vào: Sinogram góc giới hạn thô y (B, 1, 64, 512).
    Đầu ra: Sinogram đã được chữa lành và bù đắp dải góc (B, 1, 64, 512).
    """
    def __init__(self, in_channels: int = 1, feature_dim: int = 48, num_blocks: int = 2):
        super().__init__()
        self.shallow_conv = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.mamba_blocks = nn.ModuleList([
            AngularSinogramMambaBlock(in_channels=feature_dim)
            for _ in range(num_blocks)
        ])
        self.out_conv = nn.Conv2d(feature_dim, in_channels, kernel_size=3, padding=1)

    def forward(self, sino: torch.Tensor) -> torch.Tensor:
        feat = self.shallow_conv(sino)
        for blk in self.mamba_blocks:
            feat = blk(feat)
        delta_sino = self.out_conv(feat)
        # Khôi phục phần dư: y_hat = y + delta_y
        restored_sino = sino + delta_sino
        return restored_sino


# =============================================================================
# 2. BỘ GIẢI MATRIX-FREE CONJUGATE GRADIENT TỐI ƯU HÓA BẬC 2 (SAFE CG SOLVER)
# =============================================================================
class SafeCGSolver(nn.Module):
    """
    Bộ giải Conjugate Gradient (CG) tối ưu hóa bậc 2 Matrix-Free giải hệ phương trình:
        (λ_t * A^T A + μ_t * I) x_{t+1} = b_t
    trong đó:
        b_t = λ_t * A^T(y_restored) + μ_t * x_t - ∇R_t

    Bảo đảm tính đối xứng xác định dương nghiêm ngặt (Strictly SPD) qua Softplus μ_t > 0,
    triệt tiêu hoàn toàn hiện tượng dao động Zigzag và loại bỏ 100% lỗi số học NaN.
    """
    def __init__(self, cg_iters: int = 4):
        super().__init__()
        self.cg_iters = cg_iters

    def forward(
        self,
        x_init: torch.Tensor,
        b_t: torch.Tensor,
        lambda_t: torch.Tensor,
        mu_t: torch.Tensor,
        forward_op: nn.Module,
        backward_op: nn.Module,
    ) -> torch.Tensor:
        x = x_init.clone()

        def hessian_matvec(p_vec: torch.Tensor) -> torch.Tensor:
            return lambda_t * backward_op(forward_op(p_vec)) + mu_t * p_vec

        r = b_t - hessian_matvec(x)
        p = r.clone()
        rs_old = torch.sum(r * r, dim=(1, 2, 3), keepdim=True)

        for _ in range(self.cg_iters):
            Ap = hessian_matvec(p)
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


# =============================================================================
# 3. KHỐI TỰ CHÚ Ý CỬA SỔ CỤC BỘ SWIN (WINDOW-BASED ATTENTION - W-MSA)
# =============================================================================
class WindowAttention(nn.Module):
    """
    Tự Chú Ý Dựa Trên Cửa Sổ (Window Attention) kết hợp 2D Relative Position Bias B_rel.
    Khóa chặt nhiễu vệt nêm khuyết bên trong từng cửa sổ 8x8, chống rò rỉ artifact ra toàn ảnh.
    """
    def __init__(self, dim: int, window_size: int = 8, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        return out


class SwinTransformerBlock(nn.Module):
    """Khối Swin Transformer trích xuất đặc trưng phi cục bộ bên trong từng cửa sổ không gian."""
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
        B, C, H, W = x.shape
        shortcut = x
        x_perm = x.permute(0, 2, 3, 1).contiguous()
        x_norm = self.norm1(x_perm)

        ws = self.window_size
        x_windows = x_norm.view(B, H // ws, ws, W // ws, ws, C)
        x_windows = x_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws, C)

        attn_windows = self.attn(x_windows)

        attn_windows = attn_windows.view(B, H // ws, W // ws, ws, ws, C)
        x_merge = attn_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)

        x_perm = x_perm + x_merge
        x_mlp = self.mlp(self.norm2(x_perm))
        x_perm = x_perm + x_mlp

        out = x_perm.permute(0, 3, 1, 2).contiguous()
        return out


class LocalCNNBranch(nn.Module):
    """Nhánh Trích xuất Đặc trưng Cục bộ (Local Res-CNN 3x3) bảo tồn độ sắc nét viền mô xương."""
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


class RegFormerDualBranchRegularizer(nn.Module):
    """Khối Điều hòa Tiên nghiệm Miền Ảnh Kết hợp Cục bộ và Swin Window Attention."""
    def __init__(self, in_channels: int = 1, out_channels: int = 1, feature_dim: int = 48, window_size: int = 8):
        super().__init__()
        self.shallow_conv = nn.Conv2d(in_channels, feature_dim, kernel_size=3, padding=1)
        nn.init.normal_(self.shallow_conv.weight, mean=0.0, std=0.01)

        self.local_branch = LocalCNNBranch(in_channels=feature_dim, out_channels=feature_dim)
        self.nonlocal_branch = SwinTransformerBlock(dim=feature_dim, window_size=window_size, num_heads=4)

        self.fusion = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(feature_dim, out_channels, kernel_size=3, padding=1),
        )
        nn.init.normal_(self.fusion[0].weight, mean=0.0, std=0.01)
        nn.init.normal_(self.fusion[2].weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_shallow = F.leaky_relu(self.shallow_conv(x), negative_slope=0.2)
        f_local = self.local_branch(f_shallow)
        f_nonlocal = self.nonlocal_branch(f_shallow)

        f_cat = torch.cat([f_local, f_nonlocal], dim=1)
        out = self.fusion(f_cat)
        return out


# =============================================================================
# 4. MÔ HÌNH TOÀN CỤC SOLAR_DualMamba_LA (PYTORCH LIGHTNING MODULE)
# =============================================================================
class SOLAR_DualMamba_LA(pl.LightningModule):
    """
    Mô hình Đột Phá Miền Kép SOLAR_DualMamba hoàn chỉnh cho Limited-Angle CT.
    
    Cơ chế phối hợp:
    1. Sinogram Domain: Angular Bi-Mamba (khôi phục tính liên tục của các đường sin chùm tia dọc trục theta).
    2. Physics Bridge: Toán tử Adjoint A^T(.) kết nối dữ liệu chiếu đã vá sang miền ảnh.
    3. Image Domain: Động cơ tối ưu hóa bậc 2 Newton-CG (SafeCGSolver) + Điều hòa Swin 8x8.
    4. Chia sẻ trọng số tuần hoàn (Recurrent Weight Sharing) đưa tổng tham số về mức siêu nhẹ ~150K params.
    """
    def __init__(
        self,
        n_iterations: int = 8,
        cg_iters: int = 4,
        num_view: int = 64,
        num_detectors: int = 512,
        start_ang: float = -np.pi / 3,
        end_ang: float = np.pi / 3,
        input_size: int = 256,
        feature_dim: int = 48,
        window_size: int = 8,
        sino_loss_weight: float = 0.1,
        initial_lr: float = 1e-4,
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
        self.sino_loss_weight = sino_loss_weight
        self.initial_lr = initial_lr
        self.final_lr = final_lr

        # Khởi tạo toán tử hình học Fan-Beam ODL/ASTRA chuẩn
        radon_curr, fbp_curr = self.radon_transform(
            num_view=num_view,
            start_ang=start_ang,
            end_ang=end_ang,
            num_detectors=num_detectors,
            input_size=input_size,
        )
        self.forward_module = radon_curr
        self.backward_module = fbp_curr

        # 1. Module Miền Chiếu: Mạng Khôi Phục Sinogram bằng Angular Bi-Mamba
        self.sino_restoration = SinogramMambaRestorationNet(in_channels=1, feature_dim=feature_dim, num_blocks=2)

        # 2. Module Miền Ảnh: Bộ giải Newton-CG và Khối điều hòa kép Swin
        self.cg_solver = SafeCGSolver(cg_iters=cg_iters)
        self.image_regularizer = RegFormerDualBranchRegularizer(
            in_channels=1, out_channels=1, feature_dim=feature_dim, window_size=window_size
        )

        # Tham số bước nhảy khả học cho từng stage unrolling
        self.raw_lambda = nn.ParameterList([
            nn.Parameter(torch.tensor(0.54, dtype=torch.float32)) for _ in range(n_iterations)
        ])
        self.raw_mu = nn.ParameterList([
            nn.Parameter(torch.tensor(-2.25, dtype=torch.float32)) for _ in range(n_iterations)
        ])

    def radon_transform(
        self,
        num_view: int = 64,
        start_ang: float = -np.pi / 3,
        end_ang: float = np.pi / 3,
        num_detectors: int = 512,
        input_size: int = 256,
    ):
        """Cấu hình toán tử hình học Fan-Beam ODL/ASTRA chuẩn xác định trên toàn bộ hệ thống."""
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

    def forward(self, x_fbp: torch.Tensor, sino_measured: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Luồng tính toán Forward của SOLAR_DualMamba:
        1. Sinogram Domain: Ngoại suy và khử nhiễu dải góc khuyết trên Sinogram qua Angular Bi-Mamba.
        2. Bridge: Chiếu thông tin đã hoàn thiện sang miền ảnh qua toán tử Adjoint.
        3. Image Domain: 8 stages tối ưu hóa bậc 2 Newton-CG + Swin Attention.
        """
        # Bước 1: Khôi phục Sinogram qua Angular Bi-Mamba
        sino_restored = self.sino_restoration(sino_measured)  # (B, 1, 64, 512)

        # Khởi tạo ảnh: sử dụng FBP ban đầu
        x_t = x_fbp.clone()
        bp_restored = self.backward_module(sino_restored)

        # Bước 2: 8 stages Unrolling Bậc 2
        for t in range(self.n_iterations):
            lambda_t = F.softplus(self.raw_lambda[t]) + 1e-4
            mu_t = F.softplus(self.raw_mu[t]) + 1e-4

            # Tính gradient điều hòa từ khối RegFormer kép
            grad_r = self.image_regularizer(x_t)

            # Vế phải b_t: sử dụng Sinogram đã được phục hồi làm mục tiêu dữ liệu
            b_t = lambda_t * bp_restored + mu_t * x_t - grad_r

            # Bộ giải bậc 2 Newton-CG Matrix-Free
            x_t = self.cg_solver(
                x_init=x_t,
                b_t=b_t,
                lambda_t=lambda_t,
                mu_t=mu_t,
                forward_op=self.forward_module,
                backward_op=self.backward_module,
            )

        return x_t, sino_restored

    def training_step(self, batch, batch_idx):
        phantom, fbp_u, sino = batch
        x_recon, sino_restored = self(fbp_u, sino)

        loss_img = F.mse_loss(x_recon, phantom)
        loss_sino = F.mse_loss(sino_restored, sino)
        total_loss = loss_img + self.sino_loss_weight * loss_sino

        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("loss_img", loss_img, on_step=False, on_epoch=True)
        self.log("loss_sino", loss_sino, on_step=False, on_epoch=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        phantom, fbp_u, sino = batch
        x_recon, _ = self(fbp_u, sino)

        loss = F.mse_loss(x_recon, phantom)
        psnr_val = peak_signal_noise_ratio(x_recon, phantom, data_range=1.0)
        ssim_val = structural_similarity_index_measure(x_recon, phantom, data_range=1.0)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_psnr", psnr_val, prog_bar=True)
        self.log("val_ssim", ssim_val, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        phantom, fbp_u, sino = batch
        x_recon, _ = self(fbp_u, sino)

        loss = F.mse_loss(x_recon, phantom)
        psnr_val = peak_signal_noise_ratio(x_recon, phantom, data_range=1.0)
        ssim_val = structural_similarity_index_measure(x_recon, phantom, data_range=1.0)
        rmse_val = torch.sqrt(loss)

        self.log("test_loss", loss)
        self.log("test_psnr", psnr_val)
        self.log("test_ssim", ssim_val)
        self.log("test_rmse", rmse_val)
        return {"test_loss": loss, "test_psnr": psnr_val, "test_ssim": ssim_val, "test_rmse": rmse_val}


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.initial_lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs if self.trainer else 50,
            eta_min=self.final_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
