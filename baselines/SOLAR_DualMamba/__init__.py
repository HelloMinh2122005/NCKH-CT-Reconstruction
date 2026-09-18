"""
================================================================================
GÓI MODULE: SOLAR_DualMamba (LIMITED-ANGLE CT RECONSTRUCTION)
Kiến trúc Tái tạo Ảnh Cắt lớp CT Miền Kép (Dual-Domain Unrolling):
- Miền Chiếu (Sinogram Domain): Angular Bi-Mamba (Bidirectional State Space Model)
- Cầu Nối Vật Lý: Toán tử Adjoint Vi phân Fan-beam A^T(.)
- Miền Ảnh (Image Domain): Động cơ Tối ưu hóa Bậc 2 Newton-CG (SafeCGSolver)
  kết hợp Khối Điều hòa Kép Swin Window Attention (W-MSA 8x8) và Local Res-CNN 3x3.

Tác giả: MinhPD — Nhóm Nghiên cứu Tái tạo Ảnh Y tế (VNU-HCM UIT)
================================================================================
"""

from .models import (
    AngularSinogramMambaBlock,
    SinogramMambaRestorationNet,
    SafeCGSolver,
    WindowAttention,
    SwinTransformerBlock,
    LocalCNNBranch,
    RegFormerDualBranchRegularizer,
    SOLAR_DualMamba_LA,
)

__all__ = [
    "AngularSinogramMambaBlock",
    "SinogramMambaRestorationNet",
    "SafeCGSolver",
    "WindowAttention",
    "SwinTransformerBlock",
    "LocalCNNBranch",
    "RegFormerDualBranchRegularizer",
    "SOLAR_DualMamba_LA",
]
