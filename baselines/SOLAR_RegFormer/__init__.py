"""
================================================================================
GÓI MÔ HÌNH: SOLAR-RegFormer (LIMITED-ANGLE CT RECONSTRUCTION)
Second-Order Dual-Branch Newton-CG Unrolling with Swin Window Regularizer
Dự án: Nghiên cứu Tái tạo Ảnh Cắt lớp CT Góc Giới hạn (Limited-Angle CT Reconstruction)
Tác giả: MinhPD — Nhóm Nghiên cứu Tái tạo Ảnh Y tế (VNU-HCM UIT)
================================================================================
"""

from baselines.SOLAR_RegFormer.models import (
    SOLAR_RegFormer_LA,
    SafeCGSolver,
    WindowAttention,
    SwinTransformerBlock,
    LocalCNNBranch,
    RegFormerDualBranchRegularizer,
)

__all__ = [
    "SOLAR_RegFormer_LA",
    "SafeCGSolver",
    "WindowAttention",
    "SwinTransformerBlock",
    "LocalCNNBranch",
    "RegFormerDualBranchRegularizer",
]
