"""
================================================================================
SCRIPT ĐÁNH GIÁ: SOLAR_DualMamba (LIMITED-ANGLE CT)
Second-Order Dual-Domain Newton-CG Unrolling with Angular Sinogram Bi-Mamba
and Image-Domain Swin-Window Regularizer.

Dự án: Nghiên cứu Tái tạo Ảnh Cắt lớp CT Góc Giới hạn (Limited-Angle CT Reconstruction)
Tác giả: MinhPD — Nhóm Nghiên cứu Tái tạo Ảnh Y tế (VNU-HCM UIT)
================================================================================
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import numpy as np
import torch
import pytorch_lightning as pl

# Tương thích PyTorch 2.6+
_orig_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _safe_torch_load

from data.datamodule_factory import get_datamodule
from baselines.SOLAR_DualMamba.models import SOLAR_DualMamba_LA


def parse_args():
    parser = argparse.ArgumentParser(description="Kiểm thử và đánh giá mô hình đột phá miền kép SOLAR_DualMamba trên Limited-Angle CT")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Đường dẫn đến tệp trọng số checkpoint (.ckpt)")
    parser.add_argument("--dataset_type", type=str, default="aapm", choices=["aapm", "deeplesion", "lidc"])
    parser.add_argument("--dicom_dir", type=str, default="/datastore/uittogether3/LuuTru/Thanhld/CT-Reconstruction/split/")
    parser.add_argument("--cache_dir", type=str, default="/datastore/uittogether3/LuuTru/MinhPD/dataset/aapm/limited_angle/")
    parser.add_argument("--angle_range_deg", type=float, default=120.0)
    parser.add_argument("--num_view", type=int, default=64)
    parser.add_argument("--num_detectors", type=int, default=512)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--poisson_level", type=float, default=0.0)
    parser.add_argument("--gaussian_level", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()

    angle_range_deg = args.angle_range_deg
    start_ang = np.deg2rad(-angle_range_deg / 2.0)
    end_ang = np.deg2rad(angle_range_deg / 2.0)

    setting_tag = f"limited_ang_{int(angle_range_deg)}deg_numview_{args.num_view}_size_{args.input_size}"
    if args.poisson_level > 0:
        setting_tag += f"_poisson_{int(args.poisson_level)}"
    else:
        setting_tag += "_noise_0"

    print("=" * 80)
    print("🔍 ĐÁNH GIÁ MÔ HÌNH ĐỘT PHÁ MIỀN KÉP: SOLAR_DualMamba (Sinogram Bi-Mamba + Newton-CG Swin)")
    print(f"- Tệp checkpoint: {args.ckpt_path}")
    print(f"- Tập dữ liệu: {args.dataset_type.upper()} | Dải góc: {angle_range_deg}° (Views: {args.num_view})")
    print("=" * 80)

    dm = get_datamodule(
        dataset_type=args.dataset_type,
        dicom_dir=args.dicom_dir,
        cache_dir=args.cache_dir,
        setting_tag=setting_tag,
        start_ang=start_ang,
        end_ang=end_ang,
        num_view=args.num_view,
        num_detectors=args.num_detectors,
        input_size=args.input_size,
        poisson_level=args.poisson_level,
        gaussian_level=args.gaussian_level,
        use_precomputed=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = SOLAR_DualMamba_LA.load_from_checkpoint(
        args.ckpt_path,
        num_view=args.num_view,
        num_detectors=args.num_detectors,
        start_ang=start_ang,
        end_ang=end_ang,
        input_size=args.input_size,
    )
    model.eval()

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
    )

    test_results = trainer.test(model, datamodule=dm)
    print("\n📊 KẾT QUẢ ĐÁNH GIÁ ĐỊNH LƯỢNG (SOLAR_DualMamba):")
    print(test_results)


if __name__ == "__main__":
    main()
