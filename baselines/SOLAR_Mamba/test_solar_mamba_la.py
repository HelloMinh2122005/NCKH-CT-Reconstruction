"""
================================================================================
SCRIPT ĐÁNH GIÁ: SOLAR_Mamba (LIMITED-ANGLE CT)
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

_orig_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _safe_torch_load

from data.datamodule_LA import LimitedAngleCTDataModule
from baselines.SOLAR_Mamba.models import SOLAR_Mamba_LA


def parse_args():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình đề xuất SOLAR_Mamba trên Limited-Angle CT")
    
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--dicom_dir", type=str, default="/datastore/uittogether3/LuuTru/Thanhld/CT-Reconstruction/split/")
    parser.add_argument("--dataset_dir", "--data_dir", "--cache_dir", dest="cache_dir", type=str, default="/datastore/uittogether3/LuuTru/MinhPD/dataset/limited_angle/")
    parser.add_argument("--test_patients", nargs="+", default=["L310"])
    
    parser.add_argument("--angle_range_deg", type=float, default=120.0)
    parser.add_argument("--start_ang_deg", type=float, default=None)
    parser.add_argument("--end_ang_deg", type=float, default=None)
    parser.add_argument("--num_view", type=int, default=64)
    parser.add_argument("--num_detectors", type=int, default=512)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--poisson_level", type=float, default=0.0)
    parser.add_argument("--gaussian_level", type=float, default=0.0)
    parser.add_argument("--use_precomputed", action="store_true", default=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    angle_range_deg = float(args.angle_range_deg)
    if args.start_ang_deg is not None and args.end_ang_deg is not None:
        start_ang_deg = float(args.start_ang_deg)
        end_ang_deg = float(args.end_ang_deg)
    else:
        start_ang_deg = -angle_range_deg / 2.0
        end_ang_deg = angle_range_deg / 2.0
        
    start_ang = np.deg2rad(start_ang_deg)
    end_ang = np.deg2rad(end_ang_deg)
    
    setting_tag = f"limited_ang_{int(angle_range_deg)}deg_numview_{args.num_view}_size_{args.input_size}"
    if args.poisson_level > 0:
        setting_tag += f"_poisson_{int(args.poisson_level)}"
    else:
        setting_tag += "_noise_0"

    print("=" * 80)
    print("📊 BẮT ĐẦU ĐÁNH GIÁ BENCHMARK: SOLAR_Mamba")
    print(f"- Checkpoint: {args.checkpoint_path}")
    print(f"- Cung quét: [{start_ang_deg:.1f}°, {end_ang_deg:.1f}°] ({angle_range_deg:.1f}°)")
    print(f"- Views: {args.num_view} | Detectors: {args.num_detectors} | Size: {args.input_size}x{args.input_size}")
    print("=" * 80)

    datamodule = LimitedAngleCTDataModule(
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
        use_precomputed=args.use_precomputed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_patients=args.test_patients,
    )
    datamodule.setup(stage="test")

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Không tìm thấy file checkpoint: {args.checkpoint_path}")

    model = SOLAR_Mamba_LA.load_from_checkpoint(
        args.checkpoint_path,
        start_ang=start_ang,
        end_ang=end_ang,
        num_view=args.num_view,
        num_detectors=args.num_detectors,
        input_size=args.input_size,
    )
    model.eval()

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
    )

    results = trainer.test(model, datamodule=datamodule)
    
    print("\n" + "=" * 80)
    print("🏆 KẾT QUẢ BENCHMARK SOLAR_Mamba:")
    for res in results:
        for metric, val in res.items():
            print(f"  * {metric}: {val:.4f}" if isinstance(val, float) else f"  * {metric}: {val}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
