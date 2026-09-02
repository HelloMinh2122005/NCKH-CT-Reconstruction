"""
================================================================================
SCRIPT HUẤN LUYỆN: SOLAR_Mamba (LIMITED-ANGLE CT)
Second-Order Dual-Branch Unrolling with Mamba Selective SSM
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
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

_orig_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _safe_torch_load

from data.datamodule_LA import LimitedAngleCTDataModule
from baselines.SOLAR_Mamba.models import SOLAR_Mamba_LA


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình đề xuất SOLAR_Mamba trên Limited-Angle CT")
    
    # Data
    parser.add_argument("--dicom_dir", type=str, default="/datastore/uittogether3/LuuTru/Thanhld/CT-Reconstruction/split/")
    parser.add_argument("--dataset_dir", "--data_dir", "--cache_dir", dest="cache_dir", type=str, default="/datastore/uittogether3/LuuTru/MinhPD/dataset/limited_angle/")
    parser.add_argument("--train_patients", nargs="+", default=None)
    parser.add_argument("--val_patients", nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # CT Physics
    parser.add_argument("--angle_range_deg", type=float, default=120.0)
    parser.add_argument("--start_ang_deg", type=float, default=None)
    parser.add_argument("--end_ang_deg", type=float, default=None)
    parser.add_argument("--num_view", type=int, default=64)
    parser.add_argument("--num_detectors", type=int, default=512)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--window_size", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    
    # Noise
    parser.add_argument("--poisson_level", type=float, default=0.0)
    parser.add_argument("--gaussian_level", type=float, default=0.0)
    parser.add_argument("--use_precomputed", action="store_true", default=True)
    
    # SOLAR Hyperparameters
    parser.add_argument("--n_iterations", type=int, default=8)
    parser.add_argument("--cg_iters", type=int, default=4)
    
    # Training
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--lr", "--initial_lr", dest="initial_lr", type=float, default=1e-4)
    parser.add_argument("--final_lr", type=float, default=1e-5)
    
    # Checkpoint & Logging
    parser.add_argument("--output_dir", type=str, default="/datastore/uittogether3/LuuTru/MinhPD/saved_models/SOLAR_Mamba/")
    parser.add_argument("--log_dir", type=str, default="lightning_logs/SOLAR_Mamba/")
    parser.add_argument("--resume_ckpt", type=str, default=None)
    
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
    print("🚀 BẮT ĐẦU HUẤN LUYỆN: SOLAR_Mamba (Limited-Angle CT)")
    print(f"- Dải góc quét: [{start_ang_deg:.1f}°, {end_ang_deg:.1f}°] (Cung quét {angle_range_deg:.1f}°)")
    print(f"- Số views: {args.num_view} | Detectors: {args.num_detectors} | Kích thước ảnh: {args.input_size}x{args.input_size}")
    print(f"- Cấu hình Cache Tag: {setting_tag}")
    print(f"- Unrolling Bậc 2: {args.n_iterations} stages | Số bước CG mỗi stage: {args.cg_iters} steps")
    print(f"- Max Epochs: {args.max_epochs} | Batch size: {args.batch_size} | LR: {args.initial_lr} -> {args.final_lr}")
    print(f"- Checkpoint lưu tại: {args.output_dir}")
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
        train_patients=args.train_patients,
        val_patients=args.val_patients,
    )

    model = SOLAR_Mamba_LA(
        n_iterations=args.n_iterations,
        cg_iters=args.cg_iters,
        num_view=args.num_view,
        num_detectors=args.num_detectors,
        start_ang=start_ang,
        end_ang=end_ang,
        input_size=args.input_size,
        window_size=args.window_size,
        num_heads=args.num_heads,
        initial_lr=args.initial_lr,
        final_lr=args.final_lr,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name="SOLAR_Mamba_LA",
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="solar_mamba_la-{epoch:02d}-{val_psnr:.2f}-{val_ssim:.4f}",
        save_top_k=3,
        monitor="val_psnr",
        mode="max",
        save_last=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume_ckpt)
    print("\n🎉 Huấn luyện SOLAR_Mamba hoàn thành thành công!")


if __name__ == "__main__":
    main()
