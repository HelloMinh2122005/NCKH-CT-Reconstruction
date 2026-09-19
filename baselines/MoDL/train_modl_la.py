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

# Tương thích PyTorch 2.6+
_orig_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _safe_torch_load

from data.datamodule_factory import get_datamodule
from baselines.MoDL.models import MoDL_LA


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Baseline MoDL trên Limited-Angle CT")
    
    # 1. Dataset
    parser.add_argument("--dataset_type", dest="dataset_type", type=str, choices=["aapm", "deeplesion", "lidc"], default="aapm")
    parser.add_argument("--dicom_dir", type=str, default="/datastore/uittogether3/LuuTru/Thanhld/CT-Reconstruction/split/")
    parser.add_argument("--dataset_dir", "--data_dir", "--cache_dir", dest="cache_dir", type=str, default="/datastore/uittogether3/LuuTru/MinhPD/dataset/aapm/limited_angle/")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # 2. Physics LA-CT
    parser.add_argument("--angle_range_deg", type=float, default=120.0)
    parser.add_argument("--start_ang_deg", type=float, default=None)
    parser.add_argument("--end_ang_deg", type=float, default=None)
    parser.add_argument("--num_view", type=int, default=64)
    parser.add_argument("--num_detectors", type=int, default=512)
    parser.add_argument("--input_size", type=int, default=256)
    
    # 3. Model Hyperparameters
    parser.add_argument("--n_iterations", type=int, default=10, help="Số stages unrolling của MoDL (mặc định: 10)")
    parser.add_argument("--cg_iters", type=int, default=6, help="Số bước lặp Conjugate Gradient cho Data Consistency (mặc định: 6)")
    parser.add_argument("--initial_lr", type=float, default=2e-4)
    parser.add_argument("--final_lr", type=float, default=1e-5)
    
    # 4. Training
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--saved_models_dir", type=str, default="/datastore/uittogether3/LuuTru/MinhPD/saved_models/")
    parser.add_argument("--default_root_dir", type=str, default="/datastore/uittogether3/LuuTru/MinhPD/scripts/output/train_modl_la/")
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)

    if args.start_ang_deg is None:
        start_ang_deg = -args.angle_range_deg / 2.0
        end_ang_deg = args.angle_range_deg / 2.0
    else:
        start_ang_deg = args.start_ang_deg
        end_ang_deg = args.end_ang_deg

    start_ang = float(np.deg2rad(start_ang_deg))
    end_ang = float(np.deg2rad(end_ang_deg))

    # Cấu hình đường dẫn lưu model
    if args.dataset_type == "aapm":
        ckpt_dir = os.path.join(args.saved_models_dir, "MoDL")
    else:
        ckpt_dir = os.path.join(args.saved_models_dir, args.dataset_type, "MoDL")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 1. Khởi tạo DataModule
    dm = get_datamodule(
        dataset_name=args.dataset_type,
        data_dir=args.cache_dir,
        dicom_dir=args.dicom_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_view=args.num_view,
        num_detectors=args.num_detectors,
        start_ang_deg=start_ang_deg,
        end_ang_deg=end_ang_deg,
        input_size=args.input_size,
    )

    # 2. Khởi tạo Mô hình MoDL_LA
    model = MoDL_LA(
        n_iterations=args.n_iterations,
        cg_iters=args.cg_iters,
        num_view=args.num_view,
        num_detectors=args.num_detectors,
        start_ang=start_ang,
        end_ang=end_ang,
        input_size=args.input_size,
        initial_lr=args.initial_lr,
        final_lr=args.final_lr,
    )

    # 3. Callbacks & Logger
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="modl_la-{epoch:02d}-{val_psnr:.2f}-{val_ssim:.4f}",
        monitor="val_psnr",
        mode="max",
        save_top_k=3,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    tb_logger = TensorBoardLogger(save_dir=args.default_root_dir, name="tensorboard")

    # 4. Trainer
    last_ckpt_path = os.path.join(ckpt_dir, "last.ckpt")
    resume_ckpt = last_ckpt_path if os.path.exists(last_ckpt_path) else None

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=tb_logger,
        default_root_dir=args.default_root_dir,
        log_every_n_steps=10,
    )

    print(f"[INFO] Bắt đầu huấn luyện MoDL (K={args.n_iterations} stages, CG={args.cg_iters} iters) trên dải góc {args.angle_range_deg}°...")
    trainer.fit(model, datamodule=dm, ckpt_path=resume_ckpt)


if __name__ == "__main__":
    main()
