"""
================================================================================
SCRIPT HUẤN LUYỆN: SOLAR_DualMamba (LIMITED-ANGLE CT)
Second-Order Dual-Domain Newton-CG Unrolling with Angular Sinogram Bi-Mamba
and Image-Domain Swin-Window Regularizer.

Dự án: Nghiên cứu Tái tạo Ảnh Cắt lớp CT Góc Giới hạn (Limited-Angle CT Reconstruction)
Tác giả: MinhPD — Nhóm Nghiên cứu Tái tạo Ảnh Y tế (VNU-HCM UIT)
================================================================================
"""

import os
import sys
from pathlib import Path

# Đảm bảo thư mục gốc dự án luôn nằm trong sys.path để import các module data/ và baselines/
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

# Import DataModule Factory và Mô hình SOLAR_DualMamba
from data.datamodule_factory import get_datamodule
from baselines.SOLAR_DualMamba.models import SOLAR_DualMamba_LA


def parse_args():
    """
    Khai báo và phân tích toàn bộ tham số dòng lệnh phục vụ huấn luyện SOLAR_DualMamba trên Limited-Angle CT.
    """
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình đột phá miền kép SOLAR_DualMamba trên Limited-Angle CT")
    
    # 1. Cấu hình Tập dữ liệu
    parser.add_argument(
        "--dataset_type", "--dataset_name",
        dest="dataset_type",
        type=str,
        choices=["aapm", "deeplesion", "lidc"],
        default="aapm",
        help="Lựa chọn tập dữ liệu huấn luyện: 'aapm', 'deeplesion' hoặc 'lidc' (mặc định: 'aapm')"
    )
    parser.add_argument(
        "--dicom_dir",
        type=str,
        default="/datastore/uittogether3/LuuTru/Thanhld/CT-Reconstruction/split/",
        help="Đường dẫn thư mục chứa ảnh DICOM (.IMA) gốc AAPM để làm Ground Truth"
    )
    parser.add_argument(
        "--dataset_dir", "--data_dir", "--cache_dir",
        dest="dataset_dir",
        type=str,
        default="/datastore/uittogether3/LuuTru/MinhPD/dataset/aapm/limited_angle/",
        help="Đường dẫn thư mục cache chứa dữ liệu đã sinh (.npy)"
    )
    parser.add_argument(
        "--train_patients",
        nargs="+",
        default=None,
        help="Danh sách mã bệnh nhân dùng cho tập huấn luyện (Train)"
    )
    parser.add_argument(
        "--val_patients",
        nargs="+",
        default=None,
        help="Danh sách mã bệnh nhân dùng cho tập kiểm định (Validation)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Kích thước mini-batch cho mỗi bước huấn luyện (mặc định = 1 lát cắt)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Số luồng CPU nạp dữ liệu song song qua DataLoader"
    )
    
    # 2. Cấu hình Hình học & Vật lý CT góc giới hạn (LA-CT Physics)
    parser.add_argument(
        "--angle_range_deg",
        type=float,
        default=120.0,
        help="Độ rộng cung quét góc giới hạn tính bằng độ (ví dụ: 120 độ)"
    )
    parser.add_argument(
        "--start_ang_deg",
        type=float,
        default=None,
        help="Góc quét bắt đầu tính bằng độ"
    )
    parser.add_argument(
        "--end_ang_deg",
        type=float,
        default=None,
        help="Góc quét kết thúc tính bằng độ"
    )
    parser.add_argument(
        "--num_view",
        type=int,
        default=64,
        help="Số góc chiếu (views) trong dải góc giới hạn (64 views)"
    )
    parser.add_argument(
        "--num_detectors",
        type=int,
        default=512,
        help="Số lượng phần tử cảm biến trên thanh detector (512 detectors)"
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=256,
        help="Độ phân giải không gian ảnh đầu vào và đầu ra (256x256 pixel)"
    )
    parser.add_argument(
        "--feature_dim",
        type=int,
        default=48,
        help="Số kênh đặc trưng của khối điều hòa kép và khối Sinogram Mamba (mặc định = 48)"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=8,
        help="Kích thước cửa sổ Window Attention của Swin Transformer (mặc định = 8x8)"
    )
    parser.add_argument(
        "--sino_loss_weight",
        type=float,
        default=0.1,
        help="Trọng số tổn thất tái tạo sinogram miền chiếu (mặc định = 0.1)"
    )
    
    # 3. Cấu hình Nhiễu
    parser.add_argument(
        "--poisson_level",
        type=float,
        default=0.0,
        help="Mức photon mô phỏng nhiễu Poisson"
    )
    parser.add_argument(
        "--gaussian_level",
        type=float,
        default=0.0,
        help="Độ lệch chuẩn mô phỏng nhiễu Gaussian"
    )
    parser.add_argument(
        "--use_precomputed",
        action="store_true",
        default=True,
        help="Đọc trực tiếp từ file cache .npy đã tính sẵn"
    )
    
    # 4. Siêu tham số SOLAR Bậc 2
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=8,
        help="Số giai đoạn unrolling tối ưu bậc 2 của SOLAR (mặc định = 8 stages)"
    )
    parser.add_argument(
        "--cg_iters",
        type=int,
        default=4,
        help="Số bước lặp Conjugate Gradient Matrix-Free mỗi stage (mặc định = 4 steps)"
    )
    
    # 5. Cấu hình Huấn luyện
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=50,
        help="Tổng số epoch huấn luyện tối đa (mặc định = 50 epochs)"
    )
    parser.add_argument(
        "--lr", "--initial_lr",
        dest="initial_lr",
        type=float,
        default=1e-4,
        help="Tốc độ học ban đầu của bộ tối ưu Adam (mặc định = 1e-4)"
    )
    parser.add_argument(
        "--final_lr",
        type=float,
        default=1e-5,
        help="Tốc độ học tối thiểu trong lịch hạ Cosine Annealing (mặc định = 1e-5)"
    )
    
    # 6. Thư mục Checkpoint & Logging
    parser.add_argument(
        "--output_dir", "--checkpoints_dir",
        dest="output_dir",
        type=str,
        default="/datastore/uittogether3/LuuTru/MinhPD/saved_models/SOLAR_DualMamba/",
        help="Thư mục lưu trữ checkpoint trọng số mô hình (.ckpt)"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="lightning_logs/SOLAR_DualMamba/",
        help="Thư mục lưu log TensorBoard"
    )
    parser.add_argument(
        "--resume_ckpt", "--resume_from_checkpoint",
        dest="resume_ckpt",
        type=str,
        default=None,
        help="Đường dẫn đến checkpoint .ckpt để tiếp tục huấn luyện (Resume Training)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()

    # Tính toán góc bắt đầu và kết thúc (radian)
    angle_range_deg = args.angle_range_deg
    if args.start_ang_deg is not None and args.end_ang_deg is not None:
        start_ang = np.deg2rad(args.start_ang_deg)
        end_ang = np.deg2rad(args.end_ang_deg)
    else:
        start_ang = np.deg2rad(-angle_range_deg / 2.0)
        end_ang = np.deg2rad(angle_range_deg / 2.0)

    # Chuẩn hóa setting tag
    setting_tag = f"limited_ang_{int(angle_range_deg)}deg_numview_{args.num_view}_size_{args.input_size}"
    if args.poisson_level > 0:
        setting_tag += f"_poisson_{int(args.poisson_level)}"
    else:
        setting_tag += "_noise_0"

    print("=" * 80)
    print("🚀 KHỞI ĐỘNG HUẤN LUYỆN KIẾN TRÚC ĐỘT PHÁ MIỀN KÉP: SOLAR_DualMamba")
    print("   (Dual-Domain Angular Sinogram Bi-Mamba + Newton-CG Swin Regularizer)")
    print(f"- Tập dữ liệu: {args.dataset_type.upper()} ({setting_tag})")
    print(f"- Dải góc quét: {angle_range_deg}° (Views: {args.num_view}, Detectors: {args.num_detectors})")
    print(f"- Kích thước ảnh: {args.input_size}x{args.input_size} | Unrolling Stages: {args.n_iterations}")
    print(f"- Số Epochs: {args.max_epochs} | Batch size: {args.batch_size} | CG iters: {args.cg_iters}")
    print(f"- Thư mục Checkpoint: {args.output_dir}")
    print("=" * 80)

    # 1. Khởi tạo DataModule Factory
    datamodule = get_datamodule(
        dataset_type=args.dataset_type,
        dicom_dir=args.dicom_dir,
        cache_dir=args.dataset_dir,
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

    # 2. Khởi tạo Mô hình SOLAR_DualMamba
    model = SOLAR_DualMamba_LA(
        n_iterations=args.n_iterations,
        cg_iters=args.cg_iters,
        num_view=args.num_view,
        num_detectors=args.num_detectors,
        start_ang=start_ang,
        end_ang=end_ang,
        input_size=args.input_size,
        feature_dim=args.feature_dim,
        window_size=args.window_size,
        sino_loss_weight=args.sino_loss_weight,
        initial_lr=args.initial_lr,
        final_lr=args.final_lr,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name="SOLAR_DualMamba_LA",
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="solar_dualmamba_la-{epoch:02d}-{val_psnr:.2f}-{val_ssim:.4f}",
        save_top_k=3,
        monitor="val_psnr",
        mode="max",
        save_last=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # 3. Khởi tạo PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=10,
    )

    # 4. Bắt đầu quá trình huấn luyện
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume_ckpt)
    print("\n🎉 Huấn luyện SOLAR_DualMamba hoàn thành thành công!")


if __name__ == "__main__":
    main()
