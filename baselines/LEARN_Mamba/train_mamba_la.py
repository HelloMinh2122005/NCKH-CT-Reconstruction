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

# Import DataModule và Mô hình kiến trúc
from data.datamodule_LA import LimitedAngleCTDataModule
from baselines.LEARN_Mamba.models import LEARN_Mamba_LA


def parse_args():
    """
    Khai báo và phân tích toàn bộ tham số dòng lệnh phục vụ huấn luyện LEARN_Mamba trên Limited-Angle CT.
    Bao gồm 3 nhóm tham số chính:
    1. Cấu hình đường dẫn dữ liệu (DICOM gốc, Cache npy, Danh sách bệnh nhân).
    2. Cấu hình hình học & vật lý chụp CT (Dải góc quét giới hạn, Số views, Detectors, Mức nhiễu).
    3. Cấu hình mô hình Unrolling và chiến lược tối ưu hóa (Số giai đoạn, Tốc độ học, Epochs, Checkpoint).
    """
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Baseline LEARN_Mamba trên Limited-Angle CT")
    
    # -------------------------------------------------------------------------
    # 1. Cấu hình Dữ liệu (Data Configuration)
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--dicom_dir",
        type=str,
        default="/datastore/uittogether3/LuuTru/Thanhld/CT-Reconstruction/split/",
        help="Đường dẫn thư mục chứa ảnh DICOM (.IMA) gốc AAPM Mayo Clinic để làm Ground Truth"
    )
    parser.add_argument(
        "--dataset_dir", "--data_dir", "--cache_dir",
        dest="cache_dir",
        type=str,
        default="/datastore/uittogether3/LuuTru/MinhPD/dataset/limited_angle/",
        help="Đường dẫn thư mục chứa các file .npy tiền xử lý (sinogram và FBP ban đầu)"
    )
    parser.add_argument(
        "--train_patients",
        nargs="+",
        default=None,
        help="Danh sách mã bệnh nhân cho tập huấn luyện (mặc định lấy 8 bệnh nhân AAPM)"
    )
    parser.add_argument(
        "--val_patients",
        nargs="+",
        default=None,
        help="Danh sách mã bệnh nhân cho tập kiểm định (mặc định bệnh nhân L333)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Kích thước mini-batch cho mỗi bước huấn luyện (mặc định = 1 lát cắt để tối ưu VRAM)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Số luồng CPU nạp dữ liệu song song qua DataLoader"
    )
    
    # -------------------------------------------------------------------------
    # 2. Cấu hình Hình học & Vật lý CT góc giới hạn (LA-CT Physics)
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--angle_range_deg",
        type=float,
        default=120.0,
        help="Độ rộng cung quét góc giới hạn tính bằng độ (ví dụ: 120.0 độ = [-60°, +60°])"
    )
    parser.add_argument(
        "--start_ang_deg",
        type=float,
        default=None,
        help="Góc quét bắt đầu tính bằng độ (nếu None sẽ tự động tính là -angle_range_deg / 2)"
    )
    parser.add_argument(
        "--end_ang_deg",
        type=float,
        default=None,
        help="Góc quét kết thúc tính bằng độ (nếu None sẽ tự động tính là +angle_range_deg / 2)"
    )
    parser.add_argument(
        "--num_view",
        type=int,
        default=64,
        help="Số góc chiếu (projection views) trong dải góc giới hạn (64 views)"
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
        "--poisson_level",
        type=float,
        default=0.0,
        help="Mức photon mô phỏng nhiễu Poisson (0 = noise_0 không nhiễu)"
    )
    parser.add_argument(
        "--gaussian_level",
        type=float,
        default=0.0,
        help="Độ lệch chuẩn mô phỏng nhiễu Gaussian (0 = không thêm nhiễu)"
    )
    parser.add_argument(
        "--use_precomputed",
        action="store_true",
        default=True,
        help="Đọc trực tiếp từ dữ liệu cache .npy đã tiền xử lý thay vì tính toán on-the-fly qua ODL"
    )
    
    # -------------------------------------------------------------------------
    # 3. Cấu hình Mô hình & Tối ưu hóa (Model & Optimization)
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=14,
        help="Số giai đoạn lặp Unrolling của thuật toán LEARN (K = 14 stages)"
    )
    parser.add_argument(
        "--lr", "--initial_lr",
        dest="initial_lr",
        type=float,
        default=1e-4,
        help="Tốc độ học ban đầu (Initial Learning Rate) của bộ tối ưu Adam"
    )
    parser.add_argument(
        "--final_lr",
        type=float,
        default=1e-5,
        help="Tốc độ học tối thiểu (Minimum Learning Rate) trong lịch hạ Cosine Annealing"
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=50,
        help="Tổng số epoch huấn luyện tối đa"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="lightning_logs",
        help="Thư mục lưu log TensorBoard"
    )
    parser.add_argument(
        "--output_dir", "--checkpoints_dir",
        dest="output_dir",
        type=str,
        default="/datastore/uittogether3/LuuTru/MinhPD/saved_models/LEARN_Mamba/",
        help="Thư mục lưu trữ checkpoint trọng số mô hình tốt nhất (.ckpt)"
    )
    
    return parser.parse_args()


def main():
    # Bước 1: Đọc các tham số cấu hình
    args = parse_args()
    
    # Bước 2: Tính toán góc quét (chuyển đổi từ Degree sang Radian)
    if args.start_ang_deg is not None and args.end_ang_deg is not None:
        start_ang_deg = args.start_ang_deg
        end_ang_deg = args.end_ang_deg
        angle_range_deg = end_ang_deg - start_ang_deg
    else:
        angle_range_deg = args.angle_range_deg
        start_ang_deg = -angle_range_deg / 2.0
        end_ang_deg = angle_range_deg / 2.0
        
    start_ang = np.deg2rad(start_ang_deg)
    end_ang = np.deg2rad(end_ang_deg)
    
    # Bước 3: Định danh Setting Tag để khớp với cấu trúc thư mục cache npy
    setting_tag = f"limited_ang_{int(angle_range_deg)}deg_numview_{args.num_view}_size_{args.input_size}"
    if args.poisson_level > 0:
        setting_tag += f"_poisson_{int(args.poisson_level)}"
    else:
        setting_tag += "_noise_0"
    
    print("=" * 80)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN BASELINE: LEARN_Mamba (Limited-Angle CT)")
    print(f"- Dải góc quét: [{start_ang_deg:.1f}°, {end_ang_deg:.1f}°] (Cung quét {angle_range_deg:.1f}°)")
    print(f"- Số views: {args.num_view} | Số Detectors: {args.num_detectors} | Kích thước ảnh: {args.input_size}x{args.input_size}")
    print(f"- Cấu hình Cache Tag: {setting_tag}")
    print(f"- Số giai đoạn Unrolling: {args.n_iterations} stages | Max Epochs: {args.max_epochs} | Batch size: {args.batch_size}")
    print(f"- Tốc độ học: {args.initial_lr} -> {args.final_lr} (Cosine Annealing)")
    print(f"- Checkpoint lưu tại: {args.output_dir}")
    print("=" * 80)

    # Bước 4: Khởi tạo PyTorch Lightning DataModule
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

    # Bước 5: Khởi tạo Mô hình kiến trúc LEARN_Mamba_LA
    model = LEARN_Mamba_LA(
        n_iterations=args.n_iterations,
        num_view=args.num_view,
        num_detectors=args.num_detectors,
        start_ang=start_ang,
        end_ang=end_ang,
        input_size=args.input_size,
        initial_lr=args.initial_lr,
        final_lr=args.final_lr,
    )

    # Bước 6: Tạo các thư mục lưu trữ nếu chưa tồn tại
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Bước 7: Thiết lập Logger TensorBoard và Callbacks
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name="LEARN_Mamba_LA",
    )
    
    # Callback tự động lưu 3 checkpoint có chỉ số val_psnr cao nhất
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="mamba_la-{epoch:02d}-{val_psnr:.2f}-{val_ssim:.4f}",
        save_top_k=3,
        monitor="val_psnr",
        mode="max",
        save_last=True,
    )
    
    # Callback theo dõi sự thay đổi của Learning Rate qua từng bước huấn luyện
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Bước 8: Khởi tạo Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=10,
    )

    # Bước 9: Thực thi quy trình huấn luyện
    trainer.fit(model, datamodule=datamodule)
    print("\n🎉 Huấn luyện LEARN_Mamba hoàn thành thành công!")


if __name__ == "__main__":
    main()
