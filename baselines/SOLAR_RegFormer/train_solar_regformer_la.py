"""
================================================================================
SCRIPT HUẤN LUYỆN: SOLAR_RegFormer (LIMITED-ANGLE CT)
Second-Order Dual-Branch Newton-CG Unrolling with Swin Window Regularizer
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

# Import DataModule Factory và Mô hình SOLAR_RegFormer
from data.datamodule_factory import get_datamodule
from baselines.SOLAR_RegFormer.models import SOLAR_RegFormer_LA


def parse_args():
    """
    Khai báo và phân tích toàn bộ tham số dòng lệnh phục vụ huấn luyện SOLAR_RegFormer trên Limited-Angle CT.
    """
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình đề xuất SOTA SOLAR_RegFormer trên Limited-Angle CT")
    
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
        dest="cache_dir",
        type=str,
        default="/datastore/uittogether3/LuuTru/MinhPD/dataset/aapm/limited_angle/",
        help="Đường dẫn thư mục chứa dữ liệu .npy tiền xử lý"
    )
    parser.add_argument(
        "--train_patients",
        nargs="+",
        default=None,
        help="Danh sách mã bệnh nhân cho tập huấn luyện"
    )
    parser.add_argument(
        "--val_patients",
        nargs="+",
        default=None,
        help="Danh sách mã bệnh nhân cho tập kiểm định"
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
        help="Số kênh đặc trưng của khối điều hòa kép Local CNN và Swin Transformer (mặc định = 48)"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=8,
        help="Kích thước cửa sổ Window Attention của Swin Transformer (mặc định = 8x8)"
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
        default=35,
        help="Tổng số epoch huấn luyện tối đa (mặc định = 35 epochs)"
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
        default="/datastore/uittogether3/LuuTru/MinhPD/saved_models/SOLAR_RegFormer/",
        help="Thư mục lưu trữ checkpoint trọng số mô hình (.ckpt)"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="lightning_logs/SOLAR_RegFormer/",
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
    print("🚀 BẮT ĐẦU HUẤN LUYỆN: SOLAR_RegFormer (Limited-Angle CT)")
    print(f"- Dải góc quét: [{start_ang_deg:.1f}°, {end_ang_deg:.1f}°] (Cung quét {angle_range_deg:.1f}°)")
    print(f"- Số views: {args.num_view} | Detectors: {args.num_detectors} | Kích thước ảnh: {args.input_size}x{args.input_size}")
    print(f"- Cấu hình Cache Tag: {setting_tag}")
    print(f"- Unrolling Bậc 2: {args.n_iterations} stages | Số bước CG mỗi stage: {args.cg_iters} steps")
    print(f"- Điều hòa kép: Local CNN + Swin Transformer (Window size {args.window_size}x{args.window_size}, Dim {args.feature_dim})")
    print(f"- Chia sẻ trọng số tuần hoàn (Recurrent Sharing): 1 khối duy nhất (~0.28M params)")
    print(f"- Max Epochs: {args.max_epochs} | Batch size: {args.batch_size} | LR: {args.initial_lr} -> {args.final_lr}")
    print(f"- Checkpoint lưu tại: {args.output_dir}")
    print("=" * 80)

    # 1. Khởi tạo DataModule
    datamodule = get_datamodule(
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
        use_precomputed=args.use_precomputed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_patients=args.train_patients,
        val_patients=args.val_patients,
    )

    # 2. Khởi tạo Mô hình SOLAR_RegFormer
    model = SOLAR_RegFormer_LA(
        n_iterations=args.n_iterations,
        cg_iters=args.cg_iters,
        num_view=args.num_view,
        num_detectors=args.num_detectors,
        start_ang=start_ang,
        end_ang=end_ang,
        input_size=args.input_size,
        feature_dim=args.feature_dim,
        window_size=args.window_size,
        initial_lr=args.initial_lr,
        final_lr=args.final_lr,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name="SOLAR_RegFormer_LA",
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="solar_regformer_la-{epoch:02d}-{val_psnr:.2f}-{val_ssim:.4f}",
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
    print("\n🎉 Huấn luyện SOLAR_RegFormer hoàn thành thành công!")


if __name__ == "__main__":
    main()
