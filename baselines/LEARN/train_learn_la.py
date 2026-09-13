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

# Tương thích PyTorch 2.6+ (tránh lỗi WeightsUnpickler khi nạp checkpoint Lightning chứa metadata numpy)
_orig_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _safe_torch_load

# Import DataModule Factory và Mô hình LEARN
from data.datamodule_factory import get_datamodule
from baselines.LEARN.models import LEARN_LA



def parse_args():
    """
    Khai báo và phân tích toàn bộ tham số dòng lệnh phục vụ huấn luyện LEARN (Original) trên Limited-Angle CT.
    """
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Baseline LEARN (Original) trên Limited-Angle CT")
    
    # -------------------------------------------------------------------------
    # 1. Cấu hình Tập dữ liệu (Dataset Configuration)
    # -------------------------------------------------------------------------
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
        help="Kích thước mini-batch cho mỗi bước huấn luyện (mặc định = 1 lát cắt)"
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
        help="Độ rộng cung quét góc giới hạn tính bằng độ (ví dụ: 120 độ)"
    )
    parser.add_argument(
        "--start_ang_deg",
        type=float,
        default=None,
        help="Góc quét bắt đầu tính bằng độ (mặc định: -angle_range_deg/2)"
    )
    parser.add_argument(
        "--end_ang_deg",
        type=float,
        default=None,
        help="Góc quét kết thúc tính bằng độ (mặc định: +angle_range_deg/2)"
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
        help="Đọc trực tiếp từ file cache .npy đã tính sẵn để tăng tốc"
    )
    
    # -------------------------------------------------------------------------
    # 3. Cấu hình Mô hình & Tối ưu hóa (Model & Optimization)
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=14,
        help="Số giai đoạn unrolling của thuật toán LEARN (K = 14 stages)"
    )
    parser.add_argument(
        "--lr", "--initial_lr",
        dest="initial_lr",
        type=float,
        default=1e-4,
        help="Tốc độ học ban đầu của bộ tối ưu Adam"
    )
    parser.add_argument(
        "--final_lr",
        type=float,
        default=1e-5,
        help="Tốc độ học tối thiểu trong lịch hạ Cosine Annealing"
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
        default="/datastore/uittogether3/LuuTru/MinhPD/saved_models/LEARN/",
        help="Thư mục lưu trữ checkpoint trọng số mô hình (.ckpt)"
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
    # Bước 1: Đọc tham số dòng lệnh
    args = parse_args()
    
    # Bước 2: Chuyển đổi góc quét sang Radian
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
    
    # Bước 3: Định danh Setting Tag khớp với cache .npy
    setting_tag = f"limited_ang_{int(angle_range_deg)}deg_numview_{args.num_view}_size_{args.input_size}"
    if args.poisson_level > 0:
        setting_tag += f"_poisson_{int(args.poisson_level)}"
    else:
        setting_tag += "_noise_0"
    
    print("=" * 80)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN BASELINE: LEARN (Original 3-Layer CNN) (Limited-Angle CT)")
    print(f"- Dải góc quét: [{start_ang_deg:.1f}°, {end_ang_deg:.1f}°] (Cung quét {angle_range_deg:.1f}°)")
    print(f"- Số góc chiếu (num_view): {args.num_view} | Detectors: {args.num_detectors}")
    print(f"- Số giai đoạn unrolling (n_iterations): {args.n_iterations}")
    print(f"- Setting Tag: {setting_tag}")
    print(f"- Tập dữ liệu (dataset_type): {args.dataset_type.upper()}")
    print(f"- Thư mục lưu Checkpoint: {args.output_dir}")
    print("=" * 80)
    
    # Bước 4: Khởi tạo DataModule qua Nhà máy Factory
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
        use_precomputed=args.use_precomputed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_patients=args.train_patients,
        val_patients=args.val_patients,
    )
    
    # Bước 5: Khởi tạo Mô hình LEARN_LA
    model = LEARN_LA(
        n_iterations=args.n_iterations,
        num_view=args.num_view,
        num_detectors=args.num_detectors,
        start_ang=start_ang,
        end_ang=end_ang,
        input_size=args.input_size,
        initial_lr=args.initial_lr,
        final_lr=args.final_lr,
    )
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Tổng số tham số có thể huấn luyện (Trainable Parameters): {total_params:,}")
    
    # Bước 6: Cấu hình Thư mục lưu và Callbacks
    # Định dạng tên thư mục con theo đúng chuẩn của dự án:
    dataset_name_clean = args.dataset_type.upper()
    experiment_name = f"LEARN_{dataset_name_clean}_LA_{int(angle_range_deg)}deg_view{args.num_view}"
    ckpt_dir = os.path.join(args.output_dir, experiment_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch:02d}-{val_psnr:.4f}",
        monitor="val_psnr",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=experiment_name,
    )
    
    # Bước 7: Khởi tạo PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=10,
        enable_progress_bar=True,
    )
    
    # Bước 8: Bắt đầu Huấn luyện
    print("\n▶️ Bắt đầu tiến trình Trainer.fit()...")
    trainer.fit(model, datamodule=dm, ckpt_path=args.resume_ckpt)
    
    print("\n✅ HUẤN LUYỆN HOÀN TẤT THÀNH CÔNG!")
    print(f"- Checkpoint tốt nhất được lưu tại: {checkpoint_callback.best_model_path}")
    print(f"- Điểm val_psnr cao nhất đạt được: {checkpoint_callback.best_model_score:.4f} dB")


if __name__ == "__main__":
    main()
