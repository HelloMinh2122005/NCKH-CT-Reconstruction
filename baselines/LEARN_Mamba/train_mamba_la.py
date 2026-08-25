import os
import argparse
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from data.datamodule_LA import CTDataModule_LA
from baselines.LEARN_Mamba.models import LEARN_Mamba_LA


def parse_args():
    """
    Khai báo và đọc các tham số dòng lệnh phục vụ huấn luyện LEARN_Mamba trên Limited-Angle CT.
    """
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Baseline LEARN_Mamba trên Limited-Angle CT")
    
    # 1. Cấu hình Dữ liệu (Data Configuration)
    parser.add_argument("--data_dir", type=str, default="dataset/limited_angle", help="Đường dẫn thư mục chứa dữ liệu .npy cache")
    parser.add_argument("--train_patients", nargs="+", default=["L067", "L096", "L109", "L143", "L192", "L286", "L291", "L333"], help="Danh sách mã bệnh nhân dùng cho tập huấn luyện")
    parser.add_argument("--val_patients", nargs="+", default=["L506"], help="Danh sách mã bệnh nhân dùng cho tập kiểm định")
    parser.add_argument("--batch_size", type=int, default=1, help="Kích thước mini-batch (mặc định = 1 lát cắt)")
    parser.add_argument("--num_workers", type=int, default=2, help="Số luồng CPU nạp dữ liệu song song")
    
    # 2. Cấu hình Hình học Chụp CT Góc Giới Hạn (Limited-Angle CT Physics)
    parser.add_argument("--num_view", type=int, default=64, help="Số góc chiếu trong dải góc giới hạn (ví dụ: 64 views)")
    parser.add_argument("--num_detectors", type=int, default=512, help="Số lượng phần tử cảm biến detector (512 detectors)")
    parser.add_argument("--start_ang_deg", type=float, default=-60.0, help="Góc quét bắt đầu tính bằng độ (-60 độ)")
    parser.add_argument("--end_ang_deg", type=float, default=60.0, help="Góc quét kết thúc tính bằng độ (+60 độ)")
    parser.add_argument("--noise_level", type=float, default=0.0, help="Mức độ nhiễu bổ sung vào sinogram")
    
    # 3. Cấu hình Mô hình & Tối ưu hóa (Model & Optimization)
    parser.add_argument("--n_iterations", type=int, default=14, help="Số giai đoạn unrolling của thuật toán LEARN (K = 14)")
    parser.add_argument("--initial_lr", type=float, default=1e-4, help="Tốc độ học ban đầu của bộ tối ưu Adam")
    parser.add_argument("--final_lr", type=float, default=1e-5, help="Tốc độ học tối thiểu của Cosine Annealing")
    parser.add_argument("--max_epochs", type=int, default=50, help="Tổng số epoch huấn luyện")
    parser.add_argument("--log_dir", type=str, default="lightning_logs", help="Thư mục lưu log TensorBoard")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints", help="Thư mục lưu trọng số mô hình tốt nhất")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Chuyển đổi góc từ độ sang radian
    start_ang = np.deg2rad(args.start_ang_deg)
    end_ang = np.deg2rad(args.end_ang_deg)
    
    print("=" * 70)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN BASELINE: LEARN_Mamba (Limited-Angle CT)")
    print(f"- Dải góc quét: [{args.start_ang_deg}°, {args.end_ang_deg}°] | Số views: {args.num_view} | Detectors: {args.num_detectors}")
    print(f"- Số giai đoạn Unrolling: {args.n_iterations} | Epochs: {args.max_epochs} | Batch size: {args.batch_size}")
    print(f"- Tốc độ học: {args.initial_lr} -> {args.final_lr}")
    print("=" * 70)

    # 1. Khởi tạo Lightning DataModule
    datamodule = CTDataModule_LA(
        data_dir=args.data_dir,
        train_patients=args.train_patients,
        val_patients=args.val_patients,
        num_view=args.num_view,
        start_ang=start_ang,
        end_ang=end_ang,
        num_detectors=args.num_detectors,
        noise_level=args.noise_level,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # 2. Khởi tạo Mô hình LEARN_Mamba cho Limited-Angle CT
    model = LEARN_Mamba_LA(
        n_iterations=args.n_iterations,
        num_view=args.num_view,
        num_detectors=args.num_detectors,
        start_ang=start_ang,
        end_ang=end_ang,
        initial_lr=args.initial_lr,
        final_lr=args.final_lr,
    )

    # 3. Thiết lập Logger và Callbacks
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name="LEARN_Mamba_LA",
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.checkpoints_dir, "LEARN_Mamba_LA"),
        filename="mamba_la-{epoch:02d}-{val_psnr:.2f}-{val_ssim:.4f}",
        save_top_k=3,
        monitor="val_psnr",
        mode="max",
        save_last=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # 4. Khởi tạo PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=10,
    )

    # 5. Tiến hành huấn luyện
    trainer.fit(model, datamodule=datamodule)
    print("🎉 Huấn luyện LEARN_Mamba hoàn thành thành công!")


if __name__ == "__main__":
    main()
