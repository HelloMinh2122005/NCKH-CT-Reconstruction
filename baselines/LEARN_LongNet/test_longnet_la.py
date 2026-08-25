import argparse
import numpy as np
import torch
import pytorch_lightning as pl

from data.datamodule_LA import CTDataModule_LA
from baselines.LEARN_LongNet.models import LEARN_LongNet_LA


def parse_args():
    """
    Khai báo tham số dòng lệnh phục vụ đánh giá (Testing) mô hình LEARN_LongNet.
    """
    parser = argparse.ArgumentParser(description="Đánh giá mô hình Baseline LEARN_LongNet trên Limited-Angle CT")
    
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Đường dẫn đến file trọng số .ckpt")
    parser.add_argument("--data_dir", type=str, default="dataset/limited_angle", help="Đường dẫn thư mục dữ liệu .npy")
    parser.add_argument("--test_patients", nargs="+", default=["L310"], help="Mã bệnh nhân test (mặc định: L310)")
    parser.add_argument("--num_view", type=int, default=64, help="Số góc chiếu trong dải góc giới hạn")
    parser.add_argument("--num_detectors", type=int, default=512, help="Số lượng phần tử cảm biến detector")
    parser.add_argument("--start_ang_deg", type=float, default=-60.0, help="Góc bắt đầu (độ)")
    parser.add_argument("--end_ang_deg", type=float, default=60.0, help="Góc kết thúc (độ)")
    parser.add_argument("--noise_level", type=float, default=0.0, help="Mức độ nhiễu bổ sung")
    parser.add_argument("--batch_size", type=int, default=1, help="Kích thước batch khi test")
    parser.add_argument("--num_workers", type=int, default=2, help="Số worker nạp dữ liệu song song")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    start_ang = np.deg2rad(args.start_ang_deg)
    end_ang = np.deg2rad(args.end_ang_deg)

    print("=" * 70)
    print("📊 TIẾN HÀNH ĐÁNH GIÁ MÔ HÌNH: LEARN_LongNet_LA")
    print(f"- Checkpoint: {args.checkpoint_path}")
    print(f"- Bệnh nhân kiểm thử: {args.test_patients}")
    print(f"- Dải góc: [{args.start_ang_deg}°, {args.end_ang_deg}°] | {args.num_view} views | {args.num_detectors} detectors")
    print("=" * 70)

    # 1. Nạp mô hình từ checkpoint
    model = LEARN_LongNet_LA.load_from_checkpoint(args.checkpoint_path)
    model.eval()

    # 2. Khởi tạo DataModule
    datamodule = CTDataModule_LA(
        data_dir=args.data_dir,
        train_patients=[],
        val_patients=[],
        test_patients=args.test_patients,
        num_view=args.num_view,
        start_ang=start_ang,
        end_ang=end_ang,
        num_detectors=args.num_detectors,
        noise_level=args.noise_level,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # 3. Trainer đánh giá
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
    )

    results = trainer.test(model, datamodule=datamodule)
    print("=" * 70)
    print("📈 KẾT QUẢ ĐO LƯỜNG ĐỘ CHÍNH XÁC (TEST RESULTS):")
    for key, val in results[0].items():
        print(f"  * {key}: {val:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
