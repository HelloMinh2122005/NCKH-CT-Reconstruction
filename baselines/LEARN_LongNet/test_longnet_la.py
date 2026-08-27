import argparse
import numpy as np
import torch
import pytorch_lightning as pl

from data.datamodule_LA import LimitedAngleCTDataModule
from baselines.LEARN_LongNet.models import LEARN_LongNet_LA


def parse_args():
    """
    Khai báo tham số dòng lệnh phục vụ đánh giá (Testing) mô hình LEARN_LongNet.
    """
    parser = argparse.ArgumentParser(description="Đánh giá mô hình Baseline LEARN_LongNet trên Limited-Angle CT")
    
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Đường dẫn đến file trọng số .ckpt")
    parser.add_argument("--dicom_dir", type=str, default="/datastore/uittogether3/LuuTru/Thanhld/CT-Reconstruction/split/", help="Đường dẫn thư mục DICOM")
    parser.add_argument("--dataset_dir", "--data_dir", "--cache_dir", dest="cache_dir", type=str, default="/datastore/uittogether3/LuuTru/MinhPD/dataset/limited_angle/", help="Đường dẫn thư mục dữ liệu .npy")
    parser.add_argument("--test_patients", nargs="+", default=["L310"], help="Mã bệnh nhân test (mặc định: L310)")
    parser.add_argument("--angle_range_deg", type=float, default=120.0, help="Độ rộng cung quét góc giới hạn")
    parser.add_argument("--start_ang_deg", type=float, default=None, help="Góc bắt đầu (độ)")
    parser.add_argument("--end_ang_deg", type=float, default=None, help="Góc kết thúc (độ)")
    parser.add_argument("--num_view", type=int, default=64, help="Số góc chiếu trong dải góc giới hạn")
    parser.add_argument("--num_detectors", type=int, default=512, help="Số lượng phần tử cảm biến detector")
    parser.add_argument("--input_size", type=int, default=256, help="Kích thước ảnh")
    parser.add_argument("--poisson_level", type=float, default=0.0, help="Mức độ nhiễu Poisson")
    parser.add_argument("--gaussian_level", type=float, default=0.0, help="Mức độ nhiễu Gaussian")
    parser.add_argument("--batch_size", type=int, default=1, help="Kích thước batch khi test")
    parser.add_argument("--num_workers", type=int, default=4, help="Số worker nạp dữ liệu song song")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
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

    setting_tag = f"limited_ang_{int(angle_range_deg)}deg_numview_{args.num_view}_size_{args.input_size}"
    if args.poisson_level > 0:
        setting_tag += f"_poisson_{int(args.poisson_level)}"
    else:
        setting_tag += "_noise_0"

    print("=" * 70)
    print("📊 TIẾN HÀNH ĐÁNH GIÁ MÔ HÌNH: LEARN_LongNet_LA")
    print(f"- Checkpoint: {args.checkpoint_path}")
    print(f"- Bệnh nhân kiểm thử: {args.test_patients}")
    print(f"- Dải góc: [{start_ang_deg:.1f}°, {end_ang_deg:.1f}°] | {args.num_view} views | {args.num_detectors} detectors")
    print("=" * 70)

    # 1. Nạp mô hình từ checkpoint
    model = LEARN_LongNet_LA.load_from_checkpoint(args.checkpoint_path)
    model.eval()

    # 2. Khởi tạo DataModule
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
        use_precomputed=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_patients=args.test_patients,
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
