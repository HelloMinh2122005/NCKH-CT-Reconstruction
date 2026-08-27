import argparse
import numpy as np
import torch
import pytorch_lightning as pl

# Import DataModule và Mô hình
from data.datamodule_LA import LimitedAngleCTDataModule
from baselines.LEARN_Longformer.models import LEARN_Longformer_LA


def parse_args():
    """
    Khai báo và xử lý tham số dòng lệnh phục vụ đánh giá (Testing & Benchmark) mô hình LEARN_Longformer.
    """
    parser = argparse.ArgumentParser(description="Đánh giá mô hình Baseline LEARN_Longformer trên tập kiểm thử Limited-Angle CT")
    
    # Checkpoint
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Đường dẫn đến file trọng số .ckpt đã huấn luyện"
    )
    
    # Cấu hình dữ liệu
    parser.add_argument(
        "--dicom_dir",
        type=str,
        default="/datastore/uittogether3/LuuTru/Thanhld/CT-Reconstruction/split/",
        help="Đường dẫn thư mục chứa ảnh DICOM (.IMA) gốc AAPM"
    )
    parser.add_argument(
        "--dataset_dir", "--data_dir", "--cache_dir",
        dest="cache_dir",
        type=str,
        default="/datastore/uittogether3/LuuTru/MinhPD/dataset/limited_angle/",
        help="Đường dẫn thư mục chứa dữ liệu .npy cache"
    )
    parser.add_argument(
        "--test_patients",
        nargs="+",
        default=["L310"],
        help="Mã bệnh nhân kiểm thử độc lập (mặc định: L310)"
    )
    
    # Cấu hình hình học & vật lý CT
    parser.add_argument(
        "--angle_range_deg",
        type=float,
        default=120.0,
        help="Độ rộng cung quét góc giới hạn (độ)"
    )
    parser.add_argument(
        "--start_ang_deg",
        type=float,
        default=None,
        help="Góc quét bắt đầu tính theo độ"
    )
    parser.add_argument(
        "--end_ang_deg",
        type=float,
        default=None,
        help="Góc quét kết thúc tính theo độ"
    )
    parser.add_argument(
        "--num_view",
        type=int,
        default=64,
        help="Số góc chiếu trong dải góc giới hạn"
    )
    parser.add_argument(
        "--num_detectors",
        type=int,
        default=512,
        help="Số lượng phần tử cảm biến trên thanh detector"
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=256,
        help="Độ phân giải không gian ảnh (256x256 pixel)"
    )
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
        help="Độ lệch chuẩn nhiễu Gaussian"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Kích thước batch khi chạy test"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Số luồng CPU nạp dữ liệu song song"
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

    # Bước 3: Định danh cấu hình cache
    setting_tag = f"limited_ang_{int(angle_range_deg)}deg_numview_{args.num_view}_size_{args.input_size}"
    if args.poisson_level > 0:
        setting_tag += f"_poisson_{int(args.poisson_level)}"
    else:
        setting_tag += "_noise_0"

    print("=" * 80)
    print("📊 TIẾN HÀNH ĐÁNH GIÁ MÔ HÌNH: LEARN_Longformer_LA (Test & Benchmark)")
    print(f"- Checkpoint nạp: {args.checkpoint_path}")
    print(f"- Bệnh nhân kiểm thử: {args.test_patients}")
    print(f"- Cung góc quét: [{start_ang_deg:.1f}°, {end_ang_deg:.1f}°] | {args.num_view} views | {args.num_detectors} detectors")
    print(f"- Cấu hình Setting Tag: {setting_tag}")
    print("=" * 80)

    # Bước 4: Nạp trọng số mô hình từ Checkpoint
    model = LEARN_Longformer_LA.load_from_checkpoint(args.checkpoint_path)
    model.eval()

    # Bước 5: Khởi tạo DataModule cho tập kiểm thử
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

    # Bước 6: Khởi tạo Lightning Trainer cho giai đoạn Test
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
    )

    # Bước 7: Thực hiện đánh giá và xuất kết quả
    results = trainer.test(model, datamodule=datamodule)
    print("\n" + "=" * 80)
    print("📈 KẾT QUẢ ĐO LƯỜNG ĐỘ CHÍNH XÁC TÁI TẠO (TEST BENCHMARK RESULTS):")
    for key, val in results[0].items():
        print(f"  • {key:15s}: {val:.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
