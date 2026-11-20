import gc
import os
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

try:
    from data.CTSlice_Provider_LA import LimitedAngleCT_Provider
    from data.aapm.prepare_data_sinogram_LA import (
        _DICOMGroundTruthProvider,
        generate_split as _aapm_generate_split,
        validate_safe_output_dir,
    )
except ImportError:
    from CTSlice_Provider_LA import LimitedAngleCT_Provider
    from aapm.prepare_data_sinogram_LA import (
        _DICOMGroundTruthProvider,
        generate_split as _aapm_generate_split,
        validate_safe_output_dir,
    )


def parse_args():
    """
    Khai báo và xử lý các đối số dòng lệnh phục vụ việc sinh dữ liệu Sinogram và FBP góc giới hạn.
    """
    parser = argparse.ArgumentParser(description="Sinh và lưu cache dữ liệu Sinogram và FBP cho bài toán Limited-Angle CT.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/datastore/uittogether3/LuuTru/Thanhld/CT-Reconstruction/split/",
        help="Đường dẫn thư mục gốc tập dữ liệu AAPM Mayo CT chứa train/ và test/"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/datastore/uittogether3/LuuTru/MinhPD/dataset/limited_angle/",
        help="Thư mục gốc lưu trữ các file sinogram và FBP đã tiền xử lý (.npy)"
    )
    parser.add_argument(
        "--angle_range_deg",
        type=float,
        default=120.0,
        help="Dải góc quét giới hạn tính theo độ (ví dụ: 90.0, 120.0, 150.0 độ)"
    )
    parser.add_argument(
        "--num_view",
        type=int,
        default=64,
        help="Số góc chiếu (projection views) được lấy mẫu đều trong dải góc giới hạn"
    )
    parser.add_argument(
        "--num_detectors",
        type=int,
        default=512,
        help="Số lượng cảm biến trên thanh detector của hệ thống Fan-Beam"
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=256,
        help="Độ phân giải không gian ảnh đích sau resize (256x256 pixel)"
    )
    parser.add_argument(
        "--poisson_level",
        type=float,
        default=0.0,
        help="Mức photon mô phỏng nhiễu Poisson (0 nếu là chế độ noise_0 không nhiễu, 1e6 cho Low-Dose thực tế)"
    )
    parser.add_argument(
        "--gaussian_level",
        type=float,
        default=0.0,
        help="Độ lệch chuẩn của nhiễu Gaussian (0 nếu không thêm nhiễu, 0.05 cho nhiễu cảm biến)"
    )
    parser.add_argument(
        "--train_patients",
        nargs="+",
        default=None,
        help="Danh sách mã bệnh nhân cho tập huấn luyện (mặc định: None, sử dụng 8 bệnh nhân mặc định của AAPM)"
    )
    parser.add_argument(
        "--val_patients",
        nargs="+",
        default=None,
        help="Danh sách mã bệnh nhân cho tập kiểm định (mặc định: None, sử dụng bệnh nhân L333)"
    )
    parser.add_argument(
        "--test_patients",
        nargs="+",
        default=None,
        help="Danh sách mã bệnh nhân cho tập kiểm thử độc lập (mặc định: None, sử dụng bệnh nhân L310)"
    )
    parser.add_argument(
        "--gt_only",
        action="store_true",
        default=False,
        help="Chỉ tạo cache Ground Truth gt_size_{input_size}/ từ DICOM mà không khởi tạo ODL/ASTRA hay chạm sino/ và fbp_u/"
    )
    return parser.parse_args()


def generate_split(provider, save_sino_path=None, save_fbp_path=None, save_gt_path=None, gt_only: bool = False):
    """
    Lặp qua toàn bộ lát cắt trong Dataset Provider:
    - Kiểm tra trước sự tồn tại của file .npy để bỏ qua tức thì (tiết kiệm thời gian tính toán ODL).
    - Chỉ thực hiện Chiếu thuận Radon và Chiếu ngược FBP cho các lát cắt chưa được tạo.
    - Lưu file .npy độc lập cho sinogram và FBP ban đầu.
    """
    # Bỏ qua ngay lập tức nếu file đã tồn tại trên đĩa mà không cần tính Radon
    # Lấy dữ liệu và thực hiện phép chiếu Radon + FBP on-the-fly
    # Lưu mảng numpy
    return _aapm_generate_split(
        provider,
        save_sino_path=save_sino_path,
        save_fbp_path=save_fbp_path,
        save_gt_path=save_gt_path,
        gt_only=gt_only,
    )


def main():
    args = parse_args()
    # Kiểm tra bảo vệ đường dẫn đầu ra trước khi tạo thư mục hay đọc dữ liệu (T0.5)
    validate_safe_output_dir(args.output_dir, dicom_dir=args.data_dir)

    # -------------------------------------------------------------
    # 1. Tính toán góc bắt đầu và kết thúc (Radian) đối xứng qua tâm 0
    # -------------------------------------------------------------
    half_span_rad = np.deg2rad(args.angle_range_deg / 2.0)
    start_ang = -half_span_rad
    end_ang = half_span_rad

    # Tạo tag định danh cho cấu hình dữ liệu này (ví dụ: limited_ang_120deg_numview_64_size_256_noise_0)
    setting_tag = f"limited_ang_{int(args.angle_range_deg)}deg_numview_{args.num_view}_size_{args.input_size}"
    if args.poisson_level > 0:
        setting_tag += f"_poisson_{int(args.poisson_level)}"
    else:
        setting_tag += "_noise_0"

    train_gt_dir = os.path.join(args.output_dir, "train", f"gt_size_{args.input_size}")
    test_gt_dir = os.path.join(args.output_dir, "test", f"gt_size_{args.input_size}")

    print("=" * 70)
    print(f"🚀 BẮT ĐẦU TẠO SINH DATASET LIMITED-ANGLE CT")
    print(f"- Cung góc quét: [-{args.angle_range_deg/2:.1f}°, +{args.angle_range_deg/2:.1f}°] ({args.angle_range_deg}° total)")
    print(f"- Số góc chiếu (Views): {args.num_view} | Detectors: {args.num_detectors}")
    print(f"- Độ phân giải không gian: {args.input_size}x{args.input_size}")
    print(f"- Cấu hình lưu trữ (Tag): {setting_tag}")
    print("=" * 70)

    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size))
    ])

    if args.gt_only:
        train_gt_provider = _DICOMGroundTruthProvider(
            dicom_dir=args.data_dir,
            input_size=args.input_size,
            transform=transform,
            patient_list=args.train_patients,
            test=False,
            valid=False,
        )
        generate_split(train_gt_provider, save_gt_path=train_gt_dir, gt_only=True)

        val_gt_provider = _DICOMGroundTruthProvider(
            dicom_dir=args.data_dir,
            input_size=args.input_size,
            transform=transform,
            patient_list=args.val_patients,
            test=False,
            valid=True,
        )
        generate_split(val_gt_provider, save_gt_path=train_gt_dir, gt_only=True)

        test_gt_provider = _DICOMGroundTruthProvider(
            dicom_dir=args.data_dir,
            input_size=args.input_size,
            transform=transform,
            patient_list=args.test_patients,
            test=True,
            valid=False,
        )
        generate_split(test_gt_provider, save_gt_path=test_gt_dir, gt_only=True)
        return

    # -------------------------------------------------------------
    # 2. Xử lý và lưu trữ dữ liệu cho tập TRAIN (8 bệnh nhân)
    # -------------------------------------------------------------
    train_folder = os.path.join(args.output_dir, "train", setting_tag)
    print(f"\n[1/3] Đang xử lý tập TRAIN -> {train_folder}")
    train_dataset = LimitedAngleCT_Provider(
        dicom_dir=args.data_dir,
        start_ang=start_ang,
        end_ang=end_ang,
        num_view=args.num_view,
        num_detectors=args.num_detectors,
        poission_level=args.poisson_level,
        gaussian_level=args.gaussian_level,
        input_size=args.input_size,
        transform=transform,
        use_precomputed=False,
        return_path=True,
        patient_list=args.train_patients,
        test=False,
        valid=False
    )
    generate_split(
        train_dataset,
        os.path.join(train_folder, "sino"),
        os.path.join(train_folder, "fbp_u"),
        save_gt_path=train_gt_dir,
    )

    # -------------------------------------------------------------
    # 3. Xử lý và lưu trữ dữ liệu cho tập VALIDATION (Bệnh nhân L333)
    # -------------------------------------------------------------
    print(f"\n[2/3] Đang xử lý tập VALIDATION (L333) -> {train_folder}")
    val_dataset = LimitedAngleCT_Provider(
        dicom_dir=args.data_dir,
        start_ang=start_ang,
        end_ang=end_ang,
        num_view=args.num_view,
        num_detectors=args.num_detectors,
        poission_level=args.poisson_level,
        gaussian_level=args.gaussian_level,
        input_size=args.input_size,
        transform=transform,
        use_precomputed=False,
        return_path=True,
        patient_list=args.val_patients,
        test=False,
        valid=True
    )
    generate_split(
        val_dataset,
        os.path.join(train_folder, "sino"),
        os.path.join(train_folder, "fbp_u"),
        save_gt_path=train_gt_dir,
    )

    # -------------------------------------------------------------
    # 4. Xử lý và lưu trữ dữ liệu cho tập TEST (Bệnh nhân L310)
    # -------------------------------------------------------------
    test_folder = os.path.join(args.output_dir, "test", setting_tag)
    print(f"\n[3/3] Đang xử lý tập TEST (L310) -> {test_folder}")
    test_dataset = LimitedAngleCT_Provider(
        dicom_dir=args.data_dir,
        start_ang=start_ang,
        end_ang=end_ang,
        num_view=args.num_view,
        num_detectors=args.num_detectors,
        poission_level=args.poisson_level,
        gaussian_level=args.gaussian_level,
        input_size=args.input_size,
        transform=transform,
        use_precomputed=False,
        return_path=True,
        patient_list=args.test_patients,
        test=True
    )
    generate_split(
        test_dataset,
        os.path.join(test_folder, "sino"),
        os.path.join(test_folder, "fbp_u"),
        save_gt_path=test_gt_dir,
    )

    print("\n🎉 Tạo sinh và lưu cache toàn bộ dataset Limited-Angle CT hoàn thành thành công!")


if __name__ == "__main__":
    main()
