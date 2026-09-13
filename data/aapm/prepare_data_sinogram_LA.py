import os
import sys
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
except ImportError:
    from CTSlice_Provider_LA import LimitedAngleCT_Provider


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
    return parser.parse_args()


def generate_split(provider, save_sino_path, save_fbp_path):
    """
    Lặp qua toàn bộ lát cắt trong Dataset Provider:
    - Kiểm tra trước sự tồn tại của file .npy để bỏ qua tức thì (tiết kiệm thời gian tính toán ODL).
    - Chỉ thực hiện Chiếu thuận Radon và Chiếu ngược FBP cho các lát cắt chưa được tạo.
    - Lưu file .npy độc lập cho sinogram và FBP ban đầu.
    """
    os.makedirs(save_sino_path, exist_ok=True)
    os.makedirs(save_fbp_path, exist_ok=True)

    total_slices = len(provider)
    for i in tqdm(range(total_slices), desc="Generating slices"):
        slice_path = provider.slices_path[i]
        stem = os.path.basename(slice_path).split(".IMA")[0]
        sino_file = os.path.join(save_sino_path, f"{stem}.npy")
        fbp_file = os.path.join(save_fbp_path, f"{stem}.npy")

        # Bỏ qua ngay lập tức nếu file đã tồn tại trên đĩa mà không cần tính Radon
        if os.path.exists(sino_file) and os.path.exists(fbp_file):
            continue

        # Lấy dữ liệu và thực hiện phép chiếu Radon + FBP on-the-fly
        _, phantom, fbp_u, sino = provider[i]

        sino_np = sino.cpu().numpy() if isinstance(sino, torch.Tensor) else sino
        fbp_np = fbp_u.cpu().numpy() if isinstance(fbp_u, torch.Tensor) else fbp_u

        # Lưu mảng numpy
        np.save(sino_file, sino_np)
        np.save(fbp_file, fbp_np)


def main():
    args = parse_args()

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
        test=False,
        valid=False
    )
    generate_split(train_dataset, os.path.join(train_folder, "sino"), os.path.join(train_folder, "fbp_u"))

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
        test=False,
        valid=True
    )
    generate_split(val_dataset, os.path.join(train_folder, "sino"), os.path.join(train_folder, "fbp_u"))

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
        test=True
    )
    generate_split(test_dataset, os.path.join(test_folder, "sino"), os.path.join(test_folder, "fbp_u"))

    print("\n🎉 Tạo sinh và lưu cache toàn bộ dataset Limited-Angle CT hoàn thành thành công!")


if __name__ == "__main__":
    main()
