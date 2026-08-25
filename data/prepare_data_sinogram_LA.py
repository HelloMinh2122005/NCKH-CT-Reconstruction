import os
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from CTSlice_Provider_LA import LimitedAngleCT_Provider


def parse_args():
    """
    Khai báo và xử lý các đối số dòng lệnh phục vụ việc sinh dữ liệu Sinogram và FBP góc giới hạn.
    """
    parser = argparse.ArgumentParser(description="Generate and cache Limited-Angle CT Sinograms and FBPs.")
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
        help="Thư mục gốc lưu trữ các file sinogram và FBP đã tiền xử lý"
    )
    parser.add_argument(
        "--angle_range_deg",
        type=float,
        default=120.0,
        help="Dải góc quét giới hạn tính theo độ (ví dụ: 90, 120, 150)"
    )
    parser.add_argument(
        "--num_view",
        type=int,
        default=64,
        help="Số góc chiếu (projection views) được lấy mẫu trong dải góc giới hạn"
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=256,
        help="Độ phân giải không gian ảnh đích (256 hoặc 512)"
    )
    parser.add_argument(
        "--poisson_level",
        type=float,
        default=1e6,
        help="Mức photon mô phỏng nhiễu Poisson (0 nếu không muốn thêm nhiễu)"
    )
    parser.add_argument(
        "--gaussian_level",
        type=float,
        default=0.05,
        help="Độ lệch chuẩn của nhiễu Gaussian (0 nếu không muốn thêm nhiễu)"
    )
    return parser.parse_args()


def generate_split(provider, save_sino_path, save_fbp_path):
    """
    Lặp qua toàn bộ lát cắt trong Dataset Provider, chuyển đổi Tensor sang NumPy và lưu thành file .npy.
    """
    os.makedirs(save_sino_path, exist_ok=True)
    os.makedirs(save_fbp_path, exist_ok=True)

    for (slice_path, phantom, fbp_u, sino) in tqdm(provider):
        # Lấy tên định danh file gốc (bỏ đuôi .IMA)
        stem = os.path.basename(slice_path).split(".IMA")[0]
        sino_file = os.path.join(save_sino_path, f"{stem}.npy")
        fbp_file = os.path.join(save_fbp_path, f"{stem}.npy")

        # Chuyển đổi PyTorch Tensor sang mảng NumPy để lưu trữ
        sino_np = sino.cpu().numpy() if isinstance(sino, torch.Tensor) else sino
        fbp_np = fbp_u.cpu().numpy() if isinstance(fbp_u, torch.Tensor) else fbp_u

        # Lưu file sinogram và FBP góc hạn chế dưới dạng mảng npy
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

    # Tạo tag định danh cho cấu hình dữ liệu này (ví dụ: limited_ang_120deg_numview_64_size_256_poisson_1000000)
    setting_tag = f"limited_ang_{int(args.angle_range_deg)}deg_numview_{args.num_view}_size_{args.input_size}"
    if args.poisson_level > 0:
        setting_tag += f"_poisson_{int(args.poisson_level)}"
    else:
        setting_tag += "_noise_0"

    print("=" * 70)
    print(f"Generating Limited-Angle CT Dataset")
    print(f"Angle Range: [-{args.angle_range_deg/2:.1f}°, +{args.angle_range_deg/2:.1f}°] ({args.angle_range_deg}° total)")
    print(f"Number of Views: {args.num_view}")
    print(f"Image Resolution: {args.input_size}x{args.input_size}")
    print(f"Setting Tag: {setting_tag}")
    print("=" * 70)

    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size))
    ])

    # -------------------------------------------------------------
    # 2. Xử lý và lưu trữ dữ liệu cho tập TRAIN
    # -------------------------------------------------------------
    train_folder = os.path.join(args.output_dir, "train", setting_tag)
    print(f"\n[1/2] Processing TRAIN set -> {train_folder}")
    train_dataset = LimitedAngleCT_Provider(
        base_path=args.data_dir,
        start_ang=start_ang,
        end_ang=end_ang,
        num_view=args.num_view,
        poission_level=args.poisson_level,
        gaussian_level=args.gaussian_level,
        input_size=args.input_size,
        transform=transform,
        test=False
    )
    generate_split(train_dataset, os.path.join(train_folder, "sino"), os.path.join(train_folder, "fbp_u"))

    # -------------------------------------------------------------
    # 3. Xử lý và lưu trữ dữ liệu cho tập TEST
    # -------------------------------------------------------------
    test_folder = os.path.join(args.output_dir, "test", setting_tag)
    print(f"\n[2/2] Processing TEST set -> {test_folder}")
    test_dataset = LimitedAngleCT_Provider(
        base_path=args.data_dir,
        start_ang=start_ang,
        end_ang=end_ang,
        num_view=args.num_view,
        poission_level=args.poisson_level,
        gaussian_level=args.gaussian_level,
        input_size=args.input_size,
        transform=transform,
        test=True
    )
    generate_split(test_dataset, os.path.join(test_folder, "sino"), os.path.join(test_folder, "fbp_u"))

    print("\nDataset preparation completed successfully!")


if __name__ == "__main__":
    main()
