import os
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from CTSlice_Provider_LA import LimitedAngleCT_Provider


def parse_args():
    parser = argparse.ArgumentParser(description="Generate and cache Limited-Angle CT Sinograms and FBPs.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/datastore/uittogether3/LuuTru/Thanhld/CT-Reconstruction/split/",
        help="Path to AAPM Mayo CT dataset root containing train/ and test/ folders"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/datastore/uittogether3/LuuTru/MinhPD/dataset/limited_angle/",
        help="Root path to save generated limited-angle sinograms and FBPs"
    )
    parser.add_argument(
        "--angle_range_deg",
        type=float,
        default=120.0,
        help="Angular coverage range in degrees (e.g., 90, 120, 150)"
    )
    parser.add_argument(
        "--num_view",
        type=int,
        default=64,
        help="Number of projection views inside the limited angle range"
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=256,
        help="Target spatial resolution (e.g., 256 or 512)"
    )
    parser.add_argument(
        "--poisson_level",
        type=float,
        default=1e6,
        help="Poisson photon level (0 for noiseless)"
    )
    parser.add_argument(
        "--gaussian_level",
        type=float,
        default=0.05,
        help="Gaussian noise std (0 for noiseless)"
    )
    return parser.parse_args()


def generate_split(provider, save_sino_path, save_fbp_path):
    os.makedirs(save_sino_path, exist_ok=True)
    os.makedirs(save_fbp_path, exist_ok=True)

    for (slice_path, phantom, fbp_u, sino) in tqdm(provider):
        stem = os.path.basename(slice_path).split(".IMA")[0]
        sino_file = os.path.join(save_sino_path, f"{stem}.npy")
        fbp_file = os.path.join(save_fbp_path, f"{stem}.npy")

        # Convert tensors to numpy if needed
        sino_np = sino.cpu().numpy() if isinstance(sino, torch.Tensor) else sino
        fbp_np = fbp_u.cpu().numpy() if isinstance(fbp_u, torch.Tensor) else fbp_u

        np.save(sino_file, sino_np)
        np.save(fbp_file, fbp_np)


def main():
    args = parse_args()

    # Compute start and end angles in radians centered at 0
    half_span_rad = np.deg2rad(args.angle_range_deg / 2.0)
    start_ang = -half_span_rad
    end_ang = half_span_rad

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

    # 1. Generate Training Set
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

    # 2. Generate Testing Set
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
