"""
================================================================================
SCRIPT SINH HÀNG LOẠT SINOGRAM VÀ FBP THÔ CHO BÀI TOÁN LIMITED-ANGLE CT
Dành cho bộ dữ liệu: NIH DeepLesion CT Dataset (Chùm tia Fan-Beam)
Dự án: Nghiên cứu Tái tạo Ảnh Cắt lớp CT Góc Giới hạn (Limited-Angle CT Reconstruction)
Tác giả: MinhPD — Nhóm Nghiên cứu Tái tạo Ảnh Y tế (VNU-HCM UIT)
================================================================================
"""

import os
import argparse
import numpy as np
import torch
import tqdm
from CTSlice_Provider_LA_DeepLesion import LimitedAngleCT_DeepLesion_Provider


def generate_la_deeplesion_split(
    split_name="train",
    split_dir="/datastore/uittogether3/LuuTru/Thanhld/CT-Reconstruction/split_dl/",
    data_dir="/datastore/uittogether3/LuuTru/Thanhld/CT-Reconstruction/minideeplesion/minideeplesion/",
    output_base_dir="/datastore/uittogether3/LuuTru/MinhPD/dataset/nih_deep_lesion/limited_angle/",
    angle_range_deg=120.0,
    num_view=64,
    num_detectors=512,
    input_size=256,
    poisson_level=0,
    gaussian_level=0
):
    """
    Sinh các file cache .npy (Sinogram và Filtered Backprojection thô) cho một phân vùng (train/val/test).
    """
    # 1. Tính toán góc bắt đầu và kết thúc (đối xứng qua 0 độ)
    half_angle_rad = (angle_range_deg / 2.0) * (np.pi / 180.0)
    start_ang = -half_angle_rad
    end_ang = half_angle_rad

    # 2. Đặt tên thư mục cấu hình cache
    deg_str = f"{int(angle_range_deg)}deg"
    noise_str = "0" if (poisson_level == 0 and gaussian_level == 0) else "1e6"
    setting_name = f"limited_ang_{deg_str}_numview_{num_view}_size_{input_size}_noise_{noise_str}"

    save_dir = os.path.join(output_base_dir, split_name, setting_name)
    sino_save_dir = os.path.join(save_dir, "sino")
    fbp_save_dir = os.path.join(save_dir, "fbp_u")

    os.makedirs(sino_save_dir, exist_ok=True)
    os.makedirs(fbp_save_dir, exist_ok=True)

    print(f"\n[INFO] Bắt đầu sinh dữ liệu NIH DeepLesion:")
    print(f"       Phân vùng: {split_name}")
    print(f"       Dải góc quét: [-{angle_range_deg/2:.1f}°, +{angle_range_deg/2:.1f}°] ({angle_range_deg}° span)")
    print(f"       Số góc chiếu (Views): {num_view}")
    print(f"       Thư mục đích: {save_dir}")

    # 3. Khởi tạo Dataset Provider ở chế độ tính toán online (return_path=True)
    is_test = (split_name == "test")
    is_valid = (split_name == "val")

    provider = LimitedAngleCT_DeepLesion_Provider(
        data_dir=data_dir,
        split_dir=split_dir,
        start_ang=start_ang,
        end_ang=end_ang,
        num_view=num_view,
        num_detectors=num_detectors,
        poission_level=poisson_level,
        gaussian_level=gaussian_level,
        test=is_test,
        valid=is_valid,
        input_size=input_size,
        use_precomputed=False,
        return_path=True
    )

    total_slices = len(provider)
    print(f"[INFO] Tổng số lát cắt cần sinh cho {split_name}: {total_slices}")

    # 4. Duyệt qua từng lát cắt và lưu file .npy
    saved_count = 0
    skipped_count = 0

    for idx in tqdm.tqdm(range(total_slices), desc=f"Sinh {split_name} ({deg_str})"):
        slice_path, phantom, fbp_u, sino = provider[idx]

        # Tạo tên file cache từ path gốc: .../000075_02_01/049.png -> 000075_02_01_049.npy
        rel_parts = slice_path.split(os.sep)
        case_id = rel_parts[-2]
        slice_id = os.path.splitext(rel_parts[-1])[0]
        cache_filename = f"{case_id}_{slice_id}.npy"

        sino_file = os.path.join(sino_save_dir, cache_filename)
        fbp_file = os.path.join(fbp_save_dir, cache_filename)

        # Bỏ qua nếu cả 2 file đã tồn tại hợp lệ
        if os.path.exists(sino_file) and os.path.exists(fbp_file):
            skipped_count += 1
            continue

        sino_np = sino.squeeze().cpu().numpy().astype(np.float32)
        fbp_np = fbp_u.squeeze().cpu().numpy().astype(np.float32)

        np.save(sino_file, sino_np)
        np.save(fbp_file, fbp_np)
        saved_count += 1

    print(f"[HOÀN TẤT] {split_name} ({deg_str}): Đã ghi mới {saved_count} lát, bỏ qua {skipped_count} lát đã có sẵn.")


def main():
    parser = argparse.ArgumentParser(description="Sinh cache Sinogram và FBP thô cho NIH DeepLesion")
    parser.add_argument("--split_dir", type=str, default="/datastore/uittogether3/LuuTru/Thanhld/CT-Reconstruction/split_dl/")
    parser.add_argument("--data_dir", type=str, default="/datastore/uittogether3/LuuTru/Thanhld/CT-Reconstruction/minideeplesion/minideeplesion/")
    parser.add_argument("--output_base_dir", type=str, default="/datastore/uittogether3/LuuTru/MinhPD/dataset/nih_deep_lesion/limited_angle/")
    parser.add_argument("--num_view", type=int, default=64)
    parser.add_argument("--num_detectors", type=int, default=512)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--poisson_level", type=float, default=0.0)
    parser.add_argument("--gaussian_level", type=float, default=0.0)
    parser.add_argument("--angles", nargs="+", type=float, default=[120.0, 90.0], help="Danh sách dải góc quét cần sinh (ví dụ 120.0 90.0)")
    parser.add_argument("--splits", nargs="+", type=str, default=["train", "val", "test"], help="Các phân vùng cần sinh")

    args = parser.parse_args()

    for angle in args.angles:
        for split in args.splits:
            generate_la_deeplesion_split(
                split_name=split,
                split_dir=args.split_dir,
                data_dir=args.data_dir,
                output_base_dir=args.output_base_dir,
                angle_range_deg=angle,
                num_view=args.num_view,
                num_detectors=args.num_detectors,
                input_size=args.input_size,
                poisson_level=args.poisson_level,
                gaussian_level=args.gaussian_level
            )


if __name__ == "__main__":
    main()
