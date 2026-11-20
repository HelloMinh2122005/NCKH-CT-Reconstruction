"""
================================================================================
SCRIPT SINH HÀNG LOẠT SINOGRAM VÀ FBP THÔ CHO BÀI TOÁN LIMITED-ANGLE CT
Dành cho bộ dữ liệu: LIDC-IDRI Thoracic CT Dataset (Chùm tia Fan-Beam)
Dự án: Nghiên cứu Tái tạo Ảnh Cắt lớp CT Góc Giới hạn (Limited-Angle CT Reconstruction)
Tác giả: MinhPD — Nhóm Nghiên cứu Tái tạo Ảnh Y tế (VNU-HCM UIT)
================================================================================
"""

import os
import argparse
import numpy as np
import torch
import tqdm

try:
    from data.lidc_idri.CTSlice_Provider_LA_LIDC import LimitedAngleCT_LIDC_Provider
except ImportError:
    from CTSlice_Provider_LA_LIDC import LimitedAngleCT_LIDC_Provider


def save_atomic_npy(target_file: str, array: np.ndarray) -> None:
    """
    Lưu mảng numpy ra tệp nguyên tử (atomic write) nhằm ngăn chặn tệp bị cắt cụt do lỗi NFS:
    - Ghi dữ liệu mảng ra tệp tạm <target_file>.tmp bằng luồng nhị phân wb.
    - Thực hiện flush và fsync để đảm bảo toàn bộ dirty pages được commit xuống storage NFS.
    - Dùng os.replace để tráo đổi nguyên tử tệp tạm vào vị trí tệp đích cuối cùng.
    """
    tmp_file = target_file + ".tmp"
    with open(tmp_file, "wb") as f:
        np.save(f, array)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_file, target_file)


def is_valid_cache_file(sino_file: str, fbp_file: str) -> bool:
    """
    Kiểm tra tính hợp lệ và toàn vẹn của cặp tệp cache sinogram và FBP:
    - Cả 2 tệp phải tồn tại trên hệ thống tệp.
    - Cả 2 tệp phải có kích thước tối thiểu >= 1000 bytes (tránh tệp 128 bytes chỉ có header do NFS stall).
    """
    if os.path.exists(sino_file) and os.path.exists(fbp_file):
        try:
            return os.path.getsize(sino_file) >= 1000 and os.path.getsize(fbp_file) >= 1000
        except OSError:
            return False
    return False


def generate_la_lidc_split(
    split_name="train",
    dicom_dir="/datastore/uittogether3/LuuTru/MinhPD/dataset/lidc_idri/dicom/",
    output_base_dir="/datastore/uittogether3/LuuTru/MinhPD/dataset/lidc_idri/limited_angle/",
    angle_range_deg=120.0,
    num_view=64,
    num_detectors=512,
    input_size=256,
    poisson_level=0,
    gaussian_level=0
):
    """
    Sinh các file cache .npy (Sinogram và Filtered Backprojection thô) cho một phân vùng của LIDC-IDRI.
    """
    half_angle_rad = (angle_range_deg / 2.0) * (np.pi / 180.0)
    start_ang = -half_angle_rad
    end_ang = half_angle_rad

    deg_str = f"{int(angle_range_deg)}deg"
    noise_str = "0" if (poisson_level == 0 and gaussian_level == 0) else "1e6"
    setting_name = f"limited_ang_{deg_str}_numview_{num_view}_size_{input_size}_noise_{noise_str}"

    save_dir = os.path.join(output_base_dir, split_name, setting_name)
    sino_save_dir = os.path.join(save_dir, "sino")
    fbp_save_dir = os.path.join(save_dir, "fbp_u")

    os.makedirs(sino_save_dir, exist_ok=True)
    os.makedirs(fbp_save_dir, exist_ok=True)

    print(f"\n[INFO] Bắt đầu sinh dữ liệu LIDC-IDRI:")
    print(f"       Phân vùng: {split_name}")
    print(f"       Dải góc quét: [-{angle_range_deg/2:.1f}°, +{angle_range_deg/2:.1f}°] ({angle_range_deg}° span)")
    print(f"       Số góc chiếu (Views): {num_view}")
    print(f"       Thư mục đích: {save_dir}")

    is_test = (split_name == "test")
    is_valid = (split_name == "val")

    provider = LimitedAngleCT_LIDC_Provider(
        dicom_dir=dicom_dir,
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

    saved_count = 0
    skipped_count = 0
    corrupted_slices = []
    corrupted_log_path = os.path.join(output_base_dir, "corrupted_slices.log")
    split_corrupted_log_path = os.path.join(save_dir, "corrupted_slices.log")

    for idx in tqdm.tqdm(range(total_slices), desc=f"Sinh LIDC {split_name} ({deg_str})"):
        # Lấy trước đường dẫn tệp DICOM để kiểm tra cache và ghi nhận lỗi nếu có
        slice_dcm_path = provider.slices_path[idx] if hasattr(provider, "slices_path") and idx < len(provider.slices_path) else "unknown"

        # Tạo tên file cache: pat_id + slice_id
        pat_id = "UNKNOWN"
        for p in provider.patients:
            if p in slice_dcm_path:
                pat_id = p
                break
        slice_name = os.path.splitext(os.path.basename(slice_dcm_path))[0]
        cache_filename = f"{pat_id}_{slice_name}.npy"

        sino_file = os.path.join(sino_save_dir, cache_filename)
        fbp_file = os.path.join(fbp_save_dir, cache_filename)

        if is_valid_cache_file(sino_file, fbp_file):
            skipped_count += 1
            continue

        try:
            dcm_path, phantom, fbp_u, sino = provider[idx]
        except Exception as e:
            err_msg = f"[CẢNH BÁO] Lát cắt tại index {idx} ({slice_dcm_path}) bị lỗi và được bỏ qua: {e}"
            print(err_msg)
            corrupted_slices.append((idx, slice_dcm_path, str(e)))
            try:
                for lp in [corrupted_log_path, split_corrupted_log_path]:
                    with open(lp, "a", encoding="utf-8") as f_log:
                        f_log.write(f"[{split_name}_{deg_str}] Index {idx} ({slice_dcm_path}): {e}\n")
            except OSError:
                pass
            continue

        sino_np = sino.squeeze().cpu().numpy().astype(np.float32) if hasattr(sino, "squeeze") else np.asarray(sino, dtype=np.float32)
        fbp_np = fbp_u.squeeze().cpu().numpy().astype(np.float32) if hasattr(fbp_u, "squeeze") else np.asarray(fbp_u, dtype=np.float32)

        save_atomic_npy(sino_file, sino_np)
        save_atomic_npy(fbp_file, fbp_np)
        saved_count += 1

        # Thu dọn bộ nhớ định kỳ sau mỗi 500 lát cắt
        if (idx + 1) % 500 == 0:
            import gc
            gc.collect()
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

    if corrupted_slices:
        print(f"[CẢNH BÁO] Phát hiện {len(corrupted_slices)} lát cắt bị hỏng trong {split_name} ({deg_str}). Xem nhật ký: {corrupted_log_path}")
    print(f"[HOÀN TẤT] LIDC-IDRI {split_name} ({deg_str}): Đã ghi mới {saved_count} lát, bỏ qua {skipped_count} lát.")


def main():
    parser = argparse.ArgumentParser(description="Sinh cache Sinogram và FBP thô cho LIDC-IDRI")
    parser.add_argument("--dicom_dir", type=str, default="/datastore/uittogether3/LuuTru/MinhPD/dataset/lidc_idri/dicom/")
    parser.add_argument("--output_base_dir", type=str, default="/datastore/uittogether3/LuuTru/MinhPD/dataset/lidc_idri/limited_angle/")
    parser.add_argument("--num_view", type=int, default=64)
    parser.add_argument("--num_detectors", type=int, default=512)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--poisson_level", type=float, default=0.0)
    parser.add_argument("--gaussian_level", type=float, default=0.0)
    parser.add_argument("--angles", nargs="+", type=float, default=[90.0, 120.0])
    parser.add_argument("--splits", nargs="+", type=str, default=["train", "val", "test"])

    args = parser.parse_args()

    for angle in args.angles:
        for split in args.splits:
            generate_la_lidc_split(
                split_name=split,
                dicom_dir=args.dicom_dir,
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
