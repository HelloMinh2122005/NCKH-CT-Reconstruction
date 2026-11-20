import gc
import glob
import os
import sys
import time
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
    import pydicom
except ImportError:
    pydicom = None

try:
    from data.CTSlice_Provider_LA import LimitedAngleCT_Provider
except ImportError:
    from CTSlice_Provider_LA import LimitedAngleCT_Provider


def validate_safe_output_dir(output_dir: str, dicom_dir: str = None) -> str:
    """
    Bảo vệ đường dẫn đầu ra (T0.5): từ chối tuyệt đối mọi output_dir trùng hoặc nằm bên trong
    dicom_dir hay bất kỳ thư mục raw/ hoặc split/ nào để ngăn ghi đè hoặc xóa nhầm dữ liệu gốc.
    """
    if not output_dir or not str(output_dir).strip():
        raise ValueError("CRITICAL ERROR: --output_dir không được phép rỗng.")

    abs_out = os.path.abspath(output_dir)
    real_out = os.path.realpath(abs_out)
    # Kiểm tra cả đường dẫn chưa giải symlink và đường dẫn thực xem có chạm raw/ hoặc split/ không
    for candidate in (abs_out, real_out):
        parts = Path(candidate).parts
        if "raw" in parts or "split" in parts:
            raise ValueError(
                f"CRITICAL GUARD (T0.5): Từ chối output_dir='{output_dir}' (real='{real_out}') "
                f"vì nằm trong hoặc trùng thư mục 'raw'/'split'."
            )

    if dicom_dir:
        real_dicom = os.path.realpath(os.path.abspath(dicom_dir))
        try:
            common = os.path.commonpath([real_out, real_dicom])
        except ValueError:
            common = None
        if real_out == real_dicom or common == real_dicom:
            raise ValueError(
                f"CRITICAL GUARD (T0.5): Từ chối output_dir='{output_dir}' (real='{real_out}') "
                f"vì trùng hoặc nằm bên trong dicom_dir='{dicom_dir}' (real='{real_dicom}')."
            )

    return real_out


def _to_float32_3d_numpy(tensor_or_array) -> np.ndarray:
    """Chuyển đổi tensor/ndarray về mảng float32 dạng (1, H, W) để lưu cache Ground Truth."""
    if hasattr(tensor_or_array, "detach"):
        arr = tensor_or_array.detach().cpu().numpy()
    elif isinstance(tensor_or_array, torch.Tensor):
        arr = tensor_or_array.cpu().numpy()
    else:
        arr = np.asarray(tensor_or_array)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=0)
    return arr


class _DICOMGroundTruthProvider:
    """
    Dataset nhẹ chỉ đọc và chuẩn hóa ảnh Ground Truth (phantom) từ DICOM cho chế độ --gt_only.
    Không khởi tạo ODL hoặc ASTRA CUDA và không chạm vào thư mục sino/ hoặc fbp_u/.
    """

    def __init__(
        self,
        dicom_dir: str,
        input_size: int = 256,
        transform=None,
        test: bool = False,
        valid: bool = False,
        patient_list=None,
    ):
        self.dicom_dir = dicom_dir
        self.input_size = input_size
        self.transform = transform

        if patient_list is not None:
            patients = patient_list
            split_name = "test" if test else "train"
        elif valid:
            patients = ["L333"]
            split_name = "train"
        elif test:
            patients = ["L310"]
            split_name = "test"
        else:
            patients = ["L067", "L096", "L109", "L143", "L192", "L286", "L291", "L506"]
            split_name = "train"

        paths = []
        for patient_id in patients:
            patterns = [
                os.path.join(self.dicom_dir, split_name, patient_id, "full_3mm", f"{patient_id}_FD_3_1.CT.*.IMA"),
                os.path.join(self.dicom_dir, split_name, patient_id, "full_3mm", f"{patient_id}_FD_3_1.CT.*.*.*.*.*.*.*.*.*.IMA"),
                os.path.join(self.dicom_dir, split_name, patient_id, "full_3mm", "*.IMA"),
                os.path.join(self.dicom_dir, split_name, patient_id, "full_3mm", "*.dcm"),
            ]
            for p in patterns:
                matches = glob.glob(p)
                if matches:
                    paths.extend(matches)
                    break

        dangling = [p for p in paths if not os.path.exists(p)]
        if dangling or not paths:
            raise FileNotFoundError(
                f"CRITICAL ERROR (--gt_only): Không tìm thấy tệp DICOM hợp lệ tại '{self.dicom_dir}' "
                f"(split: {split_name}, symlink gãy: {len(dangling)}/{len(paths)})."
            )

        self.slices_path = sorted(paths)
        split_label = "VALID" if valid else ("TEST" if test else "TRAIN")
        print(f"[{split_label}][GT-ONLY] Loaded {len(self.slices_path)} DICOM slices from {self.dicom_dir} (split: {split_name}).")

    def __len__(self) -> int:
        return len(self.slices_path)

    def __getitem__(self, index: int):
        if pydicom is None:
            raise ImportError("Thiếu thư viện 'pydicom' để chạy chế độ --gt_only.")
        slice_path = self.slices_path[index]
        if not os.path.exists(slice_path):
            raise FileNotFoundError(
                f"Không tìm thấy tệp DICOM: '{slice_path}' (đích: '{os.path.realpath(slice_path)}')."
            )
        if os.path.getsize(slice_path) == 0:
            raise ValueError(
                f"Phát hiện tệp DICOM rỗng 0 byte: '{slice_path}' (đích: '{os.path.realpath(slice_path)}')."
            )

        dcm = pydicom.read_file(slice_path)
        data_slice = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
        data_slice = data_slice.astype(float)
        data_slice = (data_slice - np.min(data_slice)) / (np.max(data_slice) - np.min(data_slice) + 1e-8)
        # Tensor ảnh chuẩn hóa (1, 512, 512) -> resize về (1, input_size, input_size)
        phantom = torch.from_numpy(data_slice).unsqueeze(0).float()

        if tuple(phantom.shape[-2:]) != (self.input_size, self.input_size):
            if self.transform is not None:
                phantom = self.transform(phantom)
            else:
                phantom = transforms.Resize((self.input_size, self.input_size))(phantom)
        return slice_path, phantom, None, None


def save_atomic_npy(target_file: str, array: np.ndarray, max_retries: int = 3, retry_delay: float = 0.5) -> None:
    """
    Lưu mảng numpy ra tệp nguyên tử (atomic write) nhằm ngăn chặn tệp bị cắt cụt do lỗi NFS:
    - Ghi dữ liệu mảng ra tệp tạm <target_file>.tmp bằng luồng nhị phân wb.
    - Thực hiện flush và fsync để đảm bảo toàn bộ dirty pages được commit xuống storage NFS.
    - Dùng os.replace để tráo đổi nguyên tử tệp tạm vào vị trí tệp đích cuối cùng.
    """
    tmp_file = target_file + ".tmp"
    # Cơ chế thử lại tự động khi gặp sự cố I/O tạm thời trên NFS với exponential backoff
    for attempt in range(max_retries):
        try:
            with open(tmp_file, "wb") as f:
                np.save(f, array)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_file, target_file)
            return
        except OSError as e:
            # Tự động dọn dẹp tệp tạm khi gặp lỗi nhằm ngăn ngừa tệp rác và nghẽn NFS lock
            if os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except OSError:
                    pass
            if attempt == max_retries - 1:
                raise e
            time.sleep(retry_delay * (2 ** attempt))


def cleanup_stale_tmp_files(*dir_paths: str) -> None:
    """
    Dọn dẹp các tệp tạm .tmp còn sót lại từ các lần chạy trước bị ngắt quãng trên NFS:
    - Quét toàn bộ tệp kết thúc bằng .tmp trong các thư mục chỉ định.
    - Xóa an toàn để tránh xung đột khóa tệp hoặc tràn hạn ngạch đĩa.
    """
    for d in dir_paths:
        if os.path.isdir(d):
            for tmp_file in glob.glob(os.path.join(d, "*.tmp")):
                try:
                    os.remove(tmp_file)
                except OSError:
                    pass


def is_valid_single_cache_file(file_path: str) -> bool:
    """Kiểm tra một tệp cache .npy tồn tại và có kích thước tối thiểu >= 1000 bytes."""
    if file_path and os.path.exists(file_path):
        try:
            return os.path.getsize(file_path) >= 1000
        except OSError:
            return False
    return False


def is_valid_cache_file(sino_file: str, fbp_file: str, gt_file: str = None) -> bool:
    """
    Kiểm tra tính hợp lệ và toàn vẹn của cặp tệp cache sinogram và FBP:
    - Cả 2 tệp phải tồn tại trên hệ thống tệp.
    - Cả 2 tệp phải có kích thước tối thiểu >= 1000 bytes (tránh tệp 128 bytes chỉ có header do NFS stall).
    - Nếu truyền gt_file, tệp Ground Truth (.npy) cũng phải tồn tại và >= 1000 bytes.
    """
    if not (is_valid_single_cache_file(sino_file) and is_valid_single_cache_file(fbp_file)):
        return False
    if gt_file is not None and not is_valid_single_cache_file(gt_file):
        return False
    return True


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
    - Hỗ trợ save_gt_path để lưu song song ảnh Ground Truth (1, H, W) float32 và gt_only=True để chỉ sinh GT.
    """
    if gt_only:
        if not save_gt_path:
            raise ValueError("Chế độ gt_only=True yêu cầu truyền save_gt_path.")
        validate_safe_output_dir(save_gt_path)
        os.makedirs(save_gt_path, exist_ok=True)
        cleanup_stale_tmp_files(save_gt_path)
    else:
        if not save_sino_path or not save_fbp_path:
            raise ValueError("Cần truyền save_sino_path và save_fbp_path khi gt_only=False.")
        validate_safe_output_dir(save_sino_path)
        validate_safe_output_dir(save_fbp_path)
        os.makedirs(save_sino_path, exist_ok=True)
        os.makedirs(save_fbp_path, exist_ok=True)
        # Tự động dọn dẹp các tệp tạm .tmp còn sót lại từ các lần chạy trước bị gián đoạn
        tmp_dirs = [save_sino_path, save_fbp_path]
        if save_gt_path:
            validate_safe_output_dir(save_gt_path)
            os.makedirs(save_gt_path, exist_ok=True)
            tmp_dirs.append(save_gt_path)
        cleanup_stale_tmp_files(*tmp_dirs)

    total_slices = len(provider)
    for i in tqdm(range(total_slices), desc="Generating GT slices" if gt_only else "Generating slices"):
        slice_path = provider.slices_path[i]
        # Tách phần mở rộng bằng os.path.splitext để tương thích .IMA, .dcm và tên nhiều dấu chấm
        stem = os.path.splitext(os.path.basename(slice_path))[0]
        gt_file = os.path.join(save_gt_path, f"{stem}.npy") if save_gt_path else None

        if gt_only:
            if is_valid_single_cache_file(gt_file):
                continue
            sino_file = None
            fbp_file = None
        else:
            sino_file = os.path.join(save_sino_path, f"{stem}.npy")
            fbp_file = os.path.join(save_fbp_path, f"{stem}.npy")

            # Bỏ qua ngay lập tức nếu file đã tồn tại trên đĩa mà không cần tính Radon
            # Kiểm tra bổ sung kích thước file >= 1000 bytes nhằm loại bỏ các file bị cắt cụt do đứt gãy mạng NFS
            if is_valid_cache_file(sino_file, fbp_file, gt_file=gt_file):
                continue

        try:
            # Lấy dữ liệu và thực hiện phép chiếu Radon + FBP on-the-fly
            _, phantom, fbp_u, sino = provider[i]

            if gt_only:
                if phantom is None:
                    raise ValueError(f"Provider trả về phantom=None cho {slice_path}")
                # Mảng Ground Truth chuẩn hóa float32 dạng (1, H, W)
                phantom_np = _to_float32_3d_numpy(phantom)
                save_atomic_npy(gt_file, phantom_np)
            else:
                sino_np = sino.cpu().numpy() if isinstance(sino, torch.Tensor) else sino
                fbp_np = fbp_u.cpu().numpy() if isinstance(fbp_u, torch.Tensor) else fbp_u

                # Lưu mảng numpy
                # Lưu mảng numpy theo phương thức nguyên tử (atomic write qua file tạm .tmp)
                save_atomic_npy(sino_file, sino_np)
                save_atomic_npy(fbp_file, fbp_np)
                if gt_file is not None and not is_valid_single_cache_file(gt_file):
                    if phantom is None:
                        raise ValueError(f"Provider trả về phantom=None cho {slice_path}")
                    # Lưu cache Ground Truth float32 dạng (1, H, W) dùng chung cho mọi góc quét
                    phantom_np = _to_float32_3d_numpy(phantom)
                    save_atomic_npy(gt_file, phantom_np)
        except Exception as e:
            # Ghi thông tin lỗi chi tiết ra cả stdout và stderr kèm traceback và flush ngay lập tức
            err_msg = (
                f"\n[LỖI NGHIÊM TRỌNG] Thất bại tại lát cắt [{i}/{total_slices}]:\n"
                f"  - Đường dẫn DICOM gốc: {slice_path}\n"
                f"  - Sino file đích: {sino_file}\n"
                f"  - FBP file đích: {fbp_file}\n"
                f"  - GT file đích: {gt_file}\n"
                f"  - Chi tiết lỗi: {type(e).__name__}: {e}\n"
            )
            sys.stdout.write(err_msg)
            sys.stderr.write(err_msg)
            traceback.print_exc(file=sys.stdout)
            traceback.print_exc(file=sys.stderr)
            sys.stdout.flush()
            sys.stderr.flush()
            # Ném lại ngoại lệ để dừng job có kiểm soát, tránh âm thầm bỏ qua lát cắt hỏng
            raise e

        # Thu dọn rác bộ nhớ RAM và giải phóng bộ nhớ đệm VRAM GPU định kỳ sau mỗi 500 lát cắt
        if (i + 1) % 500 == 0:
            gc.collect()
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()


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
    if args.gt_only:
        print("🚀 BẮT ĐẦU TẠO CACHE GROUND TRUTH (--gt_only, KHÔNG KHỞI TẠO ODL/ASTRA)")
        print(f"- Độ phân giải không gian: {args.input_size}x{args.input_size}")
        print(f"- Thư mục GT Train/Val: {train_gt_dir}")
        print(f"- Thư mục GT Test: {test_gt_dir}")
    else:
        print(f"🚀 BẮT ĐẦU TẠO SINH DATASET LIMITED-ANGLE CT")
        print(f"- Cung góc quét: [-{args.angle_range_deg/2:.1f}°, +{args.angle_range_deg/2:.1f}°] ({args.angle_range_deg}° total)")
        print(f"- Số góc chiếu (Views): {args.num_view} | Detectors: {args.num_detectors}")
        print(f"- Độ phân giải không gian: {args.input_size}x{args.input_size}")
        print(f"- Cấu hình lưu trữ (Tag): {setting_tag}")
        print(f"- Thư mục Ground Truth chia sẻ: gt_size_{args.input_size}")
    print("=" * 70)

    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size))
    ])

    if args.gt_only:
        # Chế độ thuần Ground Truth: bỏ qua ODL/ASTRA và không chạm sino/ hoặc fbp_u/
        print(f"\n[1/3] Đang xử lý GT tập TRAIN -> {train_gt_dir}")
        train_gt_provider = _DICOMGroundTruthProvider(
            dicom_dir=args.data_dir,
            input_size=args.input_size,
            transform=transform,
            patient_list=args.train_patients,
            test=False,
            valid=False,
        )
        generate_split(train_gt_provider, save_gt_path=train_gt_dir, gt_only=True)

        print(f"\n[2/3] Đang xử lý GT tập VALIDATION -> {train_gt_dir}")
        val_gt_provider = _DICOMGroundTruthProvider(
            dicom_dir=args.data_dir,
            input_size=args.input_size,
            transform=transform,
            patient_list=args.val_patients,
            test=False,
            valid=True,
        )
        generate_split(val_gt_provider, save_gt_path=train_gt_dir, gt_only=True)

        print(f"\n[3/3] Đang xử lý GT tập TEST -> {test_gt_dir}")
        test_gt_provider = _DICOMGroundTruthProvider(
            dicom_dir=args.data_dir,
            input_size=args.input_size,
            transform=transform,
            patient_list=args.test_patients,
            test=True,
            valid=False,
        )
        generate_split(test_gt_provider, save_gt_path=test_gt_dir, gt_only=True)
        print("\n🎉 Tạo cache Ground Truth (--gt_only) hoàn thành thành công!")
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
