"""
================================================================================
MODULE TIỆN ÍCH ĐÁNH GIÁ (EVALUATION UTILITIES)
Cung cấp các hàm chuẩn hóa phục vụ:
1. Lựa chọn checkpoint tốt nhất theo chỉ số val_psnr trên tập kiểm định L110 (Q7, spec-1-decisions.md).
2. Trích xuất thông tin bệnh nhân (patient_id) và chỉ số lát cắt (slice_idx).
3. Tính toán dải động (dynamic range) theo từng lát cắt độc lập (metrics-decision.md).
4. Định dạng và ghi nhận tệp CSV kết quả đánh giá định lượng chi tiết.
5. Kiểm tra nghiêm ngặt kích thước tập kiểm định (Validation Dataset Integrity).

Dự án: Nghiên cứu Tái tạo Ảnh Cắt lớp CT Góc Giới hạn (Limited-Angle CT Reconstruction)
Tác giả: MinhPD — Nhóm Nghiên cứu Tái tạo Ảnh Y tế (VNU-HCM UIT)
================================================================================
"""

import os
import re
import glob
from typing import Tuple, List, Optional

# Danh mục các cột chuẩn hóa theo quyết định metrics-decision.md
CSV_COLUMNS: List[str] = [
    "method",
    "angle",
    "patient_id",
    "slice_idx",
    "psnr",
    "ssim",
    "rmse",
    "job_id",
]


def find_best_val_checkpoint(ckpt_dir: str) -> str:
    """
    Tìm kiếm và lựa chọn checkpoint có điểm val_psnr cao nhất trong thư mục.
    
    Quy tắc nghiêm ngặt theo quyết định Q7 (spec-1-decisions.md):
    - Chỉ chấp nhận checkpoint được lưu bởi ModelCheckpoint dựa trên val_psnr (đo trên L110).
    - Tuyệt đối loại bỏ tệp 'last.ckpt' khỏi danh sách ứng viên tốt nhất.
    - Nếu không tìm thấy checkpoint nào chứa giá trị val_psnr, dừng tiến trình và báo lỗi (NO fallback).
    
    Args:
        ckpt_dir: Đường dẫn thư mục chứa các tệp checkpoint (.ckpt).
        
    Returns:
        Đường dẫn tuyệt đối đến tệp checkpoint tốt nhất.
        
    Raises:
        FileNotFoundError: Nếu không tìm thấy checkpoint val_psnr hợp lệ nào.
    """
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(
            f"CRITICAL ERROR: Thư mục checkpoint không tồn tại: {ckpt_dir} (NO fallback)."
        )

    # Tìm kiếm toàn bộ các tệp .ckpt trong thư mục và các thư mục con
    all_ckpts = glob.glob(os.path.join(ckpt_dir, "**", "*.ckpt"), recursive=True)
    if not all_ckpts:
        # Kiểm tra thêm tại thư mục gốc nếu tìm kiếm đệ quy không thấy
        all_ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))

    val_candidates: List[Tuple[float, str]] = []
    
    # Duyệt qua từng checkpoint để trích xuất điểm val_psnr
    for ckpt_path in all_ckpts:
        basename = os.path.basename(ckpt_path)
        # Bỏ qua tệp last.ckpt chỉ dành riêng cho việc resume
        if basename == "last.ckpt":
            continue
        
        # Mẫu trích xuất điểm số val_psnr từ tên tệp (ví dụ: val_psnr=36.2681)
        match = re.search(r"val_psnr=([0-9]+(?:\.[0-9]+)?)", basename)
        if match:
            try:
                score = float(match.group(1))
                val_candidates.append((score, ckpt_path))
            except ValueError:
                continue

    # Nếu không tìm thấy bất kỳ checkpoint nào có điểm val_psnr hợp lệ
    if not val_candidates:
        raise FileNotFoundError(
            f"CRITICAL ERROR: Không tìm thấy checkpoint val_psnr hợp lệ nào trong '{ckpt_dir}'! "
            f"Tổng số tệp .ckpt tìm thấy: {len(all_ckpts)}. "
            "Quy tắc Q7 yêu cầu checkpoint phải được đánh giá hợp lệ trên L110 (tuyệt đối NO fallback về last.ckpt)."
        )

    # Sắp xếp giảm dần theo điểm số val_psnr để chọn checkpoint cao nhất
    val_candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_path = val_candidates[0]
    
    print(f"[INFO] Đã chọn checkpoint tốt nhất: {best_path} (val_psnr = {best_score:.4f} dB)")
    return best_path


def parse_patient_and_slice_idx(file_path: str, fallback_idx: int = 1) -> Tuple[str, int]:
    """
    Trích xuất mã bệnh nhân (patient_id) và chỉ số lát cắt (slice_idx) từ đường dẫn tệp ảnh.
    
    Hỗ trợ các mẫu quy ước:
    - Symlink đánh số chuẩn: L058_FD_3_1.CT.00000042.IMA -> ('L058', 42)
    - File DICOM thô: .../L064/full_3mm/00000105.dcm -> ('L064', 105)
    - File legacy chấm nhiều tầng: L094_FD_3_1.CT.1.2.3.4.5.6.7.8.18.IMA -> ('L094', 18)
    
    Args:
        file_path: Đường dẫn tệp lát cắt.
        fallback_idx: Chỉ số lát cắt dự phòng nếu không phân tích được từ tên tệp.
        
    Returns:
        Tuple chứa (patient_id, slice_idx).
    """
    basename = os.path.basename(file_path)
    
    # 1. Trích xuất Patient ID (mã bệnh nhân dạng L kèm số, ví dụ L058, L110)
    pid_match = re.search(r"\b(L\d+)\b", file_path)
    if pid_match:
        patient_id = pid_match.group(1)
    else:
        # Dự phòng trích xuất từ tên tệp
        name_match = re.search(r"^(L\d+)", basename)
        patient_id = name_match.group(1) if name_match else "UNKNOWN"

    # 2. Trích xuất Slice Index (chỉ số lát cắt)
    # Tìm kiếm chuỗi số nằm ngay trước phần mở rộng tệp (.IMA hoặc .dcm hoặc .npy)
    slice_match = re.search(r"(\d+)\.(?:IMA|dcm|npy)$", basename, re.IGNORECASE)
    if slice_match:
        slice_idx = int(slice_match.group(1))
    else:
        slice_idx = fallback_idx

    return patient_id, slice_idx


def format_csv_row(
    method: str,
    angle: float,
    patient_id: str,
    slice_idx: int,
    psnr: float,
    ssim: float,
    rmse: float,
    job_id: str = "local",
) -> str:
    """
    Định dạng dòng dữ liệu CSV theo đúng chuẩn columns của metrics-decision.md:
    method, angle, patient_id, slice_idx, psnr, ssim, rmse, job_id
    
    Args:
        method: Tên phương pháp / mô hình (ví dụ: SOLAR_DualMamba, LEARN).
        angle: Góc quét tính bằng độ (ví dụ: 90.0, 120.0).
        patient_id: Mã định danh bệnh nhân (ví dụ: L058).
        slice_idx: Chỉ số lát cắt CT (ví dụ: 42).
        psnr: Giá trị PSNR tính bằng dB.
        ssim: Giá trị SSIM cấu trúc.
        rmse: Giá trị sai số căn quân phương RMSE.
        job_id: Mã định danh Slurm Job (hoặc 'local').
        
    Returns:
        Chuỗi CSV định dạng chuẩn kết thúc bằng ký tự xuống dòng.
    """
    return f"{method},{angle:.1f},{patient_id},{slice_idx},{psnr:.4f},{ssim:.4f},{rmse:.6f},{job_id}\n"


def validate_val_dataset_size(num_slices: int) -> bool:
    """
    Kiểm tra tính toàn vẹn của tập kiểm định (Validation) trên bệnh nhân L110.
    
    Quy tắc bắt buộc:
    - Nếu tập Validation tải được 0 lát cắt, phải dừng ngay lập tức và ném ngoại lệ RuntimeError.
    - Không cho phép tiếp tục huấn luyện để tránh lưu checkpoint không có cơ sở đánh giá (NO fallback).
    
    Args:
        num_slices: Số lượng lát cắt tải được trong tập kiểm định.
        
    Returns:
        True nếu số lát cắt > 0.
        
    Raises:
        RuntimeError: Nếu num_slices == 0.
    """
    if num_slices <= 0:
        raise RuntimeError(
            "CRITICAL ERROR: Tập kiểm định (Validation) tải được 0 lát cắt từ L110! "
            "Dừng tiến trình ngay lập tức để không lưu checkpoint sai lệch (NO fallback)."
        )
    return True


def compute_slice_metrics(pred_slice, gt_slice) -> Tuple[float, float, float]:
    """
    Tính toán 3 chỉ số chất lượng ảnh PSNR, SSIM, RMSE cho một lát cắt đơn lẻ.
    Tuân thủ tuyệt đối quy định tại metrics-decision.md:
    - data_range = (gt_slice.amin(), gt_slice.amax())
    - PSNR: torchmetrics peak_signal_noise_ratio trên từng lát cắt
    - SSIM: torchmetrics structural_similarity_index_measure trên từng lát cắt
    - RMSE: sai số căn quân phương theo đơn vị cường độ gốc
    
    Args:
        pred_slice: Tensor ảnh tái tạo (1, 1, H, W).
        gt_slice: Tensor ảnh gốc Ground Truth (1, 1, H, W).
        
    Returns:
        Tuple chứa (psnr, ssim, rmse).
    """
    import torch
    from torchmetrics.functional.image import (
        peak_signal_noise_ratio,
        structural_similarity_index_measure,
    )

    # Xác định dải động riêng của lát cắt Ground Truth hiện tại
    gt_min = gt_slice.amin()
    gt_max = gt_slice.amax()
    if torch.isclose(gt_max, gt_min):
        data_range = (gt_min, gt_min + torch.tensor(1e-8, device=gt_slice.device, dtype=gt_slice.dtype))
    else:
        data_range = (gt_min, gt_max)

    # Đo lường định lượng PSNR và SSIM
    psnr_val = peak_signal_noise_ratio(pred_slice, gt_slice, data_range=data_range).item()
    ssim_val = structural_similarity_index_measure(pred_slice, gt_slice, data_range=data_range).item()
    # Đo lường RMSE
    rmse_val = torch.sqrt(torch.mean((pred_slice - gt_slice) ** 2)).item()

    return psnr_val, ssim_val, rmse_val


def run_slice_by_slice_evaluation(
    model,
    datamodule,
    method_name: str,
    angle_deg: float,
    output_csv: Optional[str] = None,
    job_id: str = "local",
    device: Optional[str] = None,
) -> dict:
    """
    Thực hiện đánh giá toàn diện mô hình trên tập kiểm thử (Test Cohort) theo từng lát cắt.
    
    Quy trình:
    1. Kích hoạt return_path để thu thập đường dẫn gốc của từng lát cắt DICOM.
    2. Chạy suy luận (Inference) và tính toán PSNR, SSIM, RMSE với dynamic range riêng của lát đó.
    3. Ghi kết quả chi tiết từng lát cắt vào tệp CSV (columns: method, angle, patient_id, slice_idx, psnr, ssim, rmse, job_id).
    4. Tổng hợp chỉ số trung bình theo từng bệnh nhân (per-patient mean).
    5. Tính toán chỉ số tổng thể toàn tập (Headline: Mean ± Std trên các bệnh nhân).
    
    Args:
        model: Mô hình PyTorch / PyTorch Lightning cần đánh giá.
        datamodule: DataModule CT góc giới hạn chứa tập test.
        method_name: Tên phương pháp đánh giá (ví dụ: 'SOLAR_DualMamba').
        angle_deg: Góc quét giới hạn tính bằng độ (ví dụ: 90.0 hoặc 120.0).
        output_csv: Đường dẫn tệp CSV lưu kết quả theo lát cắt.
        job_id: Mã định danh Slurm Job (mặc định: 'local').
        device: Thiết bị tính toán ('cuda' hoặc 'cpu').
        
    Returns:
        Dictionary chứa bảng thống kê tổng hợp kết quả.
    """
    import numpy as np
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    # Thiết lập tập kiểm thử và bật cờ trả về đường dẫn tệp
    datamodule.setup("test")
    test_dataset = getattr(datamodule, "test_dataset", None)
    if test_dataset is not None:
        test_dataset.return_path = True

    test_loader = datamodule.test_dataloader()
    total_slices = len(test_dataset) if test_dataset is not None else len(test_loader)
    print(f"🔍 Bắt đầu đánh giá {method_name} (Góc {angle_deg:.1f}°) trên {total_slices} lát cắt...")

    csv_file = None
    if output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        write_header = not os.path.exists(output_csv)
        csv_file = open(output_csv, "a", encoding="utf-8")
        if write_header:
            csv_file.write(",".join(CSV_COLUMNS) + "\n")

    patient_records = {}
    all_psnr, all_ssim, all_rmse = [], [], []

    slice_counter = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Xử lý linh hoạt cấu trúc batch (có hoặc không có đường dẫn file)
            if len(batch) == 4 and isinstance(batch[0], (list, tuple)):
                paths, phantom, fbp_u, sino = batch
            elif len(batch) == 3:
                phantom, fbp_u, sino = batch
                paths = [f"slice_{slice_counter + i + 1:04d}.dcm" for i in range(phantom.shape[0])]
            else:
                paths, phantom, fbp_u, sino = batch

            phantom = phantom.to(device)
            fbp_u = fbp_u.to(device)
            sino = sino.to(device)

            # Thực thi mô hình
            output = model(fbp_u, sino)
            pred = output[0] if isinstance(output, (tuple, list)) else output

            batch_sz = phantom.shape[0]
            for i in range(batch_sz):
                slice_counter += 1
                p_path = paths[i]
                gt_i = phantom[i:i+1]
                pred_i = pred[i:i+1]

                pid, s_idx = parse_patient_and_slice_idx(p_path, fallback_idx=slice_counter)
                psnr, ssim, rmse = compute_slice_metrics(pred_i, gt_i)

                all_psnr.append(psnr)
                all_ssim.append(ssim)
                all_rmse.append(rmse)

                if pid not in patient_records:
                    patient_records[pid] = {"psnr": [], "ssim": [], "rmse": []}
                patient_records[pid]["psnr"].append(psnr)
                patient_records[pid]["ssim"].append(ssim)
                patient_records[pid]["rmse"].append(rmse)

                if csv_file:
                    row_str = format_csv_row(method_name, angle_deg, pid, s_idx, psnr, ssim, rmse, job_id)
                    csv_file.write(row_str)
                    csv_file.flush()

    if csv_file:
        csv_file.close()
        print(f"✅ Đã ghi nhận {slice_counter} dòng kết quả lát cắt vào: {output_csv}")

    # Bảng tổng kết kết quả theo từng bệnh nhân
    print("\n" + "=" * 80)
    print(f"📊 BẢNG TỔNG KẾT ĐÁNH GIÁ ĐỊNH LƯỢNG: {method_name} ({angle_deg:.1f}°)")
    print(f"   (Slurm Job ID: {job_id} | Tổng số lát cắt: {slice_counter})")
    print("=" * 80)
    print(f"{'Patient ID':<12} | {'Slices':<8} | {'PSNR (dB)':<12} | {'SSIM':<10} | {'RMSE':<10}")
    print("-" * 80)

    patient_means_psnr = []
    patient_means_ssim = []
    patient_means_rmse = []

    for pid in sorted(patient_records.keys()):
        vals = patient_records[pid]
        p_psnr = float(np.mean(vals["psnr"]))
        p_ssim = float(np.mean(vals["ssim"]))
        p_rmse = float(np.mean(vals["rmse"]))
        n_slices = len(vals["psnr"])

        patient_means_psnr.append(p_psnr)
        patient_means_ssim.append(p_ssim)
        patient_means_rmse.append(p_rmse)

        print(f"{pid:<12} | {n_slices:<8} | {p_psnr:<12.4f} | {p_ssim:<10.4f} | {p_rmse:<10.6f}")

    print("-" * 80)
    headline_psnr_mean = float(np.mean(patient_means_psnr))
    headline_psnr_std = float(np.std(patient_means_psnr))
    headline_ssim_mean = float(np.mean(patient_means_ssim))
    headline_ssim_std = float(np.std(patient_means_ssim))
    headline_rmse_mean = float(np.mean(patient_means_rmse))
    headline_rmse_std = float(np.std(patient_means_rmse))

    print(f"OVERALL (Headline Mean ± Std qua {len(patient_means_psnr)} bệnh nhân):")
    print(f"  • PSNR: {headline_psnr_mean:.2f} ± {headline_psnr_std:.2f} dB")
    print(f"  • SSIM: {headline_ssim_mean:.4f} ± {headline_ssim_std:.4f}")
    print(f"  • RMSE: {headline_rmse_mean:.6f} ± {headline_rmse_std:.6f}")
    print("=" * 80 + "\n")

    return {
        "method": method_name,
        "angle": angle_deg,
        "job_id": job_id,
        "patients": patient_records,
        "headline": {
            "psnr_mean": headline_psnr_mean,
            "psnr_std": headline_psnr_std,
            "ssim_mean": headline_ssim_mean,
            "ssim_std": headline_ssim_std,
            "rmse_mean": headline_rmse_mean,
            "rmse_std": headline_rmse_std,
        },
    }

