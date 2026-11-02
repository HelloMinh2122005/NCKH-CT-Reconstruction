#!/usr/bin/env python3
"""
Bộ công cụ đánh giá (Evaluation Harness) thuật toán Tái tạo ảnh FBP (Filtered Backprojection)
cho bài toán CT góc giới hạn (Limited-Angle CT) trên tập kiểm thử (Test Cohort).

Quy tắc tính toán tuân thủ nghiêm ngặt theo:
- .lab/20261026-restart-intake/experiments/metrics-decision.md
- .lab/20261026-restart-intake/experiments/split.md (Test patients: L058, L064, L094, L107)
- .lab/20261026-restart-intake/theory/geometry-decision.md (src_radius=600, det_radius=290)
"""

import os
import sys
import argparse
import csv
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

# Đảm bảo đường dẫn import thư viện từ thư mục gốc của repository
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import tùy chọn PyTorch và TorchMetrics (hỗ trợ kiểm thử trên máy local không có GPU/Torch)
try:
    import torch
    import torch.nn.functional as F
    from torchmetrics.functional.image import (
        peak_signal_noise_ratio,
        structural_similarity_index_measure,
    )
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

# Import Provider dữ liệu từ module data của dự án
try:
    from data.CTSlice_Provider_LA import LimitedAngleCT_Provider
except ImportError:
    try:
        from data.aapm.CTSlice_Provider_LA import LimitedAngleCT_Provider
    except ImportError:
        LimitedAngleCT_Provider = None

# Danh sách 8 cột chuẩn của file CSV theo quy định tại metrics-decision.md
CSV_COLUMNS = [
    "method",
    "angle",
    "patient_id",
    "slice_idx",
    "psnr",
    "ssim",
    "rmse",
    "job_id",
]

# Danh sách 4 bệnh nhân kiểm thử chuẩn theo split.md
DEFAULT_TEST_PATIENTS = ["L058", "L064", "L094", "L107"]


def get_slice_data_range_numpy(gt_slice: np.ndarray) -> Tuple[float, float]:
    """
    Xác định dải động (dynamic range) từ chính lát cắt ảnh gốc Ground Truth (mảng NumPy).
    
    Quy tắc bắt buộc:
    - Tuyệt đối KHÔNG dùng data_range = 1.0 cố định.
    - Dải động = (gt_slice.min(), gt_slice.max()).
    - Nếu max xấp xỉ min (ảnh phẳng), cộng thêm epsilon 1e-8 để tránh chia cho 0.
    
    Args:
        gt_slice: Mảng numpy 2D (H, W) của lát cắt ảnh gốc.
        
    Returns:
        Tuple (min_val, max_val) đại diện cho dải động của lát cắt.
    """
    min_val = float(np.min(gt_slice))
    max_val = float(np.max(gt_slice))
    if np.isclose(max_val, min_val):
        max_val = min_val + 1e-8
    return (min_val, max_val)


def get_slice_data_range_torch(gt_slice: Any) -> Tuple[Any, Any]:
    """
    Xác định dải động của lát cắt ảnh gốc dưới dạng PyTorch Tensor.
    
    Args:
        gt_slice: Tensor PyTorch (1, 1, H, W) hoặc (1, H, W).
        
    Returns:
        Tuple (min_val, max_val) kiểu Tensor trên cùng thiết bị (device).
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch chưa được cài đặt trong môi trường hiện tại.")
    
    min_val = gt_slice.amin()
    max_val = gt_slice.amax()
    if torch.isclose(max_val, min_val):
        max_val = min_val + torch.tensor(1e-8, device=gt_slice.device, dtype=gt_slice.dtype)
    return (min_val, max_val)


def compute_slice_metrics(
    pred: Any,
    gt: Any,
) -> Tuple[float, float, float]:
    """
    Tính toán 3 chỉ số định lượng chất lượng ảnh y tế (PSNR, SSIM, RMSE) cho từng lát cắt riêng lẻ.
    
    Quy tắc tính toán (metrics-decision.md):
    - pred, gt: Tensor PyTorch (1, 1, 256, 256) ở độ phân giải mô hình.
    - data_range: tính riêng cho từng lát cắt từ (gt.min(), gt.max()), không dùng 1.0.
    - PSNR: peak_signal_noise_ratio(pred, gt, data_range=data_range).
    - SSIM: structural_similarity_index_measure(pred, gt, data_range=data_range), cửa sổ mặc định 11x11.
    - RMSE: torch.sqrt(torch.mean((gt - pred) ** 2)).
    
    Args:
        pred: Tensor ảnh tái tạo dự đoán (1, 1, H=256, W=256).
        gt: Tensor ảnh gốc Ground Truth (1, 1, H=256, W=256).
        
    Returns:
        Tuple (psnr, ssim, rmse) dạng số thực (float).
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch và TorchMetrics là bắt buộc để tính toán metric theo chuẩn.")
    
    # Đảm bảo Tensor đầu vào có đủ 4 chiều (Batch=1, Channels=1, Height=256, Width=256)
    if pred.dim() == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
    elif pred.dim() == 3:
        pred = pred.unsqueeze(0)
        
    if gt.dim() == 2:
        gt = gt.unsqueeze(0).unsqueeze(0)
    elif gt.dim() == 3:
        gt = gt.unsqueeze(0)
    
    # 1. Xác định dải động từ chính lát cắt GT
    data_range = get_slice_data_range_torch(gt)
    
    # 2. Tính PSNR lát cắt bằng torchmetrics
    psnr_tensor = peak_signal_noise_ratio(pred, gt, data_range=data_range)
    psnr_val = float(psnr_tensor.detach().cpu().item())
    
    # 3. Tính SSIM lát cắt bằng torchmetrics (cửa sổ mặc định 11x11, sigma 1.5)
    ssim_tensor = structural_similarity_index_measure(pred, gt, data_range=data_range)
    ssim_val = float(ssim_tensor.detach().cpu().item())
    
    # 4. Tính RMSE theo công thức căn bậc hai sai số toàn phương trung bình
    rmse_tensor = torch.sqrt(torch.mean((gt - pred) ** 2))
    rmse_val = float(rmse_tensor.detach().cpu().item())
    
    return psnr_val, ssim_val, rmse_val


def extract_patient_id_from_path(slice_path: str, known_patients: Optional[List[str]] = None) -> str:
    """
    Trích xuất mã định danh bệnh nhân (Patient ID) từ đường dẫn file ảnh lát cắt CT.
    
    Hỗ trợ các định dạng:
    - Đường dẫn tuyệt đối: .../split/test/L058/full_3mm/...
    - Đường dẫn tương đối: split/test/L064/full_3mm/...
    - Tên file chứa prefix bệnh nhân: L058_FD_3_1.CT...
    
    Args:
        slice_path: Chuỗi đường dẫn tới file ảnh DICOM hoặc cache.
        known_patients: Danh sách các mã bệnh nhân đã biết để ưu tiên khớp chính xác.
        
    Returns:
        Chuỗi mã bệnh nhân (ví dụ: 'L058').
    """
    norm_path = slice_path.replace("\\", "/")
    
    # Cách 1: Khớp với danh sách bệnh nhân đã biết (nếu cung cấp)
    if known_patients:
        for pid in known_patients:
            pattern = rf"(?:^|/){re.escape(pid)}(?:/|_|\.|$)"
            if re.search(pattern, norm_path):
                return pid
                
    # Cách 2: Tìm thư mục con ngay sau 'test' hoặc 'train' hoặc 'val'
    parts = norm_path.split("/")
    for i, part in enumerate(parts):
        if part in ["test", "train", "val"] and i + 1 < len(parts):
            next_part = parts[i + 1]
            if re.match(r"^L\d+$", next_part):
                return next_part
                
    # Cách 3: Tìm bất kỳ phần nào có định dạng L theo sau bởi số (ví dụ: L058, L310)
    for part in parts:
        match = re.match(r"^(L\d+)", part)
        if match:
            return match.group(1)
            
    # Trường hợp dự phòng nếu không tìm thấy mẫu
    return "UNKNOWN"


def format_csv_row(
    method: str,
    angle: Any,
    patient_id: str,
    slice_idx: int,
    psnr: float,
    ssim: float,
    rmse: float,
    job_id: str,
) -> Dict[str, Any]:
    """
    Đóng gói dữ liệu kết quả lát cắt thành từ điển đúng theo 8 cột chuẩn của file CSV.
    """
    return {
        "method": str(method),
        "angle": int(round(float(angle))) if isinstance(angle, (int, float)) else str(angle),
        "patient_id": str(patient_id),
        "slice_idx": int(slice_idx),
        "psnr": float(psnr),
        "ssim": float(ssim),
        "rmse": float(rmse),
        "job_id": str(job_id),
    }


def compute_patient_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Tổng hợp kết quả định lượng:
    1. Tính giá trị trung bình theo từng bệnh nhân qua các lát cắt (per-patient mean).
    2. Tính giá trị trung bình và độ lệch chuẩn của cả nhóm (overall mean ± sample std qua 4 bệnh nhân).
    
    Args:
        records: Danh sách các dòng dữ liệu lát cắt từ format_csv_row.
        
    Returns:
        Từ điển chứa thống kê chi tiết theo bệnh nhân và của toàn bộ nhóm kiểm thử.
    """
    if not records:
        return {
            "patient_stats": {},
            "cohort_stats": {
                "psnr_mean": 0.0, "psnr_std": 0.0,
                "ssim_mean": 0.0, "ssim_std": 0.0,
                "rmse_mean": 0.0, "rmse_std": 0.0,
                "patient_count": 0,
                "total_slices": 0,
            }
        }
    
    # Nhóm dữ liệu theo mã bệnh nhân
    patient_groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        pid = r["patient_id"]
        if pid not in patient_groups:
            patient_groups[pid] = []
        patient_groups[pid].append(r)
        
    patient_stats: Dict[str, Dict[str, Any]] = {}
    patient_psnrs: List[float] = []
    patient_ssims: List[float] = []
    patient_rmses: List[float] = []
    
    for pid in sorted(patient_groups.keys()):
        group = patient_groups[pid]
        psnrs = [item["psnr"] for item in group]
        ssims = [item["ssim"] for item in group]
        rmses = [item["rmse"] for item in group]
        
        m_psnr = float(np.mean(psnrs))
        m_ssim = float(np.mean(ssims))
        m_rmse = float(np.mean(rmses))
        
        patient_stats[pid] = {
            "slice_count": len(group),
            "mean_psnr": m_psnr,
            "mean_ssim": m_ssim,
            "mean_rmse": m_rmse,
        }
        
        patient_psnrs.append(m_psnr)
        patient_ssims.append(m_ssim)
        patient_rmses.append(m_rmse)
        
    # Tính toán chỉ số tổng hợp của nhóm (Headline: mean ± sample std qua 4 bệnh nhân)
    n_pts = len(patient_psnrs)
    cohort_psnr_mean = float(np.mean(patient_psnrs))
    cohort_psnr_std = float(np.std(patient_psnrs, ddof=1)) if n_pts > 1 else 0.0
    
    cohort_ssim_mean = float(np.mean(patient_ssims))
    cohort_ssim_std = float(np.std(patient_ssims, ddof=1)) if n_pts > 1 else 0.0
    
    cohort_rmse_mean = float(np.mean(patient_rmses))
    cohort_rmse_std = float(np.std(patient_rmses, ddof=1)) if n_pts > 1 else 0.0
    
    cohort_stats = {
        "psnr_mean": cohort_psnr_mean,
        "psnr_std": cohort_psnr_std,
        "ssim_mean": cohort_ssim_mean,
        "ssim_std": cohort_ssim_std,
        "rmse_mean": cohort_rmse_mean,
        "rmse_std": cohort_rmse_std,
        "patient_count": n_pts,
        "total_slices": len(records),
    }
    
    return {
        "patient_stats": patient_stats,
        "cohort_stats": cohort_stats,
    }


def write_per_slice_csv(records: List[Dict[str, Any]], output_csv_path: str) -> None:
    """
    Ghi danh sách kết quả lát cắt ra tệp CSV với đúng 8 cột chuẩn.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in records:
            writer.writerow(r)


def print_and_save_summary(
    summary: Dict[str, Any],
    angle: float,
    output_dir: str,
    method: str = "FBP",
) -> None:
    """
    Hiển thị bảng tổng hợp kết quả trực quan ra console và lưu tệp summary.csv và summary.json.
    """
    patient_stats = summary["patient_stats"]
    cohort = summary["cohort_stats"]
    
    print("\n" + "=" * 80)
    print(f"📊 BẢNG TỔNG HỢP KẾT QUẢ ĐÁNH GIÁ {method.upper()} (Cung quét: {int(angle)}°)")
    print("=" * 80)
    print(f"{'Bệnh nhân':<12} {'Số lát':<10} {'PSNR (dB)':<18} {'SSIM':<18} {'RMSE':<18}")
    print("-" * 80)
    for pid, s in patient_stats.items():
        print(f"{pid:<12} {s['slice_count']:<10} {s['mean_psnr']:<18.4f} {s['mean_ssim']:<18.4f} {s['mean_rmse']:<18.6f}")
    print("-" * 80)
    print(f"TRUNG BÌNH {cohort['patient_count']} BỆNH NHÂN (Mean ± Std):")
    print(f"  • PSNR: {cohort['psnr_mean']:.4f} ± {cohort['psnr_std']:.4f} dB")
    print(f"  • SSIM: {cohort['ssim_mean']:.4f} ± {cohort['ssim_std']:.4f}")
    print(f"  • RMSE: {cohort['rmse_mean']:.6f} ± {cohort['rmse_std']:.6f}")
    print(f"  • Tổng số lát cắt: {cohort['total_slices']}")
    print("=" * 80 + "\n")
    
    # Lưu tệp summary.json
    json_path = os.path.join(output_dir, "summary_test.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        
    # Lưu tệp summary.csv
    csv_summary_path = os.path.join(output_dir, "summary_test.csv")
    with open(csv_summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "slice_count", "mean_psnr", "mean_ssim", "mean_rmse"])
        for pid, s in patient_stats.items():
            writer.writerow([pid, s["slice_count"], s["mean_psnr"], s["mean_ssim"], s["mean_rmse"]])
        writer.writerow([
            "OVERALL_COHORT",
            cohort["total_slices"],
            f"{cohort['psnr_mean']:.4f} ± {cohort['psnr_std']:.4f}",
            f"{cohort['ssim_mean']:.4f} ± {cohort['ssim_std']:.4f}",
            f"{cohort['rmse_mean']:.6f} ± {cohort['rmse_std']:.6f}",
        ])


def evaluate_fbp_for_angle(
    angle_deg: float,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    Thực thi quy trình nạp dữ liệu, tái tạo FBP và đo lường định lượng cho một góc quét chỉ định.
    
    Quy trình:
    1. Cấu hình góc quét đối xứng [-angle/2, +angle/2] (đơn vị radian).
    2. Khởi tạo LimitedAngleCT_Provider cho tập test (4 bệnh nhân: L058, L064, L094, L107).
    3. Duyệt qua từng lát cắt (tổng 757 lát cắt):
       - Lấy ảnh gốc Ground Truth (phantom) và ảnh FBP (fbp_u) tại độ phân giải 256x256.
       - Tính PSNR, SSIM, RMSE theo dải động riêng của lát cắt đó.
    4. Xuất file per_slice_test.csv và in bảng tổng hợp thống kê bệnh nhân.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch và TorchMetrics là bắt buộc để chạy đánh giá trên GPU/CPU.")
        
    if LimitedAngleCT_Provider is None:
        raise ImportError("Không thể import LimitedAngleCT_Provider từ module data.")

    # 1. Cấu hình góc quét và thẻ lưu trữ cache
    half_angle_rad = np.deg2rad(angle_deg / 2.0)
    start_ang = -half_angle_rad
    end_ang = half_angle_rad
    
    setting_tag = f"limited_ang_{int(angle_deg)}deg_numview_{args.num_view}_size_{args.input_size}"
    if args.poisson_level > 0:
        setting_tag += f"_poisson_{int(args.poisson_level)}"
    else:
        setting_tag += "_noise_0"
        
    output_dir = args.output_dir if args.output_dir else os.path.join(args.output_base_dir, f"FBP_{int(angle_deg)}", "eval")
    os.makedirs(output_dir, exist_ok=True)
    per_slice_csv_path = args.output_csv if args.output_csv else os.path.join(output_dir, "per_slice_test.csv")

    print(f"\n🚀 BẮT ĐẦU ĐÁNH GIÁ FBP CHO GÓC QUÉT {int(angle_deg)}°")
    print(f"- Cung quét: [-{angle_deg/2:.1f}°, +{angle_deg/2:.1f}°] ({angle_deg}°)")
    print(f"- Tập bệnh nhân kiểm thử: {args.test_patients}")
    print(f"- Thư mục DICOM: {args.dicom_dir}")
    print(f"- Thư mục Cache: {args.cache_dir} (use_precomputed={args.use_precomputed})")
    print(f"- File CSV đầu ra: {per_slice_csv_path}")

    # Pipeline resize ảnh đảm bảo kích thước 256x256
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size))
    ])

    # Khởi tạo Dataset Provider cho tập kiểm thử độc lập
    test_dataset = LimitedAngleCT_Provider(
        dicom_dir=args.dicom_dir,
        cache_dir=args.cache_dir,
        start_ang=start_ang,
        end_ang=end_ang,
        num_view=args.num_view,
        num_detectors=args.num_detectors,
        input_size=args.input_size,
        transform=transform,
        poission_level=args.poisson_level,
        gaussian_level=args.gaussian_level,
        use_precomputed=args.use_precomputed,
        precomputed_setting=setting_tag,
        patient_list=args.test_patients,
        test=True,
        valid=False,
        return_path=True,
    )

    total_slices = len(test_dataset)
    print(f"- Tổng số lát cắt nạp được: {total_slices}")

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"- Thiết bị tính toán: {device}")

    records: List[Dict[str, Any]] = []
    
    # Theo dõi chỉ số slice theo từng bệnh nhân
    patient_slice_counters: Dict[str, int] = {pid: 0 for pid in args.test_patients}

    for i in range(total_slices):
        # Lấy dữ liệu: slice_path, phantom (GT), fbp_u (FBP), sino
        slice_path, phantom, fbp_u, sino = test_dataset[i]
        
        # Trích xuất mã bệnh nhân từ đường dẫn lát cắt
        patient_id = extract_patient_id_from_path(slice_path, args.test_patients)
        slice_idx = patient_slice_counters.get(patient_id, 0)
        patient_slice_counters[patient_id] = slice_idx + 1

        # Đưa tensor lên thiết bị tính toán
        phantom = phantom.to(device)
        fbp_u = fbp_u.to(device)

        # Tính toán metric PSNR, SSIM, RMSE theo đúng dải động riêng của lát cắt
        psnr, ssim, rmse = compute_slice_metrics(fbp_u, phantom)

        row = format_csv_row(
            method="FBP",
            angle=angle_deg,
            patient_id=patient_id,
            slice_idx=slice_idx,
            psnr=psnr,
            ssim=ssim,
            rmse=rmse,
            job_id=args.job_id,
        )
        records.append(row)

        if (i + 1) % 100 == 0 or (i + 1) == total_slices:
            print(f"  [{i + 1}/{total_slices}] BN: {patient_id} Lát #{slice_idx}: PSNR={psnr:.2f}dB, SSIM={ssim:.4f}, RMSE={rmse:.5f}")

    # Ghi tệp CSV per-slice
    write_per_slice_csv(records, per_slice_csv_path)
    print(f"✅ Đã ghi thành công {len(records)} dòng vào: {per_slice_csv_path}")

    # Tính toán tổng hợp và in kết quả
    summary = compute_patient_summary(records)
    print_and_save_summary(summary, angle=angle_deg, output_dir=output_dir, method="FBP")

    return summary


def parse_args(args_list: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Khai báo các tham số dòng lệnh phục vụ việc đánh giá FBP trên tập kiểm thử.
    """
    parser = argparse.ArgumentParser(
        description="Đánh giá FBP trên tập kiểm thử Limited-Angle CT (120° và 90°) theo chuẩn metrics-decision.md"
    )
    
    # Góc quét CT cần đánh giá
    parser.add_argument(
        "--angles",
        "--angle_range_deg",
        nargs="+",
        type=float,
        default=[120.0, 90.0],
        help="Danh sách các cung quét góc giới hạn cần đánh giá (mặc định: 120.0 90.0)"
    )
    
    # Danh sách bệnh nhân kiểm thử
    parser.add_argument(
        "--test_patients",
        nargs="+",
        default=DEFAULT_TEST_PATIENTS,
        help="Danh sách mã bệnh nhân tập kiểm thử (mặc định: L058 L064 L094 L107)"
    )
    
    # Cấu hình đường dẫn dữ liệu trên HPC-UIT
    default_dicom = "/datastore/doanhbc/minhdphan/datasets/aapm/split/"
    default_cache = "/datastore/doanhbc/minhdphan/datasets/aapm/limited_angle/"
    
    # Fallback tự động nếu đường dẫn mặc định chưa có nhưng thư mục local hoặc legacy tồn tại
    if not os.path.exists(default_dicom):
        legacy_dicom = "/datastore/uittogether3/LuuTru/Thanhld/CT-Reconstruction/split/"
        if os.path.exists(legacy_dicom):
            default_dicom = legacy_dicom
            
    if not os.path.exists(default_cache):
        legacy_cache = "/datastore/uittogether3/LuuTru/MinhPD/dataset/aapm/limited_angle/"
        if os.path.exists(legacy_cache):
            default_cache = legacy_cache

    parser.add_argument("--dicom_dir", type=str, default=default_dicom, help="Thư mục chứa ảnh DICOM tập test")
    parser.add_argument("--cache_dir", type=str, default=default_cache, help="Thư mục chứa file cache .npy")
    parser.add_argument("--dataset_type", type=str, default="aapm", choices=["aapm", "deeplesion", "lidc"])
    
    # Cấu hình hình học Fan-Beam (cố định theo geometry-decision.md)
    parser.add_argument("--num_view", type=int, default=64, help="Số góc chiếu trong dải quét giới hạn")
    parser.add_argument("--num_detectors", type=int, default=512, help="Số cảm biến detector")
    parser.add_argument("--src_radius", type=float, default=600.0, help="Khoảng cách nguồn phát tới tâm quay (mm)")
    parser.add_argument("--det_radius", type=float, default=290.0, help="Khoảng cách tâm quay tới detector (mm)")
    parser.add_argument("--input_size", type=int, default=256, help="Kích thước không gian ảnh tái tạo (256x256)")
    
    # Mức độ nhiễu mô phỏng (0.0 = không nhiễu)
    parser.add_argument("--poisson_level", type=float, default=0.0, help="Mức photon mô phỏng nhiễu Poisson")
    parser.add_argument("--gaussian_level", type=float, default=0.0, help="Độ lệch chuẩn nhiễu Gaussian")
    
    # Cơ chế nạp cache
    parser.add_argument(
        "--use_precomputed",
        action="store_true",
        default=True,
        help="Đọc dữ liệu FBP từ cache .npy đã tính sẵn (mặc định: True)"
    )
    parser.add_argument(
        "--no_precomputed",
        dest="use_precomputed",
        action="store_false",
        help="Tắt đọc cache, tính toán FBP on-the-fly qua ODL/ASTRA"
    )
    
    # Thư mục và tệp kết quả
    parser.add_argument("--output_base_dir", type=str, default="saved_models", help="Thư mục gốc lưu trữ kết quả")
    parser.add_argument("--output_dir", type=str, default=None, help="Thư mục lưu trữ kết quả cụ thể (nếu có)")
    parser.add_argument("--output_csv", type=str, default=None, help="Đường dẫn file CSV xuất ra cụ thể (nếu có)")
    parser.add_argument(
        "--job_id",
        type=str,
        default=os.environ.get("SLURM_JOB_ID", "local"),
        help="Mã Job ID của Slurm để lưu trữ nguồn gốc dữ liệu (provenance)"
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Thiết bị tính toán")
    
    if args_list is not None:
        return parser.parse_args(args_list)
    return parser.parse_args()


def main():
    """Điểm khởi nhập chính của script đánh giá FBP."""
    args = parse_args()
    
    print("=" * 80)
    print("🔬 HỆ THỐNG ĐÁNH GIÁ BENCHMARK FBP TRÊN TẬP KIỂM THỬ AAPM LIMITED-ANGLE CT")
    print(f"- Các góc quét cần đánh giá: {args.angles}")
    print(f"- Bệnh nhân kiểm thử: {args.test_patients}")
    print(f"- Job ID: {args.job_id}")
    print("=" * 80)

    for angle in args.angles:
        evaluate_fbp_for_angle(angle_deg=angle, args=args)
        
    print("\n🎉 Hoàn thành toàn bộ quy trình đánh giá FBP cho các góc quét chỉ định!")


if __name__ == "__main__":
    main()
