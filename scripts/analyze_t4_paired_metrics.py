#!/usr/bin/env python3
# ==============================================================================
# scripts/analyze_t4_paired_metrics.py
#
# Công cụ Phân tích T4 (Paired Patient-Level Analysis):
# - Đọc kết quả per-slice từ các tệp CSV (mặc định trong reports/hpc-uit-55p/).
# - Tính trung bình chỉ số từng bệnh nhân p in {L058, L064, L094, L107}.
# - Tính trung bình đoàn hệ (cohort mean) và độ lệch chuẩn mẫu (sample std với ddof=1).
# - Tính delta bắt cặp theo bệnh nhân: Delta_p = M_p(variant) - M_p(ref).
# - Tính mean và std của delta (ddof=1) qua 4 bệnh nhân.
# - Xác định tỷ lệ thắng k/4 (PSNR/SSIM: Delta > 0; RMSE: Delta < 0).
# - Bảo toàn job_id tương ứng của từng mô hình trong bảng tổng hợp.
# - Xuất bảng tổng kết ở định dạng Markdown và CSV.
# ==============================================================================

import argparse
import csv
import glob
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union


def calc_mean_std(values: List[float], ddof: int = 1) -> Tuple[float, float]:
    """
    Tính trung bình và độ lệch chuẩn mẫu (ddof=1).
    Nếu số lượng phần tử <= ddof thì std trả về 0.0.
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean_val = sum(values) / n
    if n <= ddof:
        return mean_val, 0.0
    variance = sum((x - mean_val) ** 2 for x in values) / (n - ddof)
    return mean_val, math.sqrt(variance)


def compute_t4_paired_analysis(
    csv_paths: Union[str, List[str]],
    ref_method: str = "SOLAR_DualMamba",
    patients: Optional[List[str]] = None,
    angle: Optional[str] = None,
) -> Dict[str, Dict]:
    """
    Đọc dữ liệu từ danh sách tệp CSV per-slice và thực hiện phân tích bắt cặp cấp bệnh nhân T4.

    Args:
        csv_paths: Đường dẫn hoặc danh sách đường dẫn tệp CSV hoặc thư mục.
        ref_method: Tên mô hình chuẩn làm mốc so sánh (mặc định SOLAR_DualMamba).
        patients: Danh sách mã bệnh nhân đánh giá (mặc định L058, L064, L094, L107).
        angle: Góc giới hạn (nếu muốn lọc cụ thể 90 hoặc 120).

    Returns:
        Dictionary chứa kết quả phân tích theo từng mô hình.
    """
    if patients is None:
        patients = ["L058", "L064", "L094", "L107"]

    # Chuẩn hóa danh sách tệp CSV đầu vào
    resolved_files = []
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]

    for p in csv_paths:
        if os.path.isdir(p):
            resolved_files.extend(glob.glob(os.path.join(p, "*.csv")))
        elif os.path.isfile(p):
            resolved_files.append(p)
        else:
            # Hỗ trợ glob pattern
            matched = glob.glob(p)
            if matched:
                resolved_files.extend(matched)
            else:
                raise FileNotFoundError(f"Không tìm thấy tệp hoặc thư mục CSV: {p}")

    resolved_files = sorted(list(set(resolved_files)))
    if not resolved_files:
        raise ValueError("Không tìm thấy tệp CSV hợp lệ nào để phân tích.")

    # Cấu trúc lưu trữ: raw_data[method][patient_id] = {"psnr": [], "ssim": [], "rmse": [], "job_id": ...}
    raw_data = defaultdict(lambda: defaultdict(lambda: {"psnr": [], "ssim": [], "rmse": []}))
    method_job_ids = {}

    for file_path in resolved_files:
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Kiểm tra lọc góc quét nếu có
                if angle is not None:
                    row_angle = str(row.get("angle", "")).strip()
                    if row_angle and row_angle != str(angle):
                        continue

                m = row.get("method", "").strip()
                pid = row.get("patient_id", "").strip()

                if not m or not pid:
                    continue
                if patients and pid not in patients:
                    continue

                try:
                    p_val = float(row["psnr"])
                    s_val = float(row["ssim"])
                    r_val = float(row["rmse"])
                except (ValueError, KeyError):
                    continue

                raw_data[m][pid]["psnr"].append(p_val)
                raw_data[m][pid]["ssim"].append(s_val)
                raw_data[m][pid]["rmse"].append(r_val)

                # Lưu job_id đầu tiên hoặc cập nhật nếu có
                j_id = str(row.get("job_id", "")).strip()
                if j_id and m not in method_job_ids:
                    method_job_ids[m] = j_id

    if not raw_data:
        raise ValueError("Không có dữ liệu hợp lệ phù hợp với danh sách bệnh nhân và góc quét.")

    # 1. Tính trung bình cấp bệnh nhân M_p(method)
    patient_level_metrics = {}
    for m, p_dict in raw_data.items():
        patient_level_metrics[m] = {}
        for pid in patients:
            if pid in p_dict and len(p_dict[pid]["psnr"]) > 0:
                patient_level_metrics[m][pid] = {
                    "psnr": sum(p_dict[pid]["psnr"]) / len(p_dict[pid]["psnr"]),
                    "ssim": sum(p_dict[pid]["ssim"]) / len(p_dict[pid]["ssim"]),
                    "rmse": sum(p_dict[pid]["rmse"]) / len(p_dict[pid]["rmse"]),
                }

    # 2. Tính trung bình và độ lệch chuẩn mẫu qua đoàn hệ bệnh nhân (ddof=1)
    results = {}
    num_patients = len(patients)

    for m, p_metrics in patient_level_metrics.items():
        psnr_vals = [p_metrics[pid]["psnr"] for pid in patients if pid in p_metrics]
        ssim_vals = [p_metrics[pid]["ssim"] for pid in patients if pid in p_metrics]
        rmse_vals = [p_metrics[pid]["rmse"] for pid in patients if pid in p_metrics]

        p_mean, p_std = calc_mean_std(psnr_vals, ddof=1)
        s_mean, s_std = calc_mean_std(ssim_vals, ddof=1)
        r_mean, r_std = calc_mean_std(rmse_vals, ddof=1)

        results[m] = {
            "method": m,
            "job_id": method_job_ids.get(m, "N/A"),
            "num_patients": len(psnr_vals),
            "psnr_mean": p_mean,
            "psnr_std": p_std,
            "ssim_mean": s_mean,
            "ssim_std": s_std,
            "rmse_mean": r_mean,
            "rmse_std": r_std,
            "patient_metrics": p_metrics,
        }

    # 3. Tính toán chênh lệch bắt cặp Delta_p = M_p(variant) - M_p(ref) và tỷ lệ thắng k/N
    has_ref = ref_method in patient_level_metrics
    ref_p_metrics = patient_level_metrics.get(ref_method, {})

    for m, res in results.items():
        if m == ref_method:
            res["delta_psnr_mean"] = 0.0
            res["delta_psnr_std"] = 0.0
            res["psnr_win_rate"] = "-"
            res["delta_ssim_mean"] = 0.0
            res["delta_ssim_std"] = 0.0
            res["ssim_win_rate"] = "-"
            res["delta_rmse_mean"] = 0.0
            res["delta_rmse_std"] = 0.0
            res["rmse_win_rate"] = "-"
            continue

        if not has_ref:
            res["delta_psnr_mean"] = None
            res["delta_psnr_std"] = None
            res["psnr_win_rate"] = "N/A"
            res["delta_ssim_mean"] = None
            res["delta_ssim_std"] = None
            res["ssim_win_rate"] = "N/A"
            res["delta_rmse_mean"] = None
            res["delta_rmse_std"] = None
            res["rmse_win_rate"] = "N/A"
            continue

        # Tính Delta theo từng bệnh nhân
        p_metrics = patient_level_metrics[m]
        delta_psnr_list = []
        delta_ssim_list = []
        delta_rmse_list = []

        wins_psnr = 0
        wins_ssim = 0
        wins_rmse = 0
        common_patients = 0

        for pid in patients:
            if pid in p_metrics and pid in ref_p_metrics:
                common_patients += 1
                d_p = p_metrics[pid]["psnr"] - ref_p_metrics[pid]["psnr"]
                d_s = p_metrics[pid]["ssim"] - ref_p_metrics[pid]["ssim"]
                d_r = p_metrics[pid]["rmse"] - ref_p_metrics[pid]["rmse"]

                delta_psnr_list.append(d_p)
                delta_ssim_list.append(d_s)
                delta_rmse_list.append(d_r)

                if d_p > 0:
                    wins_psnr += 1
                if d_s > 0:
                    wins_ssim += 1
                if d_r < 0:  # Với RMSE, giá trị nhỏ hơn là tốt hơn (thắng)
                    wins_rmse += 1

        d_p_mean, d_p_std = calc_mean_std(delta_psnr_list, ddof=1)
        d_s_mean, d_s_std = calc_mean_std(delta_ssim_list, ddof=1)
        d_r_mean, d_r_std = calc_mean_std(delta_rmse_list, ddof=1)

        denom = common_patients if common_patients > 0 else num_patients
        res["delta_psnr_mean"] = d_p_mean
        res["delta_psnr_std"] = d_p_std
        res["psnr_win_rate"] = f"{wins_psnr}/{denom}"

        res["delta_ssim_mean"] = d_s_mean
        res["delta_ssim_std"] = d_s_std
        res["ssim_win_rate"] = f"{wins_ssim}/{denom}"

        res["delta_rmse_mean"] = d_r_mean
        res["delta_rmse_std"] = d_r_std
        res["rmse_win_rate"] = f"{wins_rmse}/{denom}"

    return results


def generate_markdown_table(results: Dict[str, Dict], ref_method: str = "SOLAR_DualMamba") -> str:
    """
    Sinh bảng Markdown hiển thị kết quả phân tích bắt cặp cấp bệnh nhân T4.
    """
    lines = []
    lines.append("# BẢNG TỔNG HỢP ĐÁNH GIÁ BẮT CẶP CẤP BỆNH NHÂN (T4 PAIRWISE ANALYSIS)")
    lines.append(f"- **Mô hình tham chiếu (Baseline/Ref):** `{ref_method}`")
    lines.append("- **Đoàn hệ bệnh nhân kiểm thử:** L058, L064, L094, L107 (N=4)")
    lines.append("- **Quy tắc thống kê:** Sample standard deviation (ddof=1). Thắng: PSNR/SSIM Δ > 0; RMSE Δ < 0.")
    lines.append("")
    lines.append("| Method | PSNR (dB) | Δ PSNR | PSNR Win | SSIM | Δ SSIM | SSIM Win | RMSE | Δ RMSE | RMSE Win | Job ID |")
    lines.append("|:-------|:----------|:-------|:---------|:-----|:-------|:---------|:-----|:-------|:---------|:-------|")

    # Đưa ref_method lên đầu danh sách nếu có
    ordered_methods = []
    if ref_method in results:
        ordered_methods.append(ref_method)
    for m in results:
        if m != ref_method:
            ordered_methods.append(m)

    for m in ordered_methods:
        res = results[m]
        p_str = f"{res['psnr_mean']:.2f} ± {res['psnr_std']:.2f}"
        s_str = f"{res['ssim_mean']:.4f} ± {res['ssim_std']:.4f}"
        r_str = f"{res['rmse_mean']:.4f} ± {res['rmse_std']:.4f}"

        if m == ref_method:
            d_p_str = "-"
            p_win = "-"
            d_s_str = "-"
            s_win = "-"
            d_r_str = "-"
            r_win = "-"
        else:
            d_p_mean = res.get("delta_psnr_mean")
            d_p_std = res.get("delta_psnr_std", 0.0)
            d_p_str = f"{d_p_mean:+.2f} ± {d_p_std:.2f}" if d_p_mean is not None else "N/A"
            p_win = res.get("psnr_win_rate", "N/A")

            d_s_mean = res.get("delta_ssim_mean")
            d_s_std = res.get("delta_ssim_std", 0.0)
            d_s_str = f"{d_s_mean:+.4f} ± {d_s_std:.4f}" if d_s_mean is not None else "N/A"
            s_win = res.get("ssim_win_rate", "N/A")

            d_r_mean = res.get("delta_rmse_mean")
            d_r_std = res.get("delta_rmse_std", 0.0)
            d_r_str = f"{d_r_mean:+.4f} ± {d_r_std:.4f}" if d_r_mean is not None else "N/A"
            r_win = res.get("rmse_win_rate", "N/A")

        j_id = res.get("job_id", "N/A")
        lines.append(f"| {m} | {p_str} | {d_p_str} | {p_win} | {s_str} | {d_s_str} | {s_win} | {r_str} | {d_r_str} | {r_win} | {j_id} |")

    return "\n".join(lines)


def generate_summary_csv(results: Dict[str, Dict], ref_method: str = "SOLAR_DualMamba") -> str:
    """
    Sinh nội dung CSV tổng hợp kết quả phân tích bắt cặp cấp bệnh nhân T4.
    """
    fieldnames = [
        "method",
        "job_id",
        "psnr_mean",
        "psnr_std",
        "delta_psnr_mean",
        "delta_psnr_std",
        "psnr_win",
        "ssim_mean",
        "ssim_std",
        "delta_ssim_mean",
        "delta_ssim_std",
        "ssim_win",
        "rmse_mean",
        "rmse_std",
        "delta_rmse_mean",
        "delta_rmse_std",
        "rmse_win",
    ]

    import io
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    ordered_methods = []
    if ref_method in results:
        ordered_methods.append(ref_method)
    for m in results:
        if m != ref_method:
            ordered_methods.append(m)

    for m in ordered_methods:
        res = results[m]
        writer.writerow({
            "method": m,
            "job_id": res.get("job_id", "N/A"),
            "psnr_mean": f"{res['psnr_mean']:.4f}",
            "psnr_std": f"{res['psnr_std']:.4f}",
            "delta_psnr_mean": f"{res['delta_psnr_mean']:.4f}" if res.get("delta_psnr_mean") is not None else "",
            "delta_psnr_std": f"{res['delta_psnr_std']:.4f}" if res.get("delta_psnr_std") is not None else "",
            "psnr_win": res.get("psnr_win_rate", ""),
            "ssim_mean": f"{res['ssim_mean']:.4f}",
            "ssim_std": f"{res['ssim_std']:.4f}",
            "delta_ssim_mean": f"{res['delta_ssim_mean']:.4f}" if res.get("delta_ssim_mean") is not None else "",
            "delta_ssim_std": f"{res['delta_ssim_std']:.4f}" if res.get("delta_ssim_std") is not None else "",
            "ssim_win": res.get("ssim_win_rate", ""),
            "rmse_mean": f"{res['rmse_mean']:.4f}",
            "rmse_std": f"{res['rmse_std']:.4f}",
            "delta_rmse_mean": f"{res['delta_rmse_mean']:.4f}" if res.get("delta_rmse_mean") is not None else "",
            "delta_rmse_std": f"{res['delta_rmse_std']:.4f}" if res.get("delta_rmse_std") is not None else "",
            "rmse_win": res.get("rmse_win_rate", ""),
        })

    return output.getvalue()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Công cụ phân tích bắt cặp cấp bệnh nhân T4 (Paired Patient-Level Analysis)."
    )
    parser.add_argument(
        "--input_csv",
        nargs="+",
        default=[],
        help="Đường dẫn một hoặc nhiều tệp CSV kết quả per-slice.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="reports/hpc-uit-55p",
        help="Thư mục chứa các tệp CSV per-slice (mặc định: reports/hpc-uit-55p).",
    )
    parser.add_argument(
        "--ref_method",
        type=str,
        default="SOLAR_DualMamba",
        help="Tên mô hình tham chiếu chuẩn (mặc định: SOLAR_DualMamba).",
    )
    parser.add_argument(
        "--patients",
        nargs="+",
        default=["L058", "L064", "L094", "L107"],
        help="Danh sách mã bệnh nhân kiểm thử (mặc định: L058 L064 L094 L107).",
    )
    parser.add_argument(
        "--angle",
        type=str,
        default=None,
        help="Lọc theo góc quét (ví dụ: 90 hoặc 120).",
    )
    parser.add_argument(
        "--output_markdown",
        type=str,
        default=None,
        help="Đường dẫn lưu bảng Markdown tổng hợp kết quả.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Đường dẫn lưu tệp CSV tổng hợp kết quả.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    csv_paths = list(args.input_csv)
    if not csv_paths:
        if os.path.isdir(args.input_dir):
            csv_paths = [args.input_dir]
        else:
            print(f"[ERROR] Không tìm thấy thư mục đầu vào: {args.input_dir}")
            sys.exit(1)

    print(f"[INFO] Bắt đầu phân tích bắt cặp T4 với mốc tham chiếu: {args.ref_method}")
    print(f"[INFO] Danh sách bệnh nhân: {args.patients}")
    print(f"[INFO] Nguồn tệp CSV: {csv_paths}")

    results = compute_t4_paired_analysis(
        csv_paths=csv_paths,
        ref_method=args.ref_method,
        patients=args.patients,
        angle=args.angle,
    )

    md_table = generate_markdown_table(results, ref_method=args.ref_method)
    print("\n" + md_table + "\n")

    if args.output_markdown:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_markdown)), exist_ok=True)
        with open(args.output_markdown, "w", encoding="utf-8") as f:
            f.write(md_table)
        print(f"[INFO] Đã ghi bảng Markdown vào: {args.output_markdown}")

    if args.output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        csv_content = generate_summary_csv(results, ref_method=args.ref_method)
        with open(args.output_csv, "w", encoding="utf-8") as f:
            f.write(csv_content)
        print(f"[INFO] Đã ghi bảng CSV vào: {args.output_csv}")


if __name__ == "__main__":
    main()
