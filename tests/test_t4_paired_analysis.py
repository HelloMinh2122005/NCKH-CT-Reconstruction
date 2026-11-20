#!/usr/bin/env python3
# ==============================================================================
# KIỂM THỬ ĐƠN VỊ TDD CHO T1.8 (CHANGE 39)
# T1.8: Công cụ Phân tích T4 (Paired Patient-Level Analysis):
#       - scripts/analyze_t4_paired_metrics.py
#       - Trung bình từng bệnh nhân qua 4 test patients: L058, L064, L094, L107
#       - Mean ± Std qua 4 bệnh nhân với sample std (ddof=1)
#       - Delta bắt cặp Delta_p = M_p(variant) - M_p(ref)
#       - Mean ± Std của Delta (ddof=1)
#       - Tỷ lệ thắng k/4 (PSNR/SSIM: Delta > 0, RMSE: Delta < 0)
#       - Bảo toàn job_id
#       - Xuất Markdown và CSV
#       - Kiểm chứng bằng dữ liệu giả lập (synthetic mock), không dùng pilot
# ==============================================================================

import csv
import io
import math
import os
import sys
import tempfile
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class TestT4PairedAnalysis(unittest.TestCase):
    """Kiểm thử đơn vị cho công cụ phân tích bắt cặp cấp bệnh nhân T4."""

    def setUp(self):
        """Tạo dữ liệu fixture giả lập (synthetic mock) cho 2 mô hình (Ref và Variant)."""
        self.test_patients = ["L058", "L064", "L094", "L107"]
        # Giả lập dữ liệu per-slice cho 2 lát cắt mỗi bệnh nhân
        self.mock_data = []

        # Mô hình tham chiếu: SOLAR_DualMamba
        # Giả sử điểm trung bình từng bệnh nhân:
        # L058: psnr=36.0, ssim=0.920, rmse=0.015
        # L064: psnr=35.0, ssim=0.910, rmse=0.016
        # L094: psnr=37.0, ssim=0.930, rmse=0.014
        # L107: psnr=34.0, ssim=0.900, rmse=0.017
        ref_patient_metrics = {
            "L058": (36.0, 0.920, 0.015),
            "L064": (35.0, 0.910, 0.016),
            "L094": (37.0, 0.930, 0.014),
            "L107": (34.0, 0.900, 0.017),
        }
        for pid, (p, s, r) in ref_patient_metrics.items():
            for sl in [1, 2]:
                self.mock_data.append({
                    "method": "SOLAR_DualMamba",
                    "angle": "90",
                    "patient_id": pid,
                    "slice_idx": str(sl),
                    "psnr": str(p),
                    "ssim": str(s),
                    "rmse": str(r),
                    "job_id": "11111",
                })

        # Mô hình so sánh: LEARN
        # Giả sử điểm từng bệnh nhân:
        # L058: psnr=33.0 (thua), ssim=0.880 (thua), rmse=0.020 (thua, vì > ref)
        # L064: psnr=32.0 (thua), ssim=0.870 (thua), rmse=0.022 (thua)
        # L094: psnr=34.0 (thua), ssim=0.890 (thua), rmse=0.018 (thua)
        # L107: psnr=35.0 (thắng!), ssim=0.905 (thắng!), rmse=0.016 (thắng!, vì < ref)
        var_patient_metrics = {
            "L058": (33.0, 0.880, 0.020),
            "L064": (32.0, 0.870, 0.022),
            "L094": (34.0, 0.890, 0.018),
            "L107": (35.0, 0.905, 0.016),
        }
        for pid, (p, s, r) in var_patient_metrics.items():
            for sl in [1, 2]:
                self.mock_data.append({
                    "method": "LEARN",
                    "angle": "90",
                    "patient_id": pid,
                    "slice_idx": str(sl),
                    "psnr": str(p),
                    "ssim": str(s),
                    "rmse": str(r),
                    "job_id": "22222",
                })

    def _write_mock_csv(self, file_path):
        """Ghi dữ liệu fixture vào tệp CSV."""
        fieldnames = ["method", "angle", "patient_id", "slice_idx", "psnr", "ssim", "rmse", "job_id"]
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.mock_data)

    def test_paired_metrics_math_correctness(self):
        """Kiểm tra độ chính xác thuật toán: mean/std ddof=1, delta bắt cặp và tỷ lệ thắng k/4."""
        from scripts.analyze_t4_paired_metrics import compute_t4_paired_analysis

        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w+", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            self._write_mock_csv(tmp_path)
            results = compute_t4_paired_analysis(
                csv_paths=[tmp_path],
                ref_method="SOLAR_DualMamba",
                patients=["L058", "L064", "L094", "L107"],
            )

            self.assertIn("SOLAR_DualMamba", results)
            self.assertIn("LEARN", results)

            # 1. Kiểm tra mô hình tham chiếu SOLAR_DualMamba:
            # Mean PSNR = (36 + 35 + 37 + 34) / 4 = 35.5
            # Std PSNR (ddof=1) = sqrt(((0.5)^2 + (-0.5)^2 + (1.5)^2 + (-1.5)^2)/3) = sqrt((0.25+0.25+2.25+2.25)/3) = sqrt(5/3) = 1.29099
            sol = results["SOLAR_DualMamba"]
            self.assertAlmostEqual(sol["psnr_mean"], 35.5, places=3)
            self.assertAlmostEqual(sol["psnr_std"], math.sqrt(5.0 / 3.0), places=3)
            self.assertEqual(sol["job_id"], "11111")

            # 2. Kiểm tra mô hình so sánh LEARN:
            # Patient PSNRs = [33.0, 32.0, 34.0, 35.0]
            # Mean = (33 + 32 + 34 + 35) / 4 = 33.5
            # Std (ddof=1) = sqrt(((-0.5)^2 + (-1.5)^2 + (0.5)^2 + (1.5)^2)/3) = sqrt(5/3) = 1.29099
            lrn = results["LEARN"]
            self.assertAlmostEqual(lrn["psnr_mean"], 33.5, places=3)
            self.assertAlmostEqual(lrn["psnr_std"], math.sqrt(5.0 / 3.0), places=3)
            self.assertEqual(lrn["job_id"], "22222")

            # 3. Delta bắt cặp từng bệnh nhân:
            # Deltas = [33 - 36, 32 - 35, 34 - 37, 35 - 34] = [-3.0, -3.0, -3.0, +1.0]
            # Mean Delta = (-3 - 3 - 3 + 1) / 4 = -8 / 4 = -2.0 dB
            # Variance Delta (ddof=1) = ((-1)^2 + (-1)^2 + (-1)^2 + (3)^2)/3 = (1 + 1 + 1 + 9)/3 = 12/3 = 4.0
            # Std Delta = 2.0
            self.assertAlmostEqual(lrn["delta_psnr_mean"], -2.0, places=3)
            self.assertAlmostEqual(lrn["delta_psnr_std"], 2.0, places=3)

            # 4. Tỷ lệ thắng k/4:
            # LEARN chỉ thắng bệnh nhân L107 (delta = +1 > 0): thắng 1/4 bệnh nhân
            self.assertEqual(lrn["psnr_win_rate"], "1/4")

            # RMSE:
            # Ref = [0.015, 0.016, 0.014, 0.017]
            # LEARN = [0.020, 0.022, 0.018, 0.016]
            # Delta RMSE = LEARN - Ref = [+0.005, +0.006, +0.004, -0.001]
            # LEARN thắng ở L107 (Delta < 0: -0.001 < 0): thắng 1/4
            self.assertEqual(lrn["rmse_win_rate"], "1/4")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_markdown_and_csv_table_generation(self):
        """Kiểm tra xuất bảng Markdown và bảng CSV có đầy đủ cột chỉ số và job_id."""
        from scripts.analyze_t4_paired_metrics import (
            compute_t4_paired_analysis,
            generate_markdown_table,
            generate_summary_csv,
        )

        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w+", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            self._write_mock_csv(tmp_path)
            results = compute_t4_paired_analysis(csv_paths=[tmp_path], ref_method="SOLAR_DualMamba")
            md = generate_markdown_table(results, ref_method="SOLAR_DualMamba")
            csv_str = generate_summary_csv(results, ref_method="SOLAR_DualMamba")

            # Markdown chứa tên bảng và các cột chính
            self.assertIn("Method", md)
            self.assertIn("PSNR (dB)", md)
            self.assertIn("Δ PSNR", md)
            self.assertIn("PSNR Win", md)
            self.assertIn("Job ID", md)
            self.assertIn("SOLAR_DualMamba", md)
            self.assertIn("LEARN", md)
            self.assertIn("1/4", md)
            self.assertIn("11111", md)
            self.assertIn("22222", md)

            # CSV chứa header tương ứng
            self.assertIn("method", csv_str)
            self.assertIn("psnr_mean", csv_str)
            self.assertIn("delta_psnr_mean", csv_str)
            self.assertIn("psnr_win", csv_str)
            self.assertIn("job_id", csv_str)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
