#!/usr/bin/env python3
"""
Unit tests cho quy trình đánh giá FBP trên tập kiểm thử AAPM Limited-Angle CT (TDD Red-Green).
Kiểm thử các hợp đồng tính toán dải động, tổng hợp thống kê bệnh nhân, và xuất file CSV.
"""

import unittest
import sys
import os
import math
import numpy as np

# Cho phép import từ thư mục gốc của repository
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.evaluate_fbp import (
    get_slice_data_range_numpy,
    compute_patient_summary,
    format_csv_row,
    extract_patient_id_from_path,
    parse_args,
    CSV_COLUMNS,
)


class TestEvaluateFBP(unittest.TestCase):
    def test_csv_columns_specification(self):
        """Kiểm tra danh sách và thứ tự 8 cột chuẩn của file CSV theo spec."""
        expected_columns = [
            "method",
            "angle",
            "patient_id",
            "slice_idx",
            "psnr",
            "ssim",
            "rmse",
            "job_id",
        ]
        self.assertEqual(CSV_COLUMNS, expected_columns)

    def test_get_slice_data_range_rejects_one(self):
        """
        Kiểm tra dải động data_range:
        - Bắt buộc tính từ (min, max) của chính lát cắt ảnh gốc Ground Truth.
        - Tuyệt đối không dùng giá trị cố định 1.0.
        """
        # Giả lập lát cắt có dải giá trị từ 0.05 đến 0.85
        gt_slice = np.array([[0.05, 0.20], [0.50, 0.85]], dtype=np.float32)
        min_val, max_val = get_slice_data_range_numpy(gt_slice)
        
        self.assertAlmostEqual(min_val, 0.05, places=5)
        self.assertAlmostEqual(max_val, 0.85, places=5)
        self.assertNotEqual(max_val - min_val, 1.0)

        # Trường hợp lát cắt đồng nhất (min xấp xỉ max): cộng thêm epsilon 1e-8
        flat_slice = np.full((10, 10), 0.3, dtype=np.float32)
        min_flat, max_flat = get_slice_data_range_numpy(flat_slice)
        self.assertAlmostEqual(min_flat, 0.3, places=5)
        self.assertGreater(max_flat, min_flat)
        self.assertAlmostEqual(max_flat - min_flat, 1e-8, places=8)

    def test_extract_patient_id_from_path(self):
        """Kiểm tra trích xuất chính xác mã bệnh nhân từ các biến thể đường dẫn file."""
        known_patients = ["L058", "L064", "L094", "L107"]
        
        # Đường dẫn tuyệt đối chuẩn Mayo Clinic
        path1 = "/datastore/doanhbc/minhdphan/datasets/aapm/split/test/L058/full_3mm/L058_FD_3_1.CT.0001.0001.IMA"
        self.assertEqual(extract_patient_id_from_path(path1, known_patients), "L058")

        # Đường dẫn tương đối dạng symlink
        path2 = "data/split/test/L107/full_3mm/0042.IMA"
        self.assertEqual(extract_patient_id_from_path(path2, known_patients), "L107")

        # Đường dẫn chứa bệnh nhân L064
        path3 = "/tmp/test/L064/slice_10.dcm"
        self.assertEqual(extract_patient_id_from_path(path3, known_patients), "L064")

        # Đường dẫn không nằm trong danh sách định trước nhưng có cấu trúc /PID/
        path_unknown = "/data/split/test/L999/full_3mm/0001.IMA"
        self.assertEqual(extract_patient_id_from_path(path_unknown, known_patients), "L999")

    def test_format_csv_row(self):
        """Kiểm tra tạo dòng dữ liệu CSV đầy đủ và chuẩn xác các trường."""
        row = format_csv_row(
            method="FBP",
            angle=120,
            patient_id="L058",
            slice_idx=15,
            psnr=28.4521,
            ssim=0.8123,
            rmse=0.045123,
            job_id="8433",
        )
        self.assertEqual(row["method"], "FBP")
        self.assertEqual(row["angle"], 120)
        self.assertEqual(row["patient_id"], "L058")
        self.assertEqual(row["slice_idx"], 15)
        self.assertAlmostEqual(row["psnr"], 28.4521, places=4)
        self.assertAlmostEqual(row["ssim"], 0.8123, places=4)
        self.assertAlmostEqual(row["rmse"], 0.045123, places=6)
        self.assertEqual(row["job_id"], "8433")

    def test_compute_patient_summary(self):
        """
        Kiểm tra tính toán thống kê:
        1. Trung bình theo từng bệnh nhân (per-patient mean).
        2. Trung bình và độ lệch chuẩn của cả nhóm (overall mean ± sample std qua 4 bệnh nhân).
        """
        records = [
            # Bệnh nhân L058: 2 lát cắt (PSNR: 30.0, 32.0 -> mean = 31.0)
            {"patient_id": "L058", "slice_idx": 0, "psnr": 30.0, "ssim": 0.80, "rmse": 0.050, "angle": 120, "method": "FBP", "job_id": "test"},
            {"patient_id": "L058", "slice_idx": 1, "psnr": 32.0, "ssim": 0.84, "rmse": 0.040, "angle": 120, "method": "FBP", "job_id": "test"},
            # Bệnh nhân L064: 2 lát cắt (PSNR: 27.0, 29.0 -> mean = 28.0)
            {"patient_id": "L064", "slice_idx": 0, "psnr": 27.0, "ssim": 0.70, "rmse": 0.060, "angle": 120, "method": "FBP", "job_id": "test"},
            {"patient_id": "L064", "slice_idx": 1, "psnr": 29.0, "ssim": 0.74, "rmse": 0.050, "angle": 120, "method": "FBP", "job_id": "test"},
            # Bệnh nhân L094: 1 lát cắt (PSNR: 34.0 -> mean = 34.0)
            {"patient_id": "L094", "slice_idx": 0, "psnr": 34.0, "ssim": 0.90, "rmse": 0.030, "angle": 120, "method": "FBP", "job_id": "test"},
            # Bệnh nhân L107: 1 lát cắt (PSNR: 27.0 -> mean = 27.0)
            {"patient_id": "L107", "slice_idx": 0, "psnr": 27.0, "ssim": 0.68, "rmse": 0.065, "angle": 120, "method": "FBP", "job_id": "test"},
        ]

        summary = compute_patient_summary(records)

        # 1. Kiểm tra per-patient metrics
        patient_stats = summary["patient_stats"]
        self.assertEqual(len(patient_stats), 4)

        self.assertAlmostEqual(patient_stats["L058"]["mean_psnr"], 31.0, places=4)
        self.assertAlmostEqual(patient_stats["L058"]["mean_ssim"], 0.82, places=4)
        self.assertAlmostEqual(patient_stats["L058"]["mean_rmse"], 0.045, places=4)
        self.assertEqual(patient_stats["L058"]["slice_count"], 2)

        self.assertAlmostEqual(patient_stats["L064"]["mean_psnr"], 28.0, places=4)
        self.assertAlmostEqual(patient_stats["L094"]["mean_psnr"], 34.0, places=4)
        self.assertAlmostEqual(patient_stats["L107"]["mean_psnr"], 27.0, places=4)

        # 2. Kiểm tra overall cohort metrics (trung bình và độ lệch chuẩn mẫu ddof=1 qua 4 bệnh nhân)
        # Patient PSNRs: [31.0, 28.0, 34.0, 27.0] -> Mean = 30.0, Sample Std = sqrt(10) ≈ 3.162277
        cohort = summary["cohort_stats"]
        self.assertAlmostEqual(cohort["psnr_mean"], 30.0, places=4)
        expected_std = math.sqrt(((31 - 30)**2 + (28 - 30)**2 + (34 - 30)**2 + (27 - 30)**2) / 3.0)
        self.assertAlmostEqual(cohort["psnr_std"], expected_std, places=4)

    def test_cli_defaults(self):
        """Kiểm tra các giá trị mặc định của parser CLI tuân thủ nghiêm ngặt spec."""
        args = parse_args([])
        self.assertEqual(args.test_patients, ["L058", "L064", "L094", "L107"])
        self.assertEqual(args.num_view, 64)
        self.assertEqual(args.num_detectors, 512)
        self.assertEqual(args.input_size, 256)
        self.assertEqual(args.src_radius, 600.0)
        self.assertEqual(args.det_radius, 290.0)
        self.assertEqual(args.angles, [120.0, 90.0])


if __name__ == "__main__":
    unittest.main()
