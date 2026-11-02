#!/usr/bin/env python3
"""
Unit test cho quy trình lựa chọn checkpoint tốt nhất (theo val_psnr) và ghi nhận chỉ số lát cắt.
Tuân thủ quyết định spec-1-decisions.md (Q7), metrics-decision.md và split.md.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Thêm thư mục gốc repo vào sys.path để import
REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from baselines.eval_utils import (
    find_best_val_checkpoint,
    parse_patient_and_slice_idx,
    format_csv_row,
    CSV_COLUMNS,
    validate_val_dataset_size,
)


class TestCheckpointSelectionAndMetrics(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.test_dir.cleanup()

    def test_find_best_val_checkpoint_success(self):
        """Kiểm tra chọn đúng checkpoint có val_psnr cao nhất trong thư mục."""
        ckpt_dir = self.test_dir.name
        # Tạo các checkpoint giả định
        p1 = os.path.join(ckpt_dir, "epoch=10-val_psnr=28.5000.ckpt")
        p2 = os.path.join(ckpt_dir, "epoch=46-val_psnr=36.2681.ckpt")
        p3 = os.path.join(ckpt_dir, "epoch=30-val_psnr=33.1200.ckpt")
        p_last = os.path.join(ckpt_dir, "last.ckpt")
        for p in [p1, p2, p3, p_last]:
            open(p, "w").close()

        best_ckpt = find_best_val_checkpoint(ckpt_dir)
        self.assertEqual(best_ckpt, p2)

    def test_find_best_val_checkpoint_no_fallback_on_last(self):
        """
        Quyết định Q7: Nếu chỉ có last.ckpt và không có checkpoint val_psnr nào,
        phải báo lỗi và dừng (tuyệt đối NO fallback về last.ckpt).
        """
        ckpt_dir = self.test_dir.name
        p_last = os.path.join(ckpt_dir, "last.ckpt")
        open(p_last, "w").close()

        with self.assertRaises(FileNotFoundError) as ctx:
            find_best_val_checkpoint(ckpt_dir)
        self.assertIn("NO fallback", str(ctx.exception))

    def test_find_best_val_checkpoint_empty_dir(self):
        """Báo lỗi khi thư mục rỗng không có checkpoint."""
        ckpt_dir = self.test_dir.name
        with self.assertRaises(FileNotFoundError):
            find_best_val_checkpoint(ckpt_dir)

    def test_parse_patient_and_slice_idx_symlink(self):
        """Trích xuất patient_id và slice_idx từ symlink đánh số chuẩn."""
        path = "/datastore/datasets/aapm/split/test/L058/full_3mm/L058_FD_3_1.CT.00000042.IMA"
        pid, s_idx = parse_patient_and_slice_idx(path, fallback_idx=1)
        self.assertEqual(pid, "L058")
        self.assertEqual(s_idx, 42)

    def test_parse_patient_and_slice_idx_dcm(self):
        """Trích xuất patient_id và slice_idx từ file raw .dcm."""
        path = "/datastore/datasets/aapm/raw/L064/full_3mm/00000105.dcm"
        pid, s_idx = parse_patient_and_slice_idx(path, fallback_idx=1)
        self.assertEqual(pid, "L064")
        self.assertEqual(s_idx, 105)

    def test_parse_patient_and_slice_idx_legacy_dotted(self):
        """Trích xuất patient_id và slice_idx từ tên file legacy nhiều dấu chấm."""
        path = "/datastore/datasets/aapm/split/test/L094/full_3mm/L094_FD_3_1.CT.1.2.3.4.5.6.7.8.18.IMA"
        pid, s_idx = parse_patient_and_slice_idx(path, fallback_idx=1)
        self.assertEqual(pid, "L094")
        self.assertEqual(s_idx, 18)

    def test_csv_columns_schema(self):
        """Kiểm tra cấu trúc cột CSV đúng yêu cầu metrics-decision.md."""
        expected_cols = ["method", "angle", "patient_id", "slice_idx", "psnr", "ssim", "rmse", "job_id"]
        self.assertEqual(CSV_COLUMNS, expected_cols)

    def test_format_csv_row(self):
        """Kiểm tra định dạng dòng CSV xuất ra."""
        row_str = format_csv_row(
            method="SOLAR_DualMamba",
            angle=90.0,
            patient_id="L058",
            slice_idx=42,
            psnr=32.1234,
            ssim=0.91234,
            rmse=0.024567,
            job_id="8450",
        )
        self.assertEqual(
            row_str.strip(),
            "SOLAR_DualMamba,90.0,L058,42,32.1234,0.9123,0.024567,8450"
        )

    def test_validate_val_dataset_size_zero_raises(self):
        """Kiểm tra tập validation L110: nếu số lát cắt = 0 phải raise RuntimeError (NO fallback)."""
        with self.assertRaises(RuntimeError) as ctx:
            validate_val_dataset_size(0)
        self.assertIn("NO fallback", str(ctx.exception))

    def test_validate_val_dataset_size_positive_passes(self):
        """Kiểm tra tập validation L110: nếu số lát cắt > 0 thì hợp lệ."""
        self.assertTrue(validate_val_dataset_size(133))


if __name__ == "__main__":
    unittest.main()
