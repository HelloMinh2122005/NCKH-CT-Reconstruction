#!/usr/bin/env python3
"""
Unit tests cho kịch bản đo độ trễ suy luận scripts/benchmark_latency_hpcuit.py (TDD Red-Green).
Kiểm thử các hợp đồng CLI mặc định (warmup >= 20, test_runs >= 100), phân tích tham số dòng lệnh,
schema cột CSV và định dạng dữ liệu theo giao thức GT-03 / Wave E.
"""

import os
import sys
import types
import unittest
from unittest.mock import MagicMock

# Giả lập import torch nếu môi trường cục bộ chưa cài đặt (cluster-only dependency)
# Cấu trúc module hợp lệ tương thích với CTSlice_Provider_LA và các bộ test khác
if "torch" not in sys.modules:
    torch_mock = types.ModuleType("torch")
    torch_utils = types.ModuleType("torch.utils")
    torch_utils_data = types.ModuleType("torch.utils.data")
    torch_utils_data.Dataset = object
    torch_mock.utils = torch_utils
    torch_utils.data = torch_utils_data
    torch_mock.Tensor = MagicMock
    torch_mock.randn = MagicMock
    torch_mock.cuda = MagicMock()
    torch_mock.float32 = "float32"
    sys.modules["torch"] = torch_mock
    sys.modules["torch.utils"] = torch_utils
    sys.modules["torch.utils.data"] = torch_utils_data

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.benchmark_latency_hpcuit import (
    parse_args,
    CSV_COLUMNS,
    format_csv_row,
)


class TestBenchmarkLatencyHPCUIT(unittest.TestCase):
    """Bộ kiểm thử đơn vị cho scripts/benchmark_latency_hpcuit.py."""

    def test_default_cli_arguments_protocol_compliance(self):
        """
        Kiểm tra giá trị mặc định của CLI tuân thủ nghiêm ngặt giao thức GT-03 / Wave E:
        - warmup_runs: >= 20 (mặc định 20, trước đây là 10)
        - test_runs: >= 100 (mặc định 100, trước đây là 50)
        - input_size: 256
        - num_view: 64
        - num_detectors: 512
        - hardware_profile: 'HPC-UIT MIG Slice'
        """
        args = parse_args(["--model", "learn"])
        self.assertEqual(args.model, "learn")
        self.assertEqual(args.warmup_runs, 20, "Mặc định warmup_runs phải là 20 để thỏa mãn >=20 warm-up iterations")
        self.assertEqual(args.test_runs, 100, "Mặc định test_runs phải là 100 để thỏa mãn >=100 timed iterations")
        self.assertEqual(args.input_size, 256, "Mặc định input_size phải là 256")
        self.assertEqual(args.num_view, 64, "Mặc định num_view phải là 64")
        self.assertEqual(args.num_detectors, 512, "Mặc định num_detectors phải là 512")
        self.assertEqual(args.angle_range_deg, 120.0, "Mặc định angle_range_deg phải là 120.0")
        self.assertTrue(hasattr(args, "hardware_profile"), "Cần có cờ tham số --hardware_profile")
        self.assertEqual(args.hardware_profile, "HPC-UIT MIG Slice")

    def test_custom_cli_arguments(self):
        """Kiểm tra truyền tham số tùy chỉnh qua CLI."""
        custom_args = [
            "--model", "solar",
            "--warmup_runs", "30",
            "--test_runs", "200",
            "--input_size", "512",
            "--num_view", "128",
            "--num_detectors", "1024",
            "--angle_range_deg", "90.0",
            "--hardware_profile", "Custom-GPU-Slice",
            "--output_csv", "reports/latency.csv",
        ]
        args = parse_args(custom_args)
        self.assertEqual(args.model, "solar")
        self.assertEqual(args.warmup_runs, 30)
        self.assertEqual(args.test_runs, 200)
        self.assertEqual(args.input_size, 512)
        self.assertEqual(args.num_view, 128)
        self.assertEqual(args.num_detectors, 1024)
        self.assertEqual(args.angle_range_deg, 90.0)
        self.assertEqual(args.hardware_profile, "Custom-GPU-Slice")
        self.assertEqual(args.output_csv, "reports/latency.csv")

    def test_model_choices_supported(self):
        """Kiểm tra các mô hình hợp lệ và từ chối mô hình ngoài danh mục."""
        valid_models = ["learn", "regformer", "dudotrans", "longformer", "longnet", "mamba", "solar"]
        for m in valid_models:
            args = parse_args(["--model", m])
            self.assertEqual(args.model, m)

        # Mô hình không hợp lệ phải phát sinh lỗi parse (SystemExit)
        with self.assertRaises(SystemExit):
            parse_args(["--model", "unknown_model"])

    def test_csv_columns_schema(self):
        """Kiểm tra schema 9 cột xuất ra CSV theo đúng đặc tả yêu cầu."""
        expected_columns = [
            "model",
            "mean_ms",
            "std_ms",
            "fps",
            "warmup_runs",
            "test_runs",
            "input_size",
            "views",
            "hardware_profile",
        ]
        self.assertEqual(CSV_COLUMNS, expected_columns, "CSV schema phải gồm 9 cột chính xác theo yêu cầu")

    def test_format_csv_row(self):
        """Kiểm tra logic định dạng các trường dữ liệu của dòng CSV."""
        row = format_csv_row(
            model="learn",
            mean_ms=12.3456,
            std_ms=0.7891,
            fps=81.002,
            warmup_runs=20,
            test_runs=100,
            input_size=256,
            views=64,
            hardware_profile="HPC-UIT MIG Slice",
        )
        expected_row = [
            "learn",
            "12.346",
            "0.789",
            "81.00",
            20,
            100,
            256,
            64,
            "HPC-UIT MIG Slice",
        ]
        self.assertEqual(row, expected_row)


if __name__ == "__main__":
    unittest.main()
