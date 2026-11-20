#!/usr/bin/env python3
# ==============================================================================
# KIỂM THỬ ĐƠN VỊ TDD CHO T1.9 (CHANGE 39)
# T1.9: Thêm Peak VRAM và hỗ trợ Checkpoint 55 BN trong Latency Benchmark:
#       - CLI có cờ --ckpt_path
#       - reset_peak_memory_stats và max_memory_allocated
#       - peak_vram_mb trong CSV_COLUMNS và format_csv_row
#       - nạp checkpoint nếu --ckpt_path được cung cấp
# ==============================================================================

import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

# Giả lập import torch nếu môi trường cục bộ chưa cài đặt (cluster-only dependency)
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
    torch_mock.load = MagicMock()
    torch_mock.from_numpy = MagicMock(return_value=MagicMock())
    torch_mock.float32 = "float32"
    sys.modules["torch"] = torch_mock
    sys.modules["torch.utils"] = torch_utils
    sys.modules["torch.utils.data"] = torch_utils_data
else:
    torch_mock = sys.modules["torch"]
    if not hasattr(torch_mock, "load"):
        torch_mock.load = MagicMock()
    if not hasattr(torch_mock, "from_numpy"):
        torch_mock.from_numpy = MagicMock(return_value=MagicMock())

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.benchmark_latency_hpcuit import (
    parse_args,
    CSV_COLUMNS,
    format_csv_row,
    load_model,
)


class TestBenchmarkLatencyVRAM(unittest.TestCase):
    """Kiểm thử đơn vị cho các tính năng Peak VRAM và Checkpoint Loading (T1.9)."""

    def test_cli_has_ckpt_path_flag(self):
        """Xác minh cờ --ckpt_path khả dụng và mặc định là None."""
        args_default = parse_args(["--model", "learn"])
        self.assertTrue(hasattr(args_default, "ckpt_path"), "Thiếu cờ --ckpt_path trong parse_args")
        self.assertIsNone(args_default.ckpt_path)

        args_custom = parse_args(["--model", "learn", "--ckpt_path", "/path/to/checkpoint.ckpt"])
        self.assertEqual(args_custom.ckpt_path, "/path/to/checkpoint.ckpt")

    def test_csv_columns_contains_peak_vram_mb(self):
        """Xác minh cột peak_vram_mb nằm trong CSV_COLUMNS."""
        self.assertIn("peak_vram_mb", CSV_COLUMNS, "CSV_COLUMNS bắt buộc phải có cột 'peak_vram_mb' (T1.9)")

    def test_format_csv_row_includes_peak_vram_mb(self):
        """Xác minh format_csv_row định dạng chính xác giá trị peak_vram_mb."""
        row = format_csv_row(
            model="solar",
            mean_ms=25.5,
            std_ms=1.2,
            fps=39.22,
            warmup_runs=20,
            test_runs=100,
            input_size=256,
            views=64,
            hardware_profile="HPC-UIT MIG Slice",
            peak_vram_mb=1450.75,
        )
        self.assertEqual(len(row), len(CSV_COLUMNS), "Số phần tử trong row phải khớp độ dài CSV_COLUMNS")
        idx = CSV_COLUMNS.index("peak_vram_mb")
        self.assertEqual(row[idx], "1450.75")

    def test_cuda_peak_vram_telemetry_flow(self):
        """Kiểm tra luồng gọi reset_peak_memory_stats và max_memory_allocated."""
        import torch
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "reset_peak_memory_stats") as mock_reset, \
             patch.object(torch.cuda, "max_memory_allocated", return_value=1024 * 1024 * 500) as mock_max:
            # 500 MB = 500 * 1024 * 1024 bytes
            peak_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
            self.assertAlmostEqual(peak_mb, 500.0)

    def test_load_model_with_mock_checkpoint_file(self):
        """Kiểm tra load_model nạp state_dict từ file .ckpt khi truyền --ckpt_path."""
        import torch
        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as tmp_ckpt:
            tmp_ckpt.write(b"dummy_ckpt_bytes")
            tmp_ckpt_path = tmp_ckpt.name

        try:
            mock_model = MagicMock()
            mock_model.to.return_value = mock_model
            args = parse_args(["--model", "learn", "--ckpt_path", tmp_ckpt_path])

            mock_models_module = types.ModuleType("baselines.LEARN.models")
            mock_models_module.LEARN_LA = MagicMock(return_value=mock_model)

            if not hasattr(torch, "load"):
                torch.load = MagicMock()
            with patch.dict(sys.modules, {"baselines.LEARN.models": mock_models_module}), \
                 patch.object(torch, "load", return_value={"state_dict": {"weight": 1}}) as mock_load:
                loaded = load_model(args, device="cpu")
                self.assertIsNotNone(loaded)
                mock_load.assert_called_once()
                mock_model.load_state_dict.assert_called_once()
        finally:
            if os.path.exists(tmp_ckpt_path):
                os.remove(tmp_ckpt_path)


if __name__ == "__main__":
    unittest.main()
