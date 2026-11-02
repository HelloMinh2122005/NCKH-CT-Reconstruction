#!/usr/bin/env python3
# ==============================================================================
# KIỂM THỬ ĐƠN VỊ CHO SMOKE TEST HPC-UIT (T4)
# Kiểm tra tham số dòng lệnh cho LEARN & SOLAR_DualMamba, cú pháp sbatch và slurm_lint.
# ==============================================================================

import os
import re
import subprocess
import sys
import unittest
from unittest.mock import MagicMock

# Đảm bảo import giả lập khi môi trường cục bộ chưa có torch / pytorch_lightning
for mod in [
    "torch",
    "pytorch_lightning",
    "pytorch_lightning.callbacks",
    "pytorch_lightning.loggers",
    "data",
    "data.datamodule_factory",
    "baselines.LEARN.models",
    "baselines.SOLAR_DualMamba.models",
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()
if not hasattr(sys.modules["torch"], "load"):
    sys.modules["torch"].load = MagicMock()

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SCRIPT_PATH = os.path.join(REPO_ROOT, "scripts", "smoke_hpcuit.sh")


class TestSmokeArgsAndScript(unittest.TestCase):
    """Bộ kiểm thử đơn vị cho smoke test trên HPC-UIT."""

    def test_learn_args_smoke_batches_and_precomputed(self):
        """Xác minh parse_args của LEARN hỗ trợ limit_train_batches, limit_val_batches và use_precomputed=False."""
        from baselines.LEARN.train_learn_la import parse_args as parse_args_learn

        # 1. Mặc định: limit_*_batches là None, use_precomputed là True
        sys.argv = ["train_learn_la.py"]
        args_default = parse_args_learn()
        self.assertIsNone(args_default.limit_train_batches)
        self.assertIsNone(args_default.limit_val_batches)
        self.assertTrue(args_default.use_precomputed)

        # 2. Cấu hình smoke test: limit_train_batches=3, limit_val_batches=2, use_precomputed=False
        sys.argv = [
            "train_learn_la.py",
            "--limit_train_batches", "3",
            "--limit_val_batches", "2",
            "--use_precomputed", "False",
        ]
        args_smoke = parse_args_learn()
        self.assertEqual(args_smoke.limit_train_batches, 3)
        self.assertEqual(args_smoke.limit_val_batches, 2)
        self.assertFalse(args_smoke.use_precomputed)

        # 3. Cờ bật --use_precomputed không kèm tham số vẫn trả về True (tương thích ngược)
        sys.argv = ["train_learn_la.py", "--use_precomputed"]
        args_flag = parse_args_learn()
        self.assertTrue(args_flag.use_precomputed)

    def test_solar_dualmamba_args_smoke_batches_and_precomputed(self):
        """Xác minh parse_args của SOLAR_DualMamba hỗ trợ limit_train_batches, limit_val_batches và use_precomputed=False."""
        from baselines.SOLAR_DualMamba.train_solar_dualmamba_la import parse_args as parse_args_solar

        # 1. Mặc định
        sys.argv = ["train_solar_dualmamba_la.py"]
        args_default = parse_args_solar()
        self.assertIsNone(args_default.limit_train_batches)
        self.assertIsNone(args_default.limit_val_batches)
        self.assertTrue(args_default.use_precomputed)

        # 2. Cấu hình smoke test
        sys.argv = [
            "train_solar_dualmamba_la.py",
            "--limit_train_batches", "3",
            "--limit_val_batches", "2",
            "--use_precomputed", "False",
        ]
        args_smoke = parse_args_solar()
        self.assertEqual(args_smoke.limit_train_batches, 3)
        self.assertEqual(args_smoke.limit_val_batches, 2)
        self.assertFalse(args_smoke.use_precomputed)

        # 3. Cờ bật --use_precomputed không kèm tham số vẫn trả về True
        sys.argv = ["train_solar_dualmamba_la.py", "--use_precomputed"]
        args_flag = parse_args_solar()
        self.assertTrue(args_flag.use_precomputed)

    def test_smoke_script_exists_and_syntax(self):
        """Xác minh script scripts/smoke_hpcuit.sh tồn tại và vượt qua kiểm tra bash -n."""
        self.assertTrue(os.path.exists(SCRIPT_PATH), f"Không tìm thấy file {SCRIPT_PATH}")
        proc = subprocess.run(["bash", "-n", SCRIPT_PATH], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, f"Lỗi cú pháp bash trong smoke_hpcuit.sh:\n{proc.stderr}")

    def test_smoke_script_slurm_lint(self):
        """Xác minh scripts/smoke_hpcuit.sh vượt qua công cụ slurm_lint --profile hpc-uit."""
        lint_tool = os.path.abspath(os.path.join(REPO_ROOT, "..", "..", ".claude", "tools", "slurm_lint.py"))
        self.assertTrue(os.path.exists(lint_tool), f"Không tìm thấy slurm_lint.py tại {lint_tool}")
        proc = subprocess.run(
            [sys.executable, lint_tool, "--profile", "hpc-uit", SCRIPT_PATH],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, f"slurm_lint thất bại trên smoke_hpcuit.sh:\n{proc.stdout}\n{proc.stderr}")

    def test_smoke_script_directives_and_content(self):
        """Xác minh các chỉ thị Slurm và cấu trúc bắt buộc theo yêu cầu T4."""
        with open(SCRIPT_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        # Kiểm tra partition và qos
        self.assertRegex(content, r"#SBATCH\s+--partition=small\b")
        self.assertRegex(content, r"#SBATCH\s+--qos=mig1-18\b")
        self.assertRegex(content, r"#SBATCH\s+--gres=gpu:1g\.18gb:1\b")
        self.assertRegex(content, r"#SBATCH\s+--time=00:30:00\b")

        # Kiểm tra chạy tuần tự LEARN và SOLAR_DualMamba trong 1 script
        self.assertIn("train_learn_la.py", content)
        self.assertIn("train_solar_dualmamba_la.py", content)

        # Kiểm tra trên-the-fly (--use_precomputed False)
        self.assertRegex(content, r"--use_precomputed\s+False")

        # Kiểm tra giới hạn batch
        self.assertRegex(content, r"--limit_train_batches\s+\d+")
        self.assertRegex(content, r"--limit_val_batches\s+\d+")

        # Kiểm tra in git rev-parse HEAD và đo peak VRAM
        self.assertIn("rev-parse HEAD", content)
        self.assertIn("max_memory_allocated", content)

        # Kiểm tra có chú thích rõ ràng rằng smoke test không phải là kết quả thực nghiệm khoa học
        self.assertTrue(
            "không phải là kết quả" in content.lower() or "không dùng làm kết quả" in content.lower(),
            "Script cần ghi chú rõ các số liệu smoke test không phải là kết quả thực nghiệm khoa học",
        )


if __name__ == "__main__":
    unittest.main()
