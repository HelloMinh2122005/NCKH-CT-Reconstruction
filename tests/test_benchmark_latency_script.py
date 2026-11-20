#!/usr/bin/env python3
"""
Unit tests cho kịch bản Slurm scripts/benchmark_latency_hpcuit.sh (TDD Red-Green).
Kiểm thử tính hợp lệ cú pháp bash, các chỉ thị SBATCH theo chuẩn HPC-UIT,
và các tham số gọi kịch bản benchmark độ trễ suy luận Wave E.
"""

import os
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "benchmark_latency_hpcuit.sh"
SLURM_LINT_PATH = REPO_ROOT.parent.parent / ".claude" / "tools" / "slurm_lint.py"


class TestBenchmarkLatencyScript(unittest.TestCase):
    """Kiểm tra kịch bản sbatch benchmark_latency_hpcuit.sh."""

    def test_script_exists(self):
        """Kịch bản scripts/benchmark_latency_hpcuit.sh bắt buộc phải tồn tại."""
        self.assertTrue(SCRIPT_PATH.is_file(), f"Tệp không tồn tại: {SCRIPT_PATH}")

    def test_bash_syntax_validity(self):
        """Cú pháp bash phải hoàn toàn hợp lệ (kiểm tra bằng bash -n)."""
        result = subprocess.run(["bash", "-n", str(SCRIPT_PATH)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, f"Lỗi cú pháp bash: {result.stderr}")

    def test_slurm_lint_pass(self):
        """Kịch bản phải vượt qua kiểm tra của slurm_lint.py trên profile hpc-uit."""
        self.assertTrue(SLURM_LINT_PATH.is_file(), f"Không tìm thấy slurm_lint.py tại: {SLURM_LINT_PATH}")
        result = subprocess.run(
            ["python3", str(SLURM_LINT_PATH), "--profile", "hpc-uit", str(SCRIPT_PATH)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, f"slurm_lint thất bại: {result.stdout}\n{result.stderr}")
        self.assertIn("PASS", result.stdout, "Kết quả slurm_lint phải chứa PASS")

    def test_required_sbatch_directives(self):
        """Kiểm tra các chỉ thị SBATCH cấu hình phân vùng normal MIG gpu:2g.35gb:1."""
        content = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --job-name=benchmark_latency_hpcuit", content)
        self.assertIn("#SBATCH --partition=normal", content)
        self.assertIn("#SBATCH --gres=gpu:2g.35gb:1", content)
        self.assertIn("#SBATCH --mem=16G", content)
        self.assertIn("#SBATCH --time=00:30:00", content)
        self.assertIn("#SBATCH --cpus-per-task=8", content)
        self.assertIn("scripts/output-hpc-uit/benchmark_latency_hpcuit/log/%j.out", content)
        self.assertIn("scripts/output-hpc-uit/benchmark_latency_hpcuit/log/%j.err", content)

    def test_git_commit_logged(self):
        """Kịch bản phải ghi lại thông tin commit git (git rev-parse HEAD)."""
        content = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn("rev-parse HEAD", content, "Kịch bản phải chứa lệnh truy vấn git rev-parse HEAD")

    def test_no_forbidden_home_paths(self):
        """Kịch bản không được chứa đường dẫn /home/ cho dữ liệu, checkpoint hoặc log."""
        content = SCRIPT_PATH.read_text(encoding="utf-8")
        for line in content.splitlines():
            line_clean = line.strip()
            if line_clean.startswith("#") or not line_clean:
                continue
            self.assertNotIn("/home/doanhbc", line_clean, "Không dùng /home/doanhbc trong kịch bản")
            self.assertNotIn("/home/phandinhminh", line_clean, "Không dùng /home/phandinhminh trong kịch bản")

    def test_invokes_python_benchmark_with_wave_e_protocol(self):
        """Kịch bản phải gọi scripts/benchmark_latency_hpcuit.py với đầy đủ tham số Wave E."""
        content = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn("scripts/benchmark_latency_hpcuit.py", content)
        self.assertIn("--warmup_runs 20", content)
        self.assertIn("--test_runs 100", content)
        self.assertIn("--angle_range_deg 90.0", content)
        self.assertIn("--hardware_profile", content)
        self.assertIn("--output_csv", content)


if __name__ == "__main__":
    unittest.main()
