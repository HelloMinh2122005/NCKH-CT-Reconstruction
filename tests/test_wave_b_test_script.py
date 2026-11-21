#!/usr/bin/env python3
# ==============================================================================
# KIỂM THỬ HỢP ĐỒNG TDD CHO KỊCH BẢN ĐÁNH GIÁ SLURM WAVE B ABLATION
# Tệp mục tiêu: scripts/test_wave_b_ablation_hpcuit.sh & test_solar_dualmamba_la.py
# Tuân thủ: spec-1.md, spec-1-decisions.md, split.md (GT-03-wave-b-test)
# ==============================================================================

import os
import stat
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(REPO_ROOT, "..", ".."))
LINTER_PATH = os.path.join(WORKSPACE_ROOT, ".claude", "tools", "slurm_lint.py")
SCRIPT_RELPATH = "scripts/test_wave_b_ablation_hpcuit.sh"
SCRIPT_PATH = os.path.join(REPO_ROOT, SCRIPT_RELPATH)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    import numpy
except ImportError:
    pass

def _cluster_mock_modules():
    # Đảm bảo các module cluster-only được mock nếu chạy ở môi trường thiếu dependency
    mocks = {
        mod: MagicMock()
        for mod in [
            "torch",
            "torch.nn",
            "torch.nn.functional",
            "torchvision",
            "pytorch_lightning",
            "torchmetrics",
            "torchmetrics.functional",
            "torchmetrics.functional.image",
            "odl",
            "odl.contrib",
            "odl.contrib.torch",
            "data",
            "data.datamodule_factory",
        ]
        if mod not in sys.modules
    }
    if "data.split_guards" not in sys.modules:
        mocks["data.split_guards"] = MagicMock(unsafe=True)
    return mocks


class TestWaveBTestScriptContract(unittest.TestCase):
    """Kiểm thử hợp đồng cho kịch bản Slurm scripts/test_wave_b_ablation_hpcuit.sh."""

    def test_script_exists_and_executable(self):
        """Xác minh script scripts/test_wave_b_ablation_hpcuit.sh tồn tại và có quyền thực thi (+x)."""
        self.assertTrue(
            os.path.exists(SCRIPT_PATH),
            f"Kịch bản kiểm thử Wave B không tồn tại tại: {SCRIPT_PATH}"
        )
        file_mode = os.stat(SCRIPT_PATH).st_mode
        self.assertTrue(
            bool(file_mode & stat.S_IXUSR),
            f"Kịch bản {SCRIPT_PATH} chưa được gán quyền thực thi (chmod +x)."
        )

    def test_slurm_headers_contract(self):
        """Xác minh cấu hình header Slurm đáp ứng đúng đặc tả phân vùng normal và tài nguyên."""
        self.assertTrue(os.path.exists(SCRIPT_PATH), "Script chưa tồn tại.")
        with open(SCRIPT_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("#SBATCH --job-name=test_wave_b_ablation_hpcuit", content)
        self.assertIn("#SBATCH --account=uit", content)
        self.assertIn("#SBATCH --partition=normal", content)
        self.assertIn("#SBATCH --gres=gpu:2g.35gb:1", content)
        self.assertIn("#SBATCH --mem=16G", content)
        self.assertIn("#SBATCH --cpus-per-task=8", content)
        self.assertIn(
            "#SBATCH --output=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output-hpc-uit/test_wave_b_ablation_hpcuit/log/%j.out",
            content
        )
        self.assertIn(
            "#SBATCH --error=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output-hpc-uit/test_wave_b_ablation_hpcuit/log/%j.err",
            content
        )

    def test_safety_and_trap_cleanup_order(self):
        """Xác minh set -euo pipefail và trap cleanup EXIT được đặt trước mkdir -p."""
        self.assertTrue(os.path.exists(SCRIPT_PATH), "Script chưa tồn tại.")
        with open(SCRIPT_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("set -euo pipefail", content)
        trap_pos = content.find("trap cleanup EXIT")
        mkdir_pos = content.find("mkdir -p")
        self.assertNotEqual(trap_pos, -1, "Không tìm thấy 'trap cleanup EXIT' trong script.")
        self.assertNotEqual(mkdir_pos, -1, "Không tìm thấy 'mkdir -p' trong script.")
        self.assertLess(
            trap_pos,
            mkdir_pos,
            "'trap cleanup EXIT' phải được khai báo trước mọi lệnh I/O như 'mkdir -p'."
        )

    def test_git_provenance_and_test_patients(self):
        """Xác minh script in git rev-parse HEAD và sử dụng đúng 4 bệnh nhân kiểm thử đóng băng."""
        self.assertTrue(os.path.exists(SCRIPT_PATH), "Script chưa tồn tại.")
        with open(SCRIPT_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("git -C \"$REPO\" rev-parse HEAD", content)
        # Bệnh nhân test đóng băng theo split.md: L058, L064, L094, L107
        for pid in ["L058", "L064", "L094", "L107"]:
            self.assertIn(pid, content, f"Thiếu bệnh nhân kiểm thử {pid} theo split.md")

    def test_evaluates_all_4_checkpoints_with_correct_flags(self):
        """Xác minh script đánh giá cả 4 checkpoint với method label phân biệt và cờ thích hợp."""
        self.assertTrue(os.path.exists(SCRIPT_PATH), "Script chưa tồn tại.")
        with open(SCRIPT_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        # Kiểm tra 4 checkpoint dir / suffix
        for var in ["abl-a", "abl-b", "abl-c", "abl-d"]:
            self.assertIn(
                f"SOLAR_DualMamba_90_{var}",
                content,
                f"Thiếu định tuyến checkpoint SOLAR_DualMamba_90_{var}"
            )

        # Kiểm tra 4 method label riêng biệt
        for var in ["abl-a", "abl-b", "abl-c", "abl-d"]:
            self.assertIn(
                f"SOLAR_DualMamba_{var}",
                content,
                f"Thiếu nhãn phương pháp SOLAR_DualMamba_{var}"
            )

        # Kiểm tra các cờ ablation
        self.assertIn("--ablate a", content)
        self.assertIn("--ablate b", content)
        self.assertIn("--ablate c", content)
        self.assertIn("--ablate d", content)
        # Ablation b yêu cầu stage-matched K=8
        self.assertIn("--n_iterations 8", content)

    def test_output_csv_and_job_id(self):
        """Xác minh đường dẫn xuất per-slice CSV và truyền mã Job ID."""
        self.assertTrue(os.path.exists(SCRIPT_PATH), "Script chưa tồn tại.")
        with open(SCRIPT_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        # [2026-11-20 T1.6] Cho phép báo cáo tại reports/hpc-uit-55p hoặc reports/
        # [2026-11-21 P6] Cho phép báo cáo tại reports/hpc-uit-55p với ${ANGLE}
        self.assertTrue(
            "reports/hpc-uit-55p/wave_b_ablation_${ANGLE}_per_slice_results.csv" in content
            or "reports/hpc-uit-55p/wave_b_ablation_per_slice_results.csv" in content
            or "reports/wave_b_ablation_per_slice_results.csv" in content,
            "Đường dẫn kết quả CSV phải nằm trong reports/hpc-uit-55p hoặc reports/"
        )
        self.assertIn("--job_id", content)

    def test_bash_syntax_valid(self):
        """Xác minh cú pháp bash hợp lệ với bash -n."""
        self.assertTrue(os.path.exists(SCRIPT_PATH), "Script chưa tồn tại.")
        res = subprocess.run(["bash", "-n", SCRIPT_PATH], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Lỗi cú pháp bash:\n{res.stderr}")

    def test_slurm_lint_profile_hpcuit(self):
        """Xác minh slurm_lint.py thông qua với profile hpc-uit."""
        if not os.path.exists(LINTER_PATH):
            self.skipTest(f"Không tìm thấy linter tại {LINTER_PATH}")
        self.assertTrue(os.path.exists(SCRIPT_PATH), "Script chưa tồn tại.")

        cmd = [sys.executable, LINTER_PATH, "--profile", "hpc-uit", SCRIPT_PATH]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(
            res.returncode,
            0,
            f"slurm_lint thất bại trên {SCRIPT_RELPATH}:\n{res.stdout}\n{res.stderr}"
        )
        self.assertIn("PASS", res.stdout)


class TestSolarDualMambaTestCLIWiring(unittest.TestCase):
    """Kiểm thử đơn vị wiring cờ dòng lệnh trong baselines/SOLAR_DualMamba/test_solar_dualmamba_la.py."""

    def test_cli_has_ablate_and_n_iterations_flags(self):
        """Xác minh test_solar_dualmamba_la nhận diện cờ --ablate và --n_iterations với giá trị mặc định chuẩn."""
        with patch.dict(sys.modules, _cluster_mock_modules()):
            from baselines.SOLAR_DualMamba.test_solar_dualmamba_la import parse_args

            with patch.object(sys, "argv", ["test_solar_dualmamba_la.py"]):
                args = parse_args()
                self.assertTrue(hasattr(args, "ablate"))
                self.assertEqual(args.ablate, "none")
                self.assertTrue(hasattr(args, "n_iterations"))
                self.assertEqual(args.n_iterations, 8)

    def test_cli_custom_n_iterations(self):
        """Xác minh test_solar_dualmamba_la nhận diện giá trị tùy chỉnh của --n_iterations."""
        with patch.dict(sys.modules, _cluster_mock_modules()):
            from baselines.SOLAR_DualMamba.test_solar_dualmamba_la import parse_args

            with patch.object(sys, "argv", ["test_solar_dualmamba_la.py", "--n_iterations", "14"]):
                args = parse_args()
                self.assertEqual(args.n_iterations, 14)


if __name__ == "__main__":
    unittest.main()
