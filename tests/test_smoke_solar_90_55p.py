#!/usr/bin/env python3
# ==============================================================================
# KIỂM THỬ ĐƠN VỊ TDD CHO KỊCH BẢN SMOKE SOLAR_DualMamba LA-90° 55 BỆNH NHÂN (T1.4)
# File: scripts/smoke_solar_90_55p_hpcuit.sh
# ==============================================================================

import os
import re
import subprocess
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(REPO_ROOT, "..", ".."))
LINTER_PATH = os.path.join(WORKSPACE_ROOT, ".claude", "tools", "slurm_lint.py")
SMOKE_SCRIPT = os.path.join(REPO_ROOT, "scripts", "smoke_solar_90_55p_hpcuit.sh")

FROZEN_TRAIN_PATIENTS = [
    "L004", "L006", "L012", "L014", "L019", "L024", "L027", "L030", "L033", "L035",
    "L036", "L043", "L044", "L045", "L048", "L049", "L056", "L057", "L071", "L072",
    "L075", "L077", "L079", "L081", "L082", "L111", "L113", "L114", "L116", "L121",
    "L123", "L125", "L127", "L129", "L131", "L133", "L134", "L136", "L138", "L143",
    "L144", "L145", "L147", "L148", "L150", "L154", "L160", "L163", "L166", "L170",
    "L171", "L172", "L175", "L178", "L179",
]
FROZEN_VAL_PATIENTS = ["L110"]
FROZEN_TEST_PATIENTS = ["L058", "L064", "L094", "L107"]


def _extract_bash_array(content: str, var_name: str) -> list:
    """Trích xuất danh sách phần tử chuỗi trong mảng bash VAR=(...)."""
    match = re.search(rf"{var_name}=\(([^)]+)\)", content)
    if not match:
        return []
    return re.findall(r'"([^"]+)"', match.group(1))


class TestSmokeSolar90Script(unittest.TestCase):
    """Kiểm thử hợp đồng cho kịch bản smoke SOLAR_DualMamba 90° 55 BN (T1.4)."""

    def test_smoke_script_exists_and_is_executable(self):
        """Xác minh kịch bản tồn tại và có quyền thực thi (chmod +x)."""
        self.assertTrue(os.path.exists(SMOKE_SCRIPT), f"Không tìm thấy {SMOKE_SCRIPT}")
        self.assertTrue(os.access(SMOKE_SCRIPT, os.X_OK), f"{SMOKE_SCRIPT} thiếu quyền thực thi (+x)")

    def test_slurm_directives(self):
        """Xác minh toàn bộ chỉ thị SBATCH theo đúng đặc tả T1.4."""
        self.assertTrue(os.path.exists(SMOKE_SCRIPT), f"Không tìm thấy {SMOKE_SCRIPT}")
        with open(SMOKE_SCRIPT, "r", encoding="utf-8") as f:
            content = f.read()

        expected_directives = [
            "#SBATCH --job-name=smoke_solar_90_55p_hpcuit",
            "#SBATCH --output=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output-hpc-uit/smoke_solar_90_55p_hpcuit/log/%j.out",
            "#SBATCH --error=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output-hpc-uit/smoke_solar_90_55p_hpcuit/log/%j.err",
            "#SBATCH --account=uit",
            "#SBATCH --partition=normal",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --cpus-per-task=8",
            "#SBATCH --mem=16G",
            "#SBATCH --time=01:00:00",
            "#SBATCH --gres=gpu:2g.35gb:1",
        ]
        for directive in expected_directives:
            self.assertIn(directive, content, f"Thiếu chỉ thị {directive} trong {SMOKE_SCRIPT}")

        # Không dùng --qos và không trỏ vào scripts/output/
        self.assertNotRegex(content, r"#SBATCH\s+--qos=")
        self.assertNotIn("scripts/output/", content)

    def test_safety_and_cache_redirection(self):
        """Xác minh set -euo pipefail, trap cleanup, git commit hash và chuyển hướng cache."""
        with open(SMOKE_SCRIPT, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("set -euo pipefail", content)
        trap_pos = content.find("trap cleanup EXIT")
        mkdir_pos = content.find("mkdir -p")
        self.assertNotEqual(trap_pos, -1, "Thiếu trap cleanup EXIT")
        self.assertNotEqual(mkdir_pos, -1, "Thiếu mkdir -p")
        self.assertLess(trap_pos, mkdir_pos, "trap cleanup EXIT phải đứng trước mkdir -p")

        self.assertIn('git -C "$REPO" rev-parse HEAD', content)
        self.assertIn("nvidia-smi", content)

        # Kiểm tra chuyển hướng 5 biến cache về /datastore/doanhbc/minhdphan/cache
        self.assertIn("export XDG_CACHE_HOME=/datastore/doanhbc/minhdphan/cache", content)
        self.assertIn("export PIP_CACHE_DIR=$XDG_CACHE_HOME/pip", content)
        self.assertIn("export TORCH_HOME=$XDG_CACHE_HOME/torch", content)
        self.assertIn("export HF_HOME=$XDG_CACHE_HOME/hf", content)
        self.assertIn("export TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton", content)

    def test_split_configuration(self):
        """Xác minh danh sách 55 bệnh nhân train, 1 val, 4 test và assertion fail-fast."""
        with open(SMOKE_SCRIPT, "r", encoding="utf-8") as f:
            content = f.read()

        active_code = "\n".join(
            line for line in content.splitlines() if line.strip() and not line.strip().startswith("#")
        )
        self.assertNotIn("-exec basename", active_code)
        self.assertEqual(_extract_bash_array(content, "TRAIN_PATIENTS"), FROZEN_TRAIN_PATIENTS)
        self.assertEqual(_extract_bash_array(content, "VAL_PATIENTS"), FROZEN_VAL_PATIENTS)
        self.assertEqual(_extract_bash_array(content, "TEST_PATIENTS"), FROZEN_TEST_PATIENTS)
        self.assertIn('[ "${#TRAIN_PATIENTS[@]}" -eq 55 ]', content)

    def test_preflight_block_requirements(self):
        """Xác minh khối preflight Python cho LA-90° (setting_tag, angles, asserts, symlinks, 0-byte)."""
        with open(SMOKE_SCRIPT, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn('setting_tag="limited_ang_90deg_numview_64_size_256_noise_0"', content)
        self.assertIn("start_ang=float(np.deg2rad(-45.0))", content)
        self.assertIn("end_ang=float(np.deg2rad(45.0))", content)
        self.assertIn("len(dm.train_dataset) == 8218", content)
        self.assertIn("len(dm.val_dataset) == 133", content)
        self.assertIn("len(dm.test_dataset) == 757", content)

        # Kiểm tra symlink không gãy và kiểm tra tệp >= 1000 bytes
        self.assertIn("broken_paths = [p for p in ds.slices_path if not os.path.exists(p)]", content)
        self.assertIn("MIN_SLICE_BYTES = 1000", content)
        self.assertIn("undersized_paths = [(p, os.path.getsize(p)) for p in ds.slices_path if os.path.getsize(p) < MIN_SLICE_BYTES]", content)

        # Đọc __getitem__(0) và kiểm tra shape, finite
        self.assertIn("phantom, fbp_u, sino = ds.__getitem__(0)", content)
        self.assertIn("assert tuple(phantom.shape) == (1, 256, 256)", content)
        self.assertIn("assert tuple(fbp_u.shape) == (1, 256, 256)", content)
        self.assertIn("assert tuple(sino.shape) == (1, 64, 512)", content)
        self.assertIn("torch.isfinite(phantom).all()", content)

    def test_smoke_training_command_and_telemetry(self):
        """Xác minh câu lệnh huấn luyện 1 epoch SOLAR_DualMamba 90° và thu thập telemetry."""
        with open(SMOKE_SCRIPT, "r", encoding="utf-8") as f:
            content = f.read()

        # Tham số hóa batch size: BATCH_SIZE="${1:-1}"
        self.assertIn('BATCH_SIZE="${1:-1}"', content)
        self.assertIn('OUTPUT_DIR="$REPO/saved_models/SOLAR_DualMamba_90_smoke"', content)
        self.assertIn('LOG_DIR="$REPO/lightning_logs/SOLAR_DualMamba_90_smoke"', content)

        # Lệnh gọi train_solar_dualmamba_la.py
        self.assertIn("baselines/SOLAR_DualMamba/train_solar_dualmamba_la.py", content)
        self.assertIn("--angle_range_deg 90.0", content)
        self.assertIn('--batch_size "$BATCH_SIZE"', content)
        self.assertIn("--max_epochs 1", content)
        self.assertIn('--train_patients "${TRAIN_PATIENTS[@]}"', content)
        self.assertIn('--val_patients "${VAL_PATIENTS[@]}"', content)
        self.assertIn("--use_precomputed", content)

        # Đo wall-clock và peak VRAM
        self.assertIn("EPOCH_START_TS=$(date +%s)", content)
        self.assertIn("EPOCH_END_TS=$(date +%s)", content)
        self.assertIn("EPOCH_ELAPSED_SEC=$((EPOCH_END_TS - EPOCH_START_TS))", content)
        self.assertIn('PEAK_VRAM_LINE=$(grep "Peak VRAM allocated" "$TRAIN_STDOUT_LOG" | tail -n 1 || true)', content)

    def test_bash_syntax_and_slurm_lint(self):
        """Xác minh bash -n và slurm_lint --profile hpc-uit PASS trên kịch bản."""
        res_bash = subprocess.run(["bash", "-n", SMOKE_SCRIPT], capture_output=True, text=True)
        self.assertEqual(res_bash.returncode, 0, f"bash -n lỗi trên {SMOKE_SCRIPT}: {res_bash.stderr}")

        res_lint = subprocess.run(
            [sys.executable, LINTER_PATH, "--profile", "hpc-uit", SMOKE_SCRIPT],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res_lint.returncode, 0, f"slurm_lint lỗi:\n{res_lint.stdout}\n{res_lint.stderr}")
        self.assertIn("PASS", res_lint.stdout)


if __name__ == "__main__":
    unittest.main()
