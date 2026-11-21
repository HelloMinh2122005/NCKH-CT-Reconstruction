#!/usr/bin/env python3
# ==============================================================================
# KIỂM THỬ ĐƠN VỊ CHO C3 (THAM SỐ HÓA ANGLE & VARIANTS TRONG ABLATION SCRIPTS)
# Tệp mục tiêu:
#   - scripts/train_solar_dualmamba_la_ablate_{a,b,c,d}.sh
#   - scripts/test_wave_b_ablation_hpcuit.sh
# Tuân thủ: .lab/20261026-restart-intake/experiments/queue-plan-ALL.md (v2) - Mục C3
# ==============================================================================

import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent.parent
SLURM_LINT_PATH = WORKSPACE_ROOT / ".claude" / "tools" / "slurm_lint.py"

TRAIN_ABLATION_SCRIPTS = [
    REPO_ROOT / "scripts" / "train_solar_dualmamba_la_ablate_a.sh",
    REPO_ROOT / "scripts" / "train_solar_dualmamba_la_ablate_b.sh",
    REPO_ROOT / "scripts" / "train_solar_dualmamba_la_ablate_c.sh",
    REPO_ROOT / "scripts" / "train_solar_dualmamba_la_ablate_d.sh",
]

TEST_ABLATION_SCRIPT = REPO_ROOT / "scripts" / "test_wave_b_ablation_hpcuit.sh"


class TestAblationScriptsEnv(unittest.TestCase):
    """Kiểm thử tham số hóa ANGLE và VARIANTS cho các kịch bản ablation."""

    def test_train_ablation_scripts_angle_env_hierarchy(self):
        """Xác minh 4 script train ablation đọc ANGLE theo thứ bậc: $1 -> $ANGLE -> 90."""
        for script_path in TRAIN_ABLATION_SCRIPTS:
            self.assertTrue(script_path.is_file(), f"Tệp không tồn tại: {script_path}")
            content = script_path.read_text(encoding="utf-8")

            # Phải có biểu thức chuẩn bash parameter expansion: ANGLE="${1:-${ANGLE:-90}}"
            self.assertIn(
                'ANGLE="${1:-${ANGLE:-90}}"',
                content,
                f"Thiếu định nghĩa ANGLE=\"${{1:-${{ANGLE:-90}}}}\" trong {script_path.name}"
            )
            # Đường dẫn checkpoint và log phải dựa trên $ANGLE
            self.assertIn('SOLAR_DualMamba_${ANGLE}_abl-', content)

    def test_test_wave_b_ablation_script_parameterization(self):
        """Xác minh test_wave_b_ablation_hpcuit.sh đọc ANGLE và VARIANTS linh hoạt."""
        self.assertTrue(TEST_ABLATION_SCRIPT.is_file(), f"Tệp không tồn tại: {TEST_ABLATION_SCRIPT}")
        content = TEST_ABLATION_SCRIPT.read_text(encoding="utf-8")

        # Phải đọc ANGLE và VARIANTS với fallback chuẩn
        self.assertIn(
            'ANGLE="${1:-${ANGLE:-90}}"',
            content,
            "Thiếu ANGLE=\"${1:-${ANGLE:-90}}\" trong test_wave_b_ablation_hpcuit.sh"
        )
        self.assertIn(
            'VARIANTS="${2:-${VARIANTS:-a b c d}}"',
            content,
            "Thiếu VARIANTS=\"${2:-${VARIANTS:-a b c d}}\" trong test_wave_b_ablation_hpcuit.sh"
        )
        # Đường dẫn checkpoint phải tham chiếu ${ANGLE}
        self.assertIn('SOLAR_DualMamba_${ANGLE}_abl-${VARIANT}', content)

    def test_p6_csv_output_defined_after_angle_and_includes_angle(self):
        """P6: Xác minh CSV_OUTPUT đặt sau định nghĩa ANGLE và chứa ${ANGLE}."""
        content = TEST_ABLATION_SCRIPT.read_text(encoding="utf-8")
        angle_pos = content.find('ANGLE="${1:-${ANGLE:-90}}"')
        csv_pos = content.find('CSV_OUTPUT=')
        self.assertNotEqual(angle_pos, -1, "Không tìm thấy định nghĩa ANGLE")
        self.assertNotEqual(csv_pos, -1, "Không tìm thấy định nghĩa CSV_OUTPUT")
        self.assertGreater(csv_pos, angle_pos, "P6: CSV_OUTPUT phải được định nghĩa SAU dòng gán ANGLE")
        self.assertIn(
            'CSV_OUTPUT="$REPO/reports/hpc-uit-55p/wave_b_ablation_${ANGLE}_per_slice_results.csv"',
            content,
            "P6: CSV_OUTPUT phải chứa ${ANGLE} để tránh ghi đè giữa 90° và 120°"
        )

    def test_bash_syntax_validity(self):
        """Kiểm tra cú pháp bash hợp lệ trên toàn bộ 5 script ablation."""
        all_scripts = TRAIN_ABLATION_SCRIPTS + [TEST_ABLATION_SCRIPT]
        for script_path in all_scripts:
            res = subprocess.run(["bash", "-n", str(script_path)], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"Lỗi cú pháp bash tại {script_path.name}:\n{res.stderr}")

    def test_slurm_lint_profile_hpcuit(self):
        """Kiểm tra toàn bộ 5 script ablation vượt qua slurm_lint --profile hpc-uit."""
        if not SLURM_LINT_PATH.is_file():
            self.skipTest(f"Không tìm thấy slurm_lint tại {SLURM_LINT_PATH}")

        all_scripts = TRAIN_ABLATION_SCRIPTS + [TEST_ABLATION_SCRIPT]
        for script_path in all_scripts:
            res = subprocess.run(
                ["python3", str(SLURM_LINT_PATH), "--profile", "hpc-uit", str(script_path)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                res.returncode, 0,
                f"slurm_lint thất bại trên {script_path.name}:\n{res.stdout}\n{res.stderr}"
            )
            self.assertIn("PASS", res.stdout)


if __name__ == "__main__":
    unittest.main()
