#!/usr/bin/env python3
# ==============================================================================
# KIỂM THỬ ĐƠN VỊ CHO CÁC SBATCH SCRIPTS CỦA WAVE B ABLATION STUDIES (CHANGE 25)
# Kiểm tra: cú pháp Slurm, phân vùng normal, cờ --ablate, đường dẫn log/ckpt,
# cú pháp bash -n, và slurm_lint theo chuẩn HPC-UIT.
# ==============================================================================

import os
import subprocess
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(REPO_ROOT, "..", ".."))
LINTER_PATH = os.path.join(WORKSPACE_ROOT, ".claude", "tools", "slurm_lint.py")

ABLATION_VARIANTS = {
    "a": {
        "script": "scripts/train_solar_dualmamba_la_ablate_a.sh",
        "job_name": "train_solar_dualmamba_la_ablate_a",
        "flag": "--ablate a",
        "suffix": "abl-a",
        "extra_flag": None,
    },
    "b": {
        "script": "scripts/train_solar_dualmamba_la_ablate_b.sh",
        "job_name": "train_solar_dualmamba_la_ablate_b",
        "flag": "--ablate b",
        "suffix": "abl-b",
        "extra_flag": "--n_iterations 8",
    },
    "c": {
        "script": "scripts/train_solar_dualmamba_la_ablate_c.sh",
        "job_name": "train_solar_dualmamba_la_ablate_c",
        "flag": "--ablate c",
        "suffix": "abl-c",
        "extra_flag": None,
    },
    "d": {
        "script": "scripts/train_solar_dualmamba_la_ablate_d.sh",
        "job_name": "train_solar_dualmamba_la_ablate_d",
        "flag": "--ablate d",
        "suffix": "abl-d",
        "extra_flag": None,
    },
}


class TestWaveBAblationScripts(unittest.TestCase):
    """Bộ kiểm thử hợp đồng cho 4 kịch bản Slurm của Wave B Ablation."""

    def test_scripts_exist_and_readable(self):
        """Xác minh 4 file kịch bản ablation đều tồn tại trong scripts/."""
        for var, cfg in ABLATION_VARIANTS.items():
            script_path = os.path.join(REPO_ROOT, cfg["script"])
            self.assertTrue(
                os.path.exists(script_path),
                f"Kịch bản {cfg['script']} cho biến thể ablation '{var}' không tồn tại."
            )

    def test_slurm_headers_match_specification(self):
        """Xác minh các chỉ thị Slurm SBATCH tuân thủ đúng phân vùng normal và profile HPC-UIT."""
        for var, cfg in ABLATION_VARIANTS.items():
            script_path = os.path.join(REPO_ROOT, cfg["script"])
            if not os.path.exists(script_path):
                self.skipTest(f"Tệp {script_path} chưa được tạo.")

            with open(script_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn(f"#SBATCH --job-name={cfg['job_name']}", content)
            self.assertIn("#SBATCH --account=uit", content)
            self.assertIn("#SBATCH --partition=normal", content)
            self.assertIn("#SBATCH --gres=gpu:2g.35gb:1", content)
            self.assertIn("#SBATCH --mem=16G", content)
            self.assertIn("#SBATCH --cpus-per-task=8", content)
            self.assertIn("#SBATCH --time=24:00:00", content)
            self.assertIn(
                f"#SBATCH --output=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output/{cfg['job_name']}/log/%j.out",
                content
            )
            self.assertIn(
                f"#SBATCH --error=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output/{cfg['job_name']}/log/%j.err",
                content
            )

    def test_ablation_flags_and_directories(self):
        """Xác minh cờ --ablate và các đường dẫn saved_models/lightning_logs khớp spec-1.md."""
        for var, cfg in ABLATION_VARIANTS.items():
            script_path = os.path.join(REPO_ROOT, cfg["script"])
            if not os.path.exists(script_path):
                self.skipTest(f"Tệp {script_path} chưa được tạo.")

            with open(script_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn(cfg["flag"], content, f"Thiếu cờ {cfg['flag']} trong {cfg['script']}")
            if cfg["extra_flag"]:
                self.assertIn(
                    cfg["extra_flag"],
                    content,
                    f"Thiếu cờ bổ sung {cfg['extra_flag']} trong {cfg['script']}"
                )

            self.assertIn(f"SOLAR_DualMamba_${{ANGLE}}_{cfg['suffix']}", content)

    def test_trap_cleanup_order_in_ablation_scripts(self):
        """Xác minh trap cleanup EXIT được đặt trước mọi lệnh mkdir -p."""
        for var, cfg in ABLATION_VARIANTS.items():
            script_path = os.path.join(REPO_ROOT, cfg["script"])
            if not os.path.exists(script_path):
                self.skipTest(f"Tệp {script_path} chưa được tạo.")

            with open(script_path, "r", encoding="utf-8") as f:
                content = f.read()

            trap_pos = content.find("trap cleanup EXIT")
            mkdir_pos = content.find("mkdir -p")
            self.assertNotEqual(trap_pos, -1, f"Không tìm thấy 'trap cleanup EXIT' trong {cfg['script']}")
            if mkdir_pos != -1:
                self.assertLess(
                    trap_pos,
                    mkdir_pos,
                    f"'trap cleanup EXIT' phải nằm trước 'mkdir -p' trong {cfg['script']}"
                )

    def test_bash_syntax(self):
        """Xác minh bash -n không phát hiện lỗi cú pháp trên cả 4 kịch bản."""
        for var, cfg in ABLATION_VARIANTS.items():
            script_path = os.path.join(REPO_ROOT, cfg["script"])
            if not os.path.exists(script_path):
                self.skipTest(f"Tệp {script_path} chưa được tạo.")

            res = subprocess.run(["bash", "-n", script_path], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"Cú pháp bash lỗi trong {cfg['script']}:\n{res.stderr}")

    def test_slurm_lint_profile_hpcuit(self):
        """Xác minh slurm_lint.py thông qua với profile hpc-uit."""
        if not os.path.exists(LINTER_PATH):
            self.skipTest(f"Không tìm thấy linter tại {LINTER_PATH}")

        for var, cfg in ABLATION_VARIANTS.items():
            script_path = os.path.join(REPO_ROOT, cfg["script"])
            if not os.path.exists(script_path):
                self.skipTest(f"Tệp {script_path} chưa được tạo.")

            cmd = [sys.executable, LINTER_PATH, "--profile", "hpc-uit", script_path]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(
                res.returncode,
                0,
                f"slurm_lint thất bại trên {cfg['script']}:\n{res.stdout}\n{res.stderr}"
            )
            self.assertIn("PASS", res.stdout)


if __name__ == "__main__":
    unittest.main()
