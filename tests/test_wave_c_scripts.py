#!/usr/bin/env python3
# ==============================================================================
# KIỂM THỬ HỢP ĐỒNG TDD CHO CÁC SBATCH SCRIPTS CỦA WAVE C (LA-120°) (CHANGE 28)
# Kiểm tra: 5 scripts tồn tại, quyền thực thi, Slurm header, phân vùng small/normal,
# GRES MIG, RAM 8G/16G, trap cleanup trước mkdir -p, logging, git rev-parse HEAD,
# góc quét 120°, seed 42, last.ckpt resume, bash -n, và slurm_lint --profile hpc-uit.
# ==============================================================================

import os
import stat
import subprocess
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(REPO_ROOT, "..", ".."))
LINTER_PATH = os.path.join(WORKSPACE_ROOT, ".claude", "tools", "slurm_lint.py")

WAVE_C_SCRIPTS = {
    "generate_la_dataset_120": {
        "relpath": "scripts/generate_la_dataset_120.sh",
        "job_name": "generate_la_dataset_120",
        "partition": "small",
        "gres": "gpu:1g.18gb:1",
        "mem": "8G",
        "cpus": 4,
        "is_train": False,
        "target_angle": "120.0",
    },
    "train_solar_dualmamba_la_120": {
        "relpath": "scripts/train_solar_dualmamba_la_120.sh",
        "job_name": "train_solar_dualmamba_la_120",
        "partition": "normal",
        "gres": "gpu:2g.35gb:1",
        "mem": "12G",
        "cpus": 8,
        "is_train": True,
        "model_dir_name": "SOLAR_DualMamba_120",
        "target_angle": "120",
    },
    "train_learn_la_120": {
        "relpath": "scripts/train_learn_la_120.sh",
        "job_name": "train_learn_la_120",
        "partition": "normal",
        "gres": "gpu:2g.35gb:1",
        "mem": "12G",
        "cpus": 8,
        "is_train": True,
        "model_dir_name": "LEARN_120",
        "target_angle": "120",
    },
    "train_regformer_la_120": {
        "relpath": "scripts/train_regformer_la_120.sh",
        "job_name": "train_regformer_la_120",
        "partition": "normal",
        "gres": "gpu:2g.35gb:1",
        "mem": "12G",
        "cpus": 8,
        "is_train": True,
        "model_dir_name": "RegFormer_120",
        "target_angle": "120",
    },
    "train_dudotrans_la_120": {
        "relpath": "scripts/train_dudotrans_la_120.sh",
        "job_name": "train_dudotrans_la_120",
        "partition": "normal",
        "gres": "gpu:2g.35gb:1",
        "mem": "12G",
        "cpus": 8,
        "is_train": True,
        "model_dir_name": "DuDoTrans_120",
        "target_angle": "120",
    },
}


class TestWaveCScriptsContract(unittest.TestCase):
    """Bộ kiểm thử hợp đồng cho 5 kịch bản Slurm Wave C (LA-120°)."""

    def test_scripts_exist_and_executable(self):
        """Xác minh cả 5 file kịch bản Wave C tồn tại trong scripts/ và có quyền thực thi (+x)."""
        for name, cfg in WAVE_C_SCRIPTS.items():
            script_path = os.path.join(REPO_ROOT, cfg["relpath"])
            self.assertTrue(
                os.path.exists(script_path),
                f"Kịch bản {cfg['relpath']} không tồn tại."
            )
            file_mode = os.stat(script_path).st_mode
            self.assertTrue(
                bool(file_mode & stat.S_IXUSR),
                f"Kịch bản {cfg['relpath']} chưa có quyền thực thi (chmod +x)."
            )

    def test_slurm_headers_match_spec(self):
        """Xác minh các chỉ thị Slurm SBATCH tuân thủ phân vùng, gres, cpu, mem, logs và account uit."""
        for name, cfg in WAVE_C_SCRIPTS.items():
            script_path = os.path.join(REPO_ROOT, cfg["relpath"])
            if not os.path.exists(script_path):
                self.skipTest(f"Tệp {script_path} chưa được tạo.")

            with open(script_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn(f"#SBATCH --job-name={cfg['job_name']}", content)
            self.assertIn("#SBATCH --account=uit", content)
            self.assertIn(f"#SBATCH --partition={cfg['partition']}", content)
            self.assertIn(f"#SBATCH --gres={cfg['gres']}", content)
            self.assertIn(f"#SBATCH --mem={cfg['mem']}", content)
            self.assertIn(f"#SBATCH --cpus-per-task={cfg['cpus']}", content)
            self.assertIn("#SBATCH --time=24:00:00", content)
            self.assertIn(
                f"#SBATCH --output=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output-hpc-uit/{cfg['job_name']}/log/%j.out",
                content
            )
            self.assertIn(
                f"#SBATCH --error=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output-hpc-uit/{cfg['job_name']}/log/%j.err",
                content
            )

    def test_safety_and_trap_cleanup_order(self):
        """Xác minh set -euo pipefail và trap cleanup EXIT được đặt trước mọi lệnh mkdir -p."""
        for name, cfg in WAVE_C_SCRIPTS.items():
            script_path = os.path.join(REPO_ROOT, cfg["relpath"])
            if not os.path.exists(script_path):
                self.skipTest(f"Tệp {script_path} chưa được tạo.")

            with open(script_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn("set -euo pipefail", content)
            trap_pos = content.find("trap cleanup EXIT")
            mkdir_pos = content.find("mkdir -p")
            self.assertNotEqual(trap_pos, -1, f"Không tìm thấy 'trap cleanup EXIT' trong {cfg['relpath']}")
            if mkdir_pos != -1:
                self.assertLess(
                    trap_pos,
                    mkdir_pos,
                    f"'trap cleanup EXIT' phải nằm trước 'mkdir -p' trong {cfg['relpath']}"
                )

    def test_git_provenance(self):
        """Xác minh script in git rev-parse HEAD phục vụ truy xuất nguồn gốc."""
        for name, cfg in WAVE_C_SCRIPTS.items():
            script_path = os.path.join(REPO_ROOT, cfg["relpath"])
            if not os.path.exists(script_path):
                self.skipTest(f"Tệp {script_path} chưa được tạo.")

            with open(script_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn("git -C \"$REPO\" rev-parse HEAD", content)

    def test_target_angles_and_directories(self):
        """Xác minh cấu hình góc quét LA-120° và thư mục saved_models/lightning_logs tương ứng."""
        # 1. Kịch bản sinh dữ liệu generate_la_dataset_120.sh
        gen_cfg = WAVE_C_SCRIPTS["generate_la_dataset_120"]
        gen_path = os.path.join(REPO_ROOT, gen_cfg["relpath"])
        if os.path.exists(gen_path):
            with open(gen_path, "r", encoding="utf-8") as f:
                gen_content = f.read()
            self.assertTrue(
                ('TARGET_ANGLES="120.0"' in gen_content or 'TARGET_ANGLES="${TARGET_ANGLES:-120.0}"' in gen_content),
                f"Thiếu định nghĩa TARGET_ANGLES=120.0 trong {gen_cfg['relpath']}"
            )

        # 2. 4 Kịch bản huấn luyện
        for name, cfg in WAVE_C_SCRIPTS.items():
            if not cfg["is_train"]:
                continue
            script_path = os.path.join(REPO_ROOT, cfg["relpath"])
            if not os.path.exists(script_path):
                self.skipTest(f"Tệp {script_path} chưa được tạo.")

            with open(script_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Kiểm tra ANGLE mặc định 120 khi chạy không đối số
            self.assertTrue(
                ('ANGLE="120"' in content or 'ANGLE="${1:-120}"' in content),
                f"Thiếu cấu hình ANGLE mặc định 120 trong {cfg['relpath']}"
            )
            # Kiểm tra đường dẫn lưu model và log khớp Model_120
            self.assertIn(f"saved_models/{cfg['model_dir_name']}", content)
            self.assertIn(f"lightning_logs/{cfg['model_dir_name']}", content)
            # Kiểm tra cờ --seed 42
            self.assertIn("--seed 42", content)
            # Kiểm tra cơ chế resume từ last.ckpt
            self.assertIn("last.ckpt", content)
            self.assertIn("--resume_ckpt", content)

    def test_bash_syntax_valid(self):
        """Xác minh cú pháp bash hợp lệ với bash -n trên toàn bộ 5 kịch bản."""
        for name, cfg in WAVE_C_SCRIPTS.items():
            script_path = os.path.join(REPO_ROOT, cfg["relpath"])
            if not os.path.exists(script_path):
                self.skipTest(f"Tệp {script_path} chưa được tạo.")

            res = subprocess.run(["bash", "-n", script_path], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"Cú pháp bash lỗi trong {cfg['relpath']}:\n{res.stderr}")

    def test_slurm_lint_profile_hpcuit(self):
        """Xác minh slurm_lint.py thông qua với profile hpc-uit trên cả 5 kịch bản."""
        if not os.path.exists(LINTER_PATH):
            self.skipTest(f"Không tìm thấy linter tại {LINTER_PATH}")

        for name, cfg in WAVE_C_SCRIPTS.items():
            script_path = os.path.join(REPO_ROOT, cfg["relpath"])
            if not os.path.exists(script_path):
                self.skipTest(f"Tệp {script_path} chưa được tạo.")

            cmd = [sys.executable, LINTER_PATH, "--profile", "hpc-uit", script_path]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(
                res.returncode,
                0,
                f"slurm_lint thất bại trên {cfg['relpath']}:\n{res.stdout}\n{res.stderr}"
            )
            self.assertIn("PASS", res.stdout)


if __name__ == "__main__":
    unittest.main()
