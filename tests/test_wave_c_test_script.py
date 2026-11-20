#!/usr/bin/env python3
# ==============================================================================
# KIỂM THỬ HỢP ĐỒNG TDD CHO CÁC KỊCH BẢN ĐÁNH GIÁ SLURM WAVE C VÀ SOLAR LAST CKPT
# Tệp mục tiêu:
#   1. scripts/test_wave_c_120_hpcuit.sh
#   2. scripts/test_solar_last_ckpt_hpcuit.sh
# Tuân thủ: spec-1.md, spec-1-decisions.md, split.md (Change 29)
# ==============================================================================

import os
import re
import stat
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Xác định đường dẫn thư mục gốc repo và workspace
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(TEST_DIR) == "tests" and os.path.exists(os.path.join(TEST_DIR, "..", "baselines")):
    REPO_ROOT = os.path.abspath(os.path.join(TEST_DIR, ".."))
    WORKSPACE_ROOT = os.path.abspath(os.path.join(REPO_ROOT, "..", ".."))
else:
    REPO_ROOT = os.path.abspath(os.path.join(TEST_DIR, "..", "temp-github", "NCKH-CT-Reconstruction"))
    WORKSPACE_ROOT = os.path.abspath(os.path.join(TEST_DIR, ".."))

LINTER_PATH = os.path.join(WORKSPACE_ROOT, ".claude", "tools", "slurm_lint.py")
WAVE_C_TEST_SCRIPT = os.path.join(REPO_ROOT, "scripts", "test_wave_c_120_hpcuit.sh")
SOLAR_LAST_CKPT_SCRIPT = os.path.join(REPO_ROOT, "scripts", "test_solar_last_ckpt_hpcuit.sh")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

def _cluster_mock_modules():
    # Đảm bảo các module phụ thuộc cụm cluster-only được mock nếu chạy ở môi trường thiếu dependency
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


class TestWaveCTestScriptsContract(unittest.TestCase):
    """Kiểm thử hợp đồng cho kịch bản Slurm đánh giá Wave C và SOLAR last.ckpt."""

    def test_scripts_exist_and_executable(self):
        """Xác minh cả 2 kịch bản test tồn tại trong scripts/ và có quyền thực thi (+x)."""
        for script_path in [WAVE_C_TEST_SCRIPT, SOLAR_LAST_CKPT_SCRIPT]:
            self.assertTrue(
                os.path.exists(script_path),
                f"Kịch bản không tồn tại: {script_path}"
            )
            file_mode = os.stat(script_path).st_mode
            self.assertTrue(
                bool(file_mode & stat.S_IXUSR),
                f"Kịch bản {script_path} chưa có quyền thực thi (chmod +x)."
            )

    def test_slurm_headers_contract(self):
        """Xác minh cấu hình header Slurm tuân thủ đầy đủ phân vùng normal, GRES MIG, RAM 16G, CPUs 8 và time limit."""
        scripts_cfg = [
            {
                "path": WAVE_C_TEST_SCRIPT,
                "name": "test_wave_c_120_hpcuit",
                "time": "04:00:00",
            },
            {
                "path": SOLAR_LAST_CKPT_SCRIPT,
                "name": "test_solar_last_ckpt_hpcuit",
                "time": "01:00:00",
            },
        ]

        for cfg in scripts_cfg:
            script_path = cfg["path"]
            if not os.path.exists(script_path):
                self.skipTest(f"Script {script_path} chưa tồn tại.")

            with open(script_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn(f"#SBATCH --job-name={cfg['name']}", content)
            self.assertIn("#SBATCH --account=uit", content)
            self.assertIn("#SBATCH --partition=normal", content)
            self.assertIn("#SBATCH --gres=gpu:2g.35gb:1", content)
            self.assertIn("#SBATCH --mem=16G", content)
            self.assertIn("#SBATCH --cpus-per-task=8", content)
            self.assertIn(f"#SBATCH --time={cfg['time']}", content)
            self.assertIn(
                f"#SBATCH --output=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output-hpc-uit/{cfg['name']}/log/%j.out",
                content
            )
            self.assertIn(
                f"#SBATCH --error=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output-hpc-uit/{cfg['name']}/log/%j.err",
                content
            )

    def test_safety_and_trap_cleanup_order(self):
        """Xác minh set -euo pipefail và trap cleanup EXIT được đặt trước mọi lệnh mkdir -p."""
        for script_path in [WAVE_C_TEST_SCRIPT, SOLAR_LAST_CKPT_SCRIPT]:
            if not os.path.exists(script_path):
                self.skipTest(f"Script {script_path} chưa tồn tại.")

            with open(script_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn("set -euo pipefail", content)
            trap_pos = content.find("trap cleanup EXIT")
            mkdir_pos = content.find("mkdir -p")
            self.assertNotEqual(trap_pos, -1, f"Không tìm thấy 'trap cleanup EXIT' trong {script_path}")
            self.assertNotEqual(mkdir_pos, -1, f"Không tìm thấy 'mkdir -p' trong {script_path}")
            self.assertLess(
                trap_pos,
                mkdir_pos,
                f"'trap cleanup EXIT' phải nằm trước 'mkdir -p' trong {script_path}"
            )

    def test_git_provenance_and_test_patients(self):
        """Xác minh script in git rev-parse HEAD và sử dụng đúng 4 bệnh nhân kiểm thử đóng băng."""
        for script_path in [WAVE_C_TEST_SCRIPT, SOLAR_LAST_CKPT_SCRIPT]:
            if not os.path.exists(script_path):
                self.skipTest(f"Script {script_path} chưa tồn tại.")

            with open(script_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn("git -C \"$REPO\" rev-parse HEAD", content)
            for pid in ["L058", "L064", "L094", "L107"]:
                self.assertIn(pid, content, f"Thiếu bệnh nhân kiểm thử {pid} trong {script_path}")

    def test_no_arg_contract(self):
        """Xác minh cả 2 kịch bản là no-arg Slurm scripts, không đòi hỏi đối số dòng lệnh vị trí."""
        for script_path in [WAVE_C_TEST_SCRIPT, SOLAR_LAST_CKPT_SCRIPT]:
            if not os.path.exists(script_path):
                self.skipTest(f"Script {script_path} chưa tồn tại.")

            with open(script_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for idx, line in enumerate(lines, 1):
                clean_line = line.strip()
                if clean_line.startswith("#"):
                    continue
                # Bắt lỗi nếu có lệnh yêu cầu $1 mà không có giá trị mặc định ${1:-default}
                if re.search(r'(?<![:\w])\$1(?!\w)', clean_line):
                    self.fail(
                        f"Dòng {idx} trong {script_path} sử dụng $1 trần mà không có fallback mặc định: {clean_line}"
                    )

    def test_checkpoint_folder_alignment_with_train_120(self):
        """Xác minh kịch bản Wave C đánh giá đúng 4 thư mục checkpoint của 4 mô hình huấn luyện và fail loudly nếu thiếu."""
        if not os.path.exists(WAVE_C_TEST_SCRIPT):
            self.skipTest(f"Script {WAVE_C_TEST_SCRIPT} chưa tồn tại.")

        with open(WAVE_C_TEST_SCRIPT, "r", encoding="utf-8") as f:
            content = f.read()

        # Kiểm tra căn chỉnh với 4 thư mục checkpoint từ train_*_120.sh
        for model in ["SOLAR_DualMamba", "LEARN", "RegFormer", "DuDoTrans"]:
            self.assertIn(
                f"{model}_120",
                content,
                f"Thiếu định tuyến thư mục checkpoint cho mô hình {model}_120 trong test_wave_c_120_hpcuit.sh"
            )

        # Kiểm tra góc quét 120 độ
        self.assertTrue(
            ("120.0" in content or 'ANGLE="120"' in content),
            "Thiếu cấu hình góc quét LA-120° trong test_wave_c_120_hpcuit.sh"
        )
        self.assertIn("--num_view 64", content)

        # Kiểm tra fail loudly nếu thiếu checkpoint
        self.assertIn("exit 1", content)
        self.assertTrue(
            ("Thư mục checkpoint không tồn tại" in content or "Missing checkpoint" in content or "không tồn tại" in content),
            "Thiếu thông báo lỗi rõ ràng khi không tìm thấy thư mục checkpoint."
        )

        # Kiểm tra đường dẫn xuất CSV và dọn dẹp file CSV cũ
        # [2026-11-20 T1.6] Hỗ trợ thư mục reports/hpc-uit-55p/
        self.assertTrue(
            "reports/wave_c_120_per_slice_results.csv" in content or "reports/hpc-uit-55p/wave_c_120_per_slice_results.csv" in content,
            "Thiếu đường dẫn xuất CSV wave_c_120"
        )
        self.assertIn("rm -f", content)

    def test_solar_last_ckpt_target_and_fail_loudly(self):
        """Xác minh kịch bản SOLAR last.ckpt nhắm đúng tệp last.ckpt tại 90 độ và fail loudly nếu thiếu."""
        if not os.path.exists(SOLAR_LAST_CKPT_SCRIPT):
            self.skipTest(f"Script {SOLAR_LAST_CKPT_SCRIPT} chưa tồn tại.")

        with open(SOLAR_LAST_CKPT_SCRIPT, "r", encoding="utf-8") as f:
            content = f.read()

        # Kiểm tra đường dẫn chính xác tới last.ckpt
        # [2026-11-20 T1.6] Hỗ trợ gốc saved_models_55p/
        self.assertTrue(
            "saved_models/SOLAR_DualMamba_90/last.ckpt" in content or "saved_models_55p/SOLAR_DualMamba_90/last.ckpt" in content,
            "Thiếu đường dẫn tới last.ckpt của SOLAR_DualMamba_90"
        )

        # Kiểm tra góc quét 90 độ
        self.assertTrue(
            ("90.0" in content or 'ANGLE="90"' in content),
            "Thiếu cấu hình góc quét LA-90° trong test_solar_last_ckpt_hpcuit.sh"
        )
        self.assertIn("--num_view 64", content)

        # Kiểm tra fail loudly nếu thiếu last.ckpt
        self.assertIn("exit 1", content)
        self.assertTrue(
            ("không tồn tại" in content or "Missing" in content or "not found" in content),
            "Thiếu thông báo lỗi rõ ràng khi tệp last.ckpt không tồn tại."
        )

        # Kiểm tra file CSV xuất ra
        # [2026-11-20 T1.6] Hỗ trợ thư mục reports/hpc-uit-55p/
        self.assertTrue(
            "reports/solar_last_ckpt_per_slice_results.csv" in content or "reports/hpc-uit-55p/solar_last_ckpt_per_slice_results.csv" in content,
            "Thiếu đường dẫn file CSV xuất ra"
        )
        self.assertIn("rm -f", content)

    def test_cli_flags_validity_against_test_python_scripts(self):
        """Xác minh các tham số dòng lệnh truyền từ bash script đều được định nghĩa trong parse_args() của test_*.py."""
        with patch.dict(sys.modules, _cluster_mock_modules()):
            from baselines.SOLAR_DualMamba.test_solar_dualmamba_la import parse_args as parse_solar_args
            from baselines.LEARN.test_learn_la import parse_args as parse_learn_args
            from baselines.RegFormer.test_regformer_la import parse_args as parse_regformer_args
            from baselines.DuDoTrans.test_dudotrans_la import parse_args as parse_dudotrans_args

            parsers = {
                "SOLAR_DualMamba": parse_solar_args,
                "LEARN": parse_learn_args,
                "RegFormer": parse_regformer_args,
                "DuDoTrans": parse_dudotrans_args,
            }

            # Kiểm tra các cờ cơ sở được gọi trong script
            expected_flags = [
                "ckpt_path",
                "dataset_type",
                "dicom_dir",
                "cache_dir",
                "angle_range_deg",
                "num_view",
                "num_detectors",
                "input_size",
                "poisson_level",
                "gaussian_level",
                "batch_size",
                "num_workers",
                "test_patients",
                "output_csv",
                "job_id",
                "method",
            ]

            for model_name, parser_fn in parsers.items():
                with patch.object(sys, "argv", ["test.py"]):
                    args = parser_fn()
                    for flag in expected_flags:
                        self.assertTrue(
                            hasattr(args, flag),
                            f"Parser của mô hình {model_name} thiếu thuộc tính argparse cho cờ --{flag}"
                        )

    def test_bash_syntax_valid(self):
        """Xác minh cú pháp bash hợp lệ với lệnh bash -n."""
        for script_path in [WAVE_C_TEST_SCRIPT, SOLAR_LAST_CKPT_SCRIPT]:
            if not os.path.exists(script_path):
                self.skipTest(f"Script {script_path} chưa tồn tại.")

            res = subprocess.run(["bash", "-n", script_path], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"Cú pháp bash lỗi trong {script_path}:\n{res.stderr}")

    def test_slurm_lint_profile_hpcuit(self):
        """Xác minh slurm_lint.py thông qua với profile hpc-uit."""
        if not os.path.exists(LINTER_PATH):
            self.skipTest(f"Không tìm thấy linter tại {LINTER_PATH}")

        for script_path in [WAVE_C_TEST_SCRIPT, SOLAR_LAST_CKPT_SCRIPT]:
            if not os.path.exists(script_path):
                self.skipTest(f"Script {script_path} chưa tồn tại.")

            cmd = [sys.executable, LINTER_PATH, "--profile", "hpc-uit", script_path]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(
                res.returncode,
                0,
                f"slurm_lint thất bại trên {script_path}:\n{res.stdout}\n{res.stderr}"
            )
            self.assertIn("PASS", res.stdout)


if __name__ == "__main__":
    unittest.main()
