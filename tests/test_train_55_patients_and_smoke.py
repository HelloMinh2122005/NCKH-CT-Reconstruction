#!/usr/bin/env python3
# ==============================================================================
# KIỂM THỬ ĐƠN VỊ CHO ROOT CAUSE #2 (55 BỆNH NHÂN ĐÓNG BĂNG TRONG 12 SCRIPT HUẤN LUYỆN)
# VÀ KỊCH BẢN SMOKE BENCHMARK 1-EPOCH WAVE C (scripts/smoke_wave_c_55p_hpcuit.sh)
# ==============================================================================

import os
import re
import subprocess
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(REPO_ROOT, "..", ".."))
LINTER_PATH = os.path.join(WORKSPACE_ROOT, ".claude", "tools", "slurm_lint.py")
GEN_120_SCRIPT = os.path.join(REPO_ROOT, "scripts", "generate_la_dataset_120.sh")
SMOKE_SCRIPT = os.path.join(REPO_ROOT, "scripts", "smoke_wave_c_55p_hpcuit.sh")

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

TWELVE_TRAIN_SCRIPTS = [
    # Wave A (4 kịch bản)
    "scripts/train_solar_dualmamba_la.sh",
    "scripts/train_learn_la.sh",
    "scripts/train_regformer_la.sh",
    "scripts/train_dudotrans_la.sh",
    # Wave B (4 kịch bản ablation)
    "scripts/train_solar_dualmamba_la_ablate_a.sh",
    "scripts/train_solar_dualmamba_la_ablate_b.sh",
    "scripts/train_solar_dualmamba_la_ablate_c.sh",
    "scripts/train_solar_dualmamba_la_ablate_d.sh",
    # Wave C (4 kịch bản LA-120°)
    "scripts/train_solar_dualmamba_la_120.sh",
    "scripts/train_learn_la_120.sh",
    "scripts/train_regformer_la_120.sh",
    "scripts/train_dudotrans_la_120.sh",
]


def _extract_bash_array(content: str, var_name: str) -> list:
    """Trích xuất danh sách phần tử chuỗi trong mảng bash VAR=(...)."""
    match = re.search(rf"{var_name}=\(([^)]+)\)", content)
    if not match:
        return []
    return re.findall(r'"([^"]+)"', match.group(1))


class TestTrainScripts55FrozenPatients(unittest.TestCase):
    """Kiểm thử toàn bộ 12 kịch bản huấn luyện AAPM dùng đúng 55 bệnh nhân đóng băng."""

    def test_frozen_list_matches_generate_120_and_is_disjoint(self):
        """Xác minh danh sách 55 bệnh nhân khớp tuyệt đối với generate_la_dataset_120.sh và rời rạc hoàn toàn."""
        self.assertEqual(len(FROZEN_TRAIN_PATIENTS), 55)
        self.assertEqual(len(set(FROZEN_TRAIN_PATIENTS)), 55)
        self.assertTrue(set(FROZEN_TRAIN_PATIENTS).isdisjoint(FROZEN_VAL_PATIENTS))
        self.assertTrue(set(FROZEN_TRAIN_PATIENTS).isdisjoint(FROZEN_TEST_PATIENTS))
        self.assertTrue(set(FROZEN_VAL_PATIENTS).isdisjoint(FROZEN_TEST_PATIENTS))

        with open(GEN_120_SCRIPT, "r", encoding="utf-8") as f:
            gen_content = f.read()
        gen_train = _extract_bash_array(gen_content, "TRAIN_PATIENTS")
        self.assertEqual(gen_train, FROZEN_TRAIN_PATIENTS)

    def test_all_12_train_scripts_have_frozen_55_patients_no_find_basename(self):
        """Xác minh 12 script không còn find -exec basename, có assert 55 BN và truyền --train_patients vô điều kiện."""
        for relpath in TWELVE_TRAIN_SCRIPTS:
            script_path = os.path.join(REPO_ROOT, relpath)
            self.assertTrue(os.path.exists(script_path), f"Thiếu script {relpath}")

            with open(script_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Lọc bỏ các dòng comment khi kiểm tra mã bash thực thi
            active_code = "\n".join(
                line for line in content.splitlines() if line.strip() and not line.strip().startswith("#")
            )

            # Không còn mẫu find -exec basename {} + gây rơi --train_patients trong mã thực thi
            self.assertNotIn("-exec basename", active_code, f"{relpath} vẫn còn '-exec basename' trong mã thực thi")
            self.assertNotIn("TRAIN_PATIENT_ARGS", active_code, f"{relpath} vẫn còn biến điều kiện TRAIN_PATIENT_ARGS")

            # Mảng TRAIN_PATIENTS khớp chính xác 55 bệnh nhân đóng băng
            train_patients = _extract_bash_array(content, "TRAIN_PATIENTS")
            self.assertEqual(
                train_patients,
                FROZEN_TRAIN_PATIENTS,
                f"Danh sách TRAIN_PATIENTS trong {relpath} không khớp 55 bệnh nhân đóng băng.",
            )
            val_patients = _extract_bash_array(content, "VAL_PATIENTS")
            self.assertEqual(val_patients, FROZEN_VAL_PATIENTS, f"VAL_PATIENTS sai trong {relpath}")
            test_patients = _extract_bash_array(content, "TEST_PATIENTS")
            self.assertEqual(test_patients, FROZEN_TEST_PATIENTS, f"TEST_PATIENTS sai trong {relpath}")

            # Kiểm tra câu lệnh fail-fast số lượng 55 bệnh nhân và truyền vô điều kiện vào Python
            self.assertIn('[ "${#TRAIN_PATIENTS[@]}" -eq 55 ]', content, f"{relpath} thiếu assertion 55 bệnh nhân")
            self.assertIn(
                '--train_patients "${TRAIN_PATIENTS[@]}"',
                content,
                f"{relpath} phải truyền --train_patients \"${{TRAIN_PATIENTS[@]}}\" vô điều kiện",
            )

            # Bảo toàn nguyên trạng --max_epochs 50, --batch_size 1, --time=24:00:00
            self.assertIn("--max_epochs 50", content, f"{relpath} không được thay đổi --max_epochs 50")
            self.assertIn("--batch_size 1", content, f"{relpath} không được thay đổi --batch_size 1")
            self.assertIn("#SBATCH --time=24:00:00", content, f"{relpath} không được thay đổi --time=24:00:00")

    def test_bash_fail_fast_assertion_runtime(self):
        """Kiểm thử thực thi bash cho assertion [ ${#TRAIN_PATIENTS[@]} -eq 55 ] với 55 BN (pass) và 54 BN (fail)."""
        ok_cmd = (
            'set -euo pipefail; '
            'TRAIN_PATIENTS=(' + " ".join(f'"{p}"' for p in FROZEN_TRAIN_PATIENTS) + '); '
            '[ "${#TRAIN_PATIENTS[@]}" -eq 55 ] || exit 1'
        )
        res_ok = subprocess.run(["bash", "-c", ok_cmd], capture_output=True, text=True)
        self.assertEqual(res_ok.returncode, 0, f"Bash fail-fast thất bại với 55 BN: {res_ok.stderr}")

        bad_cmd = (
            'set -euo pipefail; '
            'TRAIN_PATIENTS=(' + " ".join(f'"{p}"' for p in FROZEN_TRAIN_PATIENTS[:-1]) + '); '
            '[ "${#TRAIN_PATIENTS[@]}" -eq 55 ] || exit 1'
        )
        res_bad = subprocess.run(["bash", "-c", bad_cmd], capture_output=True, text=True)
        self.assertNotEqual(res_bad.returncode, 0, "Bash fail-fast phải thất bại khi thiếu bệnh nhân (54 BN).")

    def test_all_12_train_scripts_bash_syntax_and_slurm_lint(self):
        """Xác minh bash -n và slurm_lint --profile hpc-uit đều PASS trên cả 12 kịch bản huấn luyện."""
        self.assertTrue(os.path.exists(LINTER_PATH), f"Không tìm thấy {LINTER_PATH}")
        for relpath in TWELVE_TRAIN_SCRIPTS:
            script_path = os.path.join(REPO_ROOT, relpath)
            res_bash = subprocess.run(["bash", "-n", script_path], capture_output=True, text=True)
            self.assertEqual(res_bash.returncode, 0, f"bash -n lỗi trên {relpath}: {res_bash.stderr}")

            res_lint = subprocess.run(
                [sys.executable, LINTER_PATH, "--profile", "hpc-uit", script_path],
                capture_output=True,
                text=True,
            )
            self.assertEqual(res_lint.returncode, 0, f"slurm_lint lỗi trên {relpath}:\n{res_lint.stdout}\n{res_lint.stderr}")
            self.assertIn("PASS", res_lint.stdout)


class TestSmokeWaveC55pScript(unittest.TestCase):
    """Kiểm thử hợp đồng cho kịch bản smoke + data-integration scripts/smoke_wave_c_55p_hpcuit.sh."""

    def test_smoke_script_exists_and_slurm_directives(self):
        """Xác minh chỉ thị Slurm của smoke_wave_c_55p_hpcuit.sh tuân thủ profile hpc-uit và đặc tả."""
        self.assertTrue(os.path.exists(SMOKE_SCRIPT), f"Không tìm thấy {SMOKE_SCRIPT}")
        with open(SMOKE_SCRIPT, "r", encoding="utf-8") as f:
            content = f.read()

        expected_directives = [
            "#SBATCH --job-name=smoke_wave_c_55p_hpcuit",
            "#SBATCH --output=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output-hpc-uit/smoke_wave_c_55p_hpcuit/log/%j.out",
            "#SBATCH --error=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output-hpc-uit/smoke_wave_c_55p_hpcuit/log/%j.err",
            "#SBATCH --account=uit",
            "#SBATCH --partition=normal",
            "#SBATCH --gres=gpu:2g.35gb:1",
            "#SBATCH --mem=12G",
            "#SBATCH --time=01:00:00",
            "#SBATCH --cpus-per-task=8",
        ]
        for directive in expected_directives:
            self.assertIn(directive, content, f"Thiếu chỉ thị {directive} trong smoke_wave_c_55p_hpcuit.sh")

        self.assertNotRegex(content, r"#SBATCH\s+--qos=")
        self.assertIn("set -euo pipefail", content)
        trap_pos = content.find("trap cleanup EXIT")
        mkdir_pos = content.find("mkdir -p")
        self.assertNotEqual(trap_pos, -1)
        self.assertNotEqual(mkdir_pos, -1)
        self.assertLess(trap_pos, mkdir_pos, "trap cleanup EXIT phải đứng trước mkdir -p")
        self.assertIn('git -C "$REPO" rev-parse HEAD', content)
        self.assertIn("nvidia-smi", content)

    def test_smoke_script_preflight_and_training_contract(self):
        """Xác minh preflight 8218/133/757 + __getitem__(0) và cấu hình huấn luyện 1-epoch SOLAR_DualMamba bs=4."""
        with open(SMOKE_SCRIPT, "r", encoding="utf-8") as f:
            content = f.read()

        active_code = "\n".join(
            line for line in content.splitlines() if line.strip() and not line.strip().startswith("#")
        )
        # Không sử dụng find -exec basename và dùng đúng 55 bệnh nhân đóng băng
        self.assertNotIn("-exec basename", active_code)
        self.assertEqual(_extract_bash_array(content, "TRAIN_PATIENTS"), FROZEN_TRAIN_PATIENTS)
        self.assertEqual(_extract_bash_array(content, "VAL_PATIENTS"), FROZEN_VAL_PATIENTS)
        self.assertEqual(_extract_bash_array(content, "TEST_PATIENTS"), FROZEN_TEST_PATIENTS)
        self.assertIn('[ "${#TRAIN_PATIENTS[@]}" -eq 55 ]', content)

        # Preflight kiểm tra số lát cắt và đọc __getitem__(0)
        self.assertIn("len(dm.train_dataset) == 8218", content)
        self.assertIn("len(dm.val_dataset) == 133", content)
        self.assertIn("len(dm.test_dataset) == 757", content)
        # [2026-11-20 T0.7/T0.5] Kiểm tra kích thước tệp lát cắt chống tệp 0-byte
        self.assertIn("os.path.getsize", content)
        self.assertIn("__getitem__(0)", content)

        # Huấn luyện 1-epoch SOLAR_DualMamba LA-120° bs=4 cô lập thư mục smoke
        self.assertIn("baselines/SOLAR_DualMamba/train_solar_dualmamba_la.py", content)
        self.assertIn("--angle_range_deg 120.0", content)
        self.assertIn("--batch_size 4", content)
        self.assertIn("--max_epochs 1", content)
        self.assertIn('--train_patients "${TRAIN_PATIENTS[@]}"', content)
        self.assertIn('--val_patients "${VAL_PATIENTS[@]}"', content)
        self.assertIn("saved_models/SOLAR_DualMamba_120_smoke", content)
        self.assertIn("lightning_logs/SOLAR_DualMamba_120_smoke", content)
        self.assertIn("Peak VRAM allocated", content)
        self.assertIn("EPOCH_ELAPSED_SEC", content)

    def test_smoke_script_bash_syntax_and_slurm_lint(self):
        """Xác minh bash -n và slurm_lint --profile hpc-uit PASS trên smoke_wave_c_55p_hpcuit.sh."""
        res_bash = subprocess.run(["bash", "-n", SMOKE_SCRIPT], capture_output=True, text=True)
        self.assertEqual(res_bash.returncode, 0, f"bash -n lỗi: {res_bash.stderr}")

        res_lint = subprocess.run(
            [sys.executable, LINTER_PATH, "--profile", "hpc-uit", SMOKE_SCRIPT],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res_lint.returncode, 0, f"slurm_lint lỗi:\n{res_lint.stdout}\n{res_lint.stderr}")
        self.assertIn("PASS", res_lint.stdout)


if __name__ == "__main__":
    unittest.main()
