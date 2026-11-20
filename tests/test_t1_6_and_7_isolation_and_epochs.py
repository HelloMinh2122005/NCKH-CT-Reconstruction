#!/usr/bin/env python3
# ==============================================================================
# KIỂM THỬ ĐƠN VỊ TDD CHO T1.6 & T1.7 (CHANGE 39)
# T1.6: Tách triệt để Checkpoint và CSV 55 BN khỏi Pilot (Chặn T2):
#       - saved_models_55p/<Model>_<angle>[_abl-x]/ cho 12 script train Tier 1
#       - Fail-fast khi $OUTPUT_DIR chứa bất kỳ tệp .ckpt nào
#       - 4 script test chỉ đọc từ saved_models_55p/ và ghi vào reports/hpc-uit-55p/
# T1.7: Tham số hóa MAX_EPOCHS="${MAX_EPOCHS:-50}", in log [INFO] MAX_EPOCHS: $MAX_EPOCHS,
#       và truyền --max_epochs "$MAX_EPOCHS"
# ==============================================================================

import os
import re
import subprocess
import tempfile
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

TWELVE_TRAIN_SCRIPTS = [
    # Wave A (4 scripts 90°)
    "scripts/train_solar_dualmamba_la.sh",
    "scripts/train_learn_la.sh",
    "scripts/train_regformer_la.sh",
    "scripts/train_dudotrans_la.sh",
    # Wave B (4 scripts ablation 90°)
    "scripts/train_solar_dualmamba_la_ablate_a.sh",
    "scripts/train_solar_dualmamba_la_ablate_b.sh",
    "scripts/train_solar_dualmamba_la_ablate_c.sh",
    "scripts/train_solar_dualmamba_la_ablate_d.sh",
    # Wave C (4 scripts 120°)
    "scripts/train_solar_dualmamba_la_120.sh",
    "scripts/train_learn_la_120.sh",
    "scripts/train_regformer_la_120.sh",
    "scripts/train_dudotrans_la_120.sh",
]

FOUR_TEST_SCRIPTS = [
    "scripts/test_benchmark_hpcuit.sh",
    "scripts/test_wave_b_ablation_hpcuit.sh",
    "scripts/test_wave_c_120_hpcuit.sh",
    "scripts/test_solar_last_ckpt_hpcuit.sh",
]


class TestT16AndT17IsolationAndEpochs(unittest.TestCase):
    """Kiểm thử tự động cho T1.6 và T1.7 trên 12 script train và 4 script test."""

    def test_train_scripts_use_saved_models_55p_root(self):
        """Xác minh cả 12 script train Tier 1 đều trỏ OUTPUT_DIR vào saved_models_55p/."""
        for relpath in TWELVE_TRAIN_SCRIPTS:
            path = os.path.join(REPO_ROOT, relpath)
            self.assertTrue(os.path.exists(path), f"Thiếu script: {relpath}")
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertRegex(
                content,
                r'OUTPUT_DIR="\$REPO/saved_models_55p/',
                f"{relpath} phải dùng gốc saved_models_55p/ cho OUTPUT_DIR (T1.6)"
            )
            # Không được dùng saved_models/ làm gốc cho OUTPUT_DIR
            self.assertNotRegex(
                content,
                r'OUTPUT_DIR="\$REPO/saved_models/',
                f"{relpath} không được lưu checkpoint vào saved_models/ pilot cũ (T1.6)"
            )

    def test_train_scripts_contain_ckpt_fail_fast_guard(self):
        """Xác minh cả 12 script train Tier 1 có cơ chế fail-fast khi $OUTPUT_DIR chứa .ckpt."""
        for relpath in TWELVE_TRAIN_SCRIPTS:
            path = os.path.join(REPO_ROOT, relpath)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn(
                'find "$OUTPUT_DIR" -name "*.ckpt" 2>/dev/null | grep -q .',
                content,
                f"{relpath} thiếu kiểm tra fail-fast find .ckpt (T1.6)"
            )
            self.assertIn(
                "exit 1",
                content,
                f"{relpath} phải exit 1 khi tìm thấy .ckpt trong OUTPUT_DIR (T1.6)"
            )

    def test_train_scripts_parameterize_max_epochs(self):
        """Xác minh cả 12 script train Tier 1 tham số hóa MAX_EPOCHS="${MAX_EPOCHS:-50}"."""
        for relpath in TWELVE_TRAIN_SCRIPTS:
            path = os.path.join(REPO_ROOT, relpath)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn(
                'MAX_EPOCHS="${MAX_EPOCHS:-50}"',
                content,
                f"{relpath} thiếu MAX_EPOCHS=\"${{MAX_EPOCHS:-50}}\" (T1.7)"
            )
            self.assertIn(
                '[INFO] MAX_EPOCHS: $MAX_EPOCHS',
                content,
                f"{relpath} thiếu log in ra [INFO] MAX_EPOCHS: $MAX_EPOCHS (T1.7)"
            )
            self.assertIn(
                '--max_epochs "$MAX_EPOCHS"',
                content,
                f"{relpath} phải truyền --max_epochs \"$MAX_EPOCHS\" vào lệnh python train (T1.7)"
            )
            # Không dùng hardcode --max_epochs 50 trong lệnh gọi python
            self.assertNotIn(
                "--max_epochs 50 ",
                content + " ",
                f"{relpath} vẫn còn hardcode --max_epochs 50 trong lệnh python"
            )

    def test_four_test_scripts_read_only_from_saved_models_55p(self):
        """Xác minh 4 script test chỉ đọc checkpoint từ gốc saved_models_55p/."""
        for relpath in FOUR_TEST_SCRIPTS:
            path = os.path.join(REPO_ROOT, relpath)
            self.assertTrue(os.path.exists(path), f"Thiếu script: {relpath}")
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn(
                "saved_models_55p/",
                content,
                f"{relpath} phải tham chiếu tới saved_models_55p/ (T1.6)"
            )
            # Không được tham chiếu saved_models/ (ngoại trừ comment giải thích nếu có)
            active_code = "\n".join(
                line for line in content.splitlines() if line.strip() and not line.strip().startswith("#")
            )
            self.assertNotIn(
                "saved_models/",
                active_code,
                f"{relpath} không được đọc checkpoint từ saved_models/ pilot cũ trong mã thực thi (T1.6)"
            )

    def test_four_test_scripts_output_to_reports_hpc_uit_55p(self):
        """Xác minh 4 script test tạo thư mục và ghi CSV vào reports/hpc-uit-55p/."""
        for relpath in FOUR_TEST_SCRIPTS:
            path = os.path.join(REPO_ROOT, relpath)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn(
                'mkdir -p "$REPO/reports/hpc-uit-55p"',
                content,
                f"{relpath} phải tạo thư mục reports/hpc-uit-55p (T1.6)"
            )
            self.assertIn(
                "reports/hpc-uit-55p/",
                content,
                f"{relpath} phải xuất CSV vào reports/hpc-uit-55p/ (T1.6)"
            )
            # Phải giữ nguyên cột/cờ job_id
            self.assertIn(
                '--job_id "${SLURM_JOB_ID:-local}"',
                content,
                f"{relpath} phải giữ nguyên cờ --job_id cho provenance"
            )

    def test_runtime_fail_fast_assertion(self):
        """Kiểm thử thực thi bash cho cơ chế fail-fast khi có và không có file .ckpt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = os.path.join(tmpdir, "test_model_dir")
            os.makedirs(ckpt_dir, exist_ok=True)

            bash_check = (
                'set -euo pipefail; '
                'OUTPUT_DIR="' + ckpt_dir + '"; '
                'if [ -d "$OUTPUT_DIR" ] && find "$OUTPUT_DIR" -name "*.ckpt" 2>/dev/null | grep -q .; then '
                '  echo "ERROR: found ckpt" >&2; exit 1; '
                'fi; '
                'exit 0'
            )
            # Chưa có .ckpt -> exit 0
            res1 = subprocess.run(["bash", "-c", bash_check], capture_output=True, text=True)
            self.assertEqual(res1.returncode, 0)

            # Tạo file .ckpt giả
            fake_ckpt = os.path.join(ckpt_dir, "epoch=10-val_psnr=32.0.ckpt")
            with open(fake_ckpt, "w") as f:
                f.write("fake")

            # Đã có .ckpt -> exit 1
            res2 = subprocess.run(["bash", "-c", bash_check], capture_output=True, text=True)
            self.assertEqual(res2.returncode, 1)
            self.assertIn("ERROR: found ckpt", res2.stderr)

    def test_runtime_max_epochs_override(self):
        """Kiểm thử thực thi bash cho việc nhận MAX_EPOCHS mặc định 50 và nhận giá trị override."""
        cmd_default = 'set -euo pipefail; MAX_EPOCHS="${MAX_EPOCHS:-50}"; echo "$MAX_EPOCHS"'
        res_def = subprocess.run(["bash", "-c", cmd_default], capture_output=True, text=True)
        self.assertEqual(res_def.returncode, 0)
        self.assertEqual(res_def.stdout.strip(), "50")

        cmd_override = 'set -euo pipefail; MAX_EPOCHS=10; MAX_EPOCHS="${MAX_EPOCHS:-50}"; echo "$MAX_EPOCHS"'
        res_over = subprocess.run(["bash", "-c", cmd_override], capture_output=True, text=True)
        self.assertEqual(res_over.returncode, 0)
        self.assertEqual(res_over.stdout.strip(), "10")


if __name__ == "__main__":
    unittest.main()
