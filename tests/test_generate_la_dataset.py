#!/usr/bin/env python3
"""
Unit tests cho scripts/generate_la_dataset.sh (Change 22: Selective Angle Dataset Generation):
- Kiểm tra khai báo biến môi trường TARGET_ANGLES với mặc định "90.0" (tiết kiệm hạn ngạch đĩa cho Wave A).
- Kiểm tra cấu trúc rẽ nhánh điều kiện cho khối sinh dữ liệu LA-120°.
- Kiểm tra cấu trúc rẽ nhánh điều kiện cho khối sinh dữ liệu LA-90°.
- Kiểm tra cờ --angles $TARGET_ANGLES được truyền vào evaluate_fbp.py trong khối đánh giá FBP.
- Kiểm tra cú pháp bash (bash -n) và slurm_lint --profile hpc-uit.
"""

import os
import re
import subprocess
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT_PATH = os.path.join(REPO_ROOT, "scripts", "generate_la_dataset.sh")


class TestGenerateLaDatasetScript(unittest.TestCase):
    def setUp(self):
        self.assertTrue(os.path.exists(SCRIPT_PATH), f"Không tìm thấy kịch bản {SCRIPT_PATH}")
        with open(SCRIPT_PATH, "r", encoding="utf-8") as f:
            self.content = f.read()

    def test_target_angles_default_declaration(self):
        """Xác minh biến TARGET_ANGLES được khai báo với giá trị mặc định là '90.0'."""
        match = re.search(r'TARGET_ANGLES="\$\{TARGET_ANGLES:-90\.0\}"', self.content)
        self.assertIsNotNone(
            match,
            "Không tìm thấy khai báo TARGET_ANGLES=\"${TARGET_ANGLES:-90.0}\" trong script."
        )

    def test_selective_angle_120_condition(self):
        """Xác minh khối tạo dữ liệu LA-120° được bao bọc trong điều kiện kiểm tra TARGET_ANGLES."""
        self.assertIn(
            'TARGET_ANGLES',
            self.content,
            "Script cần sử dụng biến TARGET_ANGLES để kiểm tra góc 120°."
        )
        # Kiểm tra điều kiện regex hoặc chuỗi cho góc 120
        has_120_condition = re.search(r'if\s+\[\[.*TARGET_ANGLES.*120.*\]\];\s*then', self.content)
        self.assertIsNotNone(
            has_120_condition,
            "Khối sinh dữ liệu LA-120° phải được bảo vệ bằng câu lệnh điều kiện kiểm tra TARGET_ANGLES."
        )
        self.assertIn(
            "Bỏ qua LA-120° theo cấu hình TARGET_ANGLES",
            self.content,
            "Phải có thông báo echo bỏ qua LA-120° khi TARGET_ANGLES không chứa 120."
        )

    def test_selective_angle_90_condition(self):
        """Xác minh khối tạo dữ liệu LA-90° được bao bọc trong điều kiện kiểm tra TARGET_ANGLES."""
        has_90_condition = re.search(r'if\s+\[\[.*TARGET_ANGLES.*90.*\]\];\s*then', self.content)
        self.assertIsNotNone(
            has_90_condition,
            "Khối sinh dữ liệu LA-90° phải được bảo vệ bằng câu lệnh điều kiện kiểm tra TARGET_ANGLES."
        )
        self.assertIn(
            "Bỏ qua LA-90° theo cấu hình TARGET_ANGLES",
            self.content,
            "Phải có thông báo echo bỏ qua LA-90° khi TARGET_ANGLES không chứa 90."
        )

    def test_evaluate_fbp_receives_target_angles(self):
        """Xác minh lệnh gọi evaluate_fbp nhận tham số --angles $TARGET_ANGLES."""
        match = re.search(r'--angles\s+\$TARGET_ANGLES', self.content)
        self.assertIsNotNone(
            match,
            "Lệnh gọi evaluate_fbp.py / evaluate_fbp_la.py phải nhận tham số '--angles $TARGET_ANGLES'."
        )

    def test_script_bash_syntax(self):
        """Xác minh cú pháp bash của script hợp lệ qua bash -n."""
        proc = subprocess.run(["bash", "-n", SCRIPT_PATH], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, f"Lỗi cú pháp bash:\n{proc.stderr}")

    def test_script_slurm_lint(self):
        """Xác minh script vượt qua công cụ slurm_lint --profile hpc-uit."""
        lint_tool = os.path.abspath(os.path.join(REPO_ROOT, "..", "..", ".claude", "tools", "slurm_lint.py"))
        if os.path.exists(lint_tool):
            proc = subprocess.run(
                [sys.executable, lint_tool, "--profile", "hpc-uit", SCRIPT_PATH],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, f"slurm_lint thất bại:\n{proc.stdout}\n{proc.stderr}")


if __name__ == "__main__":
    unittest.main()
