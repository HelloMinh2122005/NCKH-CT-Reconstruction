#!/usr/bin/env python3
# ==============================================================================
# KIỂM THỬ ĐƠN VỊ CHO KỊCH BẢN SETUP_ENV_HPCUIT.SH
# Xác minh tính tất định, fail-fast, ghim phiên bản chính xác và khối kiểm tra.
# ==============================================================================

import os
import re
import unittest

SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "scripts", "setup_env_hpcuit.sh"
)


class TestSetupEnvHpcuit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(SCRIPT_PATH, "r", encoding="utf-8") as f:
            cls.script_content = f.read()

    def test_walltime_adequate(self):
        """Xác minh walltime đủ lớn (tối thiểu 90 phút, khuyến nghị 120 phút) cho biên dịch mamba-ssm."""
        match = re.search(r"#SBATCH\s+--time=([0-9:]+)", self.script_content)
        self.assertIsNotNone(match, "Thiếu chỉ thị #SBATCH --time")
        time_str = match.group(1)
        parts = [int(p) for p in time_str.split(":")]
        if len(parts) == 3:
            total_minutes = parts[0] * 60 + parts[1]
        elif len(parts) == 2:
            total_minutes = parts[0]
        else:
            total_minutes = 0
        self.assertGreaterEqual(
            total_minutes, 90, f"Walltime {time_str} quá ngắn cho biên dịch mamba-ssm từ source"
        )

    def test_exact_package_pins(self):
        """Xác minh các gói cốt lõi được ghim chính xác phiên bản với toán tử ==."""
        expected_pins = {
            "torch": "2.8.0+cu128",
            "torchvision": "0.23.0+cu128",
            "torchaudio": "2.8.0+cu128",
            "mamba-ssm": "2.3.1",
            "transformers": "4.57.6",
            "odl": "0.8.3",
            "astra-toolbox": "2.3.0",
            "torchmetrics": "1.8.2",
            "pydicom": "2.4.4",
            "future": "1.0.0",
            "nvidia-cuda-runtime-cu12": "12.8.90",
            "nvidia-cufft-cu12": "11.3.3.83",
        }
        for pkg, ver in expected_pins.items():
            pattern = rf'["\']?{re.escape(pkg)}=={re.escape(ver)}["\']?'
            self.assertRegex(
                self.script_content,
                pattern,
                f"Gói {pkg} chưa được ghim chính xác phiên bản =={ver}",
            )

        # Kiểm tra không còn dải phiên bản lỏng lẻo (>=, <=, ~=) trong các lệnh pip install
        for line in self.script_content.splitlines():
            line_s = line.strip()
            if line_s.startswith("#"):
                continue
            if "pip install" in line_s or line_s.startswith('"') or line_s.startswith("'"):
                self.assertNotRegex(
                    line_s,
                    r"(>=|<=|~=)",
                    f"Phát hiện dải phiên bản chưa ghim chặt chẽ: {line_s}",
                )

    def test_no_swallowed_failures_or_fallbacks(self):
        """Xác minh fail-fast: không nuốt lỗi qua '|| echo [WARN]' hoặc fallback không ghim."""
        self.assertNotIn('|| echo "[WARN]', self.script_content, "Tìm thấy '|| echo \"[WARN]' nuốt lỗi install")
        # Không có lệnh cài đặt nào bị theo sau bởi || echo hoặc || true
        for line in self.script_content.splitlines():
            line_s = line.strip()
            if "pip install" in line_s:
                self.assertNotRegex(
                    line_s,
                    r"\|\|\s*(?:echo|true|:)",
                    f"Lệnh cài đặt có đuôi nuốt lỗi: {line_s}",
                )
        # Không có fallback pip install sau toán tử ||
        self.assertNotRegex(
            self.script_content,
            r"\|\|\s*(?:pip|pip3|python\s+-m\s+pip)\s+install",
            "Tìm thấy fallback cài đặt qua toán tử ||",
        )

    def test_cuda_module_loaded_before_mamba(self):
        """Xác minh module load cuda12.9/toolkit/12.9.1 được gọi trước khi chạy lệnh pip install mamba-ssm."""
        module_load_pos = self.script_content.find("module load cuda12.9/toolkit/12.9.1")
        mamba_install_match = re.search(r'pip install ["\']mamba-ssm', self.script_content)
        self.assertNotEqual(
            module_load_pos, -1, "Thiếu lệnh 'module load cuda12.9/toolkit/12.9.1'"
        )
        self.assertIsNotNone(mamba_install_match, "Thiếu lệnh pip install mamba-ssm")
        self.assertLess(
            module_load_pos,
            mamba_install_match.start(),
            "Lệnh 'module load cuda12.9/toolkit/12.9.1' phải nằm trước khi thực thi pip install mamba-ssm",
        )

    def test_verification_block_components(self):
        """Xác minh khối kiểm tra cuối cùng kiểm tra PyTorch CUDA, mamba_ssm, ODL/ASTRA FanBeam, và pip freeze."""
        # 1. PyTorch CUDA op
        self.assertIn("torch.cuda.is_available()", self.script_content)
        # 2. mamba_ssm selective scan
        self.assertIn("mamba_ssm", self.script_content)
        self.assertIn("selective_scan", self.script_content)
        # 3. ODL + ASTRA FanBeam với src_radius=600, det_radius=290
        self.assertIn("FanBeamGeometry", self.script_content)
        self.assertIn("src_radius=600", self.script_content)
        self.assertIn("det_radius=290", self.script_content)
        # 4. pip freeze
        self.assertIn("pip freeze", self.script_content)

    def test_clean_venv_recreation(self):
        """Xác minh xóa sạch thư mục venv cũ (rm -rf $ENV_DIR) trước khi tạo mới để tránh venv hỏng."""
        rm_match = re.search(r'rm\s+-rf\s+["\']?\$ENV_DIR["\']?', self.script_content)
        venv_match = re.search(r'python3\s+-m\s+venv\s+["\']?\$ENV_DIR["\']?', self.script_content)
        self.assertIsNotNone(rm_match, "Thiếu lệnh xóa sạch venv cũ: rm -rf \"$ENV_DIR\"")
        self.assertIsNotNone(venv_match, "Thiếu lệnh tạo venv: python3 -m venv \"$ENV_DIR\"")
        self.assertLess(
            rm_match.start(),
            venv_match.start(),
            "Lệnh 'rm -rf \"$ENV_DIR\"' phải được thực thi trước 'python3 -m venv \"$ENV_DIR\"'",
        )

    def test_no_deps_protection(self):
        """Xác minh các gói astra-toolbox, odl và nvidia CUDA runtime được cài đặt với --no-deps để chống clobbering."""
        # Kiểm tra pip install --no-deps cho astra-toolbox và odl
        astra_no_deps = re.search(
            r'pip install --no-deps[\s\\]+["\']astra-toolbox==2\.3\.0["\'][\s\\]+["\']odl==0\.8\.3["\']',
            self.script_content,
        )
        self.assertIsNotNone(
            astra_no_deps,
            "Cài đặt astra-toolbox và odl phải sử dụng cờ --no-deps để chống unpin/downgrade nvidia runtime",
        )

        # Kiểm tra pip install --no-deps cho nvidia runtime wheels từ pytorch index
        nvidia_no_deps = re.search(
            r'pip install --no-deps[\s\\]+["\']nvidia-cuda-runtime-cu12==12\.8\.90["\'][\s\\]+["\']nvidia-cufft-cu12==11\.3\.3\.83["\'][\s\\]+--index-url\s+https://download\.pytorch\.org/whl/cu128',
            self.script_content,
        )
        self.assertIsNotNone(
            nvidia_no_deps,
            "Phải tái cài đặt nvidia-cuda-runtime-cu12 và nvidia-cufft-cu12 với --no-deps từ index PyTorch cu128",
        )


if __name__ == "__main__":
    unittest.main()

