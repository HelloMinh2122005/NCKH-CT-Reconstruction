#!/usr/bin/env python3
# ==============================================================================
# KIỂM THỬ ĐƠN VỊ TDD CHO T0.7 (HYGIENE 29 KỊCH BẢN SBATCH NON-LIDC HPC-UIT)
# Kiểm tra:
#   1. #SBATCH --output và --error trỏ về scripts/output-hpc-uit/<stem>/log/%j.{out,err}
#   2. Không còn bất kỳ tham chiếu nào tới thư mục cũ scripts/output/
#   3. In git rev-parse HEAD ở phần đầu log job (trước khi chạy Python)
#   4. Có set -euo pipefail và cú pháp bash -n hợp lệ
#   5. slurm_lint.py --profile hpc-uit PASS 29/29 (exit 0, 0 lỗi, 0 cảnh báo)
#   6. smoke_hpcuit.sh truyền --allow_custom_split cho subset <55 bệnh nhân
#   7. smoke_wave_c_55p_hpcuit.sh bắt ngay tệp 0-byte (<1000B) cho cả DICOM và cache_only .npy
# ==============================================================================

import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(REPO_ROOT, "..", ".."))
LINTER_PATH = os.path.join(WORKSPACE_ROOT, ".claude", "tools", "slurm_lint.py")

TARGET_29_SCRIPTS = [
    "benchmark_latency_hpcuit.sh",
    "generate_la_dataset_120.sh",
    "generate_la_dataset.sh",
    "hello_world.sh",
    "probe_hpcuit.sh",
    "setup_env_hpcuit.sh",
    "smoke_hpcuit.sh",
    "smoke_wave_c_55p_hpcuit.sh",
    "test_benchmark_hpcuit.sh",
    "test_dudotrans_la.sh",
    "test_fbp_la.sh",
    "test_learn_la.sh",
    "test_regformer_la.sh",
    "test_solar_dualmamba_la.sh",
    "test_solar_last_ckpt_hpcuit.sh",
    "test_wave_b_ablation_hpcuit.sh",
    "test_wave_c_120_hpcuit.sh",
    "train_dudotrans_la_120.sh",
    "train_dudotrans_la.sh",
    "train_learn_la_120.sh",
    "train_learn_la.sh",
    "train_regformer_la_120.sh",
    "train_regformer_la.sh",
    "train_solar_dualmamba_la_120.sh",
    "train_solar_dualmamba_la_ablate_a.sh",
    "train_solar_dualmamba_la_ablate_b.sh",
    "train_solar_dualmamba_la_ablate_c.sh",
    "train_solar_dualmamba_la_ablate_d.sh",
    "train_solar_dualmamba_la.sh",
]


def _read_script(filename: str) -> str:
    path = os.path.join(REPO_ROOT, "scripts", filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _extract_smoke_wave_c_preflight_py(content: str) -> str:
    m = re.search(r"python -u - <<'PYEOF'\n(.*?)\nPYEOF", content, re.DOTALL)
    assert m is not None, "Không tìm thấy khối Python preflight <<'PYEOF' trong smoke_wave_c_55p_hpcuit.sh"
    return m.group(1)


class TestSbatchT07Conventions(unittest.TestCase):
    """Kiểm thử hợp đồng T0.7 cho toàn bộ 29 kịch bản Slurm non-LIDC trên HPC-UIT."""

    def test_all_29_target_scripts_exist(self):
        """Xác minh đủ 29 kịch bản Slurm mục tiêu đều tồn tại trong scripts/."""
        self.assertEqual(len(TARGET_29_SCRIPTS), 29)
        for name in TARGET_29_SCRIPTS:
            path = os.path.join(REPO_ROOT, "scripts", name)
            self.assertTrue(os.path.isfile(path), f"Thiếu kịch bản mục tiêu: {path}")

    def test_sbatch_output_and_error_use_output_hpc_uit(self):
        """Mọi script trong 29 script phải ghi #SBATCH --output và --error vào scripts/output-hpc-uit/<stem>/log/%j.{out,err}."""
        for name in TARGET_29_SCRIPTS:
            stem = os.path.splitext(name)[0]
            content = _read_script(name)
            want_out = (
                f"#SBATCH --output=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/"
                f"scripts/output-hpc-uit/{stem}/log/%j.out"
            )
            want_err = (
                f"#SBATCH --error=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/"
                f"scripts/output-hpc-uit/{stem}/log/%j.err"
            )
            self.assertIn(want_out, content, f"{name} thiếu chỉ thị {want_out}")
            self.assertIn(want_err, content, f"{name} thiếu chỉ thị {want_err}")

    def test_zero_legacy_scripts_output_references(self):
        """Không một script nào trong 29 script được còn tham chiếu tới thư mục cũ scripts/output/."""
        for name in TARGET_29_SCRIPTS:
            content = _read_script(name)
            self.assertNotIn(
                "scripts/output/",
                content,
                f"{name} vẫn còn tham chiếu tới thư mục log cũ 'scripts/output/'",
            )

    def test_git_rev_parse_head_printed_at_top_of_log(self):
        """Mọi script trong 29 script phải in git rev-parse HEAD ở đầu log trước khi chạy Python."""
        for name in TARGET_29_SCRIPTS:
            content = _read_script(name)
            lines = content.splitlines()
            git_echo_lines = [
                idx
                for idx, line in enumerate(lines, 1)
                if not line.strip().startswith("#")
                and "echo" in line
                and "rev-parse HEAD" in line
            ]
            self.assertTrue(
                git_echo_lines,
                f"{name} không in git rev-parse HEAD trong phần thân thực thi",
            )
            first_git_line = git_echo_lines[0]
            python_lines = [
                idx
                for idx, line in enumerate(lines, 1)
                if not line.strip().startswith("#")
                and re.match(r"^\s*python(?:3)?\b", line)
            ]
            if python_lines:
                self.assertLess(
                    first_git_line,
                    python_lines[0],
                    f"{name} phải in git rev-parse HEAD (dòng {first_git_line}) trước lệnh python đầu tiên (dòng {python_lines[0]})",
                )

    def test_set_euo_pipefail_and_bash_syntax_valid(self):
        """Mọi script trong 29 script phải bật set -euo pipefail và hợp lệ theo bash -n."""
        for name in TARGET_29_SCRIPTS:
            path = os.path.join(REPO_ROOT, "scripts", name)
            content = _read_script(name)
            self.assertRegex(
                content,
                r"(?m)^\s*set -euo pipefail\b",
                f"{name} thiếu 'set -euo pipefail'",
            )
            res = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"bash -n thất bại trên {name}: {res.stderr}")

    def test_slurm_lint_strict_pass_all_29(self):
        """Xác minh slurm_lint --profile hpc-uit trên cả 29 script đạt exit 0, 0 lỗi và 0 cảnh báo."""
        self.assertTrue(os.path.isfile(LINTER_PATH), f"Không tìm thấy slurm_lint.py tại {LINTER_PATH}")
        script_paths = [os.path.join(REPO_ROOT, "scripts", name) for name in TARGET_29_SCRIPTS]
        res = subprocess.run(
            [sys.executable, LINTER_PATH, "--profile", "hpc-uit", "--json", *script_paths],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0, f"slurm_lint thất bại:\n{res.stdout}\n{res.stderr}")
        payload = json.loads(res.stdout)
        self.assertEqual(payload["profile"], "hpc-uit")
        self.assertEqual(payload["passed"], 29)
        self.assertEqual(payload["failed"], 0)
        for item in payload["results"]:
            stem = os.path.splitext(os.path.basename(item["script"]))[0]
            self.assertEqual(
                item["warnings"],
                [],
                f"{stem}.sh phát sinh cảnh báo slurm_lint: {item['warnings']}",
            )
            self.assertEqual(
                item["errors"],
                [],
                f"{stem}.sh có lỗi slurm_lint: {item['errors']}",
            )

    def test_smoke_hpcuit_passes_allow_custom_split(self):
        """smoke_hpcuit.sh huấn luyện trên subset <55 bệnh nhân nên phải truyền --allow_custom_split cho cả LEARN và SOLAR."""
        content = _read_script("smoke_hpcuit.sh")
        count = content.count("--allow_custom_split")
        self.assertEqual(
            count,
            2,
            f"smoke_hpcuit.sh gọi train_learn_la.py và train_solar_dualmamba_la.py với tập nhỏ nên phải truyền --allow_custom_split đúng 2 lần (hiện có: {count})",
        )

    def test_smoke_wave_c_55p_preflight_catches_zero_byte_dicom_and_cache_npy(self):
        """Preflight trong smoke_wave_c_55p_hpcuit.sh phải bắt tệp 0-byte ở cả chế độ DICOM và cache_only .npy trước khi __getitem__(0)."""
        content = _read_script("smoke_wave_c_55p_hpcuit.sh")
        self.assertIn("os.path.getsize", content, "Thiếu kiểm tra os.path.getsize cho T0.5 trong smoke_wave_c_55p_hpcuit.sh")
        preflight_code = _extract_smoke_wave_c_preflight_py(content)

        # Kiểm thử thực thi thực tế khối preflight với các tệp thật trên đĩa tạm
        with tempfile.TemporaryDirectory() as tmpdir:
            stub_root = os.path.join(tmpdir, "stubs")
            os.makedirs(os.path.join(stub_root, "data"), exist_ok=True)

            with open(os.path.join(stub_root, "torch.py"), "w", encoding="utf-8") as f:
                f.write(
                    textwrap.dedent(
                        """
                        import numpy as np
                        class _Tensor:
                            def __init__(self, arr):
                                self._arr = np.asarray(arr, dtype=np.float32)
                                self.shape = self._arr.shape
                        def from_numpy(arr):
                            return _Tensor(arr)
                        def isfinite(t):
                            class _Mask:
                                def __init__(self, ok):
                                    self._ok = ok
                                def all(self):
                                    return self._ok
                            return _Mask(bool(np.isfinite(t._arr).all()))
                        """
                    )
                )

            with open(os.path.join(stub_root, "data", "__init__.py"), "w", encoding="utf-8") as f:
                f.write("")

            with open(os.path.join(stub_root, "data", "datamodule_factory.py"), "w", encoding="utf-8") as f:
                f.write(
                    textwrap.dedent(
                        """
                        import json
                        import os
                        import numpy as np
                        import torch

                        class _StubDataset:
                            def __init__(self, slices_path, use_precomputed=True, sino_dir=None, fbp_dir=None):
                                self.slices_path = slices_path
                                self.use_precomputed = use_precomputed
                                self.sino_dir = sino_dir
                                self.fbp_dir = fbp_dir
                            def __len__(self):
                                return len(self.slices_path)
                            def __getitem__(self, idx):
                                return (
                                    torch.from_numpy(np.zeros((1, 256, 256), dtype=np.float32)),
                                    torch.from_numpy(np.zeros((1, 256, 256), dtype=np.float32)),
                                    torch.from_numpy(np.zeros((1, 64, 512), dtype=np.float32)),
                                )

                        class _StubDM:
                            def __init__(self, cfg):
                                self.train_dataset = _StubDataset(**cfg["train"])
                                self.val_dataset = _StubDataset(**cfg["val"])
                                self.test_dataset = _StubDataset(**cfg["test"])
                            def setup(self, stage=None):
                                pass

                        def get_datamodule(**kwargs):
                            with open(os.environ["STUB_CFG_JSON"], "r", encoding="utf-8") as fh:
                                return _StubDM(json.load(fh))
                        """
                    )
                )

            train_patients = (
                "L004 L006 L012 L014 L019 L024 L027 L030 L033 L035 "
                "L036 L043 L044 L045 L048 L049 L056 L057 L071 L072 "
                "L075 L077 L079 L081 L082 L111 L113 L114 L116 L121 "
                "L123 L125 L127 L129 L131 L133 L134 L136 L138 L143 "
                "L144 L145 L147 L148 L150 L154 L160 L163 L166 L170 "
                "L171 L172 L175 L178 L179"
            )

            def _run_preflight_case(case_name: str, mode: str, zero_target: str | None):
                case_dir = os.path.join(tmpdir, case_name)
                sino_dir = os.path.join(case_dir, "sino")
                fbp_dir = os.path.join(case_dir, "fbp_u")
                raw_dir = os.path.join(case_dir, "raw")
                split_dir = os.path.join(case_dir, "split")
                gt_dir = os.path.join(case_dir, "gt_size_256")
                for d in (sino_dir, fbp_dir, raw_dir, split_dir, gt_dir):
                    os.makedirs(d, exist_ok=True)

                def _write_sized(path: str, size: int):
                    with open(path, "wb") as fh:
                        fh.write(b"\x01" * size)

                def _make_split(n_slices: int, prefix: str):
                    paths = []
                    for i in range(n_slices):
                        stem = f"{prefix}_{i:05d}"
                        sino_p = os.path.join(sino_dir, f"{stem}.npy")
                        fbp_p = os.path.join(fbp_dir, f"{stem}.npy")
                        _write_sized(sino_p, 0 if (zero_target == "sino" and i == 1 and prefix == "train") else 2048)
                        _write_sized(fbp_p, 2048)
                        if mode == "dicom":
                            dcm_p = os.path.join(raw_dir, f"{stem}.dcm")
                            ima_p = os.path.join(split_dir, f"{stem}.IMA")
                            _write_sized(dcm_p, 0 if (zero_target == "slice" and i == 1 and prefix == "train") else 2048)
                            os.symlink(dcm_p, ima_p)
                            paths.append(ima_p)
                        else:
                            gt_p = os.path.join(gt_dir, f"{stem}.npy")
                            _write_sized(gt_p, 0 if (zero_target == "slice" and i == 1 and prefix == "train") else 2048)
                            paths.append(gt_p)
                    return paths

                cfg = {
                    "train": {"slices_path": _make_split(8218, "train"), "use_precomputed": True, "sino_dir": sino_dir, "fbp_dir": fbp_dir},
                    "val": {"slices_path": _make_split(133, "val"), "use_precomputed": True, "sino_dir": sino_dir, "fbp_dir": fbp_dir},
                    "test": {"slices_path": _make_split(757, "test"), "use_precomputed": True, "sino_dir": sino_dir, "fbp_dir": fbp_dir},
                }
                cfg_path = os.path.join(case_dir, "cfg.json")
                with open(cfg_path, "w", encoding="utf-8") as fh:
                    json.dump(cfg, fh)

                env = os.environ.copy()
                env["PYTHONPATH"] = stub_root
                env["STUB_CFG_JSON"] = cfg_path
                env["SMOKE_DICOM_DIR"] = split_dir
                env["SMOKE_CACHE_DIR"] = case_dir
                env["SMOKE_TRAIN_PATIENTS"] = train_patients
                env["SMOKE_VAL_PATIENTS"] = "L110"
                env["SMOKE_TEST_PATIENTS"] = "L058 L064 L094 L107"
                return subprocess.run(
                    [sys.executable, "-c", preflight_code],
                    env=env,
                    capture_output=True,
                    text=True,
                )

            # 1. DICOM 0-byte ở lát cắt thứ 1 (index != 0) phải bị bắt ngay trước __getitem__(0)
            res_dcm_0 = _run_preflight_case("case_dcm_0byte", "dicom", "slice")
            self.assertNotEqual(res_dcm_0.returncode, 0, "Preflight phải thất bại khi có DICOM symlink trỏ tới tệp 0-byte")
            self.assertIn("AssertionError", res_dcm_0.stderr)

            # 2. cache_only .npy 0-byte ở gt và ở sino phải bị bắt ngay
            res_gt_0 = _run_preflight_case("case_gt_0byte", "cache_only", "slice")
            self.assertNotEqual(res_gt_0.returncode, 0, "Preflight phải thất bại khi có gt .npy 0-byte")
            self.assertIn("AssertionError", res_gt_0.stderr)

            res_sino_0 = _run_preflight_case("case_sino_0byte", "cache_only", "sino")
            self.assertNotEqual(res_sino_0.returncode, 0, "Preflight phải thất bại khi có sino .npy 0-byte")
            self.assertIn("AssertionError", res_sino_0.stderr)

            # 3. Cả 2 chế độ DICOM và cache_only hợp lệ đều phải PASS
            res_dcm_ok = _run_preflight_case("case_dcm_ok", "dicom", None)
            self.assertEqual(res_dcm_ok.returncode, 0, f"Preflight DICOM hợp lệ thất bại: {res_dcm_ok.stderr}")
            self.assertIn("[PREFLIGHT OK]", res_dcm_ok.stdout)

            res_cache_ok = _run_preflight_case("case_cache_ok", "cache_only", None)
            self.assertEqual(res_cache_ok.returncode, 0, f"Preflight cache_only hợp lệ thất bại: {res_cache_ok.stderr}")
            self.assertIn("[PREFLIGHT OK]", res_cache_ok.stdout)


if __name__ == "__main__":
    unittest.main()
