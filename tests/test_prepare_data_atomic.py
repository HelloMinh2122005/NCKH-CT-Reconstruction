#!/usr/bin/env python3
"""
Unit tests cho cơ chế Atomic Caching, Guard phát hiện file hỏng và tinh chỉnh cấu hình Slurm Memory:
- Kiểm tra hàm lưu mảng numpy nguyên tử save_atomic_npy (ghi .tmp rồi os.replace).
- Kiểm tra guard phát hiện file hỏng (< 1000 bytes / header-only 128 bytes do NFS write stall).
- Kiểm tra generate_split bỏ qua file hợp lệ và tự động sinh lại file bị hỏng hoặc bị thiếu.
- Kiểm tra cơ chế thu dọn rác định kỳ (gc.collect và torch.cuda.empty_cache sau mỗi 500 lát cắt).
- Kiểm tra các chỉ thị #SBATCH --mem=8G và #SBATCH --mem=16G trong 7 script Slurm.
"""

import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Đảm bảo import được module từ temp-github/NCKH-CT-Reconstruction
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Giả lập các thư viện cluster-only để kiểm thử có thể chạy thuần Python tại môi trường cục bộ
if "torch" not in sys.modules:
    torch_mock = types.ModuleType("torch")
    torch_utils = types.ModuleType("torch.utils")
    torch_utils_data = types.ModuleType("torch.utils.data")
    torch_utils_data.Dataset = object
    torch_mock.utils = torch_utils
    torch_utils.data = torch_utils_data
    torch_mock.Tensor = MagicMock
    torch_mock.cuda = MagicMock()
    torch_mock.cuda.is_available.return_value = False
    sys.modules["torch"] = torch_mock
    sys.modules["torch.utils"] = torch_utils
    sys.modules["torch.utils.data"] = torch_utils_data

# Giả lập các thư viện phụ thuộc cluster-only
for mod in ["torchvision", "torchvision.transforms", "pydicom", "odl", "odl.contrib", "odl.contrib.torch", "PIL"]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

# Đảm bảo tqdm trả về đối tượng có thể lặp (iterable)
tqdm_mock = types.ModuleType("tqdm")
tqdm_mock.tqdm = lambda x, **kwargs: x
sys.modules["tqdm"] = tqdm_mock

from data.aapm.prepare_data_sinogram_LA import generate_split, is_valid_cache_file, save_atomic_npy


class TestAtomicCacheAndCorruptFileGuard(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.sino_dir = os.path.join(self.test_dir.name, "sino")
        self.fbp_dir = os.path.join(self.test_dir.name, "fbp_u")
        os.makedirs(self.sino_dir, exist_ok=True)
        os.makedirs(self.fbp_dir, exist_ok=True)

    def tearDown(self):
        self.test_dir.cleanup()

    @patch("os.fsync")
    def test_save_atomic_npy(self, mock_fsync):
        """
        Kiểm tra hàm save_atomic_npy ghi dữ liệu an toàn:
        - Tệp đích tồn tại với đúng dữ liệu mảng numpy.
        - Tệp tạm thời .tmp không còn tồn tại sau khi hoàn tất.
        - os.fsync được kích hoạt để đảm bảo dirty pages commit xuống storage NFS.
        """
        target_path = os.path.join(self.sino_dir, "test_slice.npy")
        tmp_path = target_path + ".tmp"
        data = np.random.rand(64, 512).astype(np.float32)

        save_atomic_npy(target_path, data)

        self.assertTrue(os.path.exists(target_path), "Tệp đích không tồn tại sau khi lưu nguyên tử.")
        self.assertFalse(os.path.exists(tmp_path), "Tệp tạm thời .tmp vẫn còn sót lại sau khi lưu.")
        self.assertTrue(mock_fsync.called, "os.fsync phải được gọi để đảm bảo dữ liệu ghi bền vững xuống NFS.")

        loaded_data = np.load(target_path)
        np.testing.assert_allclose(loaded_data, data, err_msg="Dữ liệu mảng sau khi tải không khớp với dữ liệu gốc.")

    def test_is_valid_cache_file_scenarios(self):
        """
        Kiểm tra các kịch bản kiểm tra tính toàn vẹn của tệp cache:
        - Cả hai tệp không tồn tại -> False
        - Một trong hai tệp không tồn tại -> False
        - Cả hai tệp tồn tại nhưng kích thước < 1000 bytes (ví dụ 128 bytes do NFS stall) -> False
        - Một tệp >= 1000 bytes và một tệp < 1000 bytes -> False
        - Cả hai tệp tồn tại và có kích thước >= 1000 bytes -> True
        """
        sino_file = os.path.join(self.sino_dir, "slice_001.npy")
        fbp_file = os.path.join(self.fbp_dir, "slice_001.npy")

        # 1. Cả 2 tệp đều chưa tồn tại
        self.assertFalse(is_valid_cache_file(sino_file, fbp_file))

        # 2. Tạo tệp giả 128 byte (mô phỏng file bị cắt cụt do NFS write stall trong Job 9629)
        fake_corrupted_content = b"\x93NUMPY" + b"\x00" * 122
        with open(sino_file, "wb") as f:
            f.write(fake_corrupted_content)
        with open(fbp_file, "wb") as f:
            f.write(fake_corrupted_content)

        self.assertEqual(os.path.getsize(sino_file), 128)
        self.assertEqual(os.path.getsize(fbp_file), 128)
        self.assertFalse(is_valid_cache_file(sino_file, fbp_file), "Phải phát hiện file 128 byte là không hợp lệ.")

        # 3. Một file hợp lệ (>= 1000 byte), một file bị cắt cụt
        valid_array = np.zeros((64, 512), dtype=np.float32)
        save_atomic_npy(sino_file, valid_array)
        self.assertGreaterEqual(os.path.getsize(sino_file), 1000)
        self.assertFalse(is_valid_cache_file(sino_file, fbp_file), "Một trong hai file < 1000 byte phải trả về False.")

        # 4. Cả hai file đều hợp lệ (>= 1000 byte)
        save_atomic_npy(fbp_file, valid_array)
        self.assertGreaterEqual(os.path.getsize(fbp_file), 1000)
        self.assertTrue(is_valid_cache_file(sino_file, fbp_file), "Cả hai file >= 1000 byte phải trả về True.")

    def test_generate_split_skips_valid_and_regenerates_corrupt(self):
        """
        Kiểm tra generate_split:
        - Bỏ qua lát cắt khi cache hợp lệ (size >= 1000 bytes).
        - Tính toán lại và ghi đè lát cắt khi cache bị hỏng (< 1000 bytes) hoặc thiếu.
        """
        mock_provider = MagicMock()
        mock_provider.__len__.return_value = 3
        mock_provider.slices_path = [
            "/path/to/L001_slice0.IMA",
            "/path/to/L001_slice1.IMA",
            "/path/to/L001_slice2.IMA",
        ]

        # Slice 0: Cache hợp lệ đầy đủ (> 1000 bytes)
        valid_arr = np.ones((64, 512), dtype=np.float32)
        save_atomic_npy(os.path.join(self.sino_dir, "L001_slice0.npy"), valid_arr)
        save_atomic_npy(os.path.join(self.fbp_dir, "L001_slice0.npy"), valid_arr)

        # Slice 1: Cache bị hỏng (128 bytes)
        with open(os.path.join(self.sino_dir, "L001_slice1.npy"), "wb") as f:
            f.write(b"0" * 128)
        with open(os.path.join(self.fbp_dir, "L001_slice1.npy"), "wb") as f:
            f.write(b"0" * 128)

        # Slice 2: Hoàn toàn chưa có cache
        # (Không tạo file)

        # Trả về dữ liệu giả khi provider[i] được gọi
        dummy_tensor = np.zeros((64, 512), dtype=np.float32)
        mock_provider.__getitem__.return_value = ("path", None, dummy_tensor, dummy_tensor)

        generate_split(mock_provider, self.sino_dir, self.fbp_dir)

        # Provider chỉ được truy vấn cho Slice 1 (tái sinh do hỏng) và Slice 2 (sinh mới)
        called_indices = [call.args[0] for call in mock_provider.__getitem__.call_args_list]
        self.assertNotIn(0, called_indices, "Slice 0 có cache hợp lệ không được gọi tính toán lại.")
        self.assertIn(1, called_indices, "Slice 1 bị hỏng 128 bytes phải được gọi tính toán tái sinh.")
        self.assertIn(2, called_indices, "Slice 2 chưa có cache phải được gọi tính toán.")

        # Sau khi hoàn tất, Slice 1 phải được ghi đè thành công và có kích thước hợp lệ
        self.assertGreaterEqual(os.path.getsize(os.path.join(self.sino_dir, "L001_slice1.npy")), 1000)
        self.assertGreaterEqual(os.path.getsize(os.path.join(self.fbp_dir, "L001_slice1.npy")), 1000)

    @patch("gc.collect")
    def test_periodic_garbage_collection(self, mock_gc_collect):
        """
        Kiểm tra cơ chế thu dọn rác định kỳ: gc.collect() được gọi sau mỗi 500 lát cắt.
        """
        mock_provider = MagicMock()
        mock_provider.__len__.return_value = 1005
        mock_provider.slices_path = [f"/path/to/slice_{i}.IMA" for i in range(1005)]

        dummy_arr = np.zeros((10, 10), dtype=np.float32)
        mock_provider.__getitem__.return_value = ("path", None, dummy_arr, dummy_arr)

        generate_split(mock_provider, self.sino_dir, self.fbp_dir)

        # Với 1005 lát cắt, tại lát 500 và 1000 phải gọi gc.collect() ít nhất 2 lần
        self.assertGreaterEqual(mock_gc_collect.call_count, 2, "gc.collect() phải được gọi ít nhất 2 lần khi duyệt 1005 lát cắt.")


class TestSlurmScriptMemorySettings(unittest.TestCase):
    """
    Kiểm tra cấu hình bộ nhớ (#SBATCH --mem) của 7 kịch bản Slurm nhằm ngăn chặn cgroup memory starvation:
    - Nhóm sinh dữ liệu & kiểm thử: 8G (giảm từ 32G)
    - Nhóm huấn luyện mô hình: 16G (giảm từ 32G)
    """

    def _get_mem_directive(self, script_relpath: str) -> str:
        script_path = os.path.join(REPO_ROOT, script_relpath)
        self.assertTrue(os.path.exists(script_path), f"Không tìm thấy kịch bản {script_path}")
        with open(script_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str.startswith("#SBATCH") and "--mem=" in line_str:
                    return line_str.split("--mem=")[1].split()[0]
        self.fail(f"Không tìm thấy chỉ thị #SBATCH --mem trong {script_relpath}")

    def test_small_partition_scripts_mem_8g(self):
        """Xác minh các kịch bản nhẹ / sinh dữ liệu / benchmark được cấu hình --mem=8G."""
        self.assertEqual(self._get_mem_directive("scripts/generate_la_dataset.sh"), "8G")
        self.assertEqual(self._get_mem_directive("scripts/smoke_hpcuit.sh"), "8G")

    def test_train_scripts_mem_16g(self):
        """Xác minh các kịch bản huấn luyện mô hình được cấu hình --mem=16G."""
        self.assertEqual(self._get_mem_directive("scripts/train_solar_dualmamba_la.sh"), "16G")
        self.assertEqual(self._get_mem_directive("scripts/train_learn_la.sh"), "16G")
        self.assertEqual(self._get_mem_directive("scripts/train_dudotrans_la.sh"), "16G")
        self.assertEqual(self._get_mem_directive("scripts/train_regformer_la.sh"), "16G")
        self.assertEqual(self._get_mem_directive("scripts/test_benchmark_hpcuit.sh"), "16G")

    def test_benchmark_normal_partition_directives(self):
        """Xác minh test_benchmark_hpcuit.sh cấu hình đúng phân vùng normal (Change 24)."""
        script_path = os.path.join(REPO_ROOT, "scripts/test_benchmark_hpcuit.sh")
        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("#SBATCH --partition=normal", content)
        self.assertIn("#SBATCH --gres=gpu:2g.35gb:1", content)
        self.assertIn("#SBATCH --mem=16G", content)
        self.assertIn("#SBATCH --cpus-per-task=8", content)

    def test_no_obsolete_qos_in_scripts(self):
        """Xác minh không có kịch bản nào chứa chỉ thị --qos lỗi thời."""
        scripts = [
            "scripts/generate_la_dataset.sh",
            "scripts/smoke_hpcuit.sh",
            "scripts/test_benchmark_hpcuit.sh",
            "scripts/train_solar_dualmamba_la.sh",
            "scripts/train_learn_la.sh",
            "scripts/train_dudotrans_la.sh",
            "scripts/train_regformer_la.sh",
        ]
        for relpath in scripts:
            script_path = os.path.join(REPO_ROOT, relpath)
            with open(script_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertNotRegex(
                content,
                r"#SBATCH\s+--qos=",
                f"Kịch bản {relpath} vẫn còn chỉ thị #SBATCH --qos (đã lỗi thời trên HPC-UIT)."
            )

    def test_trap_cleanup_order(self):
        """Xác minh trap cleanup EXIT được khai báo trước mọi lệnh I/O như mkdir -p."""
        scripts = [
            "scripts/generate_la_dataset.sh",
            "scripts/smoke_hpcuit.sh",
            "scripts/test_benchmark_hpcuit.sh",
            "scripts/train_solar_dualmamba_la.sh",
            "scripts/train_learn_la.sh",
            "scripts/train_dudotrans_la.sh",
            "scripts/train_regformer_la.sh",
        ]
        for relpath in scripts:
            script_path = os.path.join(REPO_ROOT, relpath)
            with open(script_path, "r", encoding="utf-8") as f:
                content = f.read()
            trap_pos = content.find("trap cleanup EXIT")
            mkdir_pos = content.find("mkdir -p")
            self.assertNotEqual(trap_pos, -1, f"Không tìm thấy 'trap cleanup EXIT' trong {relpath}")
            if mkdir_pos != -1:
                self.assertLess(
                    trap_pos,
                    mkdir_pos,
                    f"Trong {relpath}, 'trap cleanup EXIT' (vị trí {trap_pos}) phải đứng trước 'mkdir -p' (vị trí {mkdir_pos})."
                )


if __name__ == "__main__":
    unittest.main()
