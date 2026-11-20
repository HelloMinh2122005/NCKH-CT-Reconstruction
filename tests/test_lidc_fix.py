#!/usr/bin/env python3
"""
Unit tests cho Change 30: Sửa chữa tính toàn vẹn dữ liệu LIDC-IDRI,
bộ nạp DICOM an toàn và pipeline sinh cache hai dải góc (LA-90° và LA-120°):
1. Kiểm tra phát hiện và loại bỏ tệp DICOM 0-byte, tệp bị cắt cụt/hỏng trong download_lidc_subset.py.
2. Kiểm tra cơ chế ghi tệp nguyên tử (.tmp -> .dcm) và thử lại (retry with backoff).
3. Kiểm tra CTSlice_Provider_LA_LIDC: tôn trọng TransferSyntaxUID (force=False trước),
   xử lý ngoại lệ pixel_array hỏng một cách tường minh, không làm sập tiến trình.
4. Kiểm tra prepare_data_sinogram_LA_LIDC: ghi npy nguyên tử, cô lập lát cắt lỗi vào corrupted_slices.log,
   và cấu hình hình học fan-beam locked (src_radius=600.0, det_radius=290.0).
5. Kiểm tra hai kịch bản Slurm HPC-UIT: download_lidc_hpcuit.sh và generate_la_dataset_lidc.sh (cả 90° và 120°).
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
# Khôi phục PIL thật nếu bị mock trước đó bởi các test khác trong bộ suite
if "PIL" in sys.modules and (isinstance(sys.modules["PIL"], MagicMock) or not hasattr(sys.modules["PIL"], "__file__")):
    del sys.modules["PIL"]
    if "PIL.Image" in sys.modules:
        del sys.modules["PIL.Image"]
try:
    import PIL
    import PIL.Image
except ImportError:
    pass
if "torch" not in sys.modules:
    torch_mock = types.ModuleType("torch")
    torch_utils = types.ModuleType("torch.utils")
    torch_utils_data = types.ModuleType("torch.utils.data")
    torch_utils_data.Dataset = object
    torch_mock.utils = torch_utils
    torch_utils.data = torch_utils_data
    torch_mock.Tensor = MagicMock
    torch_mock.from_numpy = lambda arr: MagicMock(
        squeeze=lambda: MagicMock(
            cpu=lambda: MagicMock(numpy=lambda: arr),
            numpy=lambda: arr
        ),
        unsqueeze=lambda dim: MagicMock(
            float=lambda: MagicMock(unsqueeze=lambda d2: MagicMock(float=lambda: MagicMock()))
        ),
        float=lambda: MagicMock(
            unsqueeze=lambda dim: MagicMock(float=lambda: MagicMock())
        )
    )
    torch_mock.cuda = MagicMock()
    torch_mock.cuda.is_available.return_value = False
    sys.modules["torch"] = torch_mock
    sys.modules["torch.utils"] = torch_utils
    sys.modules["torch.utils.data"] = torch_utils_data

for mod in ["torchvision", "torchvision.transforms", "odl", "odl.contrib", "odl.contrib.torch"]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

# Giả lập pydicom nếu chưa cài đặt trong môi trường cục bộ
if "pydicom" not in sys.modules:
    pydicom_mock = types.ModuleType("pydicom")
    pydicom_errors = types.ModuleType("pydicom.errors")
    class MockInvalidDicomError(Exception):
        pass
    pydicom_errors.InvalidDicomError = MockInvalidDicomError
    pydicom_mock.errors = pydicom_errors
    pydicom_mock.dcmread = MagicMock()
    sys.modules["pydicom"] = pydicom_mock
    sys.modules["pydicom.errors"] = pydicom_errors
else:
    import pydicom
    if not hasattr(pydicom, "errors"):
        pydicom.errors = types.ModuleType("pydicom.errors")
    if not hasattr(pydicom.errors, "InvalidDicomError"):
        class MockInvalidDicomError(Exception):
            pass
        pydicom.errors.InvalidDicomError = MockInvalidDicomError

# Giả lập tqdm
tqdm_mock = types.ModuleType("tqdm")
tqdm_mock.tqdm = lambda x, **kwargs: x
sys.modules["tqdm"] = tqdm_mock


class TestLidcDownloadAndValidation(unittest.TestCase):
    """
    Kiểm thử cho module data/lidc_idri/download_lidc_subset.py
    """
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.patient_dir = os.path.join(self.test_dir.name, "LIDC-IDRI-0001")
        os.makedirs(self.patient_dir, exist_ok=True)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_validate_dicom_file_rejects_zero_byte(self):
        """Tệp 0 byte phải bị từ chối bởi validate_dicom_file."""
        from data.lidc_idri.download_lidc_subset import validate_dicom_file
        zero_file = os.path.join(self.patient_dir, "zero.dcm")
        with open(zero_file, "wb") as f:
            pass
        self.assertFalse(validate_dicom_file(zero_file))

    def test_validate_dicom_file_rejects_under_100kb(self):
        """Tệp có kích thước < 100 KB (ví dụ 80 KB tệp cụt Job 12792) phải bị từ chối."""
        from data.lidc_idri.download_lidc_subset import validate_dicom_file
        small_file = os.path.join(self.patient_dir, "small.dcm")
        with open(small_file, "wb") as f:
            f.write(b"\x00" * (80 * 1024))
        self.assertFalse(validate_dicom_file(small_file))

    def test_validate_dicom_file_rejects_corrupted_pixel_array(self):
        """Tệp kích thước > 100KB nhưng pydicom ném ValueError khi đọc pixel_array phải bị từ chối."""
        from data.lidc_idri.download_lidc_subset import validate_dicom_file
        corrupt_file = os.path.join(self.patient_dir, "corrupt.dcm")
        with open(corrupt_file, "wb") as f:
            f.write(b"\x00" * (150 * 1024))

        mock_ds = MagicMock()
        type(mock_ds).pixel_array = unittest.mock.PropertyMock(
            side_effect=ValueError("The length of the pixel data doesn't match expected")
        )
        with patch("pydicom.dcmread", return_value=mock_ds):
            self.assertFalse(validate_dicom_file(corrupt_file))

    def test_validate_dicom_file_accepts_valid(self):
        """Tệp kích thước hợp lệ và giải mã pixel_array thành công phải được chấp nhận."""
        from data.lidc_idri.download_lidc_subset import validate_dicom_file
        valid_file = os.path.join(self.patient_dir, "valid.dcm")
        with open(valid_file, "wb") as f:
            f.write(b"\x00" * (526 * 1024))

        mock_ds = MagicMock()
        mock_ds.pixel_array = np.zeros((512, 512), dtype=np.int16)
        with patch("pydicom.dcmread", return_value=mock_ds):
            self.assertTrue(validate_dicom_file(valid_file))

    def test_clean_corrupt_dicoms(self):
        """Hàm clean_corrupt_dicoms phải xóa các file 0-byte và file hỏng, giữ lại file hợp lệ."""
        from data.lidc_idri.download_lidc_subset import clean_corrupt_dicoms

        f_zero = os.path.join(self.patient_dir, "001.dcm")
        with open(f_zero, "wb") as f:
            pass  # 0 bytes

        f_small = os.path.join(self.patient_dir, "002.dcm")
        with open(f_small, "wb") as f:
            f.write(b"\x00" * 500)

        f_valid = os.path.join(self.patient_dir, "003.dcm")
        with open(f_valid, "wb") as f:
            f.write(b"\x00" * (526 * 1024))

        mock_ds = MagicMock()
        mock_ds.pixel_array = np.zeros((512, 512), dtype=np.int16)

        def mock_read(path, **kwargs):
            if "003.dcm" in path:
                return mock_ds
            raise ValueError("Corrupt file")

        with patch("pydicom.dcmread", side_effect=mock_read):
            remaining = clean_corrupt_dicoms(self.patient_dir)
            self.assertFalse(os.path.exists(f_zero), "Tệp 0-byte không được phép tồn tại.")
            self.assertFalse(os.path.exists(f_small), "Tệp < 100KB không được phép tồn tại.")
            self.assertTrue(os.path.exists(f_valid), "Tệp hợp lệ phải được giữ lại.")
            self.assertEqual(remaining, 1)

    def test_download_series_atomic_write_and_retry(self):
        """Kiểm tra cơ chế retry khi mạng lỗi và ghi tệp nguyên tử (.tmp -> .dcm)."""
        import io
        import zipfile
        from data.lidc_idri.download_lidc_subset import download_series

        patient_info = {
            "patient_id": "LIDC-IDRI-0001",
            "series_uid": "1.2.3.4.5",
            "split": "train"
        }

        # Tạo gói zip bộ nhớ đệm chứa 1 file dcm 526KB
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("slice_0001.dcm", b"\x00" * (526 * 1024))
        valid_zip_bytes = zip_buffer.getvalue()

        # Giả lập lần 1 gặp timeout, lần 2 thành công
        call_count = [0]
        def mock_urlopen(req, timeout=180):
            call_count[0] += 1
            if call_count[0] == 1:
                import urllib.error
                raise urllib.error.URLError("Connection timed out (TCIA NBIA)")
            resp = MagicMock()
            resp.read.return_value = valid_zip_bytes
            resp.__enter__.return_value = resp
            resp.__exit__.return_value = None
            return resp

        mock_ds = MagicMock()
        mock_ds.pixel_array = np.zeros((512, 512), dtype=np.int16)

        with patch("urllib.request.urlopen", side_effect=mock_urlopen), \
             patch("pydicom.dcmread", return_value=mock_ds), \
             patch("time.sleep", return_value=None):
            count = download_series(patient_info, self.test_dir.name, max_retries=3)
            self.assertEqual(count, 1)
            self.assertEqual(call_count[0], 2, "Hàm phải retry sau lần đầu timeout.")

            target_file = os.path.join(self.patient_dir, "slice_0001.dcm")
            self.assertTrue(os.path.exists(target_file))
            tmp_file = target_file + ".tmp"
            self.assertFalse(os.path.exists(tmp_file), "Tệp tạm .tmp phải được chuyển nguyên tử.")

    def test_main_exits_on_zero_valid_slices(self):
        """main() trong download_lidc_subset.py phải thoát mã khác 0 nếu test hoặc val cohort có 0 lát cắt."""
        import argparse
        from data.lidc_idri.download_lidc_subset import main
        with patch("argparse.ArgumentParser.parse_args", return_value=argparse.Namespace(output_dir=self.test_dir.name, max_retries=1)), \
             patch("data.lidc_idri.download_lidc_subset.download_series", return_value=0), \
             patch("sys.exit") as mock_exit:
            main()
            self.assertTrue(mock_exit.called)
            self.assertEqual(mock_exit.call_args[0][0], 1)


class TestCTSliceProviderLIDC(unittest.TestCase):
    """
    Kiểm thử tính phòng vệ của bộ nạp dữ liệu data/lidc_idri/CTSlice_Provider_LA_LIDC.py
    """
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.dcm_file = os.path.join(self.test_dir.name, "test_slice.dcm")
        with open(self.dcm_file, "wb") as f:
            f.write(b"MOCK_DICOM_DATA" * 100)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_geometry_fanbeam_src_det_radius(self):
        """Hình học FanBeam phải được thiết lập đúng src_radius=600.0 và det_radius=290.0."""
        from data.lidc_idri.CTSlice_Provider_LA_LIDC import LimitedAngleCT_LIDC_Provider

        provider = LimitedAngleCT_LIDC_Provider.__new__(LimitedAngleCT_LIDC_Provider)
        provider.start_ang = -np.pi / 3
        provider.end_ang = np.pi / 3
        provider.num_view = 64
        provider.num_detectors = 512

        with patch("odl.tomo.FanBeamGeometry") as mock_geom, \
             patch("odl.uniform_discr"), \
             patch("odl.uniform_partition"), \
             patch("odl.tomo.RayTransform"), \
             patch("odl.tomo.fbp_op"), \
             patch("odl.contrib.torch.operator.OperatorModule"):
            provider._init_odl_geometry()
            self.assertTrue(mock_geom.called)
            kwargs = mock_geom.call_args[1]
            self.assertEqual(kwargs.get("src_radius"), 600.0)
            self.assertEqual(kwargs.get("det_radius"), 290.0)

    def test_force_false_by_default_and_fallback(self):
        """Provider phải gọi dcmread(force=False) trước, chỉ fallback sang force=True khi gặp InvalidDicomError."""
        from data.lidc_idri.CTSlice_Provider_LA_LIDC import LimitedAngleCT_LIDC_Provider

        provider = LimitedAngleCT_LIDC_Provider.__new__(LimitedAngleCT_LIDC_Provider)
        provider.input_size = 256

        mock_ds = MagicMock()
        mock_ds.RescaleSlope = 1.0
        mock_ds.RescaleIntercept = -1024.0
        mock_ds.pixel_array = np.zeros((512, 512), dtype=np.float32)

        calls = []
        def mock_dcmread(path, force=False, **kwargs):
            calls.append(force)
            if force is False:
                import pydicom.errors
                raise pydicom.errors.InvalidDicomError("Missing preamble")
            return mock_ds

        with patch("pydicom.dcmread", side_effect=mock_dcmread):
            arr = provider._read_and_process_dicom(self.dcm_file)
            self.assertEqual(calls, [False, True], "Phải gọi force=False trước, rồi fallback sang force=True.")
            self.assertEqual(arr.shape, (256, 256))

    def test_descriptive_error_on_corrupt_pixel_data(self):
        """Khi pixel_array ném ngoại lệ ValueError (như lỗi 79750 bytes), provider phải báo lỗi mô tả rõ ràng."""
        from data.lidc_idri.CTSlice_Provider_LA_LIDC import LimitedAngleCT_LIDC_Provider

        provider = LimitedAngleCT_LIDC_Provider.__new__(LimitedAngleCT_LIDC_Provider)
        provider.input_size = 256

        mock_ds = MagicMock()
        type(mock_ds).pixel_array = unittest.mock.PropertyMock(
            side_effect=ValueError("The length of the pixel data in the dataset (79750 bytes) doesn't match expected (524288 bytes)")
        )

        with patch("pydicom.dcmread", return_value=mock_ds):
            with self.assertRaises(ValueError) as ctx:
                provider._read_and_process_dicom(self.dcm_file)
            self.assertIn("Không thể giải mã pixel_array", str(ctx.exception))
            self.assertIn(self.dcm_file, str(ctx.exception))

    def test_getitem_graceful_fallback_when_not_return_path(self):
        """Khi return_path=False (chế độ nạp huấn luyện/đánh giá), nếu lát cắt hỏng, provider tự động fallback sang lát cắt liền kề."""
        from data.lidc_idri.CTSlice_Provider_LA_LIDC import LimitedAngleCT_LIDC_Provider

        provider = LimitedAngleCT_LIDC_Provider.__new__(LimitedAngleCT_LIDC_Provider)
        provider.slices_path = ["/path/to/LIDC-IDRI-0001/corrupt.dcm", "/path/to/LIDC-IDRI-0001/valid.dcm"]
        provider.patients = ["LIDC-IDRI-0001"]
        provider.return_path = False
        provider.use_precomputed = True
        provider.sino_cache_dir = self.test_dir.name
        provider.fbp_cache_dir = self.test_dir.name

        valid_arr = np.ones((256, 256), dtype=np.float32)

        def mock_read(path):
            if "corrupt.dcm" in path:
                raise ValueError("Corrupt slice")
            return valid_arr

        provider._read_and_process_dicom = mock_read

        # Tạo cache giả lập cho valid slice
        np.save(os.path.join(self.test_dir.name, "LIDC-IDRI-0001_valid.npy"), np.zeros((64, 512), dtype=np.float32))

        phantom, fbp, sino = provider[0]
        self.assertIsNotNone(phantom)
        self.assertIsNotNone(fbp)
        self.assertIsNotNone(sino)


class TestPrepareDataSinogramAtomicAndIsolation(unittest.TestCase):
    """
    Kiểm thử tính nguyên tử và cô lập lỗi trong data/lidc_idri/prepare_data_sinogram_LA_LIDC.py
    """
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.output_base = os.path.join(self.test_dir.name, "output")
        self.dicom_dir = os.path.join(self.test_dir.name, "dicom")
        os.makedirs(self.output_base, exist_ok=True)
        os.makedirs(self.dicom_dir, exist_ok=True)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_save_atomic_npy(self):
        """Hàm save_atomic_npy phải ghi dữ liệu nguyên tử qua .tmp và gọi fsync."""
        from data.lidc_idri.prepare_data_sinogram_LA_LIDC import save_atomic_npy
        target = os.path.join(self.output_base, "sample.npy")
        arr = np.ones((64, 512), dtype=np.float32)

        with patch("os.fsync") as mock_fsync:
            save_atomic_npy(target, arr)
            self.assertTrue(os.path.exists(target))
            self.assertFalse(os.path.exists(target + ".tmp"))
            self.assertTrue(mock_fsync.called)

        loaded = np.load(target)
        np.testing.assert_array_equal(loaded, arr)

    def test_is_valid_cache_file(self):
        """is_valid_cache_file phải trả về True chỉ khi cả 2 file đều >= 1000 bytes."""
        from data.lidc_idri.prepare_data_sinogram_LA_LIDC import is_valid_cache_file
        sino = os.path.join(self.output_base, "sino.npy")
        fbp = os.path.join(self.output_base, "fbp.npy")

        self.assertFalse(is_valid_cache_file(sino, fbp))

        with open(sino, "wb") as f:
            f.write(b"\x00" * 500)
        with open(fbp, "wb") as f:
            f.write(b"\x00" * 1500)
        self.assertFalse(is_valid_cache_file(sino, fbp))

        with open(sino, "wb") as f:
            f.write(b"\x00" * 1500)
        self.assertTrue(is_valid_cache_file(sino, fbp))

    def test_corrupted_slice_isolation_to_log(self):
        """Lát cắt bị lỗi không được làm sập tiến trình, phải ghi nhận vào corrupted_slices.log."""
        from data.lidc_idri.prepare_data_sinogram_LA_LIDC import generate_la_lidc_split

        mock_provider = MagicMock()
        mock_provider.patients = ["LIDC-IDRI-0001"]
        mock_provider.slices_path = [
            "/path/to/LIDC-IDRI-0001/001.dcm",
            "/path/to/LIDC-IDRI-0001/002.dcm"
        ]
        mock_provider.__len__.return_value = 2

        # Lát 0 bị lỗi corrupt pixel data, lát 1 thành công
        def mock_getitem(idx):
            if idx == 0:
                raise ValueError("Corrupt pixel data 79750 bytes")
            return (
                mock_provider.slices_path[1],
                MagicMock(),
                MagicMock(squeeze=lambda: MagicMock(cpu=lambda: MagicMock(numpy=lambda: np.zeros((256, 256), dtype=np.float32)))),
                MagicMock(squeeze=lambda: MagicMock(cpu=lambda: MagicMock(numpy=lambda: np.zeros((64, 512), dtype=np.float32))))
            )

        mock_provider.__getitem__.side_effect = mock_getitem

        with patch("data.lidc_idri.prepare_data_sinogram_LA_LIDC.LimitedAngleCT_LIDC_Provider", return_value=mock_provider):
            generate_la_lidc_split(
                split_name="test",
                dicom_dir=self.dicom_dir,
                output_base_dir=self.output_base,
                angle_range_deg=120.0,
                num_view=64,
                num_detectors=512,
                input_size=256
            )

        log_file = os.path.join(self.output_base, "corrupted_slices.log")
        self.assertTrue(os.path.exists(log_file), "corrupted_slices.log phải được tạo ra khi có lát cắt lỗi.")
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("001.dcm", content)
        self.assertIn("Corrupt pixel data 79750 bytes", content)


class TestSlurmScriptsConventions(unittest.TestCase):
    """
    Kiểm tra tính hợp chuẩn của 2 script Slurm:
    1. scripts/download_lidc_hpcuit.sh
    2. scripts/generate_la_dataset_lidc.sh
    """
    def setUp(self):
        self.download_script = os.path.join(REPO_ROOT, "scripts", "download_lidc_hpcuit.sh")
        self.gen_script = os.path.join(REPO_ROOT, "scripts", "generate_la_dataset_lidc.sh")

    def test_scripts_exist_and_executable(self):
        """Cả 2 script phải tồn tại."""
        self.assertTrue(os.path.exists(self.download_script))
        self.assertTrue(os.path.exists(self.gen_script))

    def test_download_script_checks_zero_bytes(self):
        """download_lidc_hpcuit.sh phải kiểm tra tệp 0-byte và thoát lỗi nếu phát hiện."""
        with open(self.download_script, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("#SBATCH --partition=small", content)
        self.assertIn("#SBATCH --gres=gpu:1g.18gb:1", content)
        self.assertIn("#SBATCH --mem=8G", content)
        self.assertIn("#SBATCH --time=01:00:00", content)
        self.assertIn("ZERO_COUNT", content, "Script phải đếm số file 0 byte.")
        self.assertIn("-size 0", content, "Script phải dùng find -size 0 để phát hiện file rỗng.")

    def test_generate_script_covers_both_90_and_120(self):
        """generate_la_dataset_lidc.sh phải cấu hình tạo sinh cả hai dải góc 90° và 120°."""
        with open(self.gen_script, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("#SBATCH --partition=small", content)
        self.assertIn("#SBATCH --gres=gpu:1g.18gb:1", content)
        self.assertIn("#SBATCH --mem=16G", content)
        self.assertIn("TARGET_ANGLES", content, "Script phải khai báo biến TARGET_ANGLES.")
        self.assertIn("90.0", content)
        self.assertIn("120.0", content)

    def test_slurm_lint_passes(self):
        """Cả 2 script download_lidc_hpcuit.sh và generate_la_dataset_lidc.sh phải PASS slurm_lint --profile hpc-uit."""
        import subprocess
        lint_tool = os.path.join(REPO_ROOT, "..", "..", ".claude", "tools", "slurm_lint.py")
        cmd = [sys.executable, lint_tool, "--profile", "hpc-uit", self.download_script, self.gen_script]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(res.returncode, 0, f"slurm_lint thất bại:\n{res.stdout}\n{res.stderr}")
        self.assertIn("pass=2 fail=0", res.stdout)


if __name__ == "__main__":
    unittest.main()
