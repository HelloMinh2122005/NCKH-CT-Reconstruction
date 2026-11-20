#!/usr/bin/env python3
"""
Unit tests cho T0.4 (Cache-only LimitedAngleCT_Provider & GT .npy generation)
và T0.5 (Guard chống ghi output_dir vào raw/ hoặc split/).

Ghi chú môi trường cục bộ:
- Khi torch / torchvision / pydicom / odl chưa cài đặt tại máy local, module test này
  cài đặt stub nhẹ tạm thời (chỉ trong phạm vi module test và khôi phục sys.modules khi xong)
  để chạy toàn bộ logic thuần Python + NumPy thực trên đĩa.
"""

import importlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class _FakeTensor:
    """Tensor giả lập gọn nhẹ bọc quanh numpy.ndarray thật phục vụ kiểm thử không có PyTorch."""

    def __init__(self, array: np.ndarray):
        self._arr = np.asarray(array)

    @property
    def shape(self):
        return self._arr.shape

    @property
    def ndim(self):
        return self._arr.ndim

    @property
    def dtype(self):
        return self._arr.dtype

    def float(self):
        return _FakeTensor(self._arr.astype(np.float32))

    def unsqueeze(self, dim: int):
        return _FakeTensor(np.expand_dims(self._arr, axis=dim))

    def cpu(self):
        return self

    def detach(self):
        return self

    def numpy(self):
        return self._arr

    def __array__(self, dtype=None):
        if dtype is not None:
            return self._arr.astype(dtype)
        return self._arr


def _build_fake_torch_stack():
    """Tạo bộ module giả lập tối thiểu cho torch, torchvision, odl, pydicom, tqdm."""
    torch_mod = types.ModuleType("torch")
    torch_utils = types.ModuleType("torch.utils")
    torch_utils_data = types.ModuleType("torch.utils.data")
    torch_utils_data.Dataset = object
    torch_mod.utils = torch_utils
    torch_utils.data = torch_utils_data
    torch_mod.Tensor = _FakeTensor
    torch_mod.from_numpy = lambda arr: _FakeTensor(np.asarray(arr))
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    torch_mod.randn_like = lambda t: _FakeTensor(np.zeros_like(t.numpy(), dtype=np.float32))

    tv_mod = types.ModuleType("torchvision")
    tv_transforms = types.ModuleType("torchvision.transforms")

    class _FakeResize:
        def __init__(self, size):
            self.size = tuple(size)

        def __call__(self, tensor):
            arr = tensor.numpy() if isinstance(tensor, _FakeTensor) else np.asarray(tensor)
            if arr.shape[-2:] == self.size:
                return _FakeTensor(arr)
            # Co giãn đơn giản bằng lặp/lấy mẫu để kiểm thử logic shape trong môi trường không có torchvision
            c = arr.shape[0] if arr.ndim == 3 else 1
            h_idx = np.linspace(0, arr.shape[-2] - 1, self.size[0]).astype(int)
            w_idx = np.linspace(0, arr.shape[-1] - 1, self.size[1]).astype(int)
            arr_3d = arr if arr.ndim == 3 else arr[None, ...]
            resized = arr_3d[:, h_idx[:, None], w_idx[None, :]]
            return _FakeTensor(resized.astype(np.float32))

    class _FakeCompose:
        def __init__(self, transforms_list):
            self.transforms = transforms_list

        def __call__(self, x):
            for t in self.transforms:
                x = t(x)
            return x

    tv_transforms.Resize = _FakeResize
    tv_transforms.Compose = _FakeCompose
    tv_mod.transforms = tv_transforms

    pydicom_mod = types.ModuleType("pydicom")
    pydicom_mod.read_file = MagicMock(side_effect=AssertionError("pydicom.read_file must not be called in stub default"))

    odl_mod = types.ModuleType("odl")
    odl_contrib = types.ModuleType("odl.contrib")
    odl_contrib_torch = types.ModuleType("odl.contrib.torch")
    odl_mod.contrib = odl_contrib
    odl_contrib.torch = odl_contrib_torch

    tqdm_mod = types.ModuleType("tqdm")
    tqdm_mod.tqdm = lambda x, **kwargs: x

    return {
        "torch": torch_mod,
        "torch.utils": torch_utils,
        "torch.utils.data": torch_utils_data,
        "torchvision": tv_mod,
        "torchvision.transforms": tv_transforms,
        "pydicom": pydicom_mod,
        "odl": odl_mod,
        "odl.contrib": odl_contrib,
        "odl.contrib.torch": odl_contrib_torch,
        "tqdm": tqdm_mod,
    }


_SAVED_MODULES = None
_USING_FAKE_TORCH = False


def _is_real_module(mod) -> bool:
    return mod is not None and hasattr(mod, "__file__") and mod.__file__ is not None


def setUpModule():
    global _SAVED_MODULES, _USING_FAKE_TORCH
    _SAVED_MODULES = sys.modules.copy()
    real_torch = _is_real_module(sys.modules.get("torch"))
    if not real_torch:
        try:
            import torch  # noqa: F401
            real_torch = _is_real_module(sys.modules.get("torch"))
        except ImportError:
            real_torch = False

    if not real_torch:
        _USING_FAKE_TORCH = True
        stubs = _build_fake_torch_stack()
        for name, mod in stubs.items():
            sys.modules[name] = mod
        # Xóa các module dữ liệu đã nạp trước đó với mock khác để nạp lại sạch sẽ
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("data.CTSlice_Provider_LA") or mod_name.startswith("data.aapm") or mod_name == "data.prepare_data_sinogram_LA" or mod_name == "scripts.evaluate_fbp":
                del sys.modules[mod_name]
        if "data" in sys.modules:
            for attr in ("CTSlice_Provider_LA", "aapm", "prepare_data_sinogram_LA"):
                if hasattr(sys.modules["data"], attr):
                    delattr(sys.modules["data"], attr)
    else:
        _USING_FAKE_TORCH = False


def tearDownModule():
    global _SAVED_MODULES
    if _SAVED_MODULES is not None:
        for name in list(sys.modules.keys()):
            if name not in _SAVED_MODULES:
                del sys.modules[name]
        for name, mod in _SAVED_MODULES.items():
            sys.modules[name] = mod
        _SAVED_MODULES = None


def _write_npy(path: str, array: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, array.astype(np.float32))


class TestCacheOnlyProviderAndPrepareData(unittest.TestCase):
    SETTING_TAG = "limited_ang_120deg_numview_64_size_256_noise_0"

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.cache_dir = os.path.join(self.tmp_dir.name, "limited_angle")
        self.dicom_dir = os.path.join(self.tmp_dir.name, "split")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _populate_cached_slice(
        self,
        split_name: str,
        stem: str,
        gt_subpath: str = "gt_size_256",
        shape_gt=(1, 256, 256),
        shape_fbp=(1, 256, 256),
        shape_sino=(1, 64, 512),
        seed_val: float = 0.25,
    ):
        gt_arr = np.full(shape_gt, seed_val, dtype=np.float32)
        fbp_arr = np.full(shape_fbp, seed_val + 0.1, dtype=np.float32)
        sino_arr = np.full(shape_sino, seed_val + 0.2, dtype=np.float32)

        base_setting = os.path.join(self.cache_dir, split_name, self.SETTING_TAG)
        _write_npy(os.path.join(base_setting, "sino", f"{stem}.npy"), sino_arr)
        _write_npy(os.path.join(base_setting, "fbp_u", f"{stem}.npy"), fbp_arr)
        _write_npy(os.path.join(self.cache_dir, split_name, gt_subpath, f"{stem}.npy"), gt_arr)
        return gt_arr, fbp_arr, sino_arr

    def _create_dangling_symlink(self, split_name: str, patient_id: str, filename: str) -> str:
        full_3mm = os.path.join(self.dicom_dir, split_name, patient_id, "full_3mm")
        os.makedirs(full_3mm, exist_ok=True)
        link_path = os.path.join(full_3mm, filename)
        missing_target = os.path.join(self.tmp_dir.name, "raw_deleted", patient_id, "full_3mm", "00000001.dcm")
        os.symlink(missing_target, link_path)
        return link_path

    def test_provider_files_are_identical(self):
        """Đảm bảo data/CTSlice_Provider_LA.py và data/aapm/CTSlice_Provider_LA.py giống nhau từng byte."""
        p_root = Path(REPO_ROOT) / "data" / "CTSlice_Provider_LA.py"
        p_aapm = Path(REPO_ROOT) / "data" / "aapm" / "CTSlice_Provider_LA.py"
        self.assertEqual(p_root.read_bytes(), p_aapm.read_bytes())

    def test_cache_only_init_without_dicom_or_with_dangling_symlinks(self):
        """Kịch bản 1: Khởi tạo cache-only thành công khi dicom_dir không tồn tại hoặc toàn symlink gãy."""
        from data.CTSlice_Provider_LA import LimitedAngleCT_Provider

        self._populate_cached_slice("train", "L004_FD_3_1.CT.00000001", seed_val=0.11)
        self._populate_cached_slice("train", "L004_FD_3_1.CT.00000002", seed_val=0.12)
        self._populate_cached_slice("train", "L110_FD_3_1.CT.00000001", seed_val=0.21)
        self._populate_cached_slice("test", "L058_FD_3_1.CT.00000001", seed_val=0.31)
        self._populate_cached_slice("test", "L058_FD_3_1.CT.00000002", seed_val=0.32)

        # (a) dicom_dir hoàn toàn không tồn tại
        missing_dicom = os.path.join(self.tmp_dir.name, "nonexistent_split")
        p_train = LimitedAngleCT_Provider(
            dicom_dir=missing_dicom,
            cache_dir=self.cache_dir,
            use_precomputed=True,
            precomputed_setting=self.SETTING_TAG,
            patient_list=["L004"],
            test=False,
            valid=False,
        )
        p_val = LimitedAngleCT_Provider(
            dicom_dir=missing_dicom,
            cache_dir=self.cache_dir,
            use_precomputed=True,
            precomputed_setting=self.SETTING_TAG,
            patient_list=["L110"],
            test=False,
            valid=False,
        )
        p_test = LimitedAngleCT_Provider(
            dicom_dir=missing_dicom,
            cache_dir=self.cache_dir,
            use_precomputed=True,
            precomputed_setting=self.SETTING_TAG,
            patient_list=["L058"],
            test=True,
            valid=False,
        )

        self.assertTrue(p_train.cache_only)
        self.assertTrue(p_val.cache_only)
        self.assertTrue(p_test.cache_only)
        self.assertEqual(len(p_train), 2)
        self.assertEqual(len(p_val), 1)
        self.assertEqual(len(p_test), 2)
        self.assertTrue(all(os.path.exists(p) for p in p_train.slices_path))
        self.assertTrue(all(os.path.exists(p) for p in p_val.slices_path))
        self.assertTrue(all(os.path.exists(p) for p in p_test.slices_path))

        # (b) dicom_dir chứa symlink gãy trỏ tới raw đã bị xóa
        self._create_dangling_symlink("train", "L004", "L004_FD_3_1.CT.00000001.IMA")
        p_dangling = LimitedAngleCT_Provider(
            dicom_dir=self.dicom_dir,
            cache_dir=self.cache_dir,
            use_precomputed=True,
            precomputed_setting=self.SETTING_TAG,
            patient_list=["L004"],
            test=False,
            valid=False,
        )
        self.assertTrue(p_dangling.cache_only)
        self.assertEqual(len(p_dangling), 2)
        self.assertTrue(all(os.path.exists(p) for p in p_dangling.slices_path))

    def test_cache_only_getitem_never_calls_pydicom(self):
        """Kịch bản 2: __getitem__ nạp đúng phantom/fbp_u/sino từ .npy và tuyệt đối không gọi pydicom.read_file."""
        import data.CTSlice_Provider_LA as provider_mod
        from data.CTSlice_Provider_LA import LimitedAngleCT_Provider

        gt_exp, fbp_exp, sino_exp = self._populate_cached_slice(
            "test", "L094_FD_3_1.CT.1.2.3.18", seed_val=0.42
        )
        # Kiểm tra cả trường hợp cache 2D (chưa có trục channel) được unsqueeze lên (1, H, W)
        self._populate_cached_slice(
            "test",
            "L094_FD_3_1.CT.1.2.3.19",
            shape_gt=(256, 256),
            shape_fbp=(256, 256),
            shape_sino=(64, 512),
            seed_val=0.77,
        )

        with patch.object(provider_mod.pydicom, "read_file", side_effect=AssertionError("must not read DICOM")) as mock_read:
            provider = LimitedAngleCT_Provider(
                dicom_dir=os.path.join(self.tmp_dir.name, "nonexistent_split"),
                cache_dir=self.cache_dir,
                use_precomputed=True,
                precomputed_setting=self.SETTING_TAG,
                patient_list=["L094"],
                test=True,
                return_path=False,
            )
            phantom, fbp_u, sino = provider[0]
            self.assertEqual(tuple(phantom.shape), (1, 256, 256))
            self.assertEqual(tuple(fbp_u.shape), (1, 256, 256))
            self.assertEqual(tuple(sino.shape), (1, 64, 512))
            np.testing.assert_allclose(np.asarray(phantom), gt_exp, rtol=1e-6)
            np.testing.assert_allclose(np.asarray(fbp_u), fbp_exp, rtol=1e-6)
            np.testing.assert_allclose(np.asarray(sino), sino_exp, rtol=1e-6)

            provider.return_path = True
            slice_path, phantom2, fbp_u2, sino2 = provider[1]
            self.assertTrue(slice_path.endswith("L094_FD_3_1.CT.1.2.3.19.npy"))
            self.assertEqual(tuple(phantom2.shape), (1, 256, 256))
            self.assertEqual(tuple(fbp_u2.shape), (1, 256, 256))
            self.assertEqual(tuple(sino2.shape), (1, 64, 512))
            self.assertEqual(mock_read.call_count, 0)

    def test_cache_only_paths_compatible_with_eval_utils_and_fbp(self):
        """Kịch bản 3: Đường dẫn .npy trả về bởi cache-only provider tương thích eval_utils & evaluate_fbp."""
        from data.CTSlice_Provider_LA import LimitedAngleCT_Provider
        from baselines.eval_utils import parse_patient_and_slice_idx
        from scripts.evaluate_fbp import extract_patient_id_from_path

        self._populate_cached_slice("test", "L058_FD_3_1.CT.00000042", seed_val=0.33)
        provider = LimitedAngleCT_Provider(
            dicom_dir=os.path.join(self.tmp_dir.name, "nonexistent_split"),
            cache_dir=self.cache_dir,
            use_precomputed=True,
            precomputed_setting=self.SETTING_TAG,
            patient_list=["L058"],
            test=True,
            return_path=True,
        )
        slice_path, _, _, _ = provider[0]
        pid, s_idx = parse_patient_and_slice_idx(slice_path)
        self.assertEqual((pid, s_idx), ("L058", 42))
        self.assertEqual(extract_patient_id_from_path(slice_path, ["L058", "L064"]), "L058")

    def test_cache_only_patient_prefix_disjoint_train_val(self):
        """Kịch bản 4: Lọc tiền tố bệnh nhân chặt chẽ (L1 không khớp L10; L004 và L110 tách biệt hoàn toàn)."""
        from data.CTSlice_Provider_LA import LimitedAngleCT_Provider

        self._populate_cached_slice("train", "L1_FD_3_1.CT.00000001")
        self._populate_cached_slice("train", "L10_FD_3_1.CT.00000001")
        self._populate_cached_slice("train", "L004_FD_3_1.CT.00000001")
        self._populate_cached_slice("train", "L110_FD_3_1.CT.00000001")

        p_l1 = LimitedAngleCT_Provider(
            dicom_dir=self.dicom_dir,
            cache_dir=self.cache_dir,
            use_precomputed=True,
            precomputed_setting=self.SETTING_TAG,
            patient_list=["L1"],
            test=False,
        )
        self.assertEqual(len(p_l1), 1)
        self.assertTrue(os.path.basename(p_l1.slices_path[0]).startswith("L1_"))

        p_l110 = LimitedAngleCT_Provider(
            dicom_dir=self.dicom_dir,
            cache_dir=self.cache_dir,
            use_precomputed=True,
            precomputed_setting=self.SETTING_TAG,
            patient_list=["L110"],
            test=False,
        )
        self.assertEqual(len(p_l110), 1)
        self.assertTrue(os.path.basename(p_l110.slices_path[0]).startswith("L110_"))

        # Thiếu bệnh nhân trong gt_dir phải báo lỗi rõ ràng thay vì im lặng trả về 0 lát
        with self.assertRaises(FileNotFoundError):
            LimitedAngleCT_Provider(
                dicom_dir=self.dicom_dir,
                cache_dir=self.cache_dir,
                use_precomputed=True,
                precomputed_setting=self.SETTING_TAG,
                patient_list=["L004", "L999"],
                test=False,
            )

        # Thiếu cặp sino/ hoặc fbp_u/ cho một lát cắt trong gt_dir phải báo lỗi rõ ràng
        self._populate_cached_slice("train", "L006_FD_3_1.CT.00000001")
        os.remove(os.path.join(self.cache_dir, "train", self.SETTING_TAG, "sino", "L006_FD_3_1.CT.00000001.npy"))
        with self.assertRaises(FileNotFoundError):
            LimitedAngleCT_Provider(
                dicom_dir=self.dicom_dir,
                cache_dir=self.cache_dir,
                use_precomputed=True,
                precomputed_setting=self.SETTING_TAG,
                patient_list=["L006"],
                test=False,
            )

    def test_missing_gt_and_dangling_dicom_fails_loudly(self):
        """Kịch bản 5: Khi thiếu gt cache VÀ symlink split/ bị gãy hoặc trống, __init__ phải ném FileNotFoundError."""
        from data.CTSlice_Provider_LA import LimitedAngleCT_Provider

        os.makedirs(os.path.join(self.cache_dir, "train", self.SETTING_TAG, "sino"), exist_ok=True)
        os.makedirs(os.path.join(self.cache_dir, "train", self.SETTING_TAG, "fbp_u"), exist_ok=True)
        self._create_dangling_symlink("train", "L004", "L004_FD_3_1.CT.00000001.IMA")

        with self.assertRaises(FileNotFoundError):
            LimitedAngleCT_Provider(
                dicom_dir=self.dicom_dir,
                cache_dir=self.cache_dir,
                use_precomputed=True,
                precomputed_setting=self.SETTING_TAG,
                patient_list=["L004"],
                test=False,
            )

    def test_use_precomputed_false_still_reads_dicom(self):
        """Kịch bản 6: use_precomputed=False vẫn khởi tạo toán tử ODL và đọc DICOM qua pydicom.read_file."""
        import data.CTSlice_Provider_LA as provider_mod
        from data.CTSlice_Provider_LA import LimitedAngleCT_Provider

        full_3mm = os.path.join(self.dicom_dir, "train", "L004", "full_3mm")
        os.makedirs(full_3mm, exist_ok=True)
        dcm_path = os.path.join(full_3mm, "L004_FD_3_1.CT.00000001.IMA")
        with open(dcm_path, "wb") as f:
            f.write(b"DICM_VALID_TEST_BYTES" * 10)

        fake_dcm = types.SimpleNamespace(
            pixel_array=np.linspace(0, 1000, 512 * 512, dtype=np.float32).reshape(512, 512),
            RescaleSlope=1.0,
            RescaleIntercept=-1024.0,
        )
        fake_radon = MagicMock(side_effect=lambda x: _FakeTensor(np.ones((1, 64, 512), dtype=np.float32)))
        fake_fbp = MagicMock(side_effect=lambda s: _FakeTensor(np.ones((1, 512, 512), dtype=np.float32)))

        with patch.object(provider_mod.pydicom, "read_file", return_value=fake_dcm) as mock_read, \
             patch.object(LimitedAngleCT_Provider, "_build_limited_angle_operators", return_value=(fake_radon, fake_fbp)) as mock_ops:
            provider = LimitedAngleCT_Provider(
                dicom_dir=self.dicom_dir,
                cache_dir=self.cache_dir,
                poission_level=0.0,
                gaussian_level=0.0,
                use_precomputed=False,
                patient_list=["L004"],
                test=False,
            )
            self.assertFalse(provider.cache_only)
            self.assertEqual(mock_ops.call_count, 1)
            phantom, fbp_u, sino = provider[0]
            self.assertEqual(mock_read.call_count, 1)
            self.assertEqual(tuple(phantom.shape), (1, 256, 256))
            self.assertEqual(tuple(fbp_u.shape), (1, 256, 256))
            self.assertEqual(tuple(sino.shape), (1, 64, 512))

    def test_zero_byte_dicom_fails_loudly_with_resolved_target(self):
        """Bảo vệ bổ sung (rca-raw-0byte): Tệp DICOM 0-byte bị từ chối trước pydicom và nêu rõ cả symlink lẫn đích."""
        import data.CTSlice_Provider_LA as provider_mod
        from data.CTSlice_Provider_LA import LimitedAngleCT_Provider

        raw_dir = os.path.join(self.tmp_dir.name, "raw_storage", "L004", "full_3mm")
        os.makedirs(raw_dir, exist_ok=True)
        raw_target = os.path.join(raw_dir, "00000001.dcm")
        open(raw_target, "wb").close()

        full_3mm = os.path.join(self.dicom_dir, "train", "L004", "full_3mm")
        os.makedirs(full_3mm, exist_ok=True)
        link_path = os.path.join(full_3mm, "L004_FD_3_1.CT.00000001.IMA")
        os.symlink(raw_target, link_path)

        with patch.object(provider_mod.pydicom, "read_file", side_effect=AssertionError("must fail before pydicom")), \
             patch.object(LimitedAngleCT_Provider, "_build_limited_angle_operators", return_value=(MagicMock(), MagicMock())):
            provider = LimitedAngleCT_Provider(
                dicom_dir=self.dicom_dir,
                cache_dir=self.cache_dir,
                use_precomputed=False,
                patient_list=["L004"],
                test=False,
            )
            with self.assertRaises(ValueError) as ctx:
                _ = provider[0]
            self.assertIn(link_path, str(ctx.exception))
            self.assertIn(os.path.realpath(raw_target), str(ctx.exception))

    def test_prepare_data_writes_gt_atomic_npy_and_gt_only_mode(self):
        """Kịch bản 7: generate_split ghi gt nguyên tử, giữ tương thích ngược 3 đối số, và --gt_only bỏ qua ODL."""
        from data.aapm import prepare_data_sinogram_LA as prep
        from data.CTSlice_Provider_LA import LimitedAngleCT_Provider

        sino_dir = os.path.join(self.cache_dir, "train", self.SETTING_TAG, "sino")
        fbp_dir = os.path.join(self.cache_dir, "train", self.SETTING_TAG, "fbp_u")
        gt_dir = os.path.join(self.cache_dir, "train", "gt_size_256")

        # 1) generate_split với save_gt_path: ghi cả 3 file, và chỉ bỏ qua khi cả 3 >= 1000 bytes
        mock_provider = MagicMock()
        mock_provider.__len__.return_value = 2
        mock_provider.slices_path = [
            "/fake/split/train/L004/full_3mm/L004_FD_3_1.CT.00000001.IMA",
            "/fake/split/train/L004/full_3mm/L004_FD_3_1.CT.00000002.IMA",
        ]
        gt_val = np.full((1, 256, 256), 0.5, dtype=np.float32)
        fbp_val = np.full((1, 256, 256), 0.3, dtype=np.float32)
        sino_val = np.full((1, 64, 512), 0.2, dtype=np.float32)

        # Slice 0 đã có sino + fbp_u hợp lệ nhưng thiếu gt -> KHÔNG được bỏ qua!
        os.makedirs(sino_dir, exist_ok=True)
        os.makedirs(fbp_dir, exist_ok=True)
        np.save(os.path.join(sino_dir, "L004_FD_3_1.CT.00000001.npy"), sino_val)
        np.save(os.path.join(fbp_dir, "L004_FD_3_1.CT.00000001.npy"), fbp_val)

        mock_provider.__getitem__.return_value = ("path", _FakeTensor(gt_val), _FakeTensor(fbp_val), _FakeTensor(sino_val))
        prep.generate_split(mock_provider, sino_dir, fbp_dir, save_gt_path=gt_dir)

        self.assertEqual(mock_provider.__getitem__.call_count, 2)
        gt0_path = os.path.join(gt_dir, "L004_FD_3_1.CT.00000001.npy")
        gt1_path = os.path.join(gt_dir, "L004_FD_3_1.CT.00000002.npy")
        self.assertTrue(os.path.exists(gt0_path))
        self.assertTrue(os.path.exists(gt1_path))
        self.assertFalse(os.path.exists(gt0_path + ".tmp"))
        loaded_gt0 = np.load(gt0_path)
        self.assertEqual(loaded_gt0.shape, (1, 256, 256))
        self.assertEqual(loaded_gt0.dtype, np.float32)
        np.testing.assert_allclose(loaded_gt0, gt_val)

        # Lần chạy thứ 2 khi cả 3 file đã >= 1000 bytes -> bỏ qua hoàn toàn
        mock_provider.__getitem__.reset_mock()
        prep.generate_split(mock_provider, sino_dir, fbp_dir, save_gt_path=gt_dir)
        self.assertEqual(mock_provider.__getitem__.call_count, 0)

        # 2) Kiểm tra chế độ CLI --gt_only trên dữ liệu DICOM giả: không khởi tạo ODL, không đụng sino/ hoặc fbp_u/
        gt_only_out = os.path.join(self.tmp_dir.name, "gt_only_cache")
        for split_n, pid, fname in [
            ("train", "L004", "L004_FD_3_1.CT.00000001.IMA"),
            ("train", "L110", "L110_FD_3_1.CT.00000001.IMA"),
            ("test", "L058", "L058_FD_3_1.CT.00000001.IMA"),
        ]:
            d = os.path.join(self.dicom_dir, split_n, pid, "full_3mm")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, fname), "wb") as f:
                f.write(b"DICM_DATA" * 8)

        fake_dcm = types.SimpleNamespace(
            pixel_array=np.linspace(100, 900, 512 * 512, dtype=np.float32).reshape(512, 512),
            RescaleSlope=1.0,
            RescaleIntercept=-1024.0,
        )
        import data.CTSlice_Provider_LA as provider_mod
        argv_gt_only = [
            "prepare_data_sinogram_LA.py",
            "--data_dir", self.dicom_dir,
            "--output_dir", gt_only_out,
            "--train_patients", "L004",
            "--val_patients", "L110",
            "--test_patients", "L058",
            "--gt_only",
        ]
        with patch.object(sys, "argv", argv_gt_only), \
             patch.object(provider_mod.pydicom, "read_file", return_value=fake_dcm), \
             patch.object(LimitedAngleCT_Provider, "_build_limited_angle_operators", side_effect=AssertionError("ODL must not be initialized in --gt_only")):
            prep.main()

        self.assertTrue(os.path.exists(os.path.join(gt_only_out, "train", "gt_size_256", "L004_FD_3_1.CT.00000001.npy")))
        self.assertTrue(os.path.exists(os.path.join(gt_only_out, "train", "gt_size_256", "L110_FD_3_1.CT.00000001.npy")))
        self.assertTrue(os.path.exists(os.path.join(gt_only_out, "test", "gt_size_256", "L058_FD_3_1.CT.00000001.npy")))
        # Không tạo bất kỳ thư mục sino hoặc fbp_u nào trong chế độ --gt_only
        self.assertFalse(os.path.exists(os.path.join(gt_only_out, "train", self.SETTING_TAG)))
        self.assertFalse(os.path.exists(os.path.join(gt_only_out, "test", self.SETTING_TAG)))

    def test_t05_output_dir_rejects_raw_and_split_paths(self):
        """T0.5 Guard: Từ chối ngay lập tức mọi output_dir trùng hoặc nằm trong dicom_dir, raw/, hoặc split/."""
        from data.aapm import prepare_data_sinogram_LA as aapm_prep
        from data import prepare_data_sinogram_LA as root_prep

        raw_dir = os.path.join(self.tmp_dir.name, "aapm", "raw")
        split_dir = os.path.join(self.tmp_dir.name, "aapm", "split")
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(split_dir, exist_ok=True)
        symlink_to_raw = os.path.join(self.tmp_dir.name, "alias_cache")
        os.symlink(raw_dir, symlink_to_raw)

        bad_outputs = [
            split_dir,
            os.path.join(split_dir, "train"),
            raw_dir,
            os.path.join(raw_dir, "L004"),
            symlink_to_raw,
            os.path.join(self.tmp_dir.name, "custom_dicom_root"),
        ]
        for prep_mod in (aapm_prep, root_prep):
            for bad_out in bad_outputs:
                dicom_arg = bad_out if "custom_dicom_root" in bad_out else split_dir
                with self.assertRaises(ValueError):
                    prep_mod.validate_safe_output_dir(bad_out, dicom_dir=dicom_arg)

            # Kiểm tra main() dừng ngay trước khi tạo Provider khi CLI truyền output_dir vào raw/
            with patch.object(
                sys,
                "argv",
                ["prepare_data_sinogram_LA.py", "--data_dir", split_dir, "--output_dir", raw_dir],
            ), patch.object(prep_mod, "LimitedAngleCT_Provider", side_effect=AssertionError("Should fail before Provider")):
                with self.assertRaises(ValueError):
                    prep_mod.main()

            # Thư mục cache hợp lệ phải được chấp nhận
            safe_out = os.path.join(self.tmp_dir.name, "aapm", "limited_angle")
            prep_mod.validate_safe_output_dir(safe_out, dicom_dir=split_dir)


if __name__ == "__main__":
    unittest.main()
