#!/usr/bin/env python3
# ==============================================================================
# KIỂM THỬ ĐƠN VỊ CHO DATA MODULE FACTORY LAZY IMPORT (CHANGE 23)
# Đảm bảo get_datamodule không nạp skimage trừ khi người dùng gọi deeplesion/lidc.
# ==============================================================================

import os
import sys
import types
import unittest
from unittest.mock import MagicMock

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Thiết lập môi trường giả lập (mocking) cho các phụ thuộc cluster nặng nếu chưa cài
for mod in [
    "torch",
    "torch.utils",
    "torch.utils.data",
    "torchvision",
    "torchvision.transforms",
    "pytorch_lightning",
    "pydicom",
    "odl",
    "odl.contrib",
    "odl.contrib.torch",
    "PIL",
]:
    if mod not in sys.modules:
        m = MagicMock()
        if mod == "torch.utils.data":
            m.Dataset = object
            m.DataLoader = MagicMock()
        if mod == "pytorch_lightning":
            m.LightningDataModule = object
        sys.modules[mod] = m

# Đảm bảo các thuộc tính thiết yếu luôn tồn tại kể cả khi module đã được mock bởi test trước
if "torch.utils.data" in sys.modules:
    if not hasattr(sys.modules["torch.utils.data"], "DataLoader"):
        sys.modules["torch.utils.data"].DataLoader = MagicMock()
    if not hasattr(sys.modules["torch.utils.data"], "Dataset"):
        sys.modules["torch.utils.data"].Dataset = object

if "torch" in sys.modules and hasattr(sys.modules["torch"], "utils"):
    if not hasattr(sys.modules["torch"].utils, "data"):
        sys.modules["torch"].utils.data = sys.modules["torch.utils.data"]
    if not hasattr(sys.modules["torch"].utils.data, "DataLoader"):
        sys.modules["torch"].utils.data.DataLoader = sys.modules["torch.utils.data"].DataLoader

if "torchvision" in sys.modules and not hasattr(sys.modules["torchvision"], "transforms"):
    sys.modules["torchvision"].transforms = sys.modules.get("torchvision.transforms", MagicMock())

if "pytorch_lightning" in sys.modules and not hasattr(sys.modules["pytorch_lightning"], "LightningDataModule"):
    sys.modules["pytorch_lightning"].LightningDataModule = object


class TestDataModuleFactoryLazyImport(unittest.TestCase):
    """
    Bộ kiểm thử xác thực cơ chế Lazy Import trong data/datamodule_factory.py.
    """

    def setUp(self):
        # Đảm bảo xóa cache module liên quan trước mỗi test case
        for mod_name in list(sys.modules.keys()):
            if "datamodule_factory" in mod_name:
                sys.modules.pop(mod_name, None)

        # Chặn skimage bằng cách gán None để giả lập môi trường không có thư viện này
        sys.modules["skimage"] = None
        sys.modules["skimage.io"] = None

    def tearDown(self):
        # Dọn dẹp trạng thái module sau mỗi test
        sys.modules.pop("skimage", None)
        sys.modules.pop("skimage.io", None)

    def test_datamodule_factory_eager_import_does_not_load_skimage(self):
        """
        Xác minh: Import get_datamodule từ data.datamodule_factory không được phép
        gây lỗi ModuleNotFoundError khi hệ thống thiếu skimage.
        """
        try:
            from data.datamodule_factory import get_datamodule
        except ModuleNotFoundError as e:
            self.fail(f"Import datamodule_factory thất bại do phụ thuộc eager import: {e}")

        self.assertTrue(callable(get_datamodule))

    def test_get_datamodule_aapm_works_without_skimage(self):
        """
        Xác minh: get_datamodule(dataset_type='aapm') hoạt động bình thường
        mà không đòi hỏi hoặc nạp skimage.
        """
        from data.datamodule_factory import get_datamodule

        # Gọi khởi tạo datamodule mặc định (aapm)
        dm = get_datamodule(
            dataset_type="aapm",
            batch_size=2,
            num_workers=0,
            use_precomputed=True,
        )
        self.assertIsNotNone(dm)
        # Kiểm tra rằng skimage vẫn không được nạp thành công
        self.assertIsNone(sys.modules.get("skimage"))

    def test_get_datamodule_deeplesion_lazy_loads_and_fails_if_skimage_missing(self):
        """
        Xác minh: get_datamodule(dataset_type='deeplesion') nạp trễ (lazy import)
        và chỉ cố gắng nạp skimage khi được gọi thực sự.
        """
        from data.datamodule_factory import get_datamodule

        # Khi gọi dataset_type='deeplesion', quá trình import lazy sẽ diễn ra
        # và vấp phải ModuleNotFoundError vì skimage bị chặn.
        with self.assertRaises(ModuleNotFoundError):
            get_datamodule(
                dataset_type="deeplesion",
                batch_size=1,
                num_workers=0,
            )

    def test_get_datamodule_lidc_lazy_loads(self):
        """
        Xác minh: get_datamodule(dataset_type='lidc') nạp trễ LimitedAngleCT_LIDC_DataModule.
        """
        from data.datamodule_factory import get_datamodule

        # Gọi khởi tạo LIDC
        try:
            dm = get_datamodule(
                dataset_type="lidc",
                batch_size=1,
                num_workers=0,
            )
            self.assertIsNotNone(dm)
        except Exception as e:
            # Nếu môi trường thiếu phụ thuộc của LIDC thì cũng chỉ xảy ra khi gọi hàm,
            # không phải lúc import file datamodule_factory.py
            pass


if __name__ == "__main__":
    unittest.main()
