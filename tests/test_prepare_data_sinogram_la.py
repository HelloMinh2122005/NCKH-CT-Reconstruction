#!/usr/bin/env python3
"""
Unit tests cho prepare_data_sinogram_LA.py và generate_la_dataset.sh:
- Kiểm tra khai báo tham số dòng lệnh CLI (--train_patients, --val_patients, --test_patients với nargs="+").
- Kiểm tra tính tương thích ngược (giữ nguyên mặc định None khi không truyền tham số).
- Kiểm tra truyền danh sách bệnh nhân vào LimitedAngleCT_Provider.
- Kiểm tra danh sách bệnh nhân phân chia trong scripts/generate_la_dataset.sh (55 train, 1 val, 4 test).
"""

import os
import sys
import types
import unittest
from unittest.mock import patch, MagicMock

# Đảm bảo import được module từ temp-github/NCKH-CT-Reconstruction
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Giả lập các thư viện cluster-only để test có thể chạy thuần Python tại local
if "torch" not in sys.modules:
    torch_mock = types.ModuleType("torch")
    torch_utils = types.ModuleType("torch.utils")
    torch_utils_data = types.ModuleType("torch.utils.data")
    torch_utils_data.Dataset = object
    torch_mock.utils = torch_utils
    torch_utils.data = torch_utils_data
    torch_mock.Tensor = MagicMock
    sys.modules["torch"] = torch_mock
    sys.modules["torch.utils"] = torch_utils
    sys.modules["torch.utils.data"] = torch_utils_data

for mod in ["torchvision", "torchvision.transforms", "tqdm", "pydicom", "odl", "odl.contrib", "odl.contrib.torch", "PIL"]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

from data.aapm.prepare_data_sinogram_LA import parse_args, main


class TestPrepareDataSinogramLA(unittest.TestCase):
    def test_parse_args_defaults(self):
        """
        Kiểm tra giá trị mặc định của parse_args khi không truyền tham số phân chia bệnh nhân.
        Bảo đảm tính tương thích ngược hoàn toàn (defaults = None).
        """
        with patch.object(sys, "argv", ["prepare_data_sinogram_LA.py"]):
            args = parse_args()
            self.assertIsNone(args.train_patients)
            self.assertIsNone(args.val_patients)
            self.assertIsNone(args.test_patients)
            self.assertEqual(args.angle_range_deg, 120.0)
            self.assertEqual(args.num_view, 64)
            self.assertEqual(args.num_detectors, 512)
            self.assertEqual(args.input_size, 256)

    def test_parse_args_custom_patients(self):
        """
        Kiểm tra parse_args khi truyền danh sách bệnh nhân qua CLI với nargs="+".
        """
        test_args = [
            "prepare_data_sinogram_LA.py",
            "--train_patients", "L004", "L006", "L012",
            "--val_patients", "L110",
            "--test_patients", "L058", "L064", "L094", "L107",
            "--angle_range_deg", "90.0",
        ]
        with patch.object(sys, "argv", test_args):
            args = parse_args()
            self.assertEqual(args.train_patients, ["L004", "L006", "L012"])
            self.assertEqual(args.val_patients, ["L110"])
            self.assertEqual(args.test_patients, ["L058", "L064", "L094", "L107"])
            self.assertEqual(args.angle_range_deg, 90.0)

    @patch("data.aapm.prepare_data_sinogram_LA.generate_split")
    @patch("data.aapm.prepare_data_sinogram_LA.LimitedAngleCT_Provider")
    def test_provider_patient_list_propagation_custom(self, mock_provider, mock_generate_split):
        """
        Kiểm tra việc truyền đúng danh sách bệnh nhân tùy chọn vào LimitedAngleCT_Provider.
        """
        mock_instance = MagicMock()
        mock_instance.__len__.return_value = 0
        mock_provider.return_value = mock_instance

        test_args = [
            "prepare_data_sinogram_LA.py",
            "--data_dir", "/tmp/split",
            "--output_dir", "/tmp/output",
            "--train_patients", "L004", "L006",
            "--val_patients", "L110",
            "--test_patients", "L058", "L064",
        ]
        with patch.object(sys, "argv", test_args):
            main()

        self.assertEqual(mock_provider.call_count, 3)
        # Call 1: Train
        train_call_kwargs = mock_provider.call_args_list[0].kwargs
        self.assertEqual(train_call_kwargs.get("patient_list"), ["L004", "L006"])
        self.assertFalse(train_call_kwargs.get("test"))
        self.assertFalse(train_call_kwargs.get("valid"))

        # Call 2: Val
        val_call_kwargs = mock_provider.call_args_list[1].kwargs
        self.assertEqual(val_call_kwargs.get("patient_list"), ["L110"])
        self.assertFalse(val_call_kwargs.get("test"))
        self.assertTrue(val_call_kwargs.get("valid"))

        # Call 3: Test
        test_call_kwargs = mock_provider.call_args_list[2].kwargs
        self.assertEqual(test_call_kwargs.get("patient_list"), ["L058", "L064"])
        self.assertTrue(test_call_kwargs.get("test"))

    @patch("data.aapm.prepare_data_sinogram_LA.generate_split")
    @patch("data.aapm.prepare_data_sinogram_LA.LimitedAngleCT_Provider")
    def test_provider_patient_list_propagation_defaults(self, mock_provider, mock_generate_split):
        """
        Kiểm tra việc truyền patient_list=None khi không chỉ định danh sách bệnh nhân.
        """
        mock_instance = MagicMock()
        mock_instance.__len__.return_value = 0
        mock_provider.return_value = mock_instance

        with patch.object(sys, "argv", ["prepare_data_sinogram_LA.py"]):
            main()

        self.assertEqual(mock_provider.call_count, 3)
        self.assertIsNone(mock_provider.call_args_list[0].kwargs.get("patient_list"))
        self.assertIsNone(mock_provider.call_args_list[1].kwargs.get("patient_list"))
        self.assertIsNone(mock_provider.call_args_list[2].kwargs.get("patient_list"))


class TestGenerateLaDatasetScript(unittest.TestCase):
    def test_script_patient_split_counts_and_disjoint(self):
        """
        Kiểm tra script scripts/generate_la_dataset.sh chứa cấu hình phân chia bệnh nhân
        chuẩn theo experiments/split.md:
        - 55 bệnh nhân train
        - 1 bệnh nhân val (L110)
        - 4 bệnh nhân test (L058, L064, L094, L107)
        - Không có sự trùng lặp giữa các tập (patient-disjoint).
        """
        script_path = os.path.join(REPO_ROOT, "scripts", "generate_la_dataset.sh")
        self.assertTrue(os.path.exists(script_path), f"Không tìm thấy file: {script_path}")

        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read()

        import re
        train_match = re.search(r"TRAIN_PATIENTS=\(([^)]+)\)", content)
        val_match = re.search(r"VAL_PATIENTS=\(([^)]+)\)", content)
        test_match = re.search(r"TEST_PATIENTS=\(([^)]+)\)", content)

        self.assertIsNotNone(train_match, "Không tìm thấy mảng TRAIN_PATIENTS trong script")
        self.assertIsNotNone(val_match, "Không tìm thấy mảng VAL_PATIENTS trong script")
        self.assertIsNotNone(test_match, "Không tìm thấy mảng TEST_PATIENTS trong script")

        train_patients = re.findall(r'"([^"]+)"', train_match.group(1))
        val_patients = re.findall(r'"([^"]+)"', val_match.group(1))
        test_patients = re.findall(r'"([^"]+)"', test_match.group(1))

        self.assertEqual(len(train_patients), 55, f"Số bệnh nhân train phải là 55, thực tế: {len(train_patients)}")
        self.assertEqual(val_patients, ["L110"], f"Bệnh nhân val phải là ['L110'], thực tế: {val_patients}")
        self.assertEqual(sorted(test_patients), sorted(["L058", "L064", "L094", "L107"]), f"Bệnh nhân test không đúng: {test_patients}")

        # Kiểm tra tính rời rạc (patient-disjoint)
        train_set = set(train_patients)
        val_set = set(val_patients)
        test_set = set(test_patients)
        self.assertEqual(len(train_set), 55, "Có bệnh nhân bị trùng lặp trong tập train")
        self.assertTrue(train_set.isdisjoint(val_set), "Tập train và val không rời rạc")
        self.assertTrue(train_set.isdisjoint(test_set), "Tập train và test không rời rạc")
        self.assertTrue(val_set.isdisjoint(test_set), "Tập val và test không rời rạc")


if __name__ == "__main__":
    unittest.main()
