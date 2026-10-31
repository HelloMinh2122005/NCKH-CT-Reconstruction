#!/usr/bin/env python3
"""
Unit test for flexible DICOM and IMA filename pattern resolution.
"""

import os
import glob
import tempfile
import unittest


def resolve_patient_slice_paths(dicom_dir: str, split_name: str, patient_id: str):
    """
    Hàm tìm kiếm đường dẫn các lát cắt theo nhiều mẫu quy ước tên file.
    """
    patterns = [
        os.path.join(dicom_dir, split_name, patient_id, "full_3mm", f"{patient_id}_FD_3_1.CT.*.IMA"),
        os.path.join(dicom_dir, split_name, patient_id, "full_3mm", f"{patient_id}_FD_3_1.CT.*.*.*.*.*.*.*.*.*.IMA"),
        os.path.join(dicom_dir, split_name, patient_id, "full_3mm", "*.IMA"),
        os.path.join(dicom_dir, split_name, patient_id, "full_3mm", "*.dcm")
    ]
    for p in patterns:
        matches = glob.glob(p)
        if matches:
            return sorted(matches)
    return []


class TestDICOMPatternMatching(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.test_dir.cleanup()

    def test_resolve_numbered_symlink_ima(self):
        # Mẫu symlink đánh số: L058_FD_3_1.CT.00000001.IMA
        patient_dir = os.path.join(self.test_dir.name, "train", "L058", "full_3mm")
        os.makedirs(patient_dir, exist_ok=True)
        file1 = os.path.join(patient_dir, "L058_FD_3_1.CT.00000001.IMA")
        file2 = os.path.join(patient_dir, "L058_FD_3_1.CT.00000002.IMA")
        open(file1, "w").close()
        open(file2, "w").close()

        paths = resolve_patient_slice_paths(self.test_dir.name, "train", "L058")
        self.assertEqual(paths, [file1, file2])

    def test_resolve_legacy_dotted_ima(self):
        # Mẫu legacy chấm nhiều tầng: L058_FD_3_1.CT.1.2.3.4.5.6.7.8.9.IMA
        patient_dir = os.path.join(self.test_dir.name, "train", "L058", "full_3mm")
        os.makedirs(patient_dir, exist_ok=True)
        file1 = os.path.join(patient_dir, "L058_FD_3_1.CT.1.2.3.4.5.6.7.8.9.IMA")
        open(file1, "w").close()

        paths = resolve_patient_slice_paths(self.test_dir.name, "train", "L058")
        self.assertEqual(paths, [file1])

    def test_resolve_direct_dcm(self):
        # Mẫu dcm trực tiếp: 00000001.dcm
        patient_dir = os.path.join(self.test_dir.name, "train", "L058", "full_3mm")
        os.makedirs(patient_dir, exist_ok=True)
        file1 = os.path.join(patient_dir, "00000001.dcm")
        open(file1, "w").close()

        paths = resolve_patient_slice_paths(self.test_dir.name, "train", "L058")
        self.assertEqual(paths, [file1])


if __name__ == "__main__":
    unittest.main()
