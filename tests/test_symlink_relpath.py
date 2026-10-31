#!/usr/bin/env python3
"""
Unit test for relative symlink calculation between split and raw directories.
"""

import os
import tempfile
import unittest


def create_relative_symlink(raw_file: str, link_path: str):
    link_dir = os.path.dirname(link_path)
    os.makedirs(link_dir, exist_ok=True)
    rel_src = os.path.relpath(raw_file, link_dir)
    if os.path.islink(link_path) or os.path.exists(link_path):
        os.remove(link_path)
    os.symlink(rel_src, link_path)
    return rel_src


class TestRelativeSymlinks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_relative_symlink_structure(self):
        base = self.tmp.name
        raw_file = os.path.join(base, "aapm", "raw", "L058", "full_3mm", "00000001.dcm")
        os.makedirs(os.path.dirname(raw_file), exist_ok=True)
        with open(raw_file, "w") as f:
            f.write("DICOM_TEST")

        link_path = os.path.join(base, "aapm", "split", "test", "L058", "full_3mm", "L058_FD_3_1.CT.00000001.IMA")
        rel_target = create_relative_symlink(raw_file, link_path)

        # Kiểm tra symlink là tương đối
        self.assertFalse(os.path.isabs(rel_target))
        self.assertEqual(rel_target, "../../../../raw/L058/full_3mm/00000001.dcm")

        # Kiểm tra đọc link bằng os.readlink
        target_read = os.readlink(link_path)
        self.assertEqual(target_read, "../../../../raw/L058/full_3mm/00000001.dcm")

        # Kiểm tra resolution của symlink
        self.assertTrue(os.path.exists(link_path))
        with open(link_path, "r") as f:
            self.assertEqual(f.read(), "DICOM_TEST")


if __name__ == "__main__":
    unittest.main()
