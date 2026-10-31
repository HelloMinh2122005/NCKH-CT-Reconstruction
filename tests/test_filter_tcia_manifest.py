#!/usr/bin/env python3
"""
Unit tests for TCIA manifest filtering logic (TDD Red-Green cycle).
"""

import unittest
import sys
import os

# Cho phép import từ thư mục cha hoặc từ .lab/scripts
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from scripts.filter_tcia_manifest import (
    is_target_series,
    parse_manifest_uids,
    format_clean_manifest,
)


class TestFilterTCIAManifest(unittest.TestCase):
    def test_is_target_series_accepts_valid_full_dose(self):
        # Trường hợp chuẩn: Full Dose Images, Abdomen, Patient L-prefix
        meta_valid = {
            "SeriesDescription": "Full Dose Images",
            "PatientID": "L058",
            "BodyPartExamined": "ABDOMEN",
        }
        self.assertTrue(is_target_series(meta_valid))

        # PatientID bắt đầu bằng L nhưng BodyPart để trống
        meta_valid_l = {
            "SeriesDescription": "Full Dose Images",
            "PatientID": "L333",
            "BodyPartExamined": "",
        }
        self.assertTrue(is_target_series(meta_valid_l))

        # BodyPart là ABDOMEN
        meta_valid_abdo = {
            "SeriesDescription": "Full Dose Images",
            "PatientID": "A001",
            "BodyPartExamined": "ABDOMEN",
        }
        self.assertTrue(is_target_series(meta_valid_abdo))

    def test_is_target_series_rejects_unwanted_modalities(self):
        # Loại bỏ Low dose projections
        meta_low_proj = {
            "SeriesDescription": "Low dose projections",
            "PatientID": "L058",
            "BodyPartExamined": "ABDOMEN",
        }
        self.assertFalse(is_target_series(meta_low_proj))

        # Loại bỏ Full dose projections
        meta_full_proj = {
            "SeriesDescription": "Full dose projections",
            "PatientID": "L058",
            "BodyPartExamined": "ABDOMEN",
        }
        self.assertFalse(is_target_series(meta_full_proj))

        # Loại bỏ Low Dose Images
        meta_low_img = {
            "SeriesDescription": "Low Dose Images",
            "PatientID": "L058",
            "BodyPartExamined": "ABDOMEN",
        }
        self.assertFalse(is_target_series(meta_low_img))

        # Loại bỏ bệnh nhân không phải bụng / không phải L-prefix
        meta_chest = {
            "SeriesDescription": "Full Dose Images",
            "PatientID": "C001",
            "BodyPartExamined": "CHEST",
        }
        self.assertFalse(is_target_series(meta_chest))

    def test_parse_manifest_uids(self):
        sample_manifest = """downloadServerUrl=https://nbia.cancerimagingarchive.net/nbia-download/servlet/DownloadServlet
includeAnnotation=true
noOfrRetry=4
databasketId=manifest-test.tcia
manifestVersion=3.0
ListOfSeriesToDownload=
1.2.840.113713.4.2.1111111111111111
1.2.840.113713.4.2.2222222222222222
"""
        uids = parse_manifest_uids(sample_manifest)
        self.assertEqual(uids, [
            "1.2.840.113713.4.2.1111111111111111",
            "1.2.840.113713.4.2.2222222222222222",
        ])

    def test_format_clean_manifest(self):
        uids = [
            "1.2.840.113713.4.2.1111111111111111",
            "1.2.840.113713.4.2.2222222222222222",
        ]
        text = format_clean_manifest(uids, "manifest-clean-full-dose.tcia")
        expected = (
            "downloadServerUrl=https://nbia.cancerimagingarchive.net/nbia-download/servlet/DownloadServlet\n"
            "includeAnnotation=true\n"
            "noOfrRetry=4\n"
            "databasketId=manifest-clean-full-dose.tcia\n"
            "manifestVersion=3.0\n"
            "ListOfSeriesToDownload=\n"
            "1.2.840.113713.4.2.1111111111111111\n"
            "1.2.840.113713.4.2.2222222222222222\n"
        )
        self.assertEqual(text, expected)


if __name__ == "__main__":
    unittest.main()
