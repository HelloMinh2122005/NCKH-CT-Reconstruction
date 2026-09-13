"""
================================================================================
SCRIPT TẢI TỰ ĐỘNG TẬP CON LIDC-IDRI TỪ TCIA REST API
Dự án: Nghiên cứu Tái tạo Ảnh Cắt lớp CT Góc Giới hạn (Limited-Angle CT Reconstruction)
Tác giả: MinhPD — Nhóm Nghiên cứu Tái tạo Ảnh Y tế (VNU-HCM UIT)
================================================================================
"""

import os
import io
import json
import zipfile
import urllib.request
import argparse
import tqdm


# Danh sách 12 bệnh nhân CT lồng ngực tiêu biểu kèm SeriesInstanceUID tương ứng
# Mỗi ca có từ 120 đến 260 lát cắt trục chuẩn y tế (Tổng cộng ~2,400 lát cắt)
LIDC_TARGET_PATIENTS = [
    # Tập Train: 8 bệnh nhân (~1,700 lát cắt)
    {"patient_id": "LIDC-IDRI-0001", "series_uid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192", "split": "train"},
    {"patient_id": "LIDC-IDRI-0002", "series_uid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.619372068417051974713149104919", "split": "train"},
    {"patient_id": "LIDC-IDRI-0003", "series_uid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.170706757615202213033480003264", "split": "train"},
    {"patient_id": "LIDC-IDRI-0004", "series_uid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.323541312620128092852212458228", "split": "train"},
    {"patient_id": "LIDC-IDRI-0005", "series_uid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.129007566048223160327836686225", "split": "train"},
    {"patient_id": "LIDC-IDRI-0006", "series_uid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.132817748896065918417924920957", "split": "train"},
    {"patient_id": "LIDC-IDRI-0007", "series_uid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.272348349298439120568330857680", "split": "train"},
    {"patient_id": "LIDC-IDRI-0008", "series_uid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.774060103415303828812229821954", "split": "train"},

    # Tập Validation: 2 bệnh nhân (~530 lát cắt)
    {"patient_id": "LIDC-IDRI-0009", "series_uid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.286061375572911414226912429210", "split": "val"},
    {"patient_id": "LIDC-IDRI-0010", "series_uid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.416701701108520592702405866796", "split": "val"},

    # Tập Test Benchmark độc lập: 2 bệnh nhân (~256 lát cắt)
    {"patient_id": "LIDC-IDRI-0011", "series_uid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.140253591510022414496468423138", "split": "test"},
    {"patient_id": "LIDC-IDRI-0012", "series_uid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.328789598898469177563438457842", "split": "test"}
]


def download_series(patient_info, output_base_dir):
    """
    Tải file zip chứa các lát cắt DICOM từ TCIA REST API và giải nén vào thư mục bệnh nhân.
    """
    patient_id = patient_info["patient_id"]
    series_uid = patient_info["series_uid"]
    split = patient_info["split"]

    patient_dir = os.path.join(output_base_dir, patient_id)
    os.makedirs(patient_dir, exist_ok=True)

    # Kiểm tra nếu đã có file DICOM bên trong
    existing_dcm = [f for f in os.listdir(patient_dir) if f.endswith(".dcm")]
    if len(existing_dcm) > 50:
        print(f"[BỎ QUA] {patient_id} ({split}) đã có sẵn {len(existing_dcm)} file DICOM.")
        return len(existing_dcm)

    url = f"https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID={series_uid}"
    print(f"\n[DOWNLOAD] Bắt đầu tải {patient_id} ({split})...")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, timeout=180) as response:
        content = response.read()

    # Giải nén zip file bộ nhớ đệm
    extracted_count = 0
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        for filename in z.namelist():
            if filename.endswith(".dcm"):
                target_path = os.path.join(patient_dir, os.path.basename(filename))
                with open(target_path, "wb") as f_out:
                    f_out.write(z.read(filename))
                extracted_count += 1

    print(f"[THÀNH CÔNG] {patient_id}: Đã giải nén {extracted_count} lát cắt DICOM.")
    return extracted_count


def main():
    parser = argparse.ArgumentParser(description="Tải tập con LIDC-IDRI chuẩn y tế từ TCIA")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/datastore/uittogether3/LuuTru/MinhPD/dataset/lidc_idri/dicom/",
        help="Thư mục đích lưu trữ các file DICOM"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 70)
    print("KHỞI CHẠY TIẾN TRÌNH TẢI BỘ DỮ LIỆU LIDC-IDRI TỪ TCIA REST API")
    print(f"Thư mục lưu trữ: {args.output_dir}")
    print(f"Tổng số ca bệnh nhân dự kiến: {len(LIDC_TARGET_PATIENTS)}")
    print("=" * 70)

    total_slices = 0
    for pat_info in tqdm.tqdm(LIDC_TARGET_PATIENTS, desc="Tiến trình tải LIDC-IDRI"):
        try:
            count = download_series(pat_info, args.output_dir)
            total_slices += count
        except Exception as e:
            print(f"[LỖI] Không thể tải {pat_info['patient_id']}: {e}")

    # Ghi file danh sách phân chia split
    splits_summary = {"train": [], "val": [], "test": []}
    for pat in LIDC_TARGET_PATIENTS:
        splits_summary[pat["split"]].append(pat["patient_id"])

    splits_file = os.path.join(args.output_dir, "..", "splits_summary.json")
    with open(splits_file, "w", encoding="utf-8") as f:
        json.dump(splits_summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"[HOÀN TẤT 100%] Tổng cộng đã tải và giải nén thành công {total_slices} lát cắt DICOM!")
    print(f"Bản tổng hợp split: {splits_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
