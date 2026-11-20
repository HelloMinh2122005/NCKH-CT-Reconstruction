"""
================================================================================
SCRIPT TẢI TỰ ĐỘNG TẬP CON LIDC-IDRI TỪ TCIA REST API
Dự án: Nghiên cứu Tái tạo Ảnh Cắt lớp CT Góc Giới hạn (Limited-Angle CT Reconstruction)
Tác giả: MinhPD — Nhóm Nghiên cứu Tái tạo Ảnh Y tế (VNU-HCM UIT)
================================================================================
"""

import os
import io
import sys
import time
import json
import zipfile
import urllib.request
import argparse

try:
    import pydicom
except ImportError:
    pydicom = None

try:
    import tqdm
    def progress_bar(it, desc=""):
        return tqdm.tqdm(it, desc=desc)
except ImportError:
    def progress_bar(it, desc=""):
        if desc:
            print(f"[{desc}]")
        return it


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


def validate_dicom_file(filepath: str, min_size_bytes: int = 100 * 1024) -> bool:
    """
    Kiểm định toàn vẹn tệp DICOM:
    - Kích thước tệp trên đĩa tối thiểu >= min_size_bytes (mặc định 100KB, loại trừ file 0B và 80KB cụt).
    - Đọc được header DICOM thông qua pydicom.dcmread (tôn trọng TransferSyntaxUID).
    - Giải mã thành công mảng pixel_array 2D hợp lệ mà không phát sinh lỗi lệch độ dài byte.
    """
    if not os.path.exists(filepath):
        return False
    try:
        size = os.path.getsize(filepath)
        if size < min_size_bytes:
            return False
    except OSError:
        return False

    if pydicom is None:
        # Nếu chưa cài pydicom, chỉ kiểm tra dung lượng tối thiểu
        return True

    try:
        # Thử đọc với force=False trước để giữ TransferSyntaxUID
        try:
            ds = pydicom.dcmread(filepath, force=False)
        except Exception:
            ds = pydicom.dcmread(filepath, force=True)

        arr = ds.pixel_array
        if arr is None:
            return False
        if getattr(arr, "size", 0) == 0:
            return False
        if getattr(arr, "ndim", 0) != 2:
            return False
        return True
    except Exception:
        return False


def clean_corrupt_dicoms(patient_dir: str) -> int:
    """
    Quét và dọn dẹp các tệp .dcm 0-byte, tệp .tmp còn sót lại hoặc tệp bị hỏng trong thư mục bệnh nhân.
    Trả về số lượng tệp DICOM hợp lệ còn lại sau khi dọn dẹp.
    """
    if not os.path.exists(patient_dir):
        return 0

    valid_count = 0
    for fname in sorted(os.listdir(patient_dir)):
        fpath = os.path.join(patient_dir, fname)
        if fname.endswith(".tmp"):
            try:
                os.remove(fpath)
            except OSError:
                pass
        elif fname.endswith(".dcm"):
            if not validate_dicom_file(fpath):
                try:
                    os.remove(fpath)
                    print(f"[DỌN DẸP] Đã xóa tệp DICOM hỏng/0-byte: {fpath}")
                except OSError:
                    pass
            else:
                valid_count += 1
    return valid_count


def download_series(patient_info, output_base_dir, max_retries: int = 3, backoff_base: int = 5):
    """
    Tải file zip chứa các lát cắt DICOM từ TCIA REST API và giải nén vào thư mục bệnh nhân.
    """
    patient_id = patient_info["patient_id"]
    series_uid = patient_info["series_uid"]
    split = patient_info["split"]

    patient_dir = os.path.join(output_base_dir, patient_id)
    os.makedirs(patient_dir, exist_ok=True)

    # Dọn dẹp tệp 0-byte hoặc bị hỏng trước khi kiểm tra số lượng
    valid_existing = clean_corrupt_dicoms(patient_dir)

    # Kiểm tra nếu đã có file DICOM bên trong
    existing_dcm = [f for f in os.listdir(patient_dir) if f.endswith(".dcm")]
    if len(existing_dcm) > 50:
        print(f"[BỎ QUA] {patient_id} ({split}) đã có sẵn {len(existing_dcm)} file DICOM.")
        return len(existing_dcm)

    url = f"https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID={series_uid}"
    print(f"\n[DOWNLOAD] Bắt đầu tải {patient_id} ({split})...")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    
    extracted_count = 0
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                content = response.read()

            # Giải nén zip file bộ nhớ đệm
            extracted_count = 0
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                for filename in z.namelist():
                    if filename.endswith(".dcm"):
                        target_path = os.path.join(patient_dir, os.path.basename(filename))
                        tmp_path = target_path + ".tmp"
                        # Ghi ra tệp tạm nguyên tử và kiểm định trước khi đổi tên
                        with open(tmp_path, "wb") as f_out:
                            f_out.write(z.read(filename))
                            f_out.flush()
                            os.fsync(f_out.fileno())

                        # Thẩm định tính toàn vẹn của lát cắt trước khi lưu chính thức
                        if validate_dicom_file(tmp_path):
                            os.replace(tmp_path, target_path)
                            extracted_count += 1
                        else:
                            print(f"[CẢNH BÁO] Lát cắt không hợp lệ hoặc bị hỏng, loại bỏ: {filename}")
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)

            if extracted_count > 0:
                print(f"[THÀNH CÔNG] {patient_id}: Đã giải nén {extracted_count} lát cắt DICOM.")
                return extracted_count
            else:
                print(f"[CẢNH BÁO] Không có lát cắt hợp lệ nào được giải nén cho {patient_id} ở lần thử {attempt}/{max_retries}.")
        except Exception as e:
            print(f"[THỬ LẠI] Lần {attempt}/{max_retries} gặp sự cố khi tải {patient_id}: {e}")

        if attempt < max_retries:
            wait_time = backoff_base * (2 ** (attempt - 1))
            print(f"[CHỜ] Chờ {wait_time}s trước khi thử lại...")
            time.sleep(wait_time)

    if extracted_count == 0:
        print(f"[LỖI] Thất bại toàn diện sau {max_retries} lần thử đối với {patient_id}.")
    return extracted_count


def main():
    parser = argparse.ArgumentParser(description="Tải tập con LIDC-IDRI chuẩn y tế từ TCIA")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/datastore/doanhbc/minhdphan/datasets/lidc_idri/dicom/",
        help="Thư mục đích lưu trữ các file DICOM"
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Số lần thử lại tối đa cho mỗi series khi gặp sự cố mạng TCIA"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 70)
    print("KHỞI CHẠY TIẾN TRÌNH TẢI BỘ DỮ LIỆU LIDC-IDRI TỪ TCIA REST API")
    print(f"Thư mục lưu trữ: {args.output_dir}")
    print(f"Tổng số ca bệnh nhân dự kiến: {len(LIDC_TARGET_PATIENTS)}")
    print("=" * 70)

    total_slices = 0
    patient_counts = {}
    for pat_info in progress_bar(LIDC_TARGET_PATIENTS, desc="Tiến trình tải LIDC-IDRI"):
        try:
            count = download_series(pat_info, args.output_dir, max_retries=args.max_retries)
            patient_counts[pat_info["patient_id"]] = count
            total_slices += count
        except Exception as e:
            print(f"[LỖI] Không thể tải {pat_info['patient_id']}: {e}")
            patient_counts[pat_info["patient_id"]] = 0

    # Ghi file danh sách phân chia split
    splits_summary = {"train": [], "val": [], "test": []}
    for pat in LIDC_TARGET_PATIENTS:
        splits_summary[pat["split"]].append(pat["patient_id"])

    splits_file = os.path.join(args.output_dir, "..", "splits_summary.json")
    with open(splits_file, "w", encoding="utf-8") as f:
        json.dump(splits_summary, f, indent=2)

    print("\n" + "=" * 70)
    print("BÁO CÁO TỔNG KẾT TẢI DỮ LIỆU LIDC-IDRI:")
    for pat in LIDC_TARGET_PATIENTS:
        pid = pat["patient_id"]
        cnt = patient_counts.get(pid, 0)
        print(f"  - {pid} ({pat['split']}): {cnt} lát cắt hợp lệ")
    print(f"[HOÀN TẤT 100%] Tổng cộng đã tải và giải nén thành công {total_slices} lát cắt DICOM!")
    print(f"Bản tổng hợp split: {splits_file}")
    print("=" * 70)

    # Kiểm định tính sẵn sàng của tập dữ liệu
    val_slices = sum(patient_counts.get(p["patient_id"], 0) for p in LIDC_TARGET_PATIENTS if p["split"] == "val")
    test_slices = sum(patient_counts.get(p["patient_id"], 0) for p in LIDC_TARGET_PATIENTS if p["split"] == "test")
    if val_slices == 0 or test_slices == 0 or total_slices == 0:
        print(f"[LỖI NGHIÊM TRỌNG] Dữ liệu tải về không đạt yêu cầu: val={val_slices}, test={test_slices}, total={total_slices}")
        sys.exit(1)


if __name__ == "__main__":
    main()
