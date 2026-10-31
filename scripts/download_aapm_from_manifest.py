#!/usr/bin/env python3
"""
Tải và tiền xử lý cấu trúc dữ liệu AAPM Mayo Clinic LDCT từ file manifest .tcia.
Chỉ tải các series ảnh tái tạo Full-Dose (Full Dose Images) để tiết kiệm băng thông.
Hỗ trợ tải song song đa luồng (ThreadPoolExecutor), tự động kiểm tra và bỏ qua các bệnh nhân đã tải xong.
Tự động tạo cấu trúc symlink tương thích 100% với CTSlice_Provider_LA.py và sinh mã băm SHA-256.
"""

import os
import sys
import json
import time
import zipfile
import io
import glob
import hashlib
import argparse
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional


def parse_args():
    parser = argparse.ArgumentParser(description="Tải dữ liệu ảnh AAPM CT từ manifest .tcia")
    parser.add_argument(
        "--manifest",
        type=str,
        default="/home/phandinhminh/Downloads/kltn/agents-research/manifest-clean-full-dose.tcia",
        help="Đường dẫn đến file .tcia manifest",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/phandinhminh/datasets/aapm",
        help="Thư mục lưu trữ dữ liệu tải về",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Số lượng luồng tải dữ liệu song song",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=4,
        help="Số lần thử lại tối đa cho mỗi series khi gặp lỗi mạng",
    )
    return parser.parse_args()


def get_series_metadata(uid: str, timeout: int = 15) -> Optional[Dict]:
    """
    Truy vấn metadata của một UID qua TCIA REST API.
    """
    url = f"https://services.cancerimagingarchive.net/nbia-api/services/v1/getSeries?SeriesInstanceUID={uid}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            if data and len(data) > 0:
                return data[0]
    except Exception as e:
        sys.stderr.write(f"[CẢNH BÁO] Lỗi lấy metadata cho UID {uid}: {e}\n")
    return None


def download_single_series(
    series_info: Dict,
    raw_dir: str,
    max_retries: int = 4,
) -> Tuple[str, bool, str]:
    """
    Tải và giải nén 1 series DICOM về thư mục raw/<PID>/full_3mm/.
    Có cơ chế tự động thử lại khi lỗi kết nối.
    """
    pid = series_info["PatientID"]
    desc = series_info.get("SeriesDescription", "Full Dose Images")
    uid = series_info["SeriesInstanceUID"]
    expected_count = int(series_info.get("ImageCount", 0))
    file_size_mb = int(series_info.get("FileSize", 0)) / (1024.0 * 1024.0)

    dest_patient_dir = os.path.join(raw_dir, pid, "full_3mm")
    os.makedirs(dest_patient_dir, exist_ok=True)

    # Kiểm tra xem dữ liệu đã tồn tại đủ chưa
    existing_files = [f for f in os.listdir(dest_patient_dir) if f.endswith((".dcm", ".IMA"))]
    if expected_count > 0 and len(existing_files) >= expected_count:
        return pid, True, f"Đã tồn tại đủ {len(existing_files)} files. Bỏ qua."

    download_url = f"https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID={uid}"

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(download_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = resp.read()

            with zipfile.ZipFile(io.BytesIO(data)) as z:
                # Sắp xếp tên file trong zip để đánh số tuần tự chuẩn
                dcm_names = sorted([f for f in z.namelist() if f.endswith((".dcm", ".IMA"))])
                for idx, filename in enumerate(dcm_names, 1):
                    content = z.read(filename)
                    # Chuẩn hoá tên file: 00000001.dcm
                    out_path = os.path.join(dest_patient_dir, f"{idx:08d}.dcm")
                    with open(out_path, "wb") as f_out:
                        f_out.write(content)

            actual_saved = len([f for f in os.listdir(dest_patient_dir) if f.endswith((".dcm", ".IMA"))])
            return pid, True, f"Thành công: đã tải và giải nén {actual_saved} lát cắt (~{file_size_mb:.1f} MB)."
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 * attempt)
            else:
                return pid, False, f"Thất bại sau {max_retries} lần thử: {e}"

    return pid, False, "Không thể hoàn thành tải dữ liệu."


def setup_split_symlinks(raw_dir: str, split_dir: str, patients: List[str]):
    """
    Tạo cấu trúc symlink chuẩn cho CTSlice_Provider_LA:
    - test = L058, L064, L094, L107, L310 (nếu có)
    - val = L110, L333 (nếu có)
    - train = tất cả các bệnh nhân còn lại trong danh sách
    """
    test_pids = {"L058", "L064", "L094", "L107", "L310"}
    val_pids = {"L110", "L333"}

    total_links = 0
    split_counts = {"train": 0, "val": 0, "test": 0}

    for pid in patients:
        if pid in test_pids:
            split_name = "test"
        elif pid in val_pids:
            split_name = "val"
        else:
            split_name = "train"

        link_dir = os.path.join(split_dir, split_name, pid, "full_3mm")
        os.makedirs(link_dir, exist_ok=True)

        src_dir = os.path.join(raw_dir, pid, "full_3mm")
        if not os.path.exists(src_dir):
            continue

        dcm_files = sorted([f for f in os.listdir(src_dir) if f.endswith((".dcm", ".IMA"))])
        for idx, dcm_file in enumerate(dcm_files, 1):
            src_file_path = os.path.abspath(os.path.join(src_dir, dcm_file))
            # Định dạng tên symlink chuẩn Mayo Clinic
            link_name = f"{pid}_FD_3_1.CT.{idx:08d}.IMA"
            link_path = os.path.join(link_dir, link_name)
            if os.path.islink(link_path) or os.path.exists(link_path):
                os.remove(link_path)
            # Tạo đường dẫn tương đối (relative symlink) để không bị gãy khi chuyển dữ liệu lên HPC server
            rel_src_path = os.path.relpath(src_file_path, link_dir)
            os.symlink(rel_src_path, link_path)
            total_links += 1
            split_counts[split_name] += 1

    print(f"[+] Đã tạo {total_links} symlinks trong thư mục {split_dir}:")
    print(f"    • Train: {split_counts['train']} lát cắt")
    print(f"    • Val:   {split_counts['val']} lát cắt")
    print(f"    • Test:  {split_counts['test']} lát cắt")


def generate_sha256_manifest(raw_dir: str, manifest_path: str):
    """
    Sinh file băm SHA-256 cho toàn bộ file ảnh DICOM đã tải về.
    """
    print(f"[*] Đang tính toán mã băm SHA-256 cho các file DICOM...")
    records = []
    # Quét tất cả các file .dcm trong thư mục raw
    for root, _, files in sorted(os.walk(raw_dir)):
        for f in sorted(files):
            if f.endswith((".dcm", ".IMA")):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, raw_dir)
                hasher = hashlib.sha256()
                with open(full_path, "rb") as fp:
                    while chunk := fp.read(1024 * 1024):
                        hasher.update(chunk)
                records.append(f"{hasher.hexdigest()}  {rel_path}\n")

    with open(manifest_path, "w", encoding="utf-8") as f:
        f.writelines(records)
    print(f"[+] Đã ghi file checksum {manifest_path} ({len(records)} files).")


def main():
    args = parse_args()
    manifest_path = os.path.abspath(args.manifest)
    if not os.path.exists(manifest_path):
        print(f"[LỖI] Không tìm thấy file manifest: {manifest_path}")
        sys.exit(1)

    print(f"[*] Đang đọc file manifest: {manifest_path}")
    uids = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith((
                "#",
                "downloadServerUrl",
                "includeAnnotation",
                "noOfrRetry",
                "databasketId",
                "manifestVersion",
                "ListOfSeriesToDownload",
            )):
                continue
            uids.append(line)

    print(f"[*] Tìm thấy {len(uids)} Series Instance UIDs.")
    print(f"[*] Đang truy vấn siêu dữ liệu để lập kế hoạch tải...")

    # Tra cứu metadata song song
    series_list = []
    with ThreadPoolExecutor(max_workers=16) as meta_executor:
        meta_futures = {meta_executor.submit(get_series_metadata, uid): uid for uid in uids}
        for fut in as_completed(meta_futures):
            meta = fut.result()
            if meta:
                series_list.append(meta)

    # Sắp xếp theo mã bệnh nhân
    series_list.sort(key=lambda s: s.get("PatientID", ""))
    print(f"[*] Lấy thành công metadata cho {len(series_list)} series.")

    raw_dir = os.path.join(args.output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # Phân loại series cần tải và series đã có sẵn
    to_download = []
    already_done = []
    for s in series_list:
        pid = s["PatientID"]
        exp_count = int(s.get("ImageCount", 0))
        patient_raw = os.path.join(raw_dir, pid, "full_3mm")
        if os.path.exists(patient_raw):
            n_files = len([f for f in os.listdir(patient_raw) if f.endswith((".dcm", ".IMA"))])
            if exp_count > 0 and n_files >= exp_count:
                already_done.append((pid, n_files))
                continue
        to_download.append(s)

    print(f"\n[+] Số bệnh nhân đã tải hoàn thành trước đó: {len(already_done)}")
    for pid, cnt in already_done:
        print(f"    • {pid}: {cnt} lát cắt (ĐÃ TỒN TẠI - BỎ QUA)")

    print(f"[+] Số bệnh nhân cần tải mới: {len(to_download)}")

    if to_download:
        print(f"\n🚀 Bắt đầu tải song song {len(to_download)} series với {args.workers} workers...")
        start_time = time.time()
        completed_count = 0
        total_to_dl = len(to_download)

        with ThreadPoolExecutor(max_workers=args.workers) as dl_executor:
            futures = {
                dl_executor.submit(download_single_series, s, raw_dir, args.max_retries): s
                for s in to_download
            }
            for fut in as_completed(futures):
                completed_count += 1
                pid, success, msg = fut.result()
                status_icon = "✅" if success else "❌"
                print(f"[{completed_count:02d}/{total_to_dl:02d}] {status_icon} Bệnh nhân {pid}: {msg}")

        elapsed = time.time() - start_time
        print(f"\n[+] Hoàn tất quá trình tải sau {elapsed:.1f} giây ({elapsed / 60.0:.2f} phút).")

    # Thiết lập thư mục symlinks
    split_dir = os.path.join(args.output_dir, "split")
    all_patients = sorted(list(set(s["PatientID"] for s in series_list)))
    setup_split_symlinks(raw_dir, split_dir, all_patients)

    # Sinh file checksum
    checksum_path = os.path.join(args.output_dir, "manifest.sha256")
    generate_sha256_manifest(raw_dir, checksum_path)

    print("\n" + "=" * 70)
    print("🎉 TOÀN BỘ TIẾN TRÌNH TẢI VÀ THIẾT LẬP DỮ LIỆU ĐÃ HOÀN TẤT THÀNH CÔNG!")
    print(f"   • Thư mục Raw:      {raw_dir}")
    print(f"   • Thư mục Split:    {split_dir}")
    print(f"   • File Checksum:    {checksum_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
