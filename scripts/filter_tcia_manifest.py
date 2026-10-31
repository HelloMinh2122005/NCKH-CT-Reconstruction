#!/usr/bin/env python3
"""
Bộ lọc manifest TCIA cho dữ liệu AAPM Mayo Clinic LDCT.
Chỉ giữ lại các series ảnh tái tạo liều chuẩn (Full Dose Images) thuộc vùng bụng (ABDOMEN / mã bệnh nhân L*),
loại bỏ toàn bộ dữ liệu chiếu thô (projections) và ảnh liều thấp (Low Dose Images) để tiết kiệm băng thông.
"""

import os
import sys
import json
import argparse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple


def parse_manifest_uids(content: str) -> List[str]:
    """
    Phân tích nội dung file manifest .tcia để trích xuất danh sách SeriesInstanceUID.
    Bỏ qua các dòng tiêu đề và dòng chú thích.
    """
    header_prefixes = (
        "#",
        "downloadServerUrl",
        "includeAnnotation",
        "noOfrRetry",
        "databasketId",
        "manifestVersion",
        "ListOfSeriesToDownload",
    )
    uids = []
    # Duyệt qua từng dòng của manifest
    for raw_line in content.splitlines():
        line = raw_line.strip()
        # Bỏ qua dòng trống hoặc dòng thuộc phần header
        if not line or line.startswith(header_prefixes):
            continue
        uids.append(line)
    return uids


def is_target_series(meta: Dict) -> bool:
    """
    Kiểm tra series có thỏa mãn điều kiện giữ lại hay không:
    - SeriesDescription chính xác là 'Full Dose Images'
    - BodyPartExamined là 'ABDOMEN' hoặc PatientID bắt đầu bằng 'L'
    """
    desc = meta.get("SeriesDescription", "")
    # Chỉ chấp nhận series Full Dose Images
    if desc != "Full Dose Images":
        return False

    patient_id = meta.get("PatientID", "")
    body_part = meta.get("BodyPartExamined", "").upper()

    # Thỏa mãn nếu là vùng bụng hoặc bệnh nhân thuộc nhóm L (Mayo Clinic Abdomen)
    is_abdomen = (body_part == "ABDOMEN") or patient_id.startswith("L")
    return is_abdomen


def query_series_metadata(uid: str, timeout: int = 15) -> Optional[Dict]:
    """
    Truy vấn siêu dữ liệu của một SeriesInstanceUID qua TCIA REST API.
    """
    url = f"https://services.cancerimagingarchive.net/nbia-api/services/v1/getSeries?SeriesInstanceUID={uid}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        # Gửi HTTP GET request có giới hạn thời gian chờ
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data and len(data) > 0:
                return data[0]
    except Exception as e:
        # Ghi nhận lỗi khi gọi API cho UID này
        sys.stderr.write(f"[CẢNH BÁO] Không thể lấy metadata cho UID {uid}: {e}\n")
    return None


def format_clean_manifest(uids: List[str], databasket_id: str = "manifest-clean-full-dose.tcia") -> str:
    """
    Tạo nội dung file manifest .tcia chuẩn bảo toàn đúng header gốc của TCIA.
    """
    lines = [
        "downloadServerUrl=https://nbia.cancerimagingarchive.net/nbia-download/servlet/DownloadServlet",
        "includeAnnotation=true",
        "noOfrRetry=4",
        f"databasketId={databasket_id}",
        "manifestVersion=3.0",
        "ListOfSeriesToDownload=",
    ]
    # Nối danh sách các UID hợp lệ
    for uid in uids:
        lines.append(uid)
    return "\n".join(lines) + "\n"


def filter_manifest(
    input_path: str,
    output_path: str,
    max_workers: int = 16,
    timeout: int = 15,
) -> Tuple[List[Dict], float]:
    """
    Hàm thực thi chính: đọc manifest gốc, truy vấn song song metadata, lọc series và ghi manifest sạch.
    """
    # Đọc toàn bộ nội dung manifest đầu vào
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    uids = parse_manifest_uids(content)
    total_uids = len(uids)
    print(f"[*] Đọc thành công {total_uids} UIDs từ {input_path}")
    print(f"[*] Đang truy vấn TCIA REST API với {max_workers} luồng...")

    metadata_map = {}
    # Sử dụng ThreadPoolExecutor để tăng tốc độ truy vấn metadata qua mạng
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_uid = {
            executor.submit(query_series_metadata, uid, timeout): uid
            for uid in uids
        }
        completed = 0
        for future in as_completed(future_to_uid):
            uid = future_to_uid[future]
            completed += 1
            if completed % 20 == 0 or completed == total_uids:
                print(f"    Tiến độ: {completed}/{total_uids} ({completed * 100 // total_uids}%)")
            meta = future.result()
            if meta:
                metadata_map[uid] = meta

    # Lọc danh sách series theo tiêu chuẩn đã định
    filtered_series = []
    filtered_uids = []
    for uid in uids:
        meta = metadata_map.get(uid)
        if meta and is_target_series(meta):
            filtered_series.append(meta)
            filtered_uids.append(uid)

    # Sắp xếp kết quả theo PatientID để dễ theo dõi và tái lập
    filtered_series.sort(key=lambda x: x.get("PatientID", ""))
    filtered_uids = [s["SeriesInstanceUID"] for s in filtered_series]

    # Ghi file manifest sạch với tên file làm databasketId
    output_basket_id = os.path.basename(output_path)
    clean_manifest_content = format_clean_manifest(filtered_uids, databasket_id=output_basket_id)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(clean_manifest_content)

    total_slices = sum(int(s.get("ImageCount", 0)) for s in filtered_series)
    total_bytes = sum(int(s.get("FileSize", 0)) for s in filtered_series)
    total_size_mb = total_bytes / (1024.0 * 1024.0)

    print(f"\n[+] Đã lưu manifest sạch tại: {output_path}")
    print(f"[+] Giữ lại {len(filtered_series)} / {total_uids} series.")
    print(f"[+] Tổng số lát cắt: {total_slices}")
    print(f"[+] Tổng dung lượng chưa nén: {total_size_mb:.2f} MB")

    return filtered_series, total_size_mb


def main():
    parser = argparse.ArgumentParser(description="Lọc manifest TCIA chỉ lấy Full Dose Images cho AAPM CT")
    parser.add_argument(
        "--input_manifest",
        type=str,
        default="manifest-1793466381549.tcia",
        help="Đường dẫn file manifest gốc cần lọc",
    )
    parser.add_argument(
        "--output_manifest",
        type=str,
        default="manifest-clean-full-dose.tcia",
        help="Đường dẫn file manifest sạch sau khi lọc",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Số lượng luồng HTTP truy vấn song song",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=15,
        help="Thời gian chờ tối đa cho mỗi HTTP request (giây)",
    )
    args = parser.parse_args()

    filter_manifest(
        input_path=args.input_manifest,
        output_path=args.output_manifest,
        max_workers=args.workers,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
