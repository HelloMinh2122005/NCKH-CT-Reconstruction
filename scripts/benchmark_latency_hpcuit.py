#!/usr/bin/env python3
"""
Đo lường độ trễ suy luận (Inference Latency Benchmark) trên phần cứng HPC-UIT.
Tuân thủ nghiêm ngặt giao thức khoa học:
  1. batch_size = 1
  2. Khởi động (warm-up): 10 lần suy luận bỏ qua đo lường
  # Cập nhật 2026-11-02 (GT-03 / Wave E): Nâng mặc định lên >= 20 lần chạy warm-up
  3. Đo lường chính thức: 50 lần lặp với torch.cuda.synchronize() trước và sau mỗi forward pass
  # Cập nhật 2026-11-02 (GT-03 / Wave E): Nâng mặc định lên >= 100 lần chạy đo lường chính thức
  4. Báo cáo: Mean ± Std thời gian suy luận (ms) và FPS
  5. Tuyệt đối không nội suy hay trộn lẫn với số liệu cũ trên A100.
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Danh sách 9 cột chuẩn của file CSV kết quả đo latency theo GT-03 / Wave E
CSV_COLUMNS = [
    "model",
    "mean_ms",
    "std_ms",
    "fps",
    "warmup_runs",
    "test_runs",
    "input_size",
    "views",
    "hardware_profile",
]


def format_csv_row(model, mean_ms, std_ms, fps, warmup_runs, test_runs, input_size, views, hardware_profile):
    """Định dạng một dòng kết quả đo lường độ trễ cho tệp CSV."""
    return [
        str(model),
        f"{mean_ms:.3f}",
        f"{std_ms:.3f}",
        f"{fps:.2f}",
        int(warmup_runs),
        int(test_runs),
        int(input_size),
        int(views),
        str(hardware_profile),
    ]


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Đo lường Latency trên GPU MIG HPC-UIT")
    parser.add_argument("--model", type=str, required=True,
                        choices=["learn", "regformer", "dudotrans", "longformer", "longnet", "mamba", "solar"],
                        help="Tên mô hình cần đo latency")
    parser.add_argument("--input_size", type=int, default=256, help="Kích thước ảnh vuông (mặc định 256)")
    parser.add_argument("--num_view", type=int, default=64, help="Số lượng góc chiếu (mặc định 64)")
    parser.add_argument("--num_detectors", type=int, default=512, help="Số kênh detector (mặc định 512)")
    parser.add_argument("--warmup_runs", type=int, default=20, help="Số lần chạy khởi động (mặc định 20, tuân thủ >=20 warm-up iterations)")
    parser.add_argument("--test_runs", type=int, default=100, help="Số lần đo lường chính thức (mặc định 100, tuân thủ >=100 timed iterations)")
    parser.add_argument("--angle_range_deg", type=float, default=120.0, help="Dải góc quét (mặc định 120.0)")
    parser.add_argument("--hardware_profile", type=str, default="HPC-UIT MIG Slice", help="Hồ sơ phần cứng đo lường (mặc định: HPC-UIT MIG Slice)")
    parser.add_argument("--output_csv", type=str, default=None, help="Đường dẫn file CSV xuất kết quả")
    return parser.parse_args(args)


def load_model(args, device):
    """Khởi tạo mô hình dựa trên tên được chọn."""
    start_ang = float(np.deg2rad(-args.angle_range_deg / 2.0))
    end_ang = float(np.deg2rad(args.angle_range_deg / 2.0))

    if args.model == "learn":
        from baselines.LEARN.models import LEARN_LA
        model = LEARN_LA(
            num_view=args.num_view,
            num_detectors=args.num_detectors,
            start_ang=start_ang,
            end_ang=end_ang,
            input_size=args.input_size,
        )
    elif args.model == "regformer":
        from baselines.RegFormer.models import RegFormer_LA
        model = RegFormer_LA(
            num_view=args.num_view,
            num_detectors=args.num_detectors,
            start_ang=start_ang,
            end_ang=end_ang,
            input_size=args.input_size,
        )
    elif args.model == "dudotrans":
        from baselines.DuDoTrans.models import DuDoTrans_LA
        model = DuDoTrans_LA(
            num_view=args.num_view,
            num_detectors=args.num_detectors,
            start_ang=start_ang,
            end_ang=end_ang,
            input_size=args.input_size,
        )
    elif args.model == "longformer":
        from baselines.LEARN_Longformer.models import LEARN_Longformer_LA
        model = LEARN_Longformer_LA(
            num_view=args.num_view,
            num_detectors=args.num_detectors,
            start_ang=start_ang,
            end_ang=end_ang,
            input_size=args.input_size,
        )
    elif args.model == "longnet":
        from baselines.LEARN_LongNet.models import LEARN_LongNet_LA
        model = LEARN_LongNet_LA(
            num_view=args.num_view,
            num_detectors=args.num_detectors,
            start_ang=start_ang,
            end_ang=end_ang,
            input_size=args.input_size,
        )
    elif args.model == "mamba":
        from baselines.LEARN_Mamba.models import LEARN_Mamba_LA
        model = LEARN_Mamba_LA(
            num_view=args.num_view,
            num_detectors=args.num_detectors,
            start_ang=start_ang,
            end_ang=end_ang,
            input_size=args.input_size,
        )
    elif args.model == "solar":
        from baselines.SOLAR_DualMamba.models import SOLAR_DualMamba_LA
        model = SOLAR_DualMamba_LA(
            num_view=args.num_view,
            num_detectors=args.num_detectors,
            start_ang=start_ang,
            end_ang=end_ang,
            input_size=args.input_size,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Bắt đầu benchmark latency cho mô hình: {args.model.upper()}")
    print(f"[INFO] Thiết bị: {device}")
    if torch.cuda.is_available():
        print(f"[INFO] Tên GPU: {torch.cuda.get_device_name(0)}")

    model = load_model(args, device)

    # Dữ liệu đầu vào giả lập batch=1
    # fbp: [1, 1, H, W], sino: [1, 1, num_view, num_detectors]
    # Khởi tạo tensor giả lập chuẩn xác FP32 (float32) và batch_size=1
    fbp_input = torch.randn(1, 1, args.input_size, args.input_size, dtype=torch.float32, device=device)
    sino_input = torch.randn(1, 1, args.num_view, args.num_detectors, dtype=torch.float32, device=device)

    # Ràng buộc kiểm tra nghiêm ngặt (invariants): batch size = 1, dtype = float32
    assert fbp_input.shape[0] == 1 and sino_input.shape[0] == 1, "Batch size bắt buộc phải bằng 1"
    assert fbp_input.dtype == torch.float32 and sino_input.dtype == torch.float32, "Dữ liệu đầu vào bắt buộc là FP32"
    assert fbp_input.shape[2:] == (args.input_size, args.input_size), "Kích thước ảnh fbp_input không khớp input_size"
    assert sino_input.shape[2:] == (args.num_view, args.num_detectors), "Kích thước sinogram không khớp num_view và num_detectors"

    print(f"[INFO] Thực hiện {args.warmup_runs} lượt warm-up...")
    with torch.no_grad():
        for _ in range(args.warmup_runs):
            _ = model(fbp_input, sino_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    print(f"[INFO] Đo lường chính thức qua {args.test_runs} lượt...")
    latencies_ms = []
    with torch.no_grad():
        for _ in range(args.test_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            _ = model(fbp_input, sino_input)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    mean_ms = float(np.mean(latencies_ms))
    std_ms = float(np.std(latencies_ms))
    fps = 1000.0 / mean_ms

    print("=" * 70)
    print(f"📊 KẾT QUẢ ĐỘ TRỄ SUY LUẬN (LATENCY BENCHMARK):")
    print(f"  • Mô hình:          {args.model.upper()}")
    print(f"  • Mean Latency:     {mean_ms:.2f} ms")
    print(f"  • Std Latency:      {std_ms:.2f} ms")
    print(f"  • FPS:              {fps:.2f} frames/sec")
    print(f"  • Hardware Profile: {args.hardware_profile}")
    print("=" * 70)

    if args.output_csv:
        import csv
        file_exists = os.path.isfile(args.output_csv)
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        with open(args.output_csv, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(CSV_COLUMNS)
            row = format_csv_row(
                model=args.model,
                mean_ms=mean_ms,
                std_ms=std_ms,
                fps=fps,
                warmup_runs=args.warmup_runs,
                test_runs=args.test_runs,
                input_size=args.input_size,
                views=args.num_view,
                hardware_profile=args.hardware_profile,
            )
            writer.writerow(row)
        print(f"[INFO] Đã ghi kết quả vào CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
