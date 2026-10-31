#!/usr/bin/env python3
"""
Đo lường độ trễ suy luận (Inference Latency Benchmark) trên phần cứng HPC-UIT.
Tuân thủ nghiêm ngặt giao thức khoa học:
  1. batch_size = 1
  2. Khởi động (warm-up): 10 lần suy luận bỏ qua đo lường
  3. Đo lường chính thức: 50 lần lặp với torch.cuda.synchronize() trước và sau mỗi forward pass
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


def parse_args():
    parser = argparse.ArgumentParser(description="Đo lường Latency trên GPU MIG HPC-UIT")
    parser.add_argument("--model", type=str, required=True,
                        choices=["learn", "regformer", "dudotrans", "longformer", "longnet", "mamba", "solar"],
                        help="Tên mô hình cần đo latency")
    parser.add_argument("--input_size", type=int, default=256, help="Kích thước ảnh vuông (mặc định 256)")
    parser.add_argument("--num_view", type=int, default=64, help="Số lượng góc chiếu (mặc định 64)")
    parser.add_argument("--num_detectors", type=int, default=512, help="Số kênh detector (mặc định 512)")
    parser.add_argument("--warmup_runs", type=int, default=10, help="Số lần chạy khởi động")
    parser.add_argument("--test_runs", type=int, default=50, help="Số lần đo lường chính thức")
    parser.add_argument("--angle_range_deg", type=float, default=120.0, help="Dải góc quét (mặc định 120.0)")
    parser.add_argument("--output_csv", type=str, default=None, help="Đường dẫn file CSV xuất kết quả")
    return parser.parse_args()


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
    fbp_input = torch.randn(1, 1, args.input_size, args.input_size, device=device)
    sino_input = torch.randn(1, 1, args.num_view, args.num_detectors, device=device)

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
    print(f"  • Hardware Profile: HPC-UIT MIG Slice")
    print("=" * 70)

    if args.output_csv:
        import csv
        file_exists = os.path.isfile(args.output_csv)
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        with open(args.output_csv, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["model", "mean_ms", "std_ms", "fps", "warmup_runs", "test_runs", "input_size", "views"])
            writer.writerow([args.model, f"{mean_ms:.3f}", f"{std_ms:.3f}", f"{fps:.2f}", args.warmup_runs, args.test_runs, args.input_size, args.num_view])
        print(f"[INFO] Đã ghi kết quả vào CSV: {args.output_csv}")


if __name__ == "__main__":
    main()

