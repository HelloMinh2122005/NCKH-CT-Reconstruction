"""
================================================================================
SCRIPT TRỰC QUAN HÓA & KẾT XUẤT ẢNH ĐỐI SÁNH (VISUALIZE BENCHMARK RESULTS)
Dự án: Nghiên cứu Tái tạo Ảnh Cắt lớp CT Góc Giới hạn (Limited-Angle CT Reconstruction)
Tác giả: MinhPD — Nhóm Nghiên cứu Tái tạo Ảnh Y tế (VNU-HCM UIT)
Mô tả: 
    - Nạp các mô hình AI đã huấn luyện (LEARN_LongNet, LEARN_Mamba, LEARN_Longformer, SOLAR).
    - Tái tạo các lát cắt kiểm thử độc lập (Test Patient L310).
    - Tự động tạo thư mục riêng cho từng lát cắt (slice_050, slice_100, slice_150, ...).
    - Lưu các file ảnh PNG riêng lẻ (Ground Truth, FBP thô, Tái tạo AI, Bản đồ sai số Error Map).
    - Lưu file ảnh tổng hợp (Panel đối sánh 4 khung hình) phục vụ chèn vào bài báo/báo cáo.
================================================================================
"""

import os
import sys
from pathlib import Path

# Đảm bảo thư mục gốc dự án luôn nằm trong sys.path để import các module data/ và baselines/
PROJECT_ROOT = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# Tương thích PyTorch 2.6+ (tránh lỗi WeightsUnpickler khi nạp checkpoint Lightning chứa metadata numpy)
_orig_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _safe_torch_load

from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)

# Import DataModule và Mô hình
from data.datamodule_LA import LimitedAngleCTDataModule
from baselines.LEARN_LongNet.models import LEARN_LongNet_LA
from baselines.LEARN_Mamba.models import LEARN_Mamba_LA


def parse_args():
    """
    Khai báo các tham số dòng lệnh phục vụ việc kết xuất ảnh trực quan hóa.
    """
    parser = argparse.ArgumentParser(description="Trực quan hóa và xuất ảnh đối sánh CT góc giới hạn")
    
    # Danh sách lát cắt cần trực quan hóa
    parser.add_argument(
        "--slices",
        nargs="+",
        type=int,
        default=[50, 100, 150],
        help="Danh sách chỉ số lát cắt kiểm thử cần xuất ảnh (mặc định: 50 100 150)"
    )
    
    # Cấu hình góc quét CT
    parser.add_argument(
        "--angle_range_deg",
        type=float,
        default=120.0,
        help="Độ rộng cung quét góc giới hạn (120.0 hoặc 90.0 độ)"
    )
    parser.add_argument(
        "--num_view",
        type=int,
        default=64,
        help="Số góc chiếu trong dải quét (64 views)"
    )
    parser.add_argument(
        "--num_detectors",
        type=int,
        default=512,
        help="Số lượng cảm biến detector (512)"
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=256,
        help="Kích thước ma trận ảnh (256x256 pixel)"
    )
    
    # Đường dẫn dữ liệu và thư mục xuất ảnh
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/datastore/uittogether3/LuuTru/MinhPD/dataset/limited_angle/",
        help="Đường dẫn thư mục chứa dữ liệu .npy cache"
    )
    parser.add_argument(
        "--dicom_dir",
        type=str,
        default="/datastore/uittogether3/LuuTru/Thanhld/CT-Reconstruction/split/",
        help="Đường dẫn thư mục DICOM gốc"
    )
    parser.add_argument(
        "--test_patients",
        nargs="+",
        default=["L310"],
        help="Mã bệnh nhân test (mặc định: L310)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/datastore/uittogether3/LuuTru/MinhPD/visualizations/",
        help="Thư mục gốc lưu trữ ảnh trực quan hóa"
    )
    
    # Đường dẫn Checkpoint các mô hình
    parser.add_argument(
        "--longnet_ckpt",
        type=str,
        default="/datastore/uittogether3/LuuTru/MinhPD/saved_models/LEARN_LongNet/last.ckpt",
        help="Đường dẫn file trọng số LEARN_LongNet"
    )
    parser.add_argument(
        "--mamba_ckpt",
        type=str,
        default="/datastore/uittogether3/LuuTru/MinhPD/saved_models/LEARN_Mamba/mamba_la-epoch=17-val_psnr=27.66-val_ssim=0.7373.ckpt",
        help="Đường dẫn file trọng số LEARN_Mamba"
    )
    
    return parser.parse_args()


def save_single_image(array: np.ndarray, file_path: str, is_error_map: bool = False, error_scale: float = 5.0):
    """
    Lưu một mảng 2D numpy (float32 [0, 1]) thành file ảnh PNG chuẩn 8-bit.
    Nếu là Error Map, sẽ nhân tỉ lệ khuếch đại sai số và lưu với colormap 'jet'.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if is_error_map:
        # Chuẩn hóa bản đồ sai lệch và ánh xạ màu nhiệt
        err = np.clip(array * error_scale, 0.0, 1.0)
        cmap = plt.get_cmap("jet")
        err_colored = cmap(err)[:, :, :3]  # Lấy RGB bỏ Alpha
        err_uint8 = (err_colored * 255.0).astype(np.uint8)
        img = Image.fromarray(err_uint8)
        img.save(file_path)
    else:
        # Chuẩn hóa ảnh CT xám (Grayscale [0, 255])
        clipped = np.clip(array, 0.0, 1.0)
        img_uint8 = (clipped * 255.0).astype(np.uint8)
        img = Image.fromarray(img_uint8, mode="L")
        img.save(file_path)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("🎨 BẮT ĐẦU KẾT XUẤT ẢNH TRỰC QUAN HÓA (VISUALIZATION PIPELINE)")
    print(f"- Thiết bị tính toán: {device}")
    print(f"- Cung quét góc: {args.angle_range_deg}° (Views: {args.num_view})")
    print(f"- Các lát cắt chỉ định: {args.slices}")
    print(f"- Thư mục xuất ảnh: {args.output_dir}")
    print("=" * 80)

    # Bước 1: Thiết lập góc quét và DataModule
    start_ang = np.deg2rad(-args.angle_range_deg / 2.0)
    end_ang = np.deg2rad(args.angle_range_deg / 2.0)
    setting_tag = f"limited_ang_{int(args.angle_range_deg)}deg_numview_{args.num_view}_size_{args.input_size}_noise_0"

    datamodule = LimitedAngleCTDataModule(
        dicom_dir=args.dicom_dir,
        cache_dir=args.cache_dir,
        setting_tag=setting_tag,
        start_ang=start_ang,
        end_ang=end_ang,
        num_view=args.num_view,
        num_detectors=args.num_detectors,
        input_size=args.input_size,
        poisson_level=0,
        gaussian_level=0,
        use_precomputed=True,
        batch_size=1,
        num_workers=2,
        test_patients=args.test_patients,
    )
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()
    test_dataset = test_loader.dataset

    # Bước 2: Nạp các mô hình đã huấn luyện
    models = {}
    if os.path.exists(args.longnet_ckpt):
        print(f"📦 Đang nạp LEARN_LongNet từ: {args.longnet_ckpt}")
        model_ln = LEARN_LongNet_LA.load_from_checkpoint(args.longnet_ckpt, map_location=device)
        model_ln.to(device)
        model_ln.eval()
        models["LongNet"] = model_ln
    else:
        print(f"⚠️ Không tìm thấy checkpoint LongNet tại: {args.longnet_ckpt}")

    if os.path.exists(args.mamba_ckpt):
        print(f"📦 Đang nạp LEARN_Mamba từ: {args.mamba_ckpt}")
        model_mb = LEARN_Mamba_LA.load_from_checkpoint(args.mamba_ckpt, map_location=device)
        model_mb.to(device)
        model_mb.eval()
        models["Mamba"] = model_mb
    else:
        print(f"⚠️ Không tìm thấy checkpoint Mamba tại: {args.mamba_ckpt}")

    angle_tag = f"{int(args.angle_range_deg)}deg"

    # Bước 3: Duyệt qua từng lát cắt chỉ định và thực hiện tái tạo
    for slice_idx in args.slices:
        if slice_idx >= len(test_dataset):
            print(f"⚠️ Lát cắt {slice_idx} vượt quá tổng số lát cắt ({len(test_dataset)}). Bỏ qua.")
            continue

        print(f"\n🖼️ Đang xử lý Lát cắt #{slice_idx} (Bệnh nhân {args.test_patients[0]})...")
        phantom, fbp_u, sino_noisy = test_dataset[slice_idx]

        # Đưa tensor lên GPU và thêm batch dimension: [1, 1, H, W]
        phantom_t = phantom.unsqueeze(0).to(device)
        fbp_t = fbp_u.unsqueeze(0).to(device)
        sino_t = sino_noisy.unsqueeze(0).to(device)

        gt_np = phantom_t.squeeze().detach().cpu().numpy()
        fbp_np = fbp_t.squeeze().detach().cpu().numpy()

        # Tính chỉ số của FBP thô ban đầu (data_range là độ rộng dải giá trị float)
        dr = float(phantom_t.max() - phantom_t.min())
        data_range = dr if dr > 0.0 else 1.0
        fbp_psnr = peak_signal_noise_ratio(fbp_t, phantom_t, data_range=data_range).item()
        fbp_ssim = structural_similarity_index_measure(fbp_t, phantom_t, data_range=data_range).item()


        slice_dir = os.path.join(args.output_dir, angle_tag, f"slice_{slice_idx:03d}")
        os.makedirs(slice_dir, exist_ok=True)

        # Lưu ảnh Ground Truth và FBP Input
        save_single_image(gt_np, os.path.join(slice_dir, "1_ground_truth.png"))
        save_single_image(fbp_np, os.path.join(slice_dir, "2_fbp_input.png"))

        recons = {}
        error_maps = {}
        metrics = {}

        # Chạy suy luận qua các mô hình
        with torch.no_grad():
            for m_name, net in models.items():
                pred_t = net(fbp_t, sino_t)
                pred_np = pred_t.squeeze().detach().cpu().numpy()
                err_np = np.abs(pred_np - gt_np)

                psnr_val = peak_signal_noise_ratio(pred_t, phantom_t, data_range=data_range).item()
                ssim_val = structural_similarity_index_measure(pred_t, phantom_t, data_range=data_range).item()

                recons[m_name] = pred_np
                error_maps[m_name] = err_np
                metrics[m_name] = (psnr_val, ssim_val)

                # Lưu từng file ảnh rời
                save_single_image(pred_np, os.path.join(slice_dir, f"3_recon_{m_name.lower()}.png"))
                save_single_image(err_np, os.path.join(slice_dir, f"4_error_map_{m_name.lower()}.png"), is_error_map=True)

                print(f"  • [{m_name}] PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f} -> Đã lưu PNG.")

        # Bước 4: Tạo ảnh đối sánh đa khung hình (Comparison Summary Figure)
        # Bố cục hàng ngang: Ground Truth | FBP Thô | Mamba (nếu có) | LongNet (nếu có) | Error Map
        num_cols = 2 + len(models) * 2  # GT, FBP, + [Recon, Err] cho từng mô hình
        fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4.5), dpi=300)
        
        col_idx = 0
        # 1. Ground Truth
        axes[col_idx].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
        axes[col_idx].set_title("1. Ground Truth\n(Reference)", fontsize=11, fontweight="bold")
        axes[col_idx].axis("off")
        col_idx += 1

        # 2. FBP Thô
        axes[col_idx].imshow(fbp_np, cmap="gray", vmin=0, vmax=1)
        axes[col_idx].set_title(f"2. FBP Input ({args.angle_range_deg:.0f}°)\nPSNR: {fbp_psnr:.2f} dB | SSIM: {fbp_ssim:.4f}", fontsize=10)
        axes[col_idx].axis("off")
        col_idx += 1

        # 3. Từng mô hình AI
        for m_name in models.keys():
            p_val, s_val = metrics[m_name]
            axes[col_idx].imshow(recons[m_name], cmap="gray", vmin=0, vmax=1)
            axes[col_idx].set_title(f"Recon ({m_name})\nPSNR: {p_val:.2f} dB | SSIM: {s_val:.4f}", fontsize=10, fontweight="bold", color="darkblue")
            axes[col_idx].axis("off")
            col_idx += 1

            im_err = axes[col_idx].imshow(np.clip(error_maps[m_name] * 5.0, 0, 1), cmap="jet", vmin=0, vmax=1)
            axes[col_idx].set_title(f"Error Map ({m_name})\n(|Recon - GT| x 5)", fontsize=10, color="darkred")
            axes[col_idx].axis("off")
            col_idx += 1

        plt.suptitle(f"Limited-Angle CT Reconstruction ({args.angle_range_deg:.0f}°) — Patient L310, Slice #{slice_idx}", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        
        summary_path = os.path.join(slice_dir, "comparison_summary.png")
        plt.savefig(summary_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"  ✨ Đã lưu file tổng hợp: {summary_path}")

    print("\n" + "=" * 80)
    print(f"🎉 HOÀN TẤT KẾT XUẤT ẢNH TRỰC QUAN HÓA TẠI: {args.output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
