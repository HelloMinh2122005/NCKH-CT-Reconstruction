"""
================================================================================
SCRIPT TRỰC QUAN HÓA & KẾT XUẤT ẢNH ĐỐI SÁNH: KIẾN TRÚC ĐỀ XUẤT SOLAR
(SOLAR BENCHMARK VISUALIZATION PIPELINE)
Dự án: Nghiên cứu Tái tạo Ảnh Cắt lớp CT Góc Giới hạn (Limited-Angle CT Reconstruction)
Tác giả: MinhPD — Nhóm Nghiên cứu Tái tạo Ảnh Y tế (VNU-HCM UIT)
Mô tả: 
    - Nạp các mô hình đề xuất bậc 2: SOLAR_LongNet, SOLAR_Mamba, SOLAR_Longformer.
    - Nạp các mô hình đối chứng bậc 1: LEARN_LongNet, LEARN_Mamba, LEARN_Longformer.
    - Tái tạo các lát cắt kiểm thử độc lập từ tập bệnh nhân kiểm định mù (Patient L310).
    - Tự động tạo thư mục riêng cho từng lát cắt (slice_050, slice_100, slice_150, ...).
    - Lưu các file ảnh PNG riêng lẻ (Ground Truth, FBP thô, Tái tạo AI, Bản đồ sai số Error Map x5).
    - Kết xuất các Panel đối sánh tổng hợp chuẩn bài báo khoa học (300 DPI):
        + Panel 1: Đối sánh 3 biến thể SOLAR (SOLAR_LongNet, SOLAR_Mamba, SOLAR_Longformer).
        + Panel 2: Đối sánh trực tiếp Baseline bậc 1 (LEARN) vs Đề xuất bậc 2 (SOLAR).
        + Panel 3: Panel tổng hợp toàn diện (Comparison Summary).
    - Tự động lưu trữ và đồng bộ vào thư mục báo cáo ngày mới nhất (reports/sep-05-2026/visualizations/).
================================================================================
"""

import os
import sys
import glob
import shutil
from pathlib import Path

# Đảm bảo thư mục gốc dự án luôn nằm trong sys.path
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

# Import DataModule
from data.datamodule_LA import LimitedAngleCTDataModule

# Import Mô hình Đề xuất SOLAR (Bậc 2 Newton-CG)
from baselines.SOLAR_LongNet.models import SOLAR_LongNet_LA
from baselines.SOLAR_Mamba.models import SOLAR_Mamba_LA
from baselines.SOLAR_Longformer.models import SOLAR_Longformer_LA

# Import Mô hình Đối chứng LEARN (Bậc 1 Gradient Descent)
from baselines.LEARN_LongNet.models import LEARN_LongNet_LA
from baselines.LEARN_Mamba.models import LEARN_Mamba_LA
from baselines.LEARN_Longformer.models import LEARN_Longformer_LA


def find_best_checkpoint(model_dir: str, default_pattern: str = "*epoch=*.ckpt", fallback: str = "last.ckpt") -> str:
    """
    Tự động tìm kiếm checkpoint tốt nhất trong thư mục mô hình.
    Ưu tiên theo thứ tự:
    1. File checkpoint có val_psnr cao nhất (nếu parse được từ tên file).
    2. File checkpoint mới nhất theo pattern.
    3. File last.ckpt làm phương án dự phòng.
    """
    if not os.path.isdir(model_dir):
        return ""
    
    candidates = glob.glob(os.path.join(model_dir, default_pattern))
    if candidates:
        def extract_psnr(path):
            name = os.path.basename(path)
            if "val_psnr=" in name:
                try:
                    part = name.split("val_psnr=")[1]
                    val_str = part.split("-")[0].replace(".ckpt", "")
                    return float(val_str)
                except Exception:
                    pass
            return os.path.getmtime(path)
        
        candidates.sort(key=extract_psnr, reverse=True)
        return candidates[0]
    
    fallback_path = os.path.join(model_dir, fallback)
    if os.path.isfile(fallback_path):
        return fallback_path
    return ""


def parse_args():
    """
    Khai báo các tham số dòng lệnh phục vụ việc kết xuất ảnh trực quan hóa SOLAR.
    """
    parser = argparse.ArgumentParser(description="Trực quan hóa và xuất ảnh đối sánh 3 baseline SOLAR")
    
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
    parser.add_argument(
        "--report_dir",
        type=str,
        default="/datastore/uittogether3/LuuTru/MinhPD/reports/sep-05-2026/visualizations/",
        help="Thư mục lưu trữ báo cáo trực quan hóa ngày hiện tại"
    )
    
    # Đường dẫn Checkpoint mô hình SOLAR đề xuất
    parser.add_argument(
        "--solar_longnet_ckpt",
        type=str,
        default=None,
        help="Đường dẫn checkpoint SOLAR_LongNet (mặc định: tự tìm best checkpoint)"
    )
    parser.add_argument(
        "--solar_mamba_ckpt",
        type=str,
        default=None,
        help="Đường dẫn checkpoint SOLAR_Mamba (mặc định: tự tìm best checkpoint)"
    )
    parser.add_argument(
        "--solar_longformer_ckpt",
        type=str,
        default=None,
        help="Đường dẫn checkpoint SOLAR_Longformer (mặc định: tự tìm best checkpoint)"
    )
    
    # Đường dẫn Checkpoint mô hình LEARN đối chứng
    parser.add_argument(
        "--learn_longnet_ckpt",
        type=str,
        default="/datastore/uittogether3/LuuTru/MinhPD/saved_models/LEARN_LongNet/last.ckpt",
        help="Đường dẫn checkpoint LEARN_LongNet"
    )
    parser.add_argument(
        "--learn_mamba_ckpt",
        type=str,
        default="/datastore/uittogether3/LuuTru/MinhPD/saved_models/LEARN_Mamba/mamba_la-epoch=17-val_psnr=27.66-val_ssim=0.7373.ckpt",
        help="Đường dẫn checkpoint LEARN_Mamba"
    )
    parser.add_argument(
        "--learn_longformer_ckpt",
        type=str,
        default="/datastore/uittogether3/LuuTru/MinhPD/saved_models/LEARN_Longformer/longformer_la-epoch=45-val_psnr=34.77-val_ssim=0.9383.ckpt",
        help="Đường dẫn checkpoint LEARN_Longformer"
    )
    
    parser.add_argument(
        "--include_baselines",
        action="store_true",
        default=True,
        help="Nạp cả 3 mô hình LEARN baseline để tạo panel đối sánh đa chiều Baseline vs SOLAR (mặc định: True)"
    )
    
    return parser.parse_args()


def save_single_image(array: np.ndarray, file_path: str, is_error_map: bool = False, error_scale: float = 5.0):
    """
    Lưu một mảng 2D numpy (float32 [0, 1]) thành file ảnh PNG chuẩn 8-bit.
    Nếu là Error Map, sẽ nhân tỉ lệ khuếch đại sai số và lưu với colormap 'jet'.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if is_error_map:
        err = np.clip(array * error_scale, 0.0, 1.0)
        cmap = plt.get_cmap("jet")
        err_colored = cmap(err)[:, :, :3]  # Lấy RGB bỏ Alpha
        err_uint8 = (err_colored * 255.0).astype(np.uint8)
        img = Image.fromarray(err_uint8)
        img.save(file_path)
    else:
        clipped = np.clip(array, 0.0, 1.0)
        img_uint8 = (clipped * 255.0).astype(np.uint8)
        img = Image.fromarray(img_uint8, mode="L")
        img.save(file_path)


def mirror_directory(src_dir: str, dst_dir: str):
    """
    Sao chép đồng bộ toàn bộ ảnh từ thư mục nguồn sang thư mục đích báo cáo.
    """
    if not os.path.exists(src_dir):
        return
    os.makedirs(dst_dir, exist_ok=True)
    for root, dirs, files in os.walk(src_dir):
        rel_path = os.path.relpath(root, src_dir)
        target_root = os.path.join(dst_dir, rel_path)
        os.makedirs(target_root, exist_ok=True)
        for f in files:
            if f.endswith((".png", ".jpg", ".csv")):
                s_file = os.path.join(root, f)
                d_file = os.path.join(target_root, f)
                shutil.copy2(s_file, d_file)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 80)
    print("🎨 BẮT ĐẦU KẾT XUẤT ẢNH TRỰC QUAN HÓA CHO 3 BIẾN THỂ SOLAR")
    print(f"- Thiết bị tính toán: {device}")
    print(f"- Cung quét góc: {args.angle_range_deg}° (Views: {args.num_view})")
    print(f"- Các lát cắt chỉ định: {args.slices}")
    print(f"- Thư mục xuất ảnh: {args.output_dir}")
    print(f"- Thư mục báo cáo đồng bộ: {args.report_dir}")
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
    print(f"✅ Đã nạp thành công tập kiểm thử Patient {args.test_patients[0]}: {len(test_dataset)} lát cắt.")

    # Bước 2: Xác định và nạp 3 mô hình SOLAR đề xuất
    solar_models = {}
    
    # 2.1. SOLAR_LongNet
    ckpt_solar_ln = args.solar_longnet_ckpt or find_best_checkpoint(
        "/datastore/uittogether3/LuuTru/MinhPD/saved_models/SOLAR_LongNet"
    )
    if ckpt_solar_ln and os.path.exists(ckpt_solar_ln):
        print(f"📦 Đang nạp SOLAR_LongNet từ: {ckpt_solar_ln}")
        model_s_ln = SOLAR_LongNet_LA.load_from_checkpoint(
            ckpt_solar_ln,
            start_ang=start_ang,
            end_ang=end_ang,
            num_view=args.num_view,
            num_detectors=args.num_detectors,
            input_size=args.input_size,
            map_location=device,
        )
        model_s_ln.to(device)
        model_s_ln.eval()
        solar_models["SOLAR_LongNet"] = model_s_ln
    else:
        print(f"⚠️ Không tìm thấy checkpoint SOLAR_LongNet: {ckpt_solar_ln}")

    # 2.2. SOLAR_Mamba
    ckpt_solar_mb = args.solar_mamba_ckpt or find_best_checkpoint(
        "/datastore/uittogether3/LuuTru/MinhPD/saved_models/SOLAR_Mamba"
    )
    if ckpt_solar_mb and os.path.exists(ckpt_solar_mb):
        print(f"📦 Đang nạp SOLAR_Mamba từ: {ckpt_solar_mb}")
        model_s_mb = SOLAR_Mamba_LA.load_from_checkpoint(
            ckpt_solar_mb,
            start_ang=start_ang,
            end_ang=end_ang,
            num_view=args.num_view,
            num_detectors=args.num_detectors,
            input_size=args.input_size,
            map_location=device,
        )
        model_s_mb.to(device)
        model_s_mb.eval()
        solar_models["SOLAR_Mamba"] = model_s_mb
    else:
        print(f"⚠️ Không tìm thấy checkpoint SOLAR_Mamba: {ckpt_solar_mb}")

    # 2.3. SOLAR_Longformer
    ckpt_solar_lf = args.solar_longformer_ckpt or find_best_checkpoint(
        "/datastore/uittogether3/LuuTru/MinhPD/saved_models/SOLAR_Longformer"
    )
    if ckpt_solar_lf and os.path.exists(ckpt_solar_lf):
        print(f"📦 Đang nạp SOLAR_Longformer từ: {ckpt_solar_lf}")
        model_s_lf = SOLAR_Longformer_LA.load_from_checkpoint(
            ckpt_solar_lf,
            start_ang=start_ang,
            end_ang=end_ang,
            num_view=args.num_view,
            num_detectors=args.num_detectors,
            input_size=args.input_size,
            map_location=device,
        )
        model_s_lf.to(device)
        model_s_lf.eval()
        solar_models["SOLAR_Longformer"] = model_s_lf
    else:
        print(f"⚠️ Không tìm thấy checkpoint SOLAR_Longformer: {ckpt_solar_lf}")

    # Bước 3: Nạp 3 mô hình LEARN đối chứng (nếu kích hoạt)
    learn_models = {}
    if args.include_baselines:
        print("\n--- Nạp các mô hình đối chứng LEARN (Baseline Bậc 1) ---")
        if os.path.exists(args.learn_longnet_ckpt):
            print(f"📦 Đang nạp LEARN_LongNet từ: {args.learn_longnet_ckpt}")
            m_l_ln = LEARN_LongNet_LA.load_from_checkpoint(args.learn_longnet_ckpt, map_location=device)
            m_l_ln.to(device)
            m_l_ln.eval()
            learn_models["LEARN_LongNet"] = m_l_ln

        if os.path.exists(args.learn_mamba_ckpt):
            print(f"📦 Đang nạp LEARN_Mamba từ: {args.learn_mamba_ckpt}")
            m_l_mb = LEARN_Mamba_LA.load_from_checkpoint(args.learn_mamba_ckpt, map_location=device)
            m_l_mb.to(device)
            m_l_mb.eval()
            learn_models["LEARN_Mamba"] = m_l_mb

        lf_c = args.learn_longformer_ckpt
        if not os.path.exists(lf_c):
            lf_c = "/datastore/uittogether3/LuuTru/MinhPD/saved_models/LEARN_Longformer/last.ckpt"
        if os.path.exists(lf_c):
            print(f"📦 Đang nạp LEARN_Longformer từ: {lf_c}")
            m_l_lf = LEARN_Longformer_LA.load_from_checkpoint(lf_c, map_location=device)
            m_l_lf.to(device)
            m_l_lf.eval()
            learn_models["LEARN_Longformer"] = m_l_lf

    angle_tag = f"{int(args.angle_range_deg)}deg"

    # Bước 4: Xử lý từng lát cắt chỉ định
    for slice_idx in args.slices:
        if slice_idx >= len(test_dataset):
            print(f"⚠️ Lát cắt #{slice_idx} vượt quá tổng số lát cắt ({len(test_dataset)}). Bỏ qua.")
            continue

        print(f"\n🖼️ ĐANG XỬ LÝ LÁT CẮT #{slice_idx:03d} (BỆNH NHÂN {args.test_patients[0]}, GÓC QUÉT {args.angle_range_deg:.0f}°)...")
        phantom, fbp_u, sino_noisy = test_dataset[slice_idx]

        phantom_t = phantom.unsqueeze(0).to(device)
        fbp_t = fbp_u.unsqueeze(0).to(device)
        sino_t = sino_noisy.unsqueeze(0).to(device)

        gt_np = phantom_t.squeeze().detach().cpu().numpy()
        fbp_np = fbp_t.squeeze().detach().cpu().numpy()

        dr = float(phantom_t.max() - phantom_t.min())
        data_range = dr if dr > 0.0 else 1.0
        fbp_psnr = peak_signal_noise_ratio(fbp_t, phantom_t, data_range=data_range).item()
        fbp_ssim = structural_similarity_index_measure(fbp_t, phantom_t, data_range=data_range).item()

        slice_dir = os.path.join(args.output_dir, angle_tag, f"slice_{slice_idx:03d}")
        os.makedirs(slice_dir, exist_ok=True)

        # Lưu ảnh Ground Truth và FBP Thô
        save_single_image(gt_np, os.path.join(slice_dir, "1_ground_truth.png"))
        save_single_image(fbp_np, os.path.join(slice_dir, "2_fbp_input.png"))

        # Thu thập kết quả
        all_recons = {}
        all_error_maps = {}
        all_metrics = {}

        # 4.1. Chạy suy luận cho các mô hình SOLAR
        with torch.no_grad():
            for m_name, net in solar_models.items():
                pred_t = net(fbp_t, sino_t)
                pred_np = pred_t.squeeze().detach().cpu().numpy()
                err_np = np.abs(pred_np - gt_np)

                psnr_val = peak_signal_noise_ratio(pred_t, phantom_t, data_range=data_range).item()
                ssim_val = structural_similarity_index_measure(pred_t, phantom_t, data_range=data_range).item()

                all_recons[m_name] = pred_np
                all_error_maps[m_name] = err_np
                all_metrics[m_name] = (psnr_val, ssim_val)

                file_tag = m_name.lower()
                save_single_image(pred_np, os.path.join(slice_dir, f"3_recon_{file_tag}.png"))
                save_single_image(err_np, os.path.join(slice_dir, f"4_error_map_{file_tag}.png"), is_error_map=True)
                print(f"  🌟 [{m_name}] PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f} -> Đã lưu ảnh.")

        # 4.2. Chạy suy luận cho các mô hình LEARN đối chứng
        with torch.no_grad():
            for m_name, net in learn_models.items():
                pred_t = net(fbp_t, sino_t)
                pred_np = pred_t.squeeze().detach().cpu().numpy()
                err_np = np.abs(pred_np - gt_np)

                psnr_val = peak_signal_noise_ratio(pred_t, phantom_t, data_range=data_range).item()
                ssim_val = structural_similarity_index_measure(pred_t, phantom_t, data_range=data_range).item()

                all_recons[m_name] = pred_np
                all_error_maps[m_name] = err_np
                all_metrics[m_name] = (psnr_val, ssim_val)

                file_tag = m_name.lower()
                save_single_image(pred_np, os.path.join(slice_dir, f"3_recon_{file_tag}.png"))
                save_single_image(err_np, os.path.join(slice_dir, f"4_error_map_{file_tag}.png"), is_error_map=True)
                print(f"  📌 [{m_name}] PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f} -> Đã lưu ảnh.")

        # ----------------------------------------------------------------------
        # PANEL 1: ĐỐI SÁNH 3 BIẾN THỂ SOLAR (SOLAR SUMMARY PANEL)
        # Bố cục: GT | FBP | SOLAR_LongNet (Recon, Err) | SOLAR_Mamba (Recon, Err) | SOLAR_Longformer (Recon, Err)
        # ----------------------------------------------------------------------
        n_solar = len(solar_models)
        if n_solar > 0:
            cols = 2 + n_solar * 2
            fig, axes = plt.subplots(1, cols, figsize=(3.8 * cols, 4.4), dpi=300)
            
            c = 0
            # Ground Truth
            axes[c].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
            axes[c].set_title("Ground Truth\n(Reference)", fontsize=11, fontweight="bold")
            axes[c].axis("off")
            c += 1

            # FBP Input
            axes[c].imshow(fbp_np, cmap="gray", vmin=0, vmax=1)
            axes[c].set_title(f"FBP Input ({args.angle_range_deg:.0f}°)\nPSNR: {fbp_psnr:.2f} dB | SSIM: {fbp_ssim:.4f}", fontsize=10)
            axes[c].axis("off")
            c += 1

            # Các biến thể SOLAR
            for m_name in solar_models.keys():
                p_v, s_v = all_metrics[m_name]
                axes[c].imshow(all_recons[m_name], cmap="gray", vmin=0, vmax=1)
                axes[c].set_title(f"{m_name}\nPSNR: {p_v:.2f} dB | SSIM: {s_v:.4f}", fontsize=10, fontweight="bold", color="darkgreen")
                axes[c].axis("off")
                c += 1

                axes[c].imshow(np.clip(all_error_maps[m_name] * 5.0, 0, 1), cmap="jet", vmin=0, vmax=1)
                axes[c].set_title(f"Error Map ({m_name})\n(|Recon - GT| x 5)", fontsize=10, color="darkred")
                axes[c].axis("off")
                c += 1

            plt.suptitle(
                f"SOLAR Architecture Benchmark ({args.angle_range_deg:.0f}°) — Patient L310, Slice #{slice_idx}",
                fontsize=13,
                fontweight="bold",
                y=1.03,
            )
            plt.tight_layout()
            solar_summary_path = os.path.join(slice_dir, "comparison_solar_summary.png")
            plt.savefig(solar_summary_path, bbox_inches="tight", dpi=300)
            plt.close(fig)
            print(f"  ✨ Đã lưu Panel SOLAR: {solar_summary_path}")

        # ----------------------------------------------------------------------
        # PANEL 2: ĐỐI SÁNH TRỰC DIỆN BASELINE BẬC 1 vs SOLAR BẬC 2
        # Lưới 2x4:
        # Hàng 1: Ground Truth    | LEARN_LongNet | LEARN_Mamba | LEARN_Longformer
        # Hàng 2: FBP Thô         | SOLAR_LongNet | SOLAR_Mamba | SOLAR_Longformer
        # ----------------------------------------------------------------------
        if len(learn_models) == 3 and len(solar_models) == 3:
            fig2, axes2 = plt.subplots(2, 4, figsize=(16, 8.5), dpi=300)

            # (0, 0): Ground Truth
            axes2[0, 0].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
            axes2[0, 0].set_title("Ground Truth\n(Reference CT)", fontsize=11, fontweight="bold")
            axes2[0, 0].axis("off")

            # (1, 0): FBP Thô
            axes2[1, 0].imshow(fbp_np, cmap="gray", vmin=0, vmax=1)
            axes2[1, 0].set_title(f"FBP Input ({args.angle_range_deg:.0f}°)\nPSNR: {fbp_psnr:.2f} dB | SSIM: {fbp_ssim:.4f}", fontsize=10, color="crimson")
            axes2[1, 0].axis("off")

            # Các cặp Baseline vs SOLAR
            pairs = [
                ("LEARN_LongNet", "SOLAR_LongNet", 1),
                ("LEARN_Mamba", "SOLAR_Mamba", 2),
                ("LEARN_Longformer", "SOLAR_Longformer", 3),
            ]

            for b_name, s_name, col in pairs:
                # Hàng 1: Baseline
                bp, bs = all_metrics[b_name]
                axes2[0, col].imshow(all_recons[b_name], cmap="gray", vmin=0, vmax=1)
                axes2[0, col].set_title(f"Baseline: {b_name}\nPSNR: {bp:.2f} dB | SSIM: {bs:.4f}", fontsize=10, fontweight="bold", color="darkblue")
                axes2[0, col].axis("off")

                # Hàng 2: SOLAR
                sp, ss = all_metrics[s_name]
                delta_p = sp - bp
                delta_s = ss - bs
                axes2[1, col].imshow(all_recons[s_name], cmap="gray", vmin=0, vmax=1)
                axes2[1, col].set_title(
                    f"Đề xuất: {s_name}\nPSNR: {sp:.2f} dB (Δ {delta_p:+.2f} dB)\nSSIM: {ss:.4f} (Δ {delta_s:+.4f})",
                    fontsize=10,
                    fontweight="bold",
                    color="darkgreen"
                )
                axes2[1, col].axis("off")

            plt.suptitle(
                f"Direct Comparison: 1st-Order LEARN vs 2nd-Order SOLAR ({args.angle_range_deg:.0f}°) — Patient L310, Slice #{slice_idx}",
                fontsize=14,
                fontweight="bold",
                y=0.98
            )
            plt.tight_layout()
            vs_path = os.path.join(slice_dir, "comparison_baseline_vs_solar.png")
            plt.savefig(vs_path, bbox_inches="tight", dpi=300)
            plt.close(fig2)
            print(f"  🏆 Đã lưu Panel Đối Sánh Baseline vs SOLAR: {vs_path}")

        # ----------------------------------------------------------------------
        # PANEL 3: TƯƠNG THÍCH ĐỊNH DẠNG GỐC (comparison_summary.png)
        # ----------------------------------------------------------------------
        total_models = list(solar_models.keys())
        cols3 = 2 + len(total_models) * 2
        fig3, axes3 = plt.subplots(1, cols3, figsize=(3.8 * cols3, 4.4), dpi=300)
        c3 = 0

        axes3[c3].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
        axes3[c3].set_title("Ground Truth\n(Reference)", fontsize=11, fontweight="bold")
        axes3[c3].axis("off")
        c3 += 1

        axes3[c3].imshow(fbp_np, cmap="gray", vmin=0, vmax=1)
        axes3[c3].set_title(f"FBP Input ({args.angle_range_deg:.0f}°)\nPSNR: {fbp_psnr:.2f} dB", fontsize=10)
        axes3[c3].axis("off")
        c3 += 1

        for m_name in total_models:
            p_v, s_v = all_metrics[m_name]
            axes3[c3].imshow(all_recons[m_name], cmap="gray", vmin=0, vmax=1)
            axes3[c3].set_title(f"{m_name}\nPSNR: {p_v:.2f} dB | SSIM: {s_v:.4f}", fontsize=10, fontweight="bold", color="darkgreen")
            axes3[c3].axis("off")
            c3 += 1

            axes3[c3].imshow(np.clip(all_error_maps[m_name] * 5.0, 0, 1), cmap="jet", vmin=0, vmax=1)
            axes3[c3].set_title(f"Error Map ({m_name})\n(x5)", fontsize=10, color="darkred")
            axes3[c3].axis("off")
            c3 += 1

        plt.suptitle(f"Limited-Angle CT Reconstruction ({args.angle_range_deg:.0f}°) — Patient L310, Slice #{slice_idx}", fontsize=13, fontweight="bold", y=1.03)
        plt.tight_layout()
        legacy_path = os.path.join(slice_dir, "comparison_summary.png")
        plt.savefig(legacy_path, bbox_inches="tight", dpi=300)
        plt.close(fig3)

    # Bước 5: Đồng bộ hóa toàn bộ ảnh sang thư mục báo cáo ngày sep-05-2026
    if args.report_dir:
        print(f"\n🔄 Đang đồng bộ ảnh trực quan hóa sang thư mục báo cáo: {args.report_dir}...")
        mirror_directory(args.output_dir, args.report_dir)
        print(f"✅ Đã đồng bộ hoàn tất vào {args.report_dir}")

    print("\n" + "=" * 80)
    print(f"🎉 HOÀN TẤT KẾT XUẤT ẢNH TRỰC QUAN HÓA TẠI: {args.output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
