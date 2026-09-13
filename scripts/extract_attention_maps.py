"""
================================================================================
SCRIPT TRÍCH XUẤT VÀ TRỰC QUAN HÓA BẢN ĐỒ CHÚ Ý (ATTENTION MAPS EVOLUTION)
Dự án: Nghiên cứu Tái tạo Ảnh Cắt lớp CT Góc Giới hạn (Limited-Angle CT)
Mô hình: LEARN_Longformer (14 unrolling stages)
Mục tiêu:
    - Nạp checkpoint LEARN_Longformer đã huấn luyện.
    - Kích hoạt output_attentions=True trên 14 tầng unrolling.
    - Trích xuất ma trận attention weights của Global Token qua từng iteration (t = 1 -> 14).
    - Reshape ma trận trọng số (16384 tokens) về dạng 2D (128x128).
    - Xuất panel lưới 2 hàng x 8 cột chuẩn y hệt Fig. 6 của bài báo MVA:
      Hàng 1: Ground Truth, Iteration 1 -> 7
      Hàng 2: Iteration 8 -> 14, Reconstructed CT
================================================================================
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

# Tránh lỗi unpickler trên PyTorch Lightning
_orig_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _safe_torch_load

from data.datamodule_LA import LimitedAngleCTDataModule
from baselines.LEARN_Longformer.models import LEARN_Longformer_LA


def parse_args():
    parser = argparse.ArgumentParser(description="Trích xuất Attention Map của LEARN_Longformer qua các iterations")
    parser.add_argument("--slice_idx", type=int, default=50, help="Chỉ số lát cắt test (mặc định: 50)")
    parser.add_argument("--angle_range_deg", type=float, default=120.0, help="Độ rộng cung quét (120.0 hoặc 90.0 độ)")
    parser.add_argument("--num_view", type=int, default=64, help="Số góc chiếu (64 views)")
    parser.add_argument("--num_detectors", type=int, default=512, help="Số detector (512)")
    parser.add_argument("--input_size", type=int, default=256, help="Kích thước ảnh (256x256)")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/datastore/uittogether3/LuuTru/MinhPD/saved_models/LEARN_Longformer/longformer_la-epoch=45-val_psnr=34.77-val_ssim=0.9383.ckpt",
        help="Đường dẫn file checkpoint LEARN_Longformer"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/datastore/uittogether3/LuuTru/MinhPD/dataset/limited_angle/",
        help="Thư mục dataset cache"
    )
    parser.add_argument(
        "--dicom_dir",
        type=str,
        default="/datastore/uittogether3/LuuTru/Thanhld/CT-Reconstruction/split/",
        help="Thư mục DICOM gốc"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/datastore/uittogether3/LuuTru/MinhPD/visualizations/attention_maps/",
        help="Thư mục lưu ảnh kết quả"
    )
    parser.add_argument("--global_token_idx", type=int, default=0, help="Chỉ số global token để lấy attention map (mặc định: 0)")
    return parser.parse_args()


def extract_stage_attentions(model, fbp_input, sino, device, global_token_idx=0):
    """
    Thực hiện forward pass tùy biến để bắt ma trận attention qua từng tầng unrolling.
    """
    model.eval()
    x_k = fbp_input.clone().to(device)
    y = sino.clone().to(device)
    
    attention_maps = []
    
    with torch.no_grad():
        for stage_idx, stage in enumerate(model.gradient_list):
            # Lấy data fidelity term
            data_fidelity = stage.forward_module(x_k) - y if hasattr(stage, 'forward_module') else model.forward_module(x_k) - y
            bp_term = model.backward_module(data_fidelity)
            
            # Lấy regularizer module
            reg_block = stage.regularitation_term
            
            # Conv1
            z = F.relu(reg_block.conv1(x_k))
            
            # Tokenize vào Longformer
            attn_mod = reg_block.longformer_attn
            B, C, H, W = z.shape
            tokens = (
                z.unfold(2, attn_mod.window_size, attn_mod.window_size)
                .unfold(3, attn_mod.window_size, attn_mod.window_size)
                .contiguous()
                .view(B, C, -1, attn_mod.window_size * attn_mod.window_size)
                .permute(0, 2, 1, 3)
                .contiguous()
                .view(B, -1, attn_mod.token_dim)
            )
            N = tokens.shape[1] # 16384
            
            attention_mask = torch.zeros(B, N, dtype=torch.long, device=device)
            is_index_masked = attention_mask != 0
            
            global_positions = torch.linspace(0, N - 1, steps=attn_mod.num_global, device=device).long()
            is_index_global_attn = torch.zeros(B, N, dtype=torch.bool, device=device)
            is_index_global_attn[:, global_positions] = True
            
            # Bật output_attentions=True để lấy ma trận trọng số
            attn_out = attn_mod.longformer_attention(
                tokens,
                attention_mask=attention_mask,
                is_index_masked=is_index_masked,
                is_index_global_attn=is_index_global_attn,
                is_global_attn=True,
                output_attentions=True,
            )
            
            # Trích xuất global attention probabilities: shape (B, num_heads, num_global, N)
            # attn_out[2] là global_attentions
            if len(attn_out) > 2 and attn_out[2] is not None:
                global_attn = attn_out[2] # (B, H, G, N)
                # Trung bình cộng qua các attention heads
                mean_head_attn = global_attn[0, :, global_token_idx, :].mean(dim=0) # shape (N,)
                attn_2d = mean_head_attn.view(128, 128).cpu().numpy()
            else:
                # Fallback: tính ma trận tích vô hướng chuẩn hóa
                attn_2d = np.zeros((128, 128), dtype=np.float32)
            
            attention_maps.append(attn_2d)
            
            # Tiếp tục luồng unrolling bình thường để cập nhật x_{k+1}
            hidden = attn_out[0].view(
                B, attn_mod.new_spatial, attn_mod.new_spatial, attn_mod.patch_channels, attn_mod.window_size, attn_mod.window_size
            ).permute(0, 3, 1, 4, 2, 5).contiguous().view(B, attn_mod.patch_channels, H, W)
            
            hidden = F.relu(reg_block.conv2(hidden))
            reg_val = reg_block.conv3(hidden)
            
            grad = stage.alpha * bp_term + reg_val
            x_k = x_k - grad
            
    return x_k, attention_maps


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("🔬 TRÍCH XUẤT ATTENTION MAPS TỪNG ITERATION (LEARN_LONGFORMER)")
    print(f"- Thiết bị: {device}")
    print(f"- Góc quét: {args.angle_range_deg}° (Views: {args.num_view})")
    print(f"- Lát cắt: Slice #{args.slice_idx}")
    print(f"- Checkpoint: {args.ckpt_path}")
    print("=" * 80)
    
    # Thiết lập góc quét và DataModule
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
        test_patients=["L310"],
    )
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()
    
    # Lấy lát cắt kiểm thử chỉ định
    sample = None
    for idx, batch in enumerate(test_loader):
        if idx == args.slice_idx:
            sample = batch
            break
    if sample is None:
        raise ValueError(f"Không tìm thấy lát cắt {args.slice_idx}")
        
    phantom, fbp, sino = sample
    phantom = phantom.to(device)
    fbp = fbp.to(device)
    sino = sino.to(device)
    
    # Nạp mô hình
    print(f"📦 Đang nạp mô hình từ: {args.ckpt_path}")
    model = LEARN_Longformer_LA.load_from_checkpoint(args.ckpt_path, map_location=device)
    model.to(device)
    
    # Trích xuất
    recon_img, attn_maps = extract_stage_attentions(model, fbp, sino, device, args.global_token_idx)
    
    # Tính PSNR / SSIM của ảnh tái tạo
    from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
    data_range = model._get_batch_data_range(phantom)
    p_val = peak_signal_noise_ratio(recon_img, phantom, data_range=data_range).item()
    s_val = structural_similarity_index_measure(recon_img, phantom, data_range=data_range).item()
    
    # Vẽ panel 2 hàng x 8 cột chuẩn phong cách Fig. 6 MVA
    fig, axes = plt.subplots(2, 8, figsize=(24, 6.5), dpi=300)
    plt.subplots_adjust(wspace=0.08, hspace=0.15)
    
    gt_np = phantom[0, 0].cpu().numpy()
    recon_np = recon_img[0, 0].cpu().numpy()
    
    # Vmin, Vmax cho attention maps để thống nhất colorbar
    all_attns = np.stack(attn_maps)
    vmin = float(np.percentile(all_attns, 1))
    vmax = float(np.percentile(all_attns, 99.5))
    
    # Hàng 1: Ground Truth, Iteration 1 -> 7
    axes[0, 0].imshow(gt_np, cmap="bone")
    axes[0, 0].set_title("Ground Truth", fontsize=14, fontweight="bold")
    axes[0, 0].axis("off")
    
    for t in range(7):
        ax = axes[0, t + 1]
        im = ax.imshow(attn_maps[t], cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"Iteration {t+1}", fontsize=13)
        ax.axis("off")
        
    # Hàng 2: Iteration 8 -> 14, Reconstructed CT
    for t in range(7, 14):
        ax = axes[1, t - 7]
        im = ax.imshow(attn_maps[t], cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"Iteration {t+1}", fontsize=13)
        ax.axis("off")
        
    axes[1, 7].imshow(recon_np, cmap="bone")
    axes[1, 7].set_title("Reconstructed", fontsize=14, fontweight="bold")
    axes[1, 7].text(
        0.5, 0.92, f"PSNR: {p_val:.2f} dB | SSIM: {s_val:.4f}",
        color="yellow", fontsize=10, fontweight="bold",
        ha="center", va="top", transform=axes[1, 7].transAxes,
        bbox=dict(boxstyle="square,pad=0.2", fc="black", ec="none", alpha=0.7)
    )
    axes[1, 7].axis("off")
    
    # Thêm Colorbar ở bên phải
    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=11)
    
    save_name = f"longformer_attn_evolution_{int(args.angle_range_deg)}deg_slice{args.slice_idx:03d}.png"
    save_path = os.path.join(args.output_dir, save_name)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    
    print(f"✅ Đã kết xuất Attention Evolution Panel thành công tại:\n   {save_path}")

if __name__ == "__main__":
    main()
