"""
Script tạo hình visual_comparison_roi.png 7 cột (Ground Truth, FBP, LEARN, DuDoTrans, LEARN_Mamba, LEARN_LongNet, LEARN_Longformer)
Dành cho bài báo SOICT 2026.
Bảo toàn phong cách thẩm mỹ: Nền đen, nhãn vàng ROI và insets phóng đại 3x, huy hiệu thông số PSNR/SSIM trên đỉnh mỗi lát cắt.
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def add_roi_and_inset(img_pil: Image.Image) -> Image.Image:
    """
    Thêm bounding box màu vàng ở vùng biên xương sườn (upper ROI)
    và phóng to vùng ROI đặt ở góc dưới bên trái (inset).
    Kích thước ảnh chuẩn: 256x256.
    """
    img_rgb = img_pil.convert("RGB")
    arr = np.array(img_rgb)
    
    # Tọa độ ROI trên ảnh 256x256
    # Upper box: x in [175, 215], y in [75, 115]
    x1, y1, x2, y2 = 175, 75, 215, 115
    
    # Crop vùng ROI bên trong viền (x1+2, y1+2, x2-1, y2-1)
    crop = img_rgb.crop((x1 + 1, y1 + 1, x2, y2))
    
    # Inset box ở góc dưới trái: x in [4, 96], y in [160, 252] (kích thước 93x93)
    # Vùng ảnh bên trong viền: 89x89
    ix1, iy1, ix2, iy2 = 4, 160, 96, 252
    crop_resized = crop.resize((ix2 - ix1 - 3, iy2 - iy1 - 3), Image.Resampling.BICUBIC)
    
    # Dán crop vào góc dưới trái
    img_rgb.paste(crop_resized, (ix1 + 2, iy1 + 2))
    
    # Vẽ khung chữ nhật màu vàng (outline width=2)
    draw = ImageDraw.Draw(img_rgb)
    yellow = (255, 255, 0)
    
    # Khung upper ROI
    draw.rectangle([x1, y1, x2, y2], outline=yellow, width=2)
    
    # Khung inset
    draw.rectangle([ix1, iy1, ix2, iy2], outline=yellow, width=2)
    
    return img_rgb


def create_badge(text: str, is_reference: bool = False, font=None) -> Image.Image:
    """
    Tạo huy hiệu chỉ số (metric badge) dạng hộp đen viền trắng/vàng.
    """
    # Tạo ảnh tạm để tính kích thước text
    dummy = Image.new("RGBA", (200, 50), (0, 0, 0, 0))
    draw_d = ImageDraw.Draw(dummy)
    bbox = draw_d.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    
    pad_x = 8
    pad_y = 4
    bw = tw + 2 * pad_x
    bh = th + 2 * pad_y
    
    badge = Image.new("RGBA", (bw, bh), (0, 0, 0, 255))
    draw_b = ImageDraw.Draw(badge)
    
    color = (255, 255, 0) if is_reference else (255, 255, 255)
    border_color = (255, 255, 0) if is_reference else (180, 180, 180)
    
    draw_b.rectangle([0, 0, bw - 1, bh - 1], outline=border_color, width=1)
    draw_b.text((pad_x - bbox[0], pad_y - bbox[1]), text, fill=color, font=font)
    return badge


def main():
    root_dir = Path(__file__).resolve().parent.parent
    figures_dir = root_dir / "papers" / "soict2026" / "figures"
    viz_dir = root_dir / "visualizations"
    
    # Định nghĩa cấu hình 7 cột
    columns = [
        {
            "title": "Ground Truth",
            "img_120": viz_dir / "120deg" / "slice_050" / "1_ground_truth.png",
            "img_90": viz_dir / "90deg" / "slice_050" / "1_ground_truth.png",
            "badge_120": "Reference",
            "badge_90": "Reference",
            "is_ref": True
        },
        {
            "title": "FBP",
            "img_120": viz_dir / "120deg" / "slice_050" / "2_fbp_input.png",
            "img_90": viz_dir / "90deg" / "slice_050" / "2_fbp_input.png",
            "badge_120": "17.89 dB | 0.4984",
            "badge_90": "15.20 dB | 0.4120",
            "is_ref": False
        },
        {
            "title": "LEARN (CNN)",
            "img_120": viz_dir / "120deg" / "slice_050" / "3_recon_learn.png",
            "img_90": viz_dir / "90deg" / "slice_050" / "3_recon_learn.png",
            "badge_120": "32.29 dB | 0.9383",
            "badge_90": "28.03 dB | 0.8793",
            "is_ref": False
        },
        {
            "title": "DuDoTrans",
            "img_120": viz_dir / "120deg" / "slice_050" / "3_recon_dudotrans.png",
            "img_90": viz_dir / "90deg" / "slice_050" / "3_recon_dudotrans.png",
            "badge_120": "25.15 dB | 0.7447",
            "badge_90": "21.81 dB | 0.6738",
            "is_ref": False
        },
        {
            "title": "LEARN_Mamba",
            "img_120": viz_dir / "120deg" / "slice_050" / "3_recon_learn_mamba.png",
            "img_90": viz_dir / "90deg" / "slice_050" / "3_recon_learn_mamba.png",
            "badge_120": "26.32 dB | 0.7468",
            "badge_90": "18.76 dB | 0.4292",
            "is_ref": False
        },
        {
            "title": "LEARN_LongNet",
            "img_120": viz_dir / "120deg" / "slice_050" / "3_recon_learn_longnet.png",
            "img_90": viz_dir / "90deg" / "slice_050" / "3_recon_learn_longnet.png",
            "badge_120": "31.62 dB | 0.8991",
            "badge_90": "19.19 dB | 0.5876",
            "is_ref": False
        },
        {
            "title": "LEARN_Longformer",
            "img_120": viz_dir / "120deg" / "slice_050" / "3_recon_learn_longformer.png",
            "img_90": viz_dir / "90deg" / "slice_050" / "3_recon_learn_longformer.png",
            "badge_120": "33.10 dB | 0.9237",
            "badge_90": "19.16 dB | 0.6097",
            "is_ref": False
        }
    ]
    
    # Nạp font chữ
    # Tìm kiếm font Liberation Sans hoặc DejaVu Sans Bold
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    bold_font_path = None
    for fp in font_paths:
        if os.path.exists(fp):
            bold_font_path = fp
            break
            
    regular_font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    regular_font_path = None
    for fp in regular_font_paths:
        if os.path.exists(fp):
            regular_font_path = fp
            break

    title_font = ImageFont.truetype(bold_font_path, 16) if bold_font_path else ImageFont.load_default()
    row_font = ImageFont.truetype(bold_font_path, 15) if bold_font_path else ImageFont.load_default()
    row_sub_font = ImageFont.truetype(bold_font_path, 13) if bold_font_path else ImageFont.load_default()
    badge_font = ImageFont.truetype(bold_font_path, 12) if bold_font_path else ImageFont.load_default()
    
    # Kích thước khung hình
    num_cols = len(columns)
    tile_size = 250
    gap_x = 10
    left_margin = 100
    right_margin = 15
    top_margin = 46
    row_gap = 12
    bottom_margin = 12
    
    total_w = left_margin + num_cols * tile_size + (num_cols - 1) * gap_x + right_margin
    total_h = top_margin + 2 * tile_size + row_gap + bottom_margin
    
    canvas = Image.new("RGB", (total_w, total_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    
    # Vẽ nhãn hàng bên trái (Row Labels)
    # Row 1: 120° (Moderate)
    r1_cy = top_margin + tile_size // 2
    draw.text((left_margin // 2, r1_cy - 12), "120°", fill=(255, 255, 0), font=row_font, anchor="mm")
    draw.text((left_margin // 2, r1_cy + 10), "(Moderate)", fill=(255, 255, 0), font=row_sub_font, anchor="mm")
    
    # Row 2: 90° (Severe)
    r2_cy = top_margin + tile_size + row_gap + tile_size // 2
    draw.text((left_margin // 2, r2_cy - 12), "90°", fill=(255, 255, 0), font=row_font, anchor="mm")
    draw.text((left_margin // 2, r2_cy + 10), "(Severe)", fill=(255, 255, 0), font=row_sub_font, anchor="mm")
    
    # Vẽ từng cột
    for c_idx, col in enumerate(columns):
        col_x = left_margin + c_idx * (tile_size + gap_x)
        
        # 1. Vẽ tiêu đề cột trên đỉnh
        title_x = col_x + tile_size // 2
        draw.text((title_x, 22), col["title"], fill=(255, 255, 255), font=title_font, anchor="mm")
        
        # 2. Xử lý ảnh Row 1 (120deg)
        im120_raw = Image.open(col["img_120"])
        im120_roi = add_roi_and_inset(im120_raw).resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        
        # Thêm badge cho Row 1
        badge_120 = create_badge(col["badge_120"], is_reference=col["is_ref"], font=badge_font)
        badge_x = col_x + (tile_size - badge_120.width) // 2
        
        r1_y = top_margin
        canvas.paste(im120_roi, (col_x, r1_y))
        canvas.paste(badge_120, (badge_x, r1_y + 6), badge_120)
        
        # 3. Xử lý ảnh Row 2 (90deg)
        im90_raw = Image.open(col["img_90"])
        im90_roi = add_roi_and_inset(im90_raw).resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        
        # Thêm badge cho Row 2
        badge_90 = create_badge(col["badge_90"], is_reference=col["is_ref"], font=badge_font)
        badge_x90 = col_x + (tile_size - badge_90.width) // 2
        
        r2_y = top_margin + tile_size + row_gap
        canvas.paste(im90_roi, (col_x, r2_y))
        canvas.paste(badge_90, (badge_x90, r2_y + 6), badge_90)
        
    out_path = figures_dir / "visual_comparison_roi.png"
    canvas.save(out_path, dpi=(300, 300), quality=95)
    print(f"✅ Đã tạo thành công ảnh 7 cột tại: {out_path} (Kích thước: {total_w}x{total_h})")

if __name__ == "__main__":
    main()
