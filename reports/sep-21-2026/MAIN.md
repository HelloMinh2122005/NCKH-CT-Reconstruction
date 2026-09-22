# BÁO CÁO TIẾN ĐỘ NGHIÊN CỨU ĐỊNH KỲ (21/09/2026)
**Dự án:** Limited-Angle CT Reconstruction  
**Tác giả:** MinhPD  

---

## 1. Kết Quả Benchmark Mới Hoàn Thành (NIH DeepLesion CT)
Đã hoàn tất nghiệm thu và đánh giá kiểm thử độc lập trên **300 lát cắt CT test** của bộ dữ liệu NIH DeepLesion cho 2 mô hình:

1. **`DuDoTrans` (Job `72969` - Checkpoint `epoch=49-val_psnr=23.6662.ckpt`):**
   - **Cung quét chuẩn LA-120° (64 views):** **PSNR = 23.76 dB** | **SSIM = 0.6637** | RMSE = 0.0892 | Loss = 0.0082.
   - **Cung quét hẹp LA-90° (64 views):** **PSNR = 20.38 dB** | **SSIM = 0.6692** | RMSE = 0.1084 | Loss = 0.0120.
   - *Nhận định:* Nhờ cơ chế nội suy đa miền (Sinogram Inpainting + Image Refinement), `DuDoTrans` thể hiện tính vững chãi tốt khi góc quét bị thu hẹp xuống 90° (SSIM không bị sụp đổ, giữ mức 0.6692).

2. **`LEARN_Mamba` (Job `72968` - Checkpoint `mamba_la-epoch=23-val_psnr=26.18-val_ssim=0.7009.ckpt`):**
   - **Cung quét chuẩn LA-120° (64 views):** **PSNR = 26.69 dB** | **SSIM = 0.7132** | RMSE = 0.0486 | Loss = 0.0025.
   - **Cung quét hẹp LA-90° (64 views):** **PSNR = 18.59 dB** | **SSIM = 0.3913** | RMSE = 0.1211 | Loss = 0.0150.
   - *Nhận định:* Tái lập hoàn toàn hiện tượng suy thoái tương tự trên AAPM: Selective SSM quét tuần tự 1D không thích nghi được với vùng nêm khuyết tần số 2D mở rộng ở 90°, dẫn đến sụt giảm -8.10 dB PSNR và -0.3219 SSIM.

*Chi tiết đối sánh định lượng xem tại:* [benchmark_results.csv](benchmark_results.csv).

---

## 2. Tiến Độ Huấn Luyện Các Baseline Trên Cụm DGX-A100

1. **Các Job Huấn Luyện Vừa Về Đích 100% (50/50 Epochs):**
   - **`CT-Former` (Job `72965`):** Hoàn thành lúc 18:44:10 (`rc=0`). Best checkpoint: `ct_former_la-epoch=49-val_psnr=33.64-val_ssim=0.9303.ckpt` (Val PSNR 33.64 dB, SSIM 0.9303). Đã nộp job kiểm thử benchmark `73105`.
   - **`DuDoNet` (Job `72964`):** Hoàn thành lúc 14:08:18 (`rc=0`). Best checkpoint: `dudonet_la-epoch=41-val_psnr=28.98-val_ssim=0.8536.ckpt` (Val PSNR 28.98 dB, SSIM 0.8536). Đã nộp job kiểm thử benchmark `73106`.

2. **Các Job Đang Chạy & Resume:**
   - **`LEARN` DeepLesion (Job `72822` $\to$ Resume Job `73107`):** Job `72822` chạm mốc 24h Time Limit tại Epoch 40 (31%), Val PSNR đạt **34.27 dB** (Epoch 36). Đã submit job resume `73107` nạp `last.ckpt` để hoàn tất 10 epochs còn lại.
   - **`LEARN_Longformer` DeepLesion (Job `72966`):** Đang chạy ổn định tại Epoch 41/50, Val PSNR = 32.20 dB, SSIM = 0.8954.
   - **`FISTA-Net` AAPM (Job `72963`):** Đang chạy ổn định tại Epoch 24/50, Val PSNR = 33.54 dB, SSIM = 0.9298.
   - **`LEARN_LongNet` DeepLesion (Job `72967`):** Đang chạy ổn định tại Epoch 9/50, Val PSNR = 28.91 dB, SSIM = 0.8411 (triệt để hết OOM).

3. **Xử Lý Sự Cố Mô Hình `MoDL` (Job `72962`):**
   - Đã chủ động hủy job `72962` sau khi phát hiện phân kỳ toán học: Val PSNR chỉ đạt ~3.27 dB do bước giải CG Data Consistency dùng FBP thay vì toán tử liên hợp Adjoint $A^T$.
   - Kế hoạch: Viết lại toán tử liên hợp chuẩn trong CG solver và nộp lại sau.
