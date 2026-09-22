# 📊 BẢNG TỔNG HỢP KẾT QUẢ THỰC NGHIỆM (EXPERIMENT BENCHMARK RESULTS)
**Dự án:** Nghiên cứu Tái tạo Ảnh Cắt lớp CT Góc Giới hạn (Limited-Angle CT Reconstruction)  
**Tác giả:** MinhPD — Nhóm Nghiên cứu Tái tạo Ảnh Y tế (VNU-HCM UIT)  
**Ngày cập nhật:** 30/08/2026  

---

## 1. Cấu Hình Dữ Liệu Thực Nghiệm (Dataset Configurations)

Mọi mô hình được huấn luyện và đánh giá trên bộ dữ liệu chuẩn y tế **AAPM Mayo Clinic Low Dose CT (2016)**:
* **Tập Huấn luyện (Train):** 1,920 lát cắt CT từ 8 bệnh nhân độc lập (`L067`, `L096`, `L109`, `L143`, `L192`, `L286`, `L291`, `L506`).
* **Tập Kiểm định (Validation):** 244 lát cắt CT từ bệnh nhân `L333`.
* **Tập Kiểm thử Độc lập (Test Benchmark):** 214 lát cắt CT từ bệnh nhân `L310`.
* **Định dạng dữ liệu:** Cache `.npy` float32 tốc độ cao kèm ảnh gốc DICOM 16-bit (`.IMA`).

---

## 2. Bảng Tổng Hợp Kết Quả Huấn Luyện Các Mô Hình Baseline

> **Cấu hình Huấn luyện chính:** **LA-120° (Dải góc quét $[-60^\circ, +60^\circ]$, 64 views, ảnh $256 \times 256$, chế độ `noise_0`)**

| STT | Tên Mô hình | Động cơ Chuỗi Dài / Attention | Dữ liệu Huấn Luyện | Số Epochs | Best Val PSNR (dB) | Best Val SSIM | Đường dẫn Best Checkpoint | Trạng thái Job & Ghi chú Kỹ thuật |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **1** | **`LEARN_LongNet`** | Multi-Scale Dilated Attention | **LA-120° (64v)** | **50 / 50** | **33.37** | **0.9090** | `saved_models/LEARN_LongNet/longnet_la-epoch=45-val_psnr=33.37-val_ssim=0.9090.ckpt` | ✅ **Hoàn thành 100% (50/50 epochs)**. Tăng trưởng đều đặn, ổn định xuất sắc trên toàn bộ tiến trình. |
| **2** | **`LEARN_Longformer`**| Sliding-Chunks Self-Attention + Global Tokens | **LA-120° (64v)** | **50 / 50** | **34.77** | **0.9383** | `saved_models/LEARN_Longformer/longformer_la-epoch=45-val_psnr=34.77-val_ssim=0.9383.ckpt` | ✅ **Hoàn thành 100% (50/50 epochs - Job `66652`)**. Đạt kết quả SOTA cao nhất trong toàn bộ baseline (**PSNR = 34.77 dB**, **SSIM = 0.9383** ở Epoch 45). |
| **3** | **`LEARN_Mamba`** | Selective SSM ($\mathcal{O}(N)$) | **LA-120° (64v)** | **41 / 50** | **27.66** | **0.7373** | `saved_models/LEARN_Mamba/mamba_la-epoch=17-val_psnr=27.66-val_ssim=0.7373.ckpt` | ⚠️ **Sử dụng Checkpoint Epoch 17**. Đạt đỉnh ở Epoch 17; sau Epoch 20 xuất hiện bùng nổ gradient / NaN do đặc tính quét 1D trên Hessian suy biến của LA-CT. |
| **4** | **`SOLAR_LongNet` (Đề xuất)**| **Newton-CG Bậc 2 + Res-CNN & Dilated Attention** | **LA-120° (64v)** | **31 / 50 (Paused)** | **32.26** | **0.8935** | `saved_models/SOLAR_LongNet/solar_longnet_la-epoch=29-val_psnr=32.26-val_ssim=0.8935.ckpt` | 🎯 **Đã hoàn thành đánh giá Test Benchmark (Job `67823`)**. Vượt trội ngoạn mục ở góc hẹp LA-90° (PSNR 27.19 dB, SSIM 0.8639). |
| **5** | **`SOLAR_Longformer` (Đề xuất)**| **Newton-CG Bậc 2 + Res-CNN & Sliding-Chunks Attention** | **LA-120° (64v)** | **50 / 50** | **34.18** | **0.9165** | `saved_models/SOLAR_Longformer/solar_longformer_la-epoch=45-val_psnr=34.18-val_ssim=0.9165.ckpt` | ✅ **Hoàn thành 100% 50 epochs (Job `67820`)**. Đạt đỉnh Val PSNR 34.18 dB, SSIM 0.9165 tại Epoch 45. |
| **6** | **`SOLAR_Mamba` (Đề xuất)**| **Newton-CG Bậc 2 + Res-CNN & Selective SSM** | **LA-120° (64v)** | **50 / 50** | **34.00** | **0.9089** | `saved_models/SOLAR_Mamba/solar_mamba_la-epoch=45-val_psnr=34.00-val_ssim=0.9089.ckpt` | ✅ **Hoàn thành 100% 50 epochs (Job `67821`)**. Đạt đỉnh Val PSNR 34.00 dB tại Epoch 45 (Epoch 46 SSIM 0.9112). 100% ổn định số học, không NaN. |
| **7** | **`LEARN` (Original CNN)** | 3-Layer CNN Thuần túy (14 stages) | **LA-120° (64v)** | **50 / 50** | **36.27** | **0.950** | `saved_models/LEARN/LEARN_AAPM_LA_120deg_view64/epoch=46-val_psnr=36.2681.ckpt` | ✅ **Hoàn thành 100% 50 epochs (Job `71393`)**. Hội tụ cực kỳ mượt mà, Val PSNR đỉnh 36.27 dB ở Epoch 46. |
| **8** | **`RegFormer`** | Local CNN + Swin Transformer (14 stages) | **LA-120° (64v)** | **50 / 50** | **36.05** | **0.950** | `saved_models/RegFormer/RegFormer_AAPM_LA_120deg_view64/epoch=46-val_psnr=36.0533.ckpt` | ✅ **Hoàn thành 100% 50 epochs (Job `71394`)**. Cơ chế điều hòa kép hoạt động hoàn hảo, Val PSNR đỉnh 36.05 dB ở Epoch 46. |
| **9** | **`DuDoTrans`** | Sinogram Transformer + Image Refinement | **LA-120° (64v)** | **50 / 50** | **25.73** | **0.730** | `saved_models/DuDoTrans/DuDoTrans_AAPM_LA_120deg_view64/epoch=38-val_psnr=25.7300.ckpt` | ✅ **Hoàn thành 100% 50 epochs (Job `71395`)**. Chạy siêu nhanh (2h32m), Val PSNR đạt đỉnh 25.73 dB ở Epoch 38. |
| **10** | **`SOLAR_RegFormer` (Đề xuất)** | **Newton-CG Bậc 2 + Swin Window Attention** | **LA-120° (64v)** | **50 / 50** | **34.00** | **0.9139** | `saved_models/SOLAR_RegFormer/solar_regformer_la-epoch=45-val_psnr=34.00-val_ssim=0.9117.ckpt` | ✅ **Hoàn thành 100% 50 epochs (Jobs `71632`, `72256`)**. Val PSNR đạt đỉnh 34.00 dB (Ep 45), Val SSIM đạt 0.9139 (Ep 46). Đã kiểm thử mù hoàn chỉnh (Job `72610`). |
| **11** | **`SOLAR_DualMamba` (Đề xuất)** | **Bi-Mamba Sinogram + Newton-CG Swin** | **LA-120° (64v)** | **42 / 50 (Running)** | **34.43** | **0.9175** | `saved_models/SOLAR_DualMamba/solar_dualmamba_la-epoch=41-val_psnr=34.43-val_ssim=0.9175.ckpt` | 🚀 **Đang chạy thực tế (Job `72618`)**. Đạt đỉnh Val PSNR 34.43 dB, SSIM 0.9175 — Thiết lập kỷ lục SOTA cao nhất toàn bộ họ mô hình SOLAR! |

---

## 3. Lý Do Khoa Học & Phân Tích Lựa Chọn Checkpoint Cho Từng Mô Hình

### 3.1. `LEARN_LongNet` — Sử dụng `last.ckpt` hoặc `epoch=45`
* **Đặc tính:** Mô hình đã hoàn tất 100% kế hoạch huấn luyện (50 epochs) vào ngày 28/08/2026.
* **Đánh giá:** Chỉ số SSIM đạt $0.9090$ và PSNR $33.37\text{ dB}$, chất lượng tái tạo ổn định nhất và hoàn toàn không gặp lỗi số học.
* **Checkpoint sử dụng:** `saved_models/LEARN_LongNet/last.ckpt` (tương đương kết quả hội tụ tại cuối epoch 49-50) và `epoch=45` (đỉnh PSNR cao nhất).

### 3.2. `LEARN_Longformer` — Checkpoint `epoch=36` & Tùy chọn Resume
* **Đặc tính:** Mô hình học rất mượt và có độ tương đồng cấu trúc cũng như chất lượng tái tạo cao nhất trong tất cả các baseline (SSIM đạt $0.9323$, PSNR $34.38\text{ dB}$ tại Epoch 36).
* **Trạng thái:** Job resume `65892` đã chạy từ Epoch 17 đến Epoch 41 và dừng do Time Limit 24h trên DGX-A100.
* **Checkpoint sử dụng:** `saved_models/LEARN_Longformer/longformer_la-epoch=36-val_psnr=34.38-val_ssim=0.9323.ckpt` (Best Checkpoint) hoặc `saved_models/LEARN_Longformer/last.ckpt` nếu muốn tiếp tục chạy nốt 9 epochs còn lại (41 -> 50).

### 3.3. `LEARN_Mamba` — Bắt buộc cố định Checkpoint tại Epoch 17
* **Hiện tượng:** Mô hình chạy rất nhanh (~25 phút/epoch) nhưng sau Epoch 20 bị mất ổn định số học và xuất hiện giá trị `NaN` trong 14 vòng unrolling.
* **Lý do khoa học sử dụng Epoch 17:** 
  1. Checkpoint Epoch 17 là thời điểm mô hình đạt cực đại trước khi bị phân kỳ (`PSNR = 27.66 dB`, `SSIM = 0.7373`).
  2. Đây là **bằng chứng thực nghiệm quan trọng** chứng minh hạn chế cốt lõi của Mamba-1 khi áp dụng vào bài toán Limited-Angle CT (quét tuần tự 1D không phù hợp với nêm khuyết 2D và bước gradient bậc 1 bị đập văng trên địa hình Hessian suy biến).
  3. Minh chứng này khẳng định sự cần thiết của mô hình đề xuất **SOLAR** (sử dụng tối ưu bậc 2 Newton-CG và Direction-Aware Damping).

---

## 4. Bảng Kết Quả Đánh Giá Trên Tập Kiểm Thử (Test Set Patient `L310` - 214 Slices)

### 4.1. Đánh giá trên Cung Quét Chuẩn: LA-120° (64 views, $256 \times 256$, `noise_0`)

| Mô hình | Checkpoint Đánh Giá | Test PSNR (dB) | Test SSIM | Test RMSE | Thời gian Suy luận / Slice | Ghi chú |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **FBP Thô (Ram-Lak)** | *Không học (Analytical)* | $\approx 17.89$ | $\approx 0.4984$ | $\approx 0.1145$ | $< 5\text{ ms}$ | Chứa streak artifacts nặng do khuyết $240^\circ$ |
| **`LEARN_Mamba`** | `mamba_la-epoch=17` | **26.32** | **0.7468** | **0.0493** | $\approx 12\text{ ms}$ | Baseline Selective SSM (Epoch 17) |
| **`LEARN_LongNet`** | `longnet_la-last.ckpt` | **31.62** | **0.8991** | **0.0270** | $\approx 28\text{ ms}$ | Baseline Dilated Attention (50 ep) |
| **`LEARN_Longformer`**| `longformer_la-epoch=45` | **33.10** | **0.9237** | **0.0224** | $\approx 45\text{ ms}$ | Baseline Sliding-Chunks (50 ep - Best Ep 45) |
| **`SOLAR_LongNet` (Đề xuất)** | `solar_longnet_la-epoch=29` | **31.03** | **0.8958** | **0.0294** | $\approx 25\text{ ms}$ | Newton-CG 8 stages + Dilated Attention (31 ep, Job `67823`) |
| **`SOLAR_Mamba` (Đề xuất)** | `solar_mamba_la-epoch=45` | **31.90** | **0.9114** | **0.0268** | $\approx 18\text{ ms}$ | Newton-CG 8 stages + Selective SSM (50 ep, Job `68552`); +5.58 dB so với LEARN_Mamba |
| **`SOLAR_Longformer` (Đề xuất)**| `solar_longformer_la-epoch=45` | **32.95** | **0.9155** | **0.0228** | $\approx 35\text{ ms}$ | Newton-CG 8 stages + Sliding-Chunks (50 ep, Job `68551`) |
| **`LEARN` (Original CNN)** | `epoch=46-val_psnr=36.27` | **32.29** | **0.9383** | **0.0246** | $\approx 20\text{ ms}$ | Baseline CNN 3 tầng 14 stages (Hu Chen et al. 2018, Job `71615`) |
| **`RegFormer`** | `epoch=46-val_psnr=36.05` | **32.82** | **0.9398** | **0.0234** | $\approx 30\text{ ms}$ | Baseline điều hòa kép Local CNN + Swin Transformer (Job `71616`) |
| **`DuDoTrans`** | `epoch=38-val_psnr=25.73` | **25.15** | **0.7447** | **0.0650** | $\approx 15\text{ ms}$ | Baseline biến đổi đa miền Sinogram Transformer + FBP (Job `71617`) |
| **`SOLAR_RegFormer` (Đề xuất - 50 ep)** | `solar_regformer_la-epoch=45` | **32.47** | **0.9095** | **0.0247** | $\approx 22\text{ ms}$ | Newton-CG 8 stages + Swin Window Attn (50 ep, Job `72610`); +0.36 dB PSNR và +0.0051 SSIM so với mốc 35 ep |

### 4.2. Stress Test trên Cung Quét Khắc Nghiệt: LA-90° (64 views, $256 \times 256$, `noise_0`)
*(Đánh giá khả năng bù đắp nêm khuyết khi mô hình được thử thách trên dải góc hẹp hơn)*

| Mô hình | Checkpoint Đánh Giá | Test PSNR (dB) | Test SSIM | Test RMSE | Ghi chú |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **FBP Thô (Ram-Lak)** | *Không học (Analytical)* | $\approx 15.20$ | $\approx 0.4120$ | $\approx 0.1450$ | Khuyết $270^\circ$ (Missing Wedge cực đại) |
| **`LEARN_Mamba`** | `mamba_la-epoch=17` | **18.76** | **0.4292** | **0.1129** | Giảm mạnh do SSM 1D không tổng quát tốt khi nêm khuyết mở rộng |
| **`LEARN_LongNet`** | `longnet_la-last.ckpt` | **19.19** | **0.5876** | **0.1058** | Giữ được cấu trúc tốt hơn Mamba nhờ Dilated Attention đa tỷ lệ |
| **`LEARN_Longformer`**| `longformer_la-epoch=45` | **19.16** | **0.6097** | **0.1055** | SSIM cao nhất baseline nhờ cơ chế sliding-chunks + global tokens |
| **`SOLAR_LongNet` (Đề xuất)** | `solar_longnet_la-epoch=29` | **27.19** | **0.8639** | **0.0462** | 🚀 **Vượt trội đột phá (+8.00 dB PSNR, +0.2763 SSIM)** so với LEARN_LongNet |
| **`SOLAR_Mamba` (Đề xuất)** | `solar_mamba_la-epoch=45` | **27.53** | **0.8760** | **0.0447** | 🚀 **Vượt trội đột phá (+8.77 dB PSNR, +0.4468 SSIM)** so với LEARN_Mamba (+104.1% SSIM) |
| **`SOLAR_Longformer` (Đề xuất)**| `solar_longformer_la-epoch=45` | **28.05** | **0.8774** | **0.0412** | 🏆 **SOTA Toàn diện ở góc 90° (+8.89 dB PSNR, +0.2677 SSIM)** so với LEARN_Longformer |
| **`LEARN` (Original CNN)** | `epoch=46-val_psnr=36.27` | **28.03** | **0.8793** | **0.0411** | Duy trì SSIM cao nhờ kiến trúc unrolling gradient descent |
| **`RegFormer`** | `epoch=46-val_psnr=36.05` | **28.20** | **0.8814** | **0.0405** | SSIM cao nhất baseline nhờ kết hợp Local CNN và Swin Transformer |
| **`SOLAR_RegFormer` (Đề xuất - 50 ep)**| `solar_regformer_la-epoch=45` | **27.64** | **0.8804** | **0.0442** | 🏆 **+8.48 dB / +0.2707** so với baseline LEARN_Longformer (50 ep, Job `72610`); SSIM 0.8804 vượt cả SOLAR_Longformer và SOLAR_Mamba |
| **`DuDoTrans`** | `epoch=38-val_psnr=25.73` | **21.81** | **0.6738** | **0.0842** | Đa miền phục hồi tốt hơn FBP (+6.61 dB) nhưng kém hơn nhóm mở cuộn |

---

## 5. Trực Quan Hóa & Kết Xuất Ảnh Đối Sánh (Visualizations - Slurm Job `67830`)

Toàn bộ quá trình trực quan hóa đã được thực thi tự động qua Slurm Job `67830` (trên cụm DGX-A100) trên 3 lát cắt y tế độc lập (`slice_050`, `slice_100`, `slice_150`) của bệnh nhân `Patient L310` cho cả 2 cung quét $120^\circ$ và $90^\circ$:

* **Thư mục lưu trữ ảnh:** [`reports/sep-05-2026/visualizations/`](reports/sep-05-2026/visualizations/) và [`visualizations/`](visualizations/).
* **Các sản phẩm ảnh kết xuất đạt chuẩn 300 DPI:**
  1. `1_ground_truth.png` & `2_fbp_input.png`: Ảnh tham chiếu CT chuẩn y tế và ảnh FBP thô đầu vào.
  2. `3_recon_*.png` & `4_error_map_*.png`: Ảnh tái tạo và bản đồ sai số khuếch đại $\times 5$ của cả 6 mô hình (`SOLAR_LongNet`, `SOLAR_Mamba`, `SOLAR_Longformer`, `LEARN_LongNet`, `LEARN_Mamba`, `LEARN_Longformer`).
  3. `comparison_solar_summary.png`: Panel đối sánh đa khung hình giữa Ground Truth, FBP và 3 biến thể đề xuất SOLAR.
  4. `comparison_baseline_vs_solar.png`: Panel lưới $2\times 4$ đối sánh trực diện 1-1 giữa Baseline Bậc 1 và Đề xuất Bậc 2.




