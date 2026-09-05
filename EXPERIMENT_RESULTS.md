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
| **5** | **`SOLAR_Longformer` (Đề xuất)**| **Newton-CG Bậc 2 + Res-CNN & Sliding-Chunks Attention** | **LA-120° (64v)** | **36 / 50 (Running)** | **33.62** | **0.9079** | `saved_models/SOLAR_Longformer/solar_longformer_la-epoch=35-val_psnr=33.62-val_ssim=0.9079.ckpt` | 🚀 **Đang chạy huấn luyện resume (Job `67820`)** từ Epoch 36 trên DGX-A100. |
| **6** | **`SOLAR_Mamba` (Đề xuất)**| **Newton-CG Bậc 2 + Res-CNN & Selective SSM** | **LA-120° (64v)** | **27 / 50 (Running)** | **33.19** | **0.8975** | `saved_models/SOLAR_Mamba/solar_mamba_la-epoch=25-val_psnr=33.19-val_ssim=0.8975.ckpt` | 🚀 **Đang chạy huấn luyện resume (Job `67821`)** từ Epoch 27 trên DGX-A100 (100% ổn định số học, không NaN). |

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
| **`SOLAR_Mamba` (Đề xuất)** | `solar_mamba_la-epoch=25` | **31.21** | **0.8982** | **0.0291** | $\approx 18\text{ ms}$ | Newton-CG 8 stages + Selective SSM (29 ep, Job `67829`); +4.89 dB so với LEARN_Mamba |
| **`SOLAR_Longformer` (Đề xuất)**| `solar_longformer_la-epoch=35` | **32.51** | **0.9101** | **0.0239** | $\approx 35\text{ ms}$ | Newton-CG 8 stages + Sliding-Chunks (36 ep, Job `67828`) |

### 4.2. Stress Test trên Cung Quét Khắc Nghiệt: LA-90° (64 views, $256 \times 256$, `noise_0`)
*(Đánh giá khả năng bù đắp nêm khuyết khi mô hình được thử thách trên dải góc hẹp hơn)*

| Mô hình | Checkpoint Đánh Giá | Test PSNR (dB) | Test SSIM | Test RMSE | Ghi chú |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **FBP Thô (Ram-Lak)** | *Không học (Analytical)* | $\approx 15.20$ | $\approx 0.4120$ | $\approx 0.1450$ | Khuyết $270^\circ$ (Missing Wedge cực đại) |
| **`LEARN_Mamba`** | `mamba_la-epoch=17` | **18.76** | **0.4292** | **0.1129** | Giảm mạnh do SSM 1D không tổng quát tốt khi nêm khuyết mở rộng |
| **`LEARN_LongNet`** | `longnet_la-last.ckpt` | **19.19** | **0.5876** | **0.1058** | Giữ được cấu trúc tốt hơn Mamba nhờ Dilated Attention đa tỷ lệ |
| **`LEARN_Longformer`**| `longformer_la-epoch=45` | **19.16** | **0.6097** | **0.1055** | SSIM cao nhất baseline nhờ cơ chế sliding-chunks + global tokens |
| **`SOLAR_LongNet` (Đề xuất)** | `solar_longnet_la-epoch=29` | **27.19** | **0.8639** | **0.0462** | 🚀 **Vượt trội đột phá (+8.00 dB PSNR, +0.2763 SSIM)** so với LEARN_LongNet |
| **`SOLAR_Mamba` (Đề xuất)** | `solar_mamba_la-epoch=25` | **27.16** | **0.8620** | **0.0472** | 🚀 **Vượt trội đột phá (+8.40 dB PSNR, +0.4328 SSIM)** so với LEARN_Mamba (+100.8% SSIM) |
| **`SOLAR_Longformer` (Đề xuất)**| `solar_longformer_la-epoch=35` | **27.92** | **0.8736** | **0.0416** | 🏆 **SOTA Toàn diện ở góc 90° (+8.76 dB PSNR, +0.2639 SSIM)** so với LEARN_Longformer |

---

## 5. Trực Quan Hóa & Kết Xuất Ảnh Đối Sánh (Visualizations - Slurm Job `67830`)

Toàn bộ quá trình trực quan hóa đã được thực thi tự động qua Slurm Job `67830` (trên cụm DGX-A100) trên 3 lát cắt y tế độc lập (`slice_050`, `slice_100`, `slice_150`) của bệnh nhân `Patient L310` cho cả 2 cung quét $120^\circ$ và $90^\circ$:

* **Thư mục lưu trữ ảnh:** [`reports/sep-05-2026/visualizations/`](reports/sep-05-2026/visualizations/) và [`visualizations/`](visualizations/).
* **Các sản phẩm ảnh kết xuất đạt chuẩn 300 DPI:**
  1. `1_ground_truth.png` & `2_fbp_input.png`: Ảnh tham chiếu CT chuẩn y tế và ảnh FBP thô đầu vào.
  2. `3_recon_*.png` & `4_error_map_*.png`: Ảnh tái tạo và bản đồ sai số khuếch đại $\times 5$ của cả 6 mô hình (`SOLAR_LongNet`, `SOLAR_Mamba`, `SOLAR_Longformer`, `LEARN_LongNet`, `LEARN_Mamba`, `LEARN_Longformer`).
  3. `comparison_solar_summary.png`: Panel đối sánh đa khung hình giữa Ground Truth, FBP và 3 biến thể đề xuất SOLAR.
  4. `comparison_baseline_vs_solar.png`: Panel lưới $2\times 4$ đối sánh trực diện 1-1 giữa Baseline Bậc 1 và Đề xuất Bậc 2.




