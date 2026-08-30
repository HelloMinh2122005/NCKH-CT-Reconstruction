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
| **2** | **`LEARN_Longformer`**| Sliding-Chunks Self-Attention + Global Tokens | **LA-120° (64v)** | **19 / 50** | **32.80** | **0.9109** | `saved_models/LEARN_Longformer/longformer_la-epoch=15-val_psnr=32.80-val_ssim=0.9109.ckpt` | ⏱️ **Đạt Time Limit 24h ở Epoch 19**. Đạt SSIM cao nhất trong các baseline ($0.9109$ ở Epoch 15). Đang tiến hành resume từ `last.ckpt`. |
| **3** | **`LEARN_Mamba`** | Selective SSM ($\mathcal{O}(N)$) | **LA-120° (64v)** | **41 / 50** | **27.66** | **0.7373** | `saved_models/LEARN_Mamba/mamba_la-epoch=17-val_psnr=27.66-val_ssim=0.7373.ckpt` | ⚠️ **Sử dụng Checkpoint Epoch 17**. Đạt đỉnh ở Epoch 17; sau Epoch 20 xuất hiện bùng nổ gradient / NaN do đặc tính quét 1D trên Hessian suy biến của LA-CT. |

---

## 3. Lý Do Khoa Học & Phân Tích Lựa Chọn Checkpoint Cho Từng Mô Hình

### 3.1. `LEARN_LongNet` — Sử dụng `last.ckpt` hoặc `epoch=45`
* **Đặc tính:** Mô hình đã hoàn tất 100% kế hoạch huấn luyện (50 epochs) vào ngày 28/08/2026.
* **Đánh giá:** Chỉ số SSIM đạt $0.9090$ và PSNR $33.37\text{ dB}$, chất lượng tái tạo ổn định nhất và hoàn toàn không gặp lỗi số học.
* **Checkpoint sử dụng:** `saved_models/LEARN_LongNet/last.ckpt` (tương đương kết quả hội tụ tại cuối epoch 49-50) và `epoch=45` (đỉnh PSNR cao nhất).

### 3.2. `LEARN_Longformer` — Tiến hành Resume Training từ `last.ckpt`
* **Đặc tính:** Mô hình học rất mượt và có độ tương đồng cấu trúc cao nhất (SSIM đạt $0.9109$ chỉ sau 15 epochs).
* **Lý do dừng:** Do cơ chế chú ý sliding-chunks tính toán kỹ lưỡng, thời gian chạy ~52 phút/epoch $\to$ hết hạn 24h Slurm tại Epoch 19.
* **Xử lý:** Kích hoạt tính năng `--resume_ckpt` nạp `saved_models/LEARN_Longformer/last.ckpt` để tiếp tục huấn luyện lên 40-50 epochs.

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
| **`LEARN_Longformer`**| `longformer_la-epoch=15` | *Đang tiếp tục huấn luyện...* | *Đang tiếp tục huấn luyện...* | *Đang tiếp tục huấn luyện...* | $\approx 45\text{ ms}$ | Job `65892` đang resume |
| **`SOLAR` (Đề xuất)** | *Đang triển khai* | *Mục tiêu: > 35.0 dB* | *Mục tiêu: > 0.9300* | *Mục tiêu: < 0.0150* | $\approx 20\text{ ms}$ | Newton-CG Unrolling + Dual-Branch |

### 4.2. Stress Test trên Cung Quét Khắc Nghiệt: LA-90° (64 views, $256 \times 256$, `noise_0`)
*(Đánh giá khả năng bù đắp nêm khuyết khi mô hình được thử thách trên dải góc hẹp hơn)*

| Mô hình | Checkpoint Đánh Giá | Test PSNR (dB) | Test SSIM | Test RMSE | Ghi chú |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **FBP Thô (Ram-Lak)** | *Không học (Analytical)* | $\approx 15.20$ | $\approx 0.4120$ | $\approx 0.1450$ | Khuyết $270^\circ$ (Missing Wedge cực đại) |
| **`LEARN_Mamba`** | `mamba_la-epoch=17` | **18.76** | **0.4292** | **0.1129** | Giảm mạnh do SSM 1D không tổng quát tốt khi nêm khuyết mở rộng |
| **`LEARN_LongNet`** | `longnet_la-last.ckpt` | **19.19** | **0.5876** | **0.1058** | Giữ được cấu trúc tốt hơn Mamba nhờ Dilated Attention đa tỷ lệ |
| **`SOLAR` (Đề xuất)** | *Đang triển khai* | *Mục tiêu: > 32.5 dB* | *Mục tiêu: > 0.8800* | *Mục tiêu: < 0.0200* | Damping thích nghi theo hướng khuyết |

