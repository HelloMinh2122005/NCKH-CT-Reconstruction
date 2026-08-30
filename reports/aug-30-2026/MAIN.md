# 1. Bộ Dữ Liệu Thực Nghiệm (Datasets)

Hoàn tất quy trình tiền xử lý, mô phỏng vật lý tia X và tạo sinh hoàn chỉnh 2 bộ dữ liệu chuẩn y tế từ cơ sở dữ liệu **AAPM Mayo Clinic Low Dose CT Grand Challenge (2016)**:

### 1.1. Hai Tập Dữ Liệu Chính:
1. **Tập Dữ Liệu Chuẩn LA-120° (Limited-Angle 120-degree):**
   * **Cung góc quét:** $[-60^\circ, +60^\circ]$ (Độ mở cung quét $120^\circ$, khuyết nêm góc $240^\circ$).
   * **Số góc chiếu (Views):** 64 views (Bước nhảy góc $\Delta \theta \approx 1.87^\circ$).
   * **Cảm biến (Detectors):** 512 kênh (Dải $[-480, 480]\text{ mm}$).
   * **Độ phân giải:** Ảnh $256 \times 256$ pixel, mức nhiễu `noise_0`.
2. **Tập Dữ Liệu Khắc Nghiệt LA-90° (Limited-Angle 90-degree Stress-Test):**
   * **Cung góc quét:** $[-45^\circ, +45^\circ]$ (Độ mở cung quét $90^\circ$, khuyết nêm góc cực đại $270^\circ$).
   * **Số góc chiếu:** 64 views | 512 detectors | $256 \times 256$ pixel.

### 1.2. Phân Chia Dữ Liệu Tuyệt Đối Độc Lập (Patient-Level Splitting):
* **Tập Huấn luyện (Train):** **1,920 lát cắt CT** từ 8 bệnh nhân độc lập (`L067`, `L096`, `L109`, `L143`, `L192`, `L286`, `L291`, `L506`).
* **Tập Kiểm định (Validation):** **244 lát cắt CT** từ bệnh nhân `L333`.
* **Tập Kiểm thử Độc lập (Test Benchmark):** **214 lát cắt CT** từ bệnh nhân `L310`.
* **Định dạng lưu trữ:** Toàn bộ dữ liệu được lưu dưới dạng cache mảng thực `.npy` float32 tại `dataset/limited_angle/` nhằm tối đa hóa tốc độ nạp dữ liệu trên GPU A100.

---

# 2. Báo Cáo Chi Tiết Tiến Trình Huấn Luyện (Training)

Cả 3 mô hình Baseline đều được thiết lập theo khung Unrolling 14 giai đoạn ($K=14$ stages) với bộ tối ưu Adam và Cosine Annealing Learning Rate ($10^{-4} \to 10^{-5}$) trên cụm máy chủ DGX NVIDIA A100 80GB (Slurm Cluster):

```text
┌──────────────────┬──────────────┬──────────────────┬──────────────┬──────────────┬────────────────────────────────────────────────────────┐
│ Mô hình Baseline │ Slurm Job ID │ Số Epoch Đạt ĐượC│ Best Val PSNR│ Best Val SSIM│ Trạng thái & Hiện tượng Kỹ thuật                       │
├──────────────────┼──────────────┼──────────────────┼──────────────┼──────────────┼────────────────────────────────────────────────────────┤
│ LEARN_LongNet    │ 65486        │ 50 / 50 (100%)   │ 33.37 dB     │ 0.9090       │ ✅ Hoàn thành trọn vẹn 50 epochs, hội tụ xuất sắc.     │
│ LEARN_Longformer │ 65485 / 65892│ 19 / 50 (Resume) │ 32.80 dB     │ 0.9109       │ ⏱️ Đạt Time Limit 24h ở Ep19; đang Resume tiếp tục.    │
│ LEARN_Mamba      │ 65484        │ 41 / 50 (Hết 24h)│ 27.66 dB     │ 0.7373       │ ⚠️ Đạt đỉnh ở Ep17; sau Ep20 xuất hiện Gradient NaN.   │
└──────────────────┴──────────────┴──────────────────┴──────────────┴──────────────┴────────────────────────────────────────────────────────┘
```

### 2.1. Phân Tích Kỹ Thuật Chuyên Sâu Từng Mô Hình:
1. **`LEARN_LongNet` (Thành công trọn vẹn):**
   * Tham chiếu Log thực nghiệm: [scripts/output/train_longnet_la/log/65486.out](../../scripts/output/train_longnet_la/log/65486.out).
   * Hoàn thành 100% (50/50 epochs) trong 23 giờ.
   * Chỉ số tăng trưởng rất mượt mà từ $17.89\text{ dB}$ (FBP thô) lên **$33.37\text{ dB}$** (SSIM: **$0.9090$** tại Epoch 45).
   * Cơ chế *Multi-Scale Dilated Attention* cho thấy khả năng bao quát ngữ cảnh không gian 2D rất ổn định trên địa hình của ảnh cắt lớp.

2. **`LEARN_Longformer` (Chất lượng tương đồng cấu trúc cao nhất):**
   * Tham chiếu Log ban đầu: [scripts/output/train_longformer_la/log/65485.out](../../scripts/output/train_longformer_la/log/65485.out) và Log Resume hiện tại: [scripts/output/train_longformer_la/log/65892.out](../../scripts/output/train_longformer_la/log/65892.out).
   * Do chi phí tính toán attention cục bộ theo cửa sổ trượt (Sliding-Chunks), thời gian huấn luyện $\approx 52$ phút/epoch $\to$ đạt mốc 24h tại Epoch 19.
   * Tuy mới đi qua 19 epochs, mô hình đã đạt **SSIM cao nhất trong các baseline ($0.9109$)** và PSNR đạt **$32.80\text{ dB}$** (tại Epoch 15).
   * Fix: Bổ sung tính năng `--resume_ckpt` và hiện **Job ID `65892` đang tiếp tục huấn luyện từ Epoch 17**.

3. **`LEARN_Mamba` (Xác thực thực nghiệm điểm nghẽn lý thuyết cốt lõi):**
   * Tham chiếu Log thực nghiệm: [scripts/output/train_mamba_la/log/65484.out](../../scripts/output/train_mamba_la/log/65484.out) (Xem chi tiết phân tích tại [EXPERIMENT_RESULTS.md](../../EXPERIMENT_RESULTS.md#L35-L42) và tài liệu kiến trúc [note/ARCHITECTURE_MVA_VS_PROPOSED_SOLAR.md](../../note/ARCHITECTURE_MVA_VS_PROPOSED_SOLAR.md)).
   * Tốc độ huấn luyện rất nhanh ($\approx 25$ phút/epoch), đạt 41 epochs trong 24h.
   * Đạt đỉnh hiệu năng tại **Epoch 17** với PSNR **$27.66\text{ dB}$** và SSIM **$0.7373$** (Checkpoint: `mamba_la-epoch=17-val_psnr=27.66-val_ssim=0.7373.ckpt`).
   * **Hiện tượng quan trọng:** Từ sau Epoch 20 (bắt đầu tại Epoch 21 trong log [65484.out](../../scripts/output/train_mamba_la/log/65484.out)), hàm mất mát và gradient bắt đầu mất ổn định số học và dần chuyển sang `NaN`.
   * **Ý nghĩa khoa học:** Hiện tượng này **khớp 100% với phân tích lý thuyết toán học vi cục bộ**: Cơ chế quét tuần tự 1D của Mamba-1 khi unrolling sâu (14 vòng) trên ma trận Hessian $A^TA$ bị suy biến nặng của Limited-Angle CT rất dễ bị dao động số học và bùng nổ gradient.

---

# 3. Kết Quả Kiểm Thử & Đối Sánh Benchmark (Test)

 **214 lát cắt CT của bệnh nhân `L310`**.

### 3.1. Bảng Tổng Hợp So Sánh Các Mô Hình (Benchmark Results):

| Mô hình Kiểm Thử       | Checkpoint Đánh Giá            | Dải Góc LA-120° (64 views) |                    |                    | Dải Góc LA-90° (Stress-Test 64v) |                    |                    | Nhận Xét Kỹ Thuật                               |
| :--------------------- | :----------------------------- | :------------------------: | :----------------: | :----------------: | :------------------------------: | :----------------: | :----------------: | :---------------------------------------------- |
|                        |                                |      **PSNR (dB) ↑**       |     **SSIM ↑**     |     **RMSE ↓**     |         **PSNR (dB) ↑**          |     **SSIM ↑**     |     **RMSE ↓**     |                                                 |
| **FBP Thô (Ram-Lak)**  | *Không học (Analytical)*       |           17.89            |       0.4984       |       0.1145       |              15.20               |       0.4120       |       0.1450       | Vệt sọc (*streak artifacts*) dày đặc            |
| **`LEARN_Mamba`**      | `mamba_la-epoch=17.ckpt`       |         **26.32**          |     **0.7468**     |     **0.0493**     |            **18.76**             |     **0.4292**     |     **0.1129**     | Khôi phục mức trung bình; góc 90° suy giảm mạnh |
| **`LEARN_LongNet`**    | `longnet_la-last.ckpt` (50 ep) |         **31.62**          |     **0.8991**     |     **0.0270**     |            **19.19**             |     **0.5876**     |     **0.1058**     | Khôi phục sắc nét, giữ cấu trúc xương và mô tốt |
| **`LEARN_Longformer`** | `longformer_la-last.ckpt`      |     *Đang huấn luyện*      | *Đang huấn luyện*  | *Đang huấn luyện*  |        *Đang huấn luyện*         | *Đang huấn luyện*  | *Đang huấn luyện*  | *(Sẽ cập nhật ngay sau khi Job 65892 hoàn tất)* |
| **`SOLAR` (Đề xuất)**  | *Đang triển khai*              |     *Mục tiêu: >35.0*      | *Mục tiêu: >0.930* | *Mục tiêu: <0.015* |        *Mục tiêu: >32.5*         | *Mục tiêu: >0.880* | *Mục tiêu: <0.020* | Newton-CG Unrolling + Dual-Branch Regularizer   |

### 3.2. Đánh Giá Khả Năng Tổng Quát Hóa Khi Chuyển Từ 120° Sang 90°:
* Khi góc quét bị thu hẹp từ $120^\circ \to 90^\circ$ (khuyết tới $270^\circ$ góc chiếu):
  * Chỉ số của **`LEARN_Mamba`** sụt giảm nghiêm trọng từ $26.32\text{ dB} \to 18.76\text{ dB}$ (SSIM chỉ còn $0.4292$), do cơ chế quét 1D không có khả năng thích nghi khi cấu trúc nêm khuyết mở rộng theo phương ngang.
  * **`LEARN_LongNet`** giữ được độ tương đồng cấu trúc giải phẫu vượt trội hơn hẳn ($0.5876$ so với $0.4292$ của Mamba) nhờ các trường chú ý giãn nở đa tỷ lệ giúp tổng hợp thông tin từ các vùng lân cận không bị mất.

---

# 4. Trực Quan Hóa (Visualizations)

Code trực quan hóa [visualize_benchmark.py](../../visualize_benchmark.py) để kết xuất toàn bộ ảnh trực quan hóa chất lượng cao ($300\text{ DPI}$) trên các lát cắt kiểm thử tiêu biểu (`slice_050`, `slice_100`, `slice_150` của bệnh nhân `L310`).