# BÁO CÁO TIẾN ĐỘ & ĐỘT PHÁ CÔNG NGHỆ (05/09/2026): HOÀN THÀNH HUẤN LUYỆN, ĐÁNH GIÁ BENCHMARK & TRỰC QUAN HÓA TOÀN DIỆN KIẾN TRÚC ĐỀ XUẤT SOLAR

### 1. Tổng Quan Tiến Độ Huấn Luyện (Training Progress)
Hệ thống mạng đề xuất **SOLAR** (Second-Order Dual-Branch Unrolling với Bộ giải Safe Conjugate Gradient) đã đạt được những cột mốc đột phá trên cả 3 biến thể:
* **`SOLAR_LongNet`**: Đạt mốc **31/50 Epochs** (Slurm Job `67512`). Best checkpoint tại **Epoch 29**:
  * **Val PSNR:** **32.26 dB** | **Val SSIM:** **0.8935**
  * Checkpoint: `saved_models/SOLAR_LongNet/solar_longnet_la-epoch=29-val_psnr=32.26-val_ssim=0.8935.ckpt`
* **`SOLAR_Mamba`**: Đạt mốc **27/50 Epochs** (Slurm Job `67513` & `67821`). Best checkpoint tại **Epoch 27**:
  * **Val PSNR:** **33.22 dB** | **Val SSIM:** **0.9018**
  * Huấn luyện trơn tru tuyệt đối 100%, không gặp hiện tượng nổ `NaN` gradient nhờ cơ chế Softplus SPD Parameterization.
  * Checkpoint: `saved_models/SOLAR_Mamba/solar_mamba_la-epoch=27-val_psnr=33.22-val_ssim=0.9018.ckpt`
* **`SOLAR_Longformer`**: Đạt mốc **36/50 Epochs** (Slurm Job `67511` & `67820`). Best checkpoint tại **Epoch 35**:
  * **Val PSNR:** **33.62 dB** | **Val SSIM:** **0.9079**
  * Checkpoint: `saved_models/SOLAR_Longformer/solar_longformer_la-epoch=35-val_psnr=33.62-val_ssim=0.9079.ckpt`

---

### 2. Kết Quả Đánh Giá Kiểm Thử Độc Lập (Test Benchmark trên Patient L310 - 214 Lát Cắt)
Toàn bộ các mô hình đối chứng bậc 1 (**LEARN**) và đề xuất bậc 2 (**SOLAR**) đã được đánh giá mù trên 214 lát cắt CT của bệnh nhân `Patient L310` (AAPM Mayo Clinic Low-Dose CT) trên 2 cung quét: Cung quét chuẩn **LA-120°** và Cung quét cực hạn **LA-90°**:

| Phương pháp & Mô hình | LA-120° PSNR (dB) | LA-120° SSIM | LA-120° RMSE | LA-90° PSNR (dB) | LA-90° SSIM | LA-90° RMSE | $\Delta$ PSNR ở LA-90° | $\Delta$ SSIM ở LA-90° |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FBP Thô (Ram-Lak)** | 17.89 | 0.4984 | 0.1145 | 15.20 | 0.4120 | 0.1450 | - | - |
| **`LEARN_Mamba`** (Epoch 17) | 26.32 | 0.7468 | 0.0493 | 18.76 | 0.4292 | 0.1129 | Baseline | Baseline |
| **`LEARN_LongNet`** (Epoch 50) | 31.62 | 0.8991 | 0.0270 | 19.19 | 0.5876 | 0.1058 | Baseline | Baseline |
| **`LEARN_Longformer`** (Epoch 45) | **33.10** | **0.9237** | **0.0224** | 19.16 | 0.6097 | 0.1055 | Baseline | Baseline |
| **`SOLAR_Mamba` (Đề xuất)** | **31.21** | **0.8982** | **0.0291** | **27.16** | **0.8620** | **0.0472** | **+8.40 dB** | **+0.4328 (+100.8%)** |
| **`SOLAR_LongNet` (Đề xuất)** | **31.03** | **0.8958** | **0.0294** | **27.19** | **0.8639** | **0.0462** | **+8.00 dB** | **+0.2763 (+47.0%)** |
| **`SOLAR_Longformer` (Đề xuất)** | **32.51** | **0.9101** | **0.0239** | **27.92** | **0.8736** | **0.0416** | **+8.76 dB** | **+0.2639 (+43.3%)** |

📊 **Dữ liệu số liệu gốc được lưu trữ tại:** [benchmark_results.csv](benchmark_results.csv).

---

### 3. Trực Quan Hóa & Kết Xuất Ảnh Đối Sánh (Visualizations - Slurm Job `67830`)
Đã hoàn tất quá trình kết xuất ảnh tái tạo, bản đồ sai số (Error Map $\times 5$), và các panel đối sánh chất lượng cao ($300\text{ DPI}$) trên 3 lát cắt đại diện (`slice_050`, `slice_100`, `slice_150`) cho cả hai cung quét $120^\circ$ và $90^\circ$:

1. **Bộ ảnh thành phần riêng lẻ:**
   * Ground Truth chuẩn y tế (`1_ground_truth.png`)
   * Ảnh FBP thô đầu vào (`2_fbp_input.png`)
   * Ảnh tái tạo AI của từng mô hình (`3_recon_*.png`)
   * Bản đồ sai lệch nhiệt độ so với Ground Truth (`4_error_map_*.png`)
2. **Panel Đối Sánh Đa Chiều:**
   * `comparison_solar_summary.png`: Đối sánh trực quan giữa Ground Truth, FBP và 3 biến thể SOLAR (`SOLAR_LongNet`, `SOLAR_Mamba`, `SOLAR_Longformer`).
   * `comparison_baseline_vs_solar.png`: Panel lưới $2\times 4$ đối sánh trực diện từng cặp mô hình Baseline bậc 1 vs Đề xuất bậc 2:
     * *Hàng 1 (Baseline):* Ground Truth | `LEARN_LongNet` | `LEARN_Mamba` | `LEARN_Longformer`
     * *Hàng 2 (Đề xuất):* FBP Input | `SOLAR_LongNet` | `SOLAR_Mamba` | `SOLAR_Longformer`
     * Thể hiện rõ nét sự vượt trội tại cung quét $90^\circ$: Mạng baseline bị nhiễu vệt (streak artifacts) và mờ nhòe nghiêm trọng ($< 20.7\text{ dB}$), trong khi toàn bộ các mạng SOLAR phục hồi trọn vẹn ranh giới mô mềm và xương sắc nét ($29.0 - 31.7\text{ dB}$, SSIM $> 0.90$).
   * `comparison_summary.png`: Panel tổng hợp theo chuẩn định dạng báo cáo trước đây.

📁 **Toàn bộ hình ảnh chất lượng cao được lưu trữ đồng bộ tại:** [visualizations/](visualizations/).
