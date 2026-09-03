# BÁO CÁO TIẾN ĐỘ (03/09/2026): HOÀN THÀNH HUẤN LUYỆN & KIỂM THỬ LEARN_LONGFORMER

### 1. Tiến Độ Huấn Luyện (Training)
* Mô hình **`LEARN_Longformer`** đã hoàn thành trọn vẹn **50/50 Epochs** (Slurm Job `66652`).
* Đạt kết quả SOTA cao nhất trong toàn bộ nhóm Baseline tại **Epoch 45**:
  * **Val PSNR:** **34.77 dB**
  * **Val SSIM:** **0.9383**
* Checkpoint lưu trữ: `saved_models/LEARN_Longformer/longformer_la-epoch=45-val_psnr=34.77-val_ssim=0.9383.ckpt` và `last.ckpt`.

---

### 2. Kết Quả Đánh Giá Kiểm Thử (Test Benchmark)
* Đã thực hiện kiểm thử độc lập trên 214 lát cắt CT của bệnh nhân `L310` (Slurm Job `67223`) trên cả hai cấu hình:
  * **Cung quét chuẩn LA-120° (64 views):** **PSNR = 33.10 dB** | **SSIM = 0.9237** | **RMSE = 0.0224** (Cao nhất nhóm Baseline).
  * **Stress test LA-90° (64 views):** **PSNR = 19.16 dB** | **SSIM = 0.6097** | **RMSE = 0.1055** (Bảo toàn cấu trúc SSIM tốt nhất).
* 📊 **Tổng quan toàn bộ kết quả đối sánh chi tiết xem tại file:** [benchmark_results.csv](benchmark_results.csv).

---

### 3. Trực Quan Hóa (Visualizations)
* Đã hoàn tất kết xuất ảnh tái tạo, Error Map và Panel đối sánh đa mô hình (Slurm Job `67225`) trên 3 lát cắt kiểm thử (`slice_050`, `slice_100`, `slice_150`) cho cả hai dải góc quét $120^\circ$ và $90^\circ$.
* Toàn bộ hình ảnh chất lượng cao ($300\text{ DPI}$) được lưu trữ tại thư mục: [visualizations/](visualizations/).
