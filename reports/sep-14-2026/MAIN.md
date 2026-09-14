# BÁO CÁO TIẾN ĐỘ & ĐÁNH GIÁ ĐỐI SÁNH HỆ THỐNG MÔ HÌNH BASELINE (14/09/2026)

### 1. Tổng Quan Tiến Độ Huấn Luyện (Training Progress)

* **`LEARN` (Hu Chen et al., IEEE TMI 2018)** — Slurm Job `71393`:
  * **Thời gian thực thi:** 13 giờ 10 phút.
  * **Checkpoint tối ưu:** `saved_models/LEARN/LEARN_AAPM_LA_120deg_view64/epoch=46-val_psnr=36.2681.ckpt`
  * **Đỉnh Validation:** **Val PSNR = 36.27 dB** | **Val SSIM = 0.950** tại Epoch 46. Quá trình hội tụ mượt mà, không gặp bất kỳ lỗi số học nào.
* **`RegFormer` (Local CNN + Swin Transformer)** — Slurm Job `71394`:
  * **Thời gian thực thi:** 14 giờ 58 phút.
  * **Checkpoint tối ưu:** `saved_models/RegFormer/RegFormer_AAPM_LA_120deg_view64/epoch=46-val_psnr=36.0533.ckpt`
  * **Đỉnh Validation:** **Val PSNR = 36.05 dB** | **Val SSIM = 0.950** tại Epoch 46. Cơ chế điều hòa kép hoạt động ổn định xuyên suốt 50 epochs.
* **`DuDoTrans` (Dual-Domain Transformer)** — Slurm Job `71395`:
  * **Thời gian thực thi:** 02 giờ 32 phút (tốc độ xử lý cực nhanh nhờ số tham số nhỏ 0.13M).
  * **Checkpoint tối ưu:** `saved_models/DuDoTrans/DuDoTrans_AAPM_LA_120deg_view64/epoch=38-val_psnr=25.7300.ckpt`
  * **Đỉnh Validation:** **Val PSNR = 25.73 dB** | **Val SSIM = 0.730** tại Epoch 38.

---

### 2. Kết Quả Đánh Giá Kiểm Thử Độc Lập (Test Benchmark trên Patient L310 - 214 Lát Cắt)
Cả 3 mô hình đã được kiểm thử mù độc lập trên 214 lát cắt CT của bệnh nhân `Patient L310` (AAPM Mayo Clinic Low-Dose CT) trên 2 cung quét: Chuẩn **LA-120°** và Khắc nghiệt **LA-90°** (Jobs `71615`, `71616`, `71617`):

| Mô hình / Phương pháp            | Cung quét LA-120°<br>PSNR (dB) / SSIM / RMSE | Cung quét LA-90°<br>PSNR (dB) / SSIM / RMSE | Số lượng Stages ($K$) | Số Params | Ghi chú kỹ thuật                                                  |
| :------------------------------- | :------------------------------------------: | :-----------------------------------------: | :-------------------: | :-------: | :---------------------------------------------------------------- |
| **`LEARN (Original CNN)`**       |       **32.29** / **0.9383** / 0.0246        |       **28.03** / **0.8793** / 0.0411       |       14 stages       |   0.84M   | Unrolling CNN 3 tầng thuần túy (Hu Chen et al. 2018, Job `71615`) |
| **`RegFormer`**                  |       **32.82** / **0.9398** / 0.0234        |       **28.20** / **0.8814** / 0.0405       |       14 stages       |   1.23M   | Điều hòa kép Local CNN + Swin Transformer (Job `71616`)           |
| **`DuDoTrans`**                  |           25.15 / 0.7447 / 0.0650            |           21.81 / 0.6738 / 0.0842           |  1 pass (No unroll)   | **0.13M** | Biến đổi đa miền Sino Transformer + FBP + Image Net (Job `71617`) |
| **`LEARN_Longformer`**           |       **33.10** / 0.9237 / **0.0224**        |           19.16 / 0.6097 / 0.1055           |       14 stages       |   4.00M   | Baseline Sliding-Chunks (Best Ep 45)                              |
| **`LEARN_LongNet`**              |           31.62 / 0.8991 / 0.0270            |           19.19 / 0.5876 / 0.1058           |       14 stages       |   2.40M   | Baseline Dilated Attention (50 ep)                                |
| **`LEARN_Mamba`**                |           26.32 / 0.7468 / 0.0493            |           18.76 / 0.4292 / 0.1129           |       14 stages       |   2.90M   | Baseline Selective SSM (Epoch 17)                                 |
| **`SOLAR_Longformer` (Đề xuất)** |           32.95 / 0.9155 / 0.0228            |       **28.05** / **0.8774** / 0.0412       |     **8 stages**      |   4.00M   | Tối ưu bậc 2 Newton-CG + Sliding-Chunks (Job `68551`)             |
| **`SOLAR_Mamba` (Đề xuất)**      |           31.90 / 0.9114 / 0.0268            |           27.53 / 0.8760 / 0.0447           |     **8 stages**      |   2.90M   | Tối ưu bậc 2 Newton-CG + Selective SSM (Job `68552`)              |

📊 **Bảng dữ liệu đo lường định lượng chi tiết được lưu trữ tại:** [benchmark_results.csv](benchmark_results.csv).

---

### 3. Phân Tích & Đối Sánh Chuyên Sâu Giữa Các Trường Phái Kiến Trúc

#### A. Sức mạnh của Ràng buộc Cục bộ (`RegFormer` & `LEARN`) so với Nhóm Chuỗi Dài Bậc 1
1. **Tại LA-120°:** `LEARN_Longformer` dẫn đầu về PSNR (**33.10 dB**), nhưng `RegFormer` (**32.82 dB**) và `LEARN` (**32.29 dB**) lại chiếm ưu thế tuyệt đối về độ tương đồng cấu trúc giải phẫu (**SSIM đạt 0.9398 và 0.9383**, cao hơn mức 0.9237 của Longformer).
2. **Tại LA-90° (Stress Test góc khuyết cực đại 270°):**
   * Trong khi nhóm mô hình chuỗi dài bậc 1 (`LEARN_Longformer`, `LEARN_LongNet`, `LEARN_Mamba`) đều bị **sụp đổ về ~18.8 - 19.2 dB** do trường tiếp nhận toàn cục bị nhiễu bởi vệt sọc giả, thì `RegFormer` (**28.20 dB**) và `LEARN` (**28.03 dB**) giữ vững phong độ xuất sắc.
   * **Bản chất vật lý:** Tính chất cục bộ (Locality của Conv $3\times 3$) và cửa sổ chú ý cục bộ (Local Window $7\times 7$ của Swin Transformer) hoạt động như một màng lọc, ngăn chặn việc lan truyền lỗi của các vệt sọc nêm khuyết ra toàn bộ trường ảnh.

#### B. Giới hạn của Kiến trúc Đa miền Không Mở Cuộn (`DuDoTrans`)
* `DuDoTrans` chỉ đạt **25.15 dB ở 120°** và **21.81 dB ở 90°**.
* Mô hình chỉ thực thi tuần tự 1 chiều: *Sinogram Restoration $\to$ FBP $\to$ Image Refinement*. Do không có các vòng lặp unrolling liên tục ép ảnh nghiệm số quay lại kiểm chứng với phép chiếu tia X $A^T(Ax - y)$, sai số miền sinogram chuyển hóa thành sai số không thể phục hồi trên miền ảnh.