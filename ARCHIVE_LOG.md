# LƯU TRỮ LỊCH SỬ TIẾN ĐỘ DỰ ÁN (ARCHIVE PROGRESS LOG)
# CÁC MỐC THỬ NGHIỆM ĐÃ HOÀN TẤT TỪ THÁNG 08/2026 ĐẾN 14/09/2026

> **Mục đích:** Lưu trữ đầy đủ chi tiết toàn bộ các mốc nghiên cứu, job ID, chỉ số val/test và quá trình tạo dữ liệu ban đầu. Tài liệu này được tách ra từ `CHECKPOINT.md` để tối ưu ngữ cảnh làm việc hàng ngày mà vẫn bảo toàn 100% bằng chứng nghiên cứu lịch sử.

---

## 1. Các Mốc Khởi Tạo & Chuẩn Hóa Baseline (Cuối Tháng 08/2026)
- [x] Submit Job Slurm sinh dữ liệu: **Job ID `64295`** (`generate_la_dataset.sh`), cấu hình log chuẩn `scripts/output/generate_la_dataset/log/`.
- [x] **Sao chép và chuẩn hóa 3 mô hình Baseline từ paper của Thành sang Limited-Angle CT (`baselines/`):**
  - **`LEARN_Mamba`:** `models.py`, `train_mamba_la.py`, `test_mamba_la.py`, `scripts/train_mamba_la.sh`.
  - **`LEARN_Longformer`:** `models.py`, `train_longformer_la.py`, `test_longformer_la.py`, `scripts/train_longformer_la.sh`.
  - **`LEARN_LongNet`:** `models.py`, `train_longnet_la.py`, `test_longnet_la.py`, `long_net.py`, `scripts/train_longnet_la.sh`.
- [x] **Xác thực toàn diện môi trường Slurm A100:** Kiểm tra trực tiếp trên cluster `10.204.1.52`, tất cả 14 thư viện lõi đều đạt chuẩn `[OK]`.
- [x] **Bổ sung chú thích (Comments & Docstrings) chi tiết 100%:** Cho từng hàm, từng khối logic toán học, từng phép toán reshape/permute và từng tham số trong toàn bộ 3 thư mục baseline và thư mục `data/`.
- [x] **Chuẩn hóa cấu hình Requirements:** Xóa các file requirements con thừa, tập trung vào [requirements.txt](requirements.txt) ở thư mục gốc.
- [x] **Biên soạn Báo cáo Q&A Tạo sinh Dữ liệu & Vật lý CT (Local File):** Phục vụ báo cáo với Giáo sư.
- [x] **Sinh dữ liệu hoàn tất & Nghiệm thu thành công 100% (Job ID `65483`):** Đã tạo đầy đủ cả 2 bộ dữ liệu `120deg` và `90deg` (Train: 1,920 slices, Validation L333: 244 slices, Test L310: 214 slices) tại `/datastore/uittogether3/LuuTru/MinhPD/dataset/limited_angle/`.

---

## 2. Huấn Luyện & Đánh Giá 3 Baseline Unrolling Bậc 1 (AAPM 120-degree LA-CT)
- [x] **Nghiệm thu kết quả Huấn luyện 3 Slurm Job Baseline (120-degree LA-CT):**
  - Xem bảng chi tiết và phân tích tại: [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md)
  - **`LEARN_LongNet`:** Job ID `65486` (`scripts/output/train_longnet_la/log/65486.out`) — ✅ **Hoàn thành 100% (50/50 Epochs)**. Best Val: **PSNR = 33.37 dB**, **SSIM = 0.9090** (Checkpoint: `longnet_la-epoch=45-val_psnr=33.37-val_ssim=0.9090.ckpt`).
  - **`LEARN_Longformer`:** Job ID `65485` / Resume Jobs `65892`, `66652` (`scripts/output/train_longformer_la/log/66652.out`) — ✅ **Hoàn thành 100% (50/50 Epochs)** trên A100 GPU (`DGX-A100`). Đạt kết quả SOTA cao nhất toàn bộ baseline: **PSNR = 34.77 dB, SSIM = 0.9383** tại Epoch 45 (Checkpoint: `longformer_la-epoch=45-val_psnr=34.77-val_ssim=0.9383.ckpt`).
  - **`LEARN_Mamba`:** Job ID `65484` (`scripts/output/train_mamba_la/log/65484.out`) — ⏱️ Chạy 41/50 Epochs (đạt Time Limit 24h). Sử dụng Best Checkpoint: `mamba_la-epoch=17-val_psnr=27.66-val_ssim=0.7373.ckpt` (do sau Epoch 20 xuất hiện mất ổn định gradient/NaN theo đúng lý thuyết).
- [x] **Đánh giá Test Benchmark Độc Lập (Patient L310 - 214 Slices) & Trực Quan Hóa:**
  - **Script Test & Batch Slurm:** `test_longnet_la.sh` (Job `65899`), `test_mamba_la.sh` (Job `65900`), `test_longformer_la.sh` (Job `67223` - Hoàn thành 100%).
  - **Kết quả Test Benchmark LEARN_Longformer (Job `67223`):**
    - *Cung quét chuẩn LA-120°:* **PSNR = 33.10 dB**, **SSIM = 0.9237**, **RMSE = 0.0224** (Cao nhất nhóm baseline).
    - *Stress test LA-90°:* **PSNR = 19.16 dB**, **SSIM = 0.6097**, **RMSE = 0.1055** (SSIM cao nhất baseline).
  - **Script Trực quan hóa & Kết xuất ảnh đối sánh:** [visualize_benchmark.py](visualize_benchmark.py) (Slurm Scripts: [scripts/visualize_benchmark.sh](scripts/visualize_benchmark.sh), [scripts/visualize_longformer_la.sh](scripts/visualize_longformer_la.sh) - Job `67225` ✅ **Hoàn thành 100%**).
  - **Thư mục ảnh PNG đã xuất:** `visualizations/120deg/` và `visualizations/90deg/`.

---

## 3. Khởi Xướng Đột Phá Kiến Trúc Bậc 2: SOLAR (Đầu Tháng 09/2026)
- [x] **Hiện thực hóa 3 Biến thể Kiến trúc Đề xuất SOLAR theo Cấu trúc Baseline Chuẩn Hóa:**
  - **`baselines/SOLAR_LongNet/`:** Tối ưu hóa bậc 2 Newton-CG + Nhánh kép Res-CNN & Dilated Attention (LongNet). Script: `train_solar_longnet_la.py`, `test_solar_longnet_la.py`, `scripts/train_solar_longnet_la.sh`.
  - **`baselines/SOLAR_Longformer/`:** Tối ưu hóa bậc 2 Newton-CG + Nhánh kép Res-CNN & Sliding-Chunks Attention (Longformer). Script: `train_solar_longformer_la.py`, `test_solar_longformer_la.py`, `scripts/train_solar_longformer_la.sh`.
  - **`baselines/SOLAR_Mamba/`:** Tối ưu hóa bậc 2 Newton-CG + Nhánh kép Res-CNN & Selective SSM (Mamba). Script: `train_solar_mamba_la.py`, `test_solar_mamba_la.py`, `scripts/train_solar_mamba_la.sh`.
- [x] **Hoàn thành Huấn luyện 50 Epochs cho SOLAR_Longformer (Job `67820`) & SOLAR_Mamba (Job `67821`):**
  - Cả hai mô hình đã hoàn tất trọn vẹn 50/50 Epochs trên cụm `DGX-A100` không lỗi số học:
    - `SOLAR_Longformer`: Best Val PSNR = 34.18 dB, Val SSIM = 0.9165 (Checkpoint: `solar_longformer_la-epoch=45-val_psnr=34.18-val_ssim=0.9165.ckpt`).
    - `SOLAR_Mamba`: Best Val PSNR = 34.00 dB, Val SSIM = 0.9089 (Checkpoint: `solar_mamba_la-epoch=45-val_psnr=34.00-val_ssim=0.9089.ckpt`).
- [x] **Nghiệm thu Test Benchmark 50 Epochs của SOLAR trên Patient L310 (Jobs `68551`, `68552`, `68553`):**
  - **`SOLAR_Longformer` (50 ep - Job `68551`):**
    - LA-120°: **PSNR = 32.95 dB**, **SSIM = 0.9155**, **RMSE = 0.0228**
    - LA-90°: **PSNR = 28.05 dB**, **SSIM = 0.8774**, **RMSE = 0.0412** (🏆 **SOTA Toàn diện ở góc 90°: +8.89 dB PSNR và +0.2677 SSIM** so với baseline)
  - **`SOLAR_Mamba` (50 ep - Job `68552`):**
    - LA-120°: **PSNR = 31.90 dB**, **SSIM = 0.9114**, **RMSE = 0.0268**
    - LA-90°: **PSNR = 27.53 dB**, **SSIM = 0.8760**, **RMSE = 0.0447** (🚀 **+8.77 dB PSNR và +0.4468 SSIM (+104.1%)** so với LEARN_Mamba)
  - **Visualization:** Job `68553` hoàn thành 100% xuất ảnh panel đối sánh chất lượng cao.

---

## 4. Dữ Liệu Đa Miền LIDC-IDRI & Bài Báo SOICT 2026
- [x] **Tải dữ liệu LIDC-IDRI từ TCIA REST API:** 1,975 lát cắt DICOM, phân chia `splits_summary.json`.
- [x] **Sinh dữ liệu LIDC-IDRI (Job Slurm ID `67943`):** Sinh đầy đủ sinogram và FBP cho cả 120° và 90°.
- [x] **Khởi tạo Bản thảo Bài báo Hội nghị SOICT 2026 (`papers/soict2026/`):**
  - Hoàn tất bản thảo `main.tex`, 2 Bảng thực nghiệm đối sánh (Table 1 cho 120°, Table 2 cho 90°), Fig 1, Fig 2, Fig 3 và trích xuất Attention Maps 14 tầng (Job `67967`).
  - **Chốt nguyên tắc:** SOICT 2026 chỉ so sánh đối chứng baseline, tuyệt đối không đưa SOLAR vào.

---

## 5. Nghiệm Thu 3 Baseline Mới (LEARN, RegFormer, DuDoTrans) - Ngày 14/09/2026
- [x] **Huấn luyện hoàn thành 100% 50/50 Epochs:**
  - `LEARN` (Job `71393`): Best Val PSNR = 36.2681 dB.
  - `RegFormer` (Job `71394`): Best Val PSNR = 36.0533 dB.
  - `DuDoTrans` (Job `71395`): Best Val PSNR = 25.7300 dB.
- [x] **Test Benchmark trên Patient L310 (Jobs `71615`, `71616`, `71617`):**
  - `RegFormer` đạt SOTA baseline ở góc 90°: **PSNR = 28.20 dB**, **SSIM = 0.8814** (động lực để phát triển SOLAR-RegFormer).
  - Đã cập nhật đầy đủ số liệu vào `reports/sep-14-2026/benchmark_results.csv` và `EXPERIMENT_RESULTS.md`.
