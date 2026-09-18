# CHECKPOINT DỰ ÁN NCKH: LIMITED-ANGLE CT RECONSTRUCTION
> **TRẠNG THÁI ACTIVE (Cập nhật: 17/09/2026)**
> 
> *Lưu ý cho AI Agent:* File này lưu trữ **trạng thái công việc tức thời, các job đang chạy và nhiệm vụ cần làm tiếp theo**.
> - Thông số toán học/hình học CT & phần cứng: Tham chiếu [SYSTEM_SPEC.md](SYSTEM_SPEC.md).
> - Lịch sử các mốc thử nghiệm cũ từ tháng 8: Tham chiếu [ARCHIVE_LOG.md](ARCHIVE_LOG.md).
> - Quy tắc thực thi và điều cấm kỵ: Tham chiếu [AGENTS.md](AGENTS.md).

---

## 1. Trạng Thái Các Slurm Job Cluster (DGX-A100)

### 🚀 A. Các Job Đang Chạy / Chờ Xử Lý (In-Progress & Queue)
1. **`SOLAR_RegFormer` (Test Benchmark Độc Lập 50 Epochs Patient L310):**
   - **Job ID Slurm:** **`72610`** (Submit lúc 22:23 ngày 17/09/2026 trên node `DGX-A100`)
   - **Trạng thái:** Đang trong hàng đợi ưu tiên `PD (Priority)`, chờ cấp phát tài nguyên MPS GPU A100.
   - **Script & Log:** `scripts/test_solar_regformer_la.sh` | Log: `scripts/output/test_solar_regformer_la/log/72610.out`
   - **Checkpoint kiểm thử:** `solar_regformer_la-epoch=45-val_psnr=34.00-val_ssim=0.9117.ckpt`
   - **Mục tiêu:** Đánh giá hiệu năng SOTA của bản full 50 epochs trên cấu hình chuẩn LA-120° và stress-test góc khuyết cực đoan LA-90°.

2. **`SOLAR_DualMamba` (Huấn luyện Resume Giai đoạn 2: Epoch 15 $\to$ 50):**
   - **Job ID Slurm:** **`72618`** (Submit lúc 22:43 ngày 17/09/2026 trên node `DGX-A100`)
   - **Trạng thái:** Đang trong hàng đợi `PD (Priority)`, tự động nối tiếp từ checkpoint `last.ckpt` (Epoch 14/15).
   - **Script & Log:** `scripts/train_solar_dualmamba_la.sh` | Log: `scripts/output/train_solar_dualmamba_la/log/72618.out`
   - **Tiền nhiệm (Giai đoạn 1):** Job `72275` (Hoàn thành 24h, chạm TimeLimit lúc 22:30:48 ngày 17/09; Best Val PSNR: **32.60 dB**).
   - **Mục tiêu:** Tiếp tục huấn luyện hội tụ đến 50 epochs.

### ✅ B. Các Job Vừa Hoàn Thành & Nghiệm Thu Gần Nhất
1. **`SOLAR_RegFormer` - Huấn luyện Giai đoạn 2: Epoch 35 $\to$ 50 (Job ID `72256`):**
   - ✅ Hoàn thành 100% 50/50 Epochs lúc 18:59 ngày 17/09/2026.
   - Best Checkpoint: `solar_regformer_la-epoch=45-val_psnr=34.00-val_ssim=0.9117.ckpt` (Val PSNR: **34.00 dB**, SSIM: **0.9117**) và `solar_regformer_la-epoch=46-val_psnr=33.96-val_ssim=0.9139.ckpt` (Val SSIM: **0.9139**).
   - Tăng trưởng vượt bậc: **+0.36 dB PSNR** và **+0.0036 SSIM** so với mốc 35 epochs của Giai đoạn 1.
2. **`SOLAR_RegFormer` - Test Benchmark Độc Lập Patient L310 (Mốc 35 ep) (Job ID `72251`):**
   - ✅ Hoàn thành 100% ngày 16/09/2026 trên 214 lát cắt kiểm thử độc lập:
     - *Cung quét chuẩn LA-120° (64 views):* **PSNR = 32.11 dB**, **SSIM = 0.9044**, **RMSE = 0.0259**
     - *Stress test LA-90° (góc khuyết 270°):* **PSNR = 27.68 dB**, **SSIM = 0.8776**, **RMSE = 0.0440** (🏆 **+8.52 dB PSNR và +0.2679 SSIM** so với baseline LEARN_Longformer).

---

## 2. Bảng Tóm Tắt Best Checkpoints SOTA Hiện Tại (AAPM LA-CT)

| Dòng Mô Hình | Checkpoint Tốt Nhất | Val PSNR / SSIM | Test LA-120° (PSNR/SSIM) | Test LA-90° (PSNR/SSIM) |
| :--- | :--- | :---: | :---: | :---: |
| **`LEARN_Longformer` (Baseline)** | `longformer_la-epoch=45.ckpt` | 34.77 dB / 0.9383 | 33.10 dB / 0.9237 | 19.16 dB / 0.6097 |
| **`RegFormer` (Baseline)** | `epoch=46-val_psnr=36.0533.ckpt` | 36.05 dB / - | 32.82 dB / 0.9398 | 28.20 dB / 0.8814 |
| **`SOLAR_Longformer` (SOTA)** | `solar_longformer_la-epoch=45.ckpt` | 34.18 dB / 0.9165 | 32.95 dB / 0.9155 | **28.05 dB / 0.8774** |
| **`SOLAR_Mamba` (SOTA)** | `solar_mamba_la-epoch=45.ckpt` | 34.00 dB / 0.9089 | 31.90 dB / 0.9114 | **27.53 dB / 0.8760** |
| **`SOLAR_RegFormer` (50 ep)** | `solar_regformer_la-epoch=45.ckpt` | **34.00 dB / 0.9139** | Đang test (Job `72610`) | Đang test (Job `72610`) |
| **`SOLAR_DualMamba` (Mới)** | Đang train (Job `72275`) | Đang cập nhật | Đang cập nhật | Đang cập nhật |

*Dữ liệu chi tiết đối sánh xem tại:* [benchmark_results.csv](reports/sep-14-2026/benchmark_results.csv) và [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md).

---

## 3. Danh Mục Nhiệm Vụ Tiếp Theo (Actionable TODOs)

- [x] **1. Nghiệm thu Huấn luyện `SOLAR_RegFormer` 50 Epochs (Job `72256`):**
  - Đã chạm mốc 50 epochs lúc 18:59 ngày 17/09/2026. Best Val PSNR đỉnh 34.00 dB (Epoch 45).
- [ ] **2. Resume Huấn luyện `SOLAR_DualMamba` Giai đoạn 2 (Epoch 15 $\to$ 50):**
  - Submit lại qua `scripts/train_solar_dualmamba_la.sh` để tự động nối tiếp từ `last.ckpt` (Epoch 14/15) sau khi Job 72275 chạm TimeLimit 24h.
- [ ] **3. Theo dõi & Nghiệm thu Benchmark `SOLAR_RegFormer` 50 Epochs (Job `72610`):**
  - Kiểm tra log `scripts/output/test_solar_regformer_la/log/72610.out` khi kiểm thử LA-120° và LA-90° hoàn tất.
  - Cập nhật số liệu vào `benchmark_results.csv` và `EXPERIMENT_RESULTS.md`.
- [ ] **4. Bài báo SOICT 2026 (`papers/soict2026/`):**
  - Rà soát bản thảo `main.tex`, kiểm tra số trang LNCS (12-15 trang) và biên dịch TeX.
  - *Nhắc lại điều cấm kỵ:* Tuyệt đối không đưa SOLAR vào bài báo SOICT 2026.
- [ ] **5. Resume các Job Đa Dataset (DeepLesion & LIDC-IDRI):**
  - Hạ `REQUIRED_VRAM` từ 20000MB xuống 15000MB trong các script `train_*_lidc.sh` và `train_*_deeplesion.sh` để tránh bị hủy job oan.
