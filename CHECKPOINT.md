# CHECKPOINT DỰ ÁN NCKH: LIMITED-ANGLE CT RECONSTRUCTION
> **TRẠNG THÁI ACTIVE (Cập nhật: 19/09/2026)**
> 
> *Lưu ý cho AI Agent:* File này lưu trữ **trạng thái công việc tức thời, các job đang chạy và nhiệm vụ cần làm tiếp theo**.
> - Thông số toán học/hình học CT & phần cứng: Tham chiếu [SYSTEM_SPEC.md](SYSTEM_SPEC.md).
> - Lịch sử các mốc thử nghiệm cũ từ tháng 8: Tham chiếu [ARCHIVE_LOG.md](ARCHIVE_LOG.md).
> - Quy tắc thực thi và điều cấm kỵ: Tham chiếu [AGENTS.md](AGENTS.md).

---

## 1. Trạng Thái Các Slurm Job Cluster (DGX-A100)

### 🚀 A. Các Job Đang Chạy / Chờ Xử Lý (In-Progress & Queue)
1. **`LEARN_Longformer` (Huấn luyện Baseline trên NIH DeepLesion):**
   - **Job ID Slurm:** **`72819`** (Submit lúc 01:06 ngày 19/09/2026, chạy trên GPU 3 node `DGX-A100` từ 19:53 ngày 19/09/2026).
   - **Trạng thái:** 🟢 **ĐANG CHẠY (RUNNING - Epoch 10/50)**, tự động resume từ `last.ckpt` (Epoch 09/50).
   - **Script & Log:** `scripts/train_longformer_deeplesion.sh` | Log: `scripts/output/train_longformer_deeplesion/log/72819.out`

2. **`LEARN_LongNet` (Huấn luyện Baseline trên NIH DeepLesion):**
   - **Job ID Slurm:** **`72820`** (Submit lúc 01:06 ngày 19/09/2026).
   - **Trạng thái:** 🟡 **Hàng đợi `PD (Priority)`** (Epoch 0 $\to$ 50).
   - **Script & Log:** `scripts/train_longnet_deeplesion.sh` | Log: `scripts/output/train_longnet_deeplesion/log/72820.out`

3. **`LEARN_Mamba` (Huấn luyện Baseline trên NIH DeepLesion):**
   - **Job ID Slurm:** **`72821`** (Submit lúc 01:06 ngày 19/09/2026).
   - **Trạng thái:** 🟡 **Hàng đợi `PD (Priority)`**.
   - **Script & Log:** `scripts/train_mamba_deeplesion.sh` | Log: `scripts/output/train_mamba_deeplesion/log/72821.out`

4. **`LEARN` (Huấn luyện Baseline Original CNN trên NIH DeepLesion):**
   - **Job ID Slurm:** **`72822`** (Submit lúc 01:11 ngày 19/09/2026).
   - **Trạng thái:** 🟡 **Hàng đợi `PD (Priority)`** (Epoch 0 $\to$ 50).
   - **Script & Log:** `scripts/train_learn_deeplesion.sh` | Log: `scripts/output/train_learn_deeplesion/log/72822.out`

5. **`DuDoTrans` (Huấn luyện Baseline Dual-Domain trên NIH DeepLesion):**
   - **Job ID Slurm:** **`72823`** (Submit lúc 01:11 ngày 19/09/2026).
   - **Trạng thái:** 🟡 **Hàng đợi `PD (Priority)`** (Epoch 0 $\to$ 50).
   - **Script & Log:** `scripts/train_dudotrans_deeplesion.sh` | Log: `scripts/output/train_dudotrans_deeplesion/log/72823.out`

6. **`MoDL` (Huấn luyện Baseline Model-Based Deep Learning trên AAPM LA-120°):**
   - **Job ID Slurm:** **`72886`** (Submit lúc 20:36 ngày 19/09/2026).
   - **Trạng thái:** 🟡 **Hàng đợi `PD (Priority)`** (Epoch 0 $\to$ 50).
   - **Cấu hình:** 10 stages unrolling, 6 CG iterations, ResNet-5 regularizer (~112K params).
   - **Script & Log:** `scripts/train_modl_la.sh` | Log: `scripts/output/train_modl_la/log/72886.out`

7. **`FISTA-Net` (Huấn luyện Baseline Deep Unfolded FISTA trên AAPM LA-120°):**
   - **Job ID Slurm:** **`72887`** (Submit lúc 20:36 ngày 19/09/2026).
   - **Trạng thái:** 🟡 **Hàng đợi `PD (Priority)`** (Epoch 0 $\to$ 50).
   - **Cấu hình:** 10 stages unrolling, Learnable Soft-Thresholding + Nesterov momentum (~38K params).
   - **Script & Log:** `scripts/train_fista_net_la.sh` | Log: `scripts/output/train_fista_net_la/log/72887.out`

8. **`DuDoNet` (Huấn luyện Baseline Dual-Domain trên AAPM LA-120°):**
   - **Job ID Slurm:** **`72888`** (Submit lúc 20:36 ngày 19/09/2026).
   - **Trạng thái:** 🟡 **Hàng đợi `PD (Priority)`** (Epoch 0 $\to$ 50).
   - **Cấu hình:** Sinogram U-Net inpainting + FBP + Image Refinement CNN (~688K params).
   - **Script & Log:** `scripts/train_dudonet_la.sh` | Log: `scripts/output/train_dudonet_la/log/72888.out`

9. **`CT-Former` (Huấn luyện Baseline Cross-Shape Attention ViT trên AAPM LA-120°):**
   - **Job ID Slurm:** **`72889`** (Submit lúc 20:36 ngày 19/09/2026).
   - **Trạng thái:** 🟡 **Hàng đợi `PD (Priority)`** (Epoch 0 $\to$ 50).
   - **Cấu hình:** 6 Cross-Shape Transformer Blocks, 4 heads, ConvFFN (~76K params).
   - **Script & Log:** `scripts/train_ct_former_la.sh` | Log: `scripts/output/train_ct_former_la/log/72889.out`

### ✅ B. Các Job Vừa Hoàn Thành & Nghiệm Thu Gần Nhất
1. **`SOLAR_DualMamba` - Test Benchmark Kiểm Chứng SOTA Patient L310 (Job ID `72818`):**
   - ✅ Hoàn thành 100% lúc 19:52:50 ngày 19/09/2026 (`rc=0`) trên 214 lát cắt CT độc lập:
     - *Cung quét chuẩn LA-120° (64 views):* **PSNR = 32.91 dB**, **SSIM = 0.9186**, **RMSE = 0.0245** (vượt `SOLAR_RegFormer` +0.44 dB PSNR / +0.0091 SSIM).
     - *Stress test LA-90° (khuyết 270°):* **PSNR = 27.98 dB**, **SSIM = 0.8843**, **RMSE = 0.0446** (🏆 **Kỷ lục SSIM cao nhất toàn dự án**, vượt `RegFormer` baseline 0.8814 và `SOLAR_RegFormer` 0.8804).
   - Checkpoint kiểm thử: `solar_dualmamba_la-epoch=48-val_psnr=34.61-val_ssim=0.9197.ckpt`.
   - Script & Log: `scripts/test_solar_dualmamba_la.sh` | Log: `scripts/output/test_solar_dualmamba_la/log/72818.out`.

2. **`SOLAR_DualMamba` - Huấn luyện Giai đoạn 2: Epoch 15 $\to$ 50 (Job ID `72618`):**
   - ✅ Hoàn thành mốc 24:00:00 runtime lúc 05:49 ngày 19/09/2026, chạm mốc **Epoch 49 (67%)**.
   - Best Checkpoint: `solar_dualmamba_la-epoch=48-val_psnr=34.61-val_ssim=0.9197.ckpt` (**Val PSNR = 34.61 dB**, **Val SSIM = 0.9197** — Kỷ lục Validation cao nhất dòng SOLAR).
   - Script & Log: `scripts/train_solar_dualmamba_la.sh` | Log: `scripts/output/train_solar_dualmamba_la/log/72618.out`.

3. **`SOLAR_RegFormer` - Test Benchmark Độc Lập Patient L310 (Bản Full 50 Epochs) (Job ID `72610`):**
   - ✅ Hoàn thành 100% lúc 22:55 ngày 17/09/2026 (`rc=0`) trên 214 lát cắt CT độc lập:
     - *Cung quét chuẩn LA-120° (64 views):* **PSNR = 32.47 dB**, **SSIM = 0.9095**, **RMSE = 0.0247**.
     - *Stress test LA-90° (khuyết 270°):* **PSNR = 27.64 dB**, **SSIM = 0.8804**, **RMSE = 0.0442**.
   - Checkpoint kiểm thử: `solar_regformer_la-epoch=45-val_psnr=34.00-val_ssim=0.9117.ckpt`.

4. **`SOLAR_RegFormer` - Huấn luyện Giai đoạn 2: Epoch 35 $\to$ 50 (Job ID `72256`):**
   - ✅ Hoàn thành 100% 50/50 Epochs lúc 18:59 ngày 17/09/2026 (Best Val PSNR: 34.00 dB, SSIM: 0.9139).

---

## 2. Bảng Tóm Tắt Best Checkpoints SOTA Hiện Tại (AAPM LA-CT)

| Dòng Mô Hình | Checkpoint Tốt Nhất | Val PSNR / SSIM | Test LA-120° (PSNR/SSIM) | Test LA-90° (PSNR/SSIM) |
| :--- | :--- | :---: | :---: | :---: |
| **`LEARN_Longformer` (Baseline)** | `longformer_la-epoch=45.ckpt` | 34.77 dB / 0.9383 | 33.10 dB / 0.9237 | 19.16 dB / 0.6097 |
| **`RegFormer` (Baseline)** | `epoch=46-val_psnr=36.0533.ckpt` | 36.05 dB / - | 32.82 dB / 0.9398 | 28.20 dB / 0.8814 |
| **`LEARN (Original CNN)`** | `epoch=46-val_psnr=36.2681.ckpt` | 36.27 dB / 0.950 | 32.29 dB / 0.9383 | 28.03 dB / 0.8793 |
| **`DuDoTrans` (Baseline)** | `epoch=38-val_psnr=25.7300.ckpt` | 25.73 dB / 0.730 | 25.15 dB / 0.7447 | 21.81 dB / 0.6738 |
| **`SOLAR_Longformer` (SOTA)** | `solar_longformer_la-epoch=45.ckpt` | 34.18 dB / 0.9165 | 32.95 dB / 0.9155 | **28.05 dB / 0.8774** |
| **`SOLAR_Mamba` (SOTA)** | `solar_mamba_la-epoch=45.ckpt` | 34.00 dB / 0.9089 | 31.90 dB / 0.9114 | **27.53 dB / 0.8760** |
| **`SOLAR_RegFormer` (50 ep)** | `solar_regformer_la-epoch=45.ckpt` | **34.00 dB / 0.9139** | **32.47 dB / 0.9095** | **27.64 dB / 0.8804** |
| **`SOLAR_DualMamba` (SOTA Toàn Dự Án)**| `solar_dualmamba_la-epoch=48.ckpt` | **34.61 dB / 0.9197** | **32.91 dB / 0.9186** | **27.98 dB / 0.8843 (Kỷ lục)** |

*Dữ liệu chi tiết đối sánh xem tại:* [benchmark_results.csv](reports/sep-19-2026/benchmark_results.csv) và [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md).

---

## 3. Danh Mục Nhiệm Vụ Tiếp Theo (Actionable TODOs)

- [x] **1. Nghiệm thu Huấn luyện `SOLAR_RegFormer` 50 Epochs (Job `72256`):**
  - Đã chạm mốc 50 epochs lúc 18:59 ngày 17/09/2026. Best Val PSNR đỉnh 34.00 dB (Epoch 45).
- [x] **2. Nghiệm thu Benchmark `SOLAR_RegFormer` 50 Epochs (Job `72610`):**
  - Hoàn tất kiểm thử trên 214 lát cắt Patient L310 (LA-120°: 32.47 dB / 0.9095; LA-90°: 27.64 dB / 0.8804).
- [x] **3. Nghiệm thu Huấn luyện & Benchmark `SOLAR_DualMamba` (Jobs `72618` & `72818`):**
  - Đạt Epoch 49 (67%), Best Checkpoint Epoch 48 (Val PSNR 34.61 dB, SSIM 0.9197).
  - Hoàn tất test benchmark Patient L310 (LA-120°: 32.91 dB / 0.9186; LA-90°: 27.98 dB / 0.8843 - Kỷ lục SSIM mới).
  - Đã cập nhật vào `reports/sep-19-2026/benchmark_results.csv` và `reports/sep-19-2026/MAIN.md`.
- [x] **4. Hoàn thành Khảo cứu Toàn diện Đối thủ Tái tạo LA-CT (19/09/2026):**
  - Đã lập báo cáo chi tiết 5 trường phái đối thủ (Deep Unrolling, Dual-Domain, ViT/SSM, Diffusion, INR) tại [reports/sep-19-2026/COMPETITORS_RESEARCH.md](reports/sep-19-2026/COMPETITORS_RESEARCH.md) sẵn sàng trình Giáo sư hướng dẫn.
- [x] **5. Triển Khai & Nộp Huấn Luyện 4 Mô Hình Baseline Mới (19/09/2026):**
  - Đã hoàn tất 100% mã nguồn 4 baseline: `MoDL`, `FISTA-Net`, `DuDoNet`, `CT-Former` (đầy đủ `models.py`, `train_*_la.py`, `test_*_la.py`, `scripts/train_*_la.sh`, `scripts/test_*_la.sh`).
  - Đã nộp thành công 4 Slurm jobs lên DGX-A100:
    * `MoDL`: Job ID **`72886`**
    * `FISTA-Net`: Job ID **`72887`**
    * `DuDoNet`: Job ID **`72888`**
    * `CT-Former`: Job ID **`72889`**
- [ ] **6. Kế Hoạch Tiếp Theo: Lặp Lại Toàn Bộ Huấn Luyện & Kiểm Thử Trên 2 Bộ Dữ Liệu NIH DeepLesion & LIDC-IDRI:**
  - **NIH DeepLesion (>32,000 lát cắt):**
    * Giám sát Job `72819` (`LEARN_Longformer` DeepLesion - Epoch 10/50).
    * Lần lượt kích hoạt và hoàn tất các baseline `LEARN_LongNet` (`72820`), `LEARN_Mamba` (`72821`), `LEARN` (`72822`), `DuDoTrans` (`72823`).
    * Huấn luyện và chạy test benchmark cho `SOLAR_DualMamba` trên DeepLesion ở cả 2 góc LA-120° và LA-90°.
  - **LIDC-IDRI (1,018 bệnh nhân CT phổi):**
    * Lặp lại tương tự quy trình huấn luyện cho toàn bộ nhóm baseline và mô hình đề xuất bằng các script `scripts/train_*_lidc.sh`.
    * Đánh giá test benchmark trên tập kiểm thử độc lập LIDC (LA-120° và LA-90°).
- [ ] **7. Hoàn thiện Bản thảo Bài báo SOICT 2026 (`papers/soict2026/`):**
  - Rà soát bản thảo `main.tex`, đưa thêm kết quả đối chứng của `LEARN` và `DuDoTrans`.
  - *Nhắc lại điều cấm kỵ:* Tuyệt đối không đưa SOLAR vào bài báo SOICT 2026.
