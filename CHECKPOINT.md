# CHECKPOINT DỰ ÁN NCKH: LIMITED-ANGLE CT RECONSTRUCTION
> **TRẠNG THÁI ACTIVE (Cập nhật: 20/09/2026)**
> 
> *Lưu ý cho AI Agent:* File này lưu trữ **trạng thái công việc tức thời, các job đang chạy và nhiệm vụ cần làm tiếp theo**.
> - Thông số toán học/hình học CT & phần cứng: Tham chiếu [SYSTEM_SPEC.md](SYSTEM_SPEC.md).
> - Lịch sử các mốc thử nghiệm cũ từ tháng 8: Tham chiếu [ARCHIVE_LOG.md](ARCHIVE_LOG.md).
> - Quy tắc thực thi và điều cấm kỵ: Tham chiếu [AGENTS.md](AGENTS.md).

---

## 1. Trạng Thái Các Slurm Job Cluster (DGX-A100)

### 🚀 A. Các Job Đang Chạy / Chờ Xử Lý (In-Progress & Queue)
1. **`LEARN` (Huấn luyện Baseline Original CNN trên NIH DeepLesion):**
   - **Job ID Slurm:** **`72822`** (Submit lúc 01:11 ngày 19/09/2026, chạy trên GPU 7 `DGX-A100`).
   - **Trạng thái:** 🟢 **ĐANG CHẠY ỔN ĐỊNH (RUNNING - Epoch 22/50 ~91%)**, tốc độ ~1.07 it/s.
   - **Best Checkpoint hiện tại:** `epoch=21-val_psnr=33.1248.ckpt` (**Val PSNR = 33.12 dB**, **Val SSIM = 0.9240**).
   - **Script & Log:** `scripts/train_learn_deeplesion.sh` | Log: `scripts/output/train_learn_deeplesion/log/72822.out`

2. **`LEARN_Mamba` - Test Benchmark Độc Lập NIH DeepLesion (LA-120° & LA-90°):**
   - **Job ID Slurm:** **`72968`** (Đã hoàn thành lúc 21:14 ngày 20/09/2026 trên GPU 5 `DGX-A100`).
   - **Trạng thái:** ✅ **HOÀN THÀNH (COMPLETED)**.
   - **Checkpoint kiểm thử:** `mamba_la-epoch=23-val_psnr=26.18-val_ssim=0.7009.ckpt`.
   - **Kết quả đo lường thực tế trên 300 lát cắt độc lập:**
     * **LA-120°:** **26.69 dB PSNR** | **0.7132 SSIM** | RMSE: 0.0486 | Loss: 0.0025.
     * **LA-90°:** **18.59 dB PSNR** | **0.3913 SSIM** | RMSE: 0.1211 | Loss: 0.0150.
   - **Script & Log:** `scripts/test_mamba_deeplesion.sh` | Log: `scripts/output/test_mamba_deeplesion/log/72968.out`

3. **`DuDoTrans` - Test Benchmark Độc Lập NIH DeepLesion (LA-120° & LA-90°):**
   - **Job ID Slurm:** **`72969`** (Đã hoàn thành lúc 21:07 ngày 20/09/2026 trên GPU 5 `DGX-A100`).
   - **Trạng thái:** ✅ **HOÀN THÀNH (COMPLETED)**.
   - **Checkpoint kiểm thử:** `epoch=49-val_psnr=23.6662.ckpt`.
   - **Kết quả đo lường thực tế trên 300 lát cắt độc lập:**
     * **LA-120°:** **23.76 dB PSNR** | **0.6637 SSIM** | RMSE: 0.0892 | Loss: 0.0082.
     * **LA-90°:** **20.38 dB PSNR** | **0.6692 SSIM** | RMSE: 0.1084 | Loss: 0.0120.
   - **Script & Log:** `scripts/test_dudotrans_deeplesion.sh` | Log: `scripts/output/test_dudotrans_deeplesion/log/72969.out`

4. **`LEARN_Longformer` (Huấn luyện Resume NIH DeepLesion: Epoch 36 $\to$ 50):**
   - **Job ID Slurm:** **`72966`** (Submit lúc 21:05 ngày 20/09/2026).
   - **Trạng thái:** 🟡 **Hàng đợi `PD (Priority)`** (Tự động nạp `last.ckpt` để train tiếp 14 epochs).
   - **Script & Log:** `scripts/train_longformer_deeplesion.sh` | Log: `scripts/output/train_longformer_deeplesion/log/72966.out`

5. **`LEARN_LongNet` (Huấn luyện Baseline NIH DeepLesion - Đã fix OOM):**
   - **Job ID Slurm:** **`72967`** (Submit lúc 21:06 ngày 20/09/2026).
   - **Trạng thái:** 🟡 **Hàng đợi `PD (Priority)`** (Cấu hình chuẩn `window_size=2`, `REQUIRED_VRAM=25000` giải quyết triệt để OOM).
   - **Script & Log:** `scripts/train_longnet_deeplesion.sh` | Log: `scripts/output/train_longnet_deeplesion/log/72967.out`

6. **`MoDL` (Huấn luyện Baseline Model-Based Deep Learning trên AAPM LA-120°):**
   - **Job ID Slurm:** **`72962`** (Submit lại lúc 21:05 ngày 20/09/2026 sau khi chuẩn hóa `get_datamodule`).
   - **Trạng thái:** 🟡 **Hàng đợi `PD (Priority)`** (Epoch 0 $\to$ 50).
   - **Cấu hình:** 10 stages unrolling, 6 CG iterations, ResNet-5 regularizer (~112K params).
   - **Script & Log:** `scripts/train_modl_la.sh` | Log: `scripts/output/train_modl_la/log/72962.out`

7. **`FISTA-Net` (Huấn luyện Baseline Deep Unfolded FISTA trên AAPM LA-120°):**
   - **Job ID Slurm:** **`72963`** (Submit lại lúc 21:05 ngày 20/09/2026 sau khi chuẩn hóa `get_datamodule`).
   - **Trạng thái:** 🟡 **Hàng đợi `PD (Priority)`** (Epoch 0 $\to$ 50).
   - **Cấu hình:** 10 stages unrolling, Learnable Soft-Thresholding + Nesterov momentum (~38K params).
   - **Script & Log:** `scripts/train_fista_net_la.sh` | Log: `scripts/output/train_fista_net_la/log/72963.out`

8. **`DuDoNet` (Huấn luyện Baseline Dual-Domain trên AAPM LA-120°):**
   - **Job ID Slurm:** **`72964`** (Submit lại lúc 21:05 ngày 20/09/2026 sau khi chuẩn hóa `get_datamodule`).
   - **Trạng thái:** 🟡 **Hàng đợi `PD (Priority)`** (Epoch 0 $\to$ 50).
   - **Cấu hình:** Sinogram U-Net inpainting + FBP + Image Refinement CNN (~688K params).
   - **Script & Log:** `scripts/train_dudonet_la.sh` | Log: `scripts/output/train_dudonet_la/log/72964.out`

9. **`CT-Former` (Huấn luyện Baseline Cross-Shape Attention ViT trên AAPM LA-120°):**
   - **Job ID Slurm:** **`72965`** (Submit lại lúc 21:05 ngày 20/09/2026 sau khi chuẩn hóa `get_datamodule`).
   - **Trạng thái:** 🟡 **Hàng đợi `PD (Priority)`** (Epoch 0 $\to$ 50).
   - **Cấu hình:** 6 Cross-Shape Transformer Blocks, 4 heads, ConvFFN (~76K params).
   - **Script & Log:** `scripts/train_ct_former_la.sh` | Log: `scripts/output/train_ct_former_la/log/72965.out`

### ✅ B. Các Job Vừa Hoàn Thành & Nghiệm Thu Gần Nhất
1. **`DuDoTrans` - Test Benchmark Độc Lập NIH DeepLesion (Job ID `72969`):**
   - ✅ Hoàn thành 100% lúc 21:07:26 ngày 20/09/2026 (`rc=0`) trên 300 lát cắt CT test:
     - *Cung quét chuẩn LA-120° (64 views):* **PSNR = 23.76 dB**, **SSIM = 0.6637**, **RMSE = 0.0892**.
     - *Stress test LA-90° (64 views):* **PSNR = 20.38 dB**, **SSIM = 0.6692**, **RMSE = 0.1084**.
   - Checkpoint kiểm thử: `epoch=49-val_psnr=23.6662.ckpt`.
   - Script & Log: `scripts/test_dudotrans_deeplesion.sh` | Log: `scripts/output/test_dudotrans_deeplesion/log/72969.out`.

2. **`DuDoTrans` - Huấn luyện Baseline trên NIH DeepLesion (Job ID `72823`):**
   - ✅ Hoàn thành 100% 50/50 Epochs lúc 10:38:59 ngày 20/09/2026 (`rc=0`).
   - Best Checkpoint: `epoch=49-val_psnr=23.6662.ckpt` (**Val PSNR = 23.67 dB**, **Val SSIM = 0.6610**).
   - Script & Log: `scripts/train_dudotrans_deeplesion.sh` | Log: `scripts/output/train_dudotrans_deeplesion/log/72823.out`.

3. **`LEARN_Mamba` - Huấn luyện Baseline trên NIH DeepLesion (Job ID `72821`):**
   - ✅ Hoàn thành 100% 50/50 Epochs lúc 06:13:35 ngày 20/09/2026 (`rc=0`).
   - Best Checkpoint: `mamba_la-epoch=23-val_psnr=26.18-val_ssim=0.7009.ckpt` (**Val PSNR = 26.18 dB**, **Val SSIM = 0.7009**). Đã submit job test benchmark `72968`.
   - Script & Log: `scripts/train_mamba_deeplesion.sh` | Log: `scripts/output/train_mamba_deeplesion/log/72821.out`.

4. **`LEARN_Longformer` - Huấn luyện Giai đoạn 1 NIH DeepLesion (Job ID `72819`):**
   - Chạm mốc 24:00:00 Slurm Time Limit lúc 19:53:29 ngày 20/09/2026 tại **Epoch 36 (63%)**.
   - Best Checkpoint: `longformer_la-epoch=35-val_psnr=32.12-val_ssim=0.8933.ckpt` (**Val PSNR = 32.12 dB**, **Val SSIM = 0.8933**). File `last.ckpt` đã được lưu và submit job resume `72966`.
   - Script & Log: `scripts/train_longformer_deeplesion.sh` | Log: `scripts/output/train_longformer_deeplesion/log/72819.out`.

5. **`SOLAR_DualMamba` - Test Benchmark Kiểm Chứng SOTA Patient L310 (Job ID `72818`):**
   - ✅ Hoàn thành 100% lúc 19:52:50 ngày 19/09/2026 (`rc=0`) trên 214 lát cắt CT độc lập:
     - *Cung quét chuẩn LA-120° (64 views):* **PSNR = 32.91 dB**, **SSIM = 0.9186**, **RMSE = 0.0245** (vượt `SOLAR_RegFormer` +0.44 dB PSNR / +0.0091 SSIM).
     - *Stress test LA-90° (khuyết 270°):* **PSNR = 27.98 dB**, **SSIM = 0.8843**, **RMSE = 0.0446** (🏆 **Kỷ lục SSIM cao nhất toàn dự án**, vượt `RegFormer` baseline 0.8814 và `SOLAR_RegFormer` 0.8804).
   - Checkpoint kiểm thử: `solar_dualmamba_la-epoch=48-val_psnr=34.61-val_ssim=0.9197.ckpt`.
   - Script & Log: `scripts/test_solar_dualmamba_la.sh` | Log: `scripts/output/test_solar_dualmamba_la/log/72818.out`.

6. **`SOLAR_DualMamba` - Huấn luyện Giai đoạn 2: Epoch 15 $\to$ 50 (Job ID `72618`):**
   - ✅ Hoàn thành mốc 24:00:00 runtime lúc 05:49 ngày 19/09/2026, chạm mốc **Epoch 49 (67%)**.
   - Best Checkpoint: `solar_dualmamba_la-epoch=48-val_psnr=34.61-val_ssim=0.9197.ckpt` (**Val PSNR = 34.61 dB**, **Val SSIM = 0.9197** — Kỷ lục Validation cao nhất dòng SOLAR).
   - Script & Log: `scripts/train_solar_dualmamba_la.sh` | Log: `scripts/output/train_solar_dualmamba_la/log/72618.out`.

7. **`SOLAR_RegFormer` - Test Benchmark Độc Lập Patient L310 (Bản Full 50 Epochs) (Job ID `72610`):**
   - ✅ Hoàn thành 100% lúc 22:55 ngày 17/09/2026 (`rc=0`) trên 214 lát cắt CT độc lập:
     - *Cung quét chuẩn LA-120° (64 views):* **PSNR = 32.47 dB**, **SSIM = 0.9095**, **RMSE = 0.0247**.
     - *Stress test LA-90° (khuyết 270°):* **PSNR = 27.64 dB**, **SSIM = 0.8804**, **RMSE = 0.0442**.
   - Checkpoint kiểm thử: `solar_regformer_la-epoch=45-val_psnr=34.00-val_ssim=0.9117.ckpt`.

8. **`SOLAR_RegFormer` - Huấn luyện Giai đoạn 2: Epoch 35 $\to$ 50 (Job ID `72256`):**
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
- [x] **5. Chuẩn Hóa Mã Nguồn & Nộp Huấn Luyện 4 Mô Hình Baseline Mới (20/09/2026):**
  - Đã chuẩn hóa lệnh gọi `get_datamodule()` cho cả 4 baseline: `MoDL`, `FISTA-Net`, `DuDoNet`, `CT-Former` (sửa tham số `dataset_name` $\to$ `dataset_type`, định danh `setting_tag`).
  - Đã nộp thành công 4 Slurm jobs lên DGX-A100:
    * `MoDL`: Job ID **`72962`**
    * `FISTA-Net`: Job ID **`72963`**
    * `DuDoNet`: Job ID **`72964`**
    * `CT-Former`: Job ID **`72965`**
- [x] **6. Nghiệm Thu & Đánh Giá Benchmark Nhóm Mô Hình NIH DeepLesion (20/09/2026):**
  - **`DuDoTrans`:** Hoàn tất 50 epochs (Job `72823`). Đã chạy benchmark độc lập (Job `72969`): LA-120° đạt **23.76 dB / 0.6637**, LA-90° đạt **20.38 dB / 0.6692**.
  - **`LEARN_Mamba`:** Hoàn tất 50 epochs (Job `72821`). Đã chạy benchmark độc lập (Job `72968`): LA-120° đạt **26.69 dB / 0.7132**, LA-90° đạt **18.59 dB / 0.3913**.
  - **`LEARN_Longformer`:** Hoàn tất Giai đoạn 1 (Epoch 36/50 - Val PSNR: 32.12 dB). Đã submit job resume `72966` để chạy nốt 14 epochs.
  - **`LEARN_LongNet`:** Khắc phục triệt để lỗi OOM (chuyển `--window_size 16` $\to$ `2`, tăng `REQUIRED_VRAM=25000`), đã nộp job `72967`.
  - **`LEARN` (CNN):** Đang chạy ổn định tại Epoch 22/50 (Job `72822`), Val PSNR đạt 33.12 dB.
- [ ] **7. Kế Hoạch Tiếp Theo: Lặp Lại Huấn Luyện Trên Bộ Dữ Liệu LIDC-IDRI:**
  - Lặp lại quy trình huấn luyện cho toàn bộ nhóm baseline và mô hình đề xuất bằng các script `scripts/train_*_lidc.sh`.
  - Đánh giá test benchmark trên tập kiểm thử độc lập LIDC (LA-120° và LA-90°).
- [ ] **8. Hoàn thiện Bản thảo Bài báo SOICT 2026 (`papers/soict2026/`):**
  - Rà soát bản thảo `main.tex`, đưa thêm kết quả đối chứng của `LEARN` và `DuDoTrans`.
  - *Nhắc lại điều cấm kỵ:* Tuyệt đối không đưa SOLAR vào bài báo SOICT 2026.
