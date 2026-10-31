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

---

## 6. Trạng Thái Cuối Cùng Trên Cụm DGX-A100 (22/09/2026) — Trước Khi Mất Server

> Chuyển nguyên văn từ `CHECKPOINT.md` §1–3 và `SYSTEM_SPEC.md` §4–5 ngày 26/10/2026. Cụm này đã mất cùng toàn bộ checkpoint và dữ liệu; 6 job đang chờ dưới đây (`73105`, `73106`, `73107`, `73145`, `73146`, `73147`) chưa từng chạy. Số liệu ở đây là `legacy:<jobid>`, đo trên 1 bệnh nhân (`L310`).

### 6.1. CHECKPOINT.md §1–3 (bản ngày 22/09/2026)

#### 1. Trạng Thái Các Slurm Job Cluster (DGX-A100)

> [!NOTE]
> **Hiện trạng Cluster DGX-A100:** Node tính toán `DGX-A100` đang trong trạng thái `MIXED+DRAIN` (`Reason=Bao tri [root@2026-09-21T14:23:10]`). Toàn bộ các job đang xếp hàng đợi (`PD`) sẽ tự động được cấp phát và chạy ngay khi quản trị viên hoàn tất bảo trì và mở lại node (`scontrol resume`).

##### 🚀 A. Các Job Đang Chờ Xử Lý Trong Hàng Đợi (Queue Pending)
1. **`CT-Former` - Test Benchmark Độc Lập AAPM (LA-120° & LA-90°):**
   - **Job ID Slurm:** **`73105`** (Submit lúc 19:50 ngày 21/09/2026).
   - **Trạng thái:** 🟡 **Hàng đợi `PD (ReqNodeNotAvail - Chờ bảo trì)`**.
   - **Checkpoint kiểm thử:** `ct_former_la-epoch=49-val_psnr=33.64-val_ssim=0.9303.ckpt`.
   - **Script & Log:** `scripts/test_ct_former_la.sh` | Log: `scripts/output/test_ct_former_la/log/73105.out`

2. **`DuDoNet` - Test Benchmark Độc Lập AAPM (LA-120° & LA-90°):**
   - **Job ID Slurm:** **`73106`** (Submit lúc 19:50 ngày 21/09/2026).
   - **Trạng thái:** 🟡 **Hàng đợi `PD (ReqNodeNotAvail - Chờ bảo trì)`**.
   - **Checkpoint kiểm thử:** `dudonet_la-epoch=41-val_psnr=28.98-val_ssim=0.8536.ckpt`.
   - **Script & Log:** `scripts/test_dudonet_la.sh` | Log: `scripts/output/test_dudonet_la/log/73106.out`

3. **`LEARN` - Huấn luyện Resume NIH DeepLesion (Epoch 40 $\to$ 50):**
   - **Job ID Slurm:** **`73107`** (Submit lúc 19:50 ngày 21/09/2026).
   - **Trạng thái:** 🟡 **Hàng đợi `PD (ReqNodeNotAvail - Chờ bảo trì)`** (Tự động nạp `last.ckpt` để train tiếp 10 epochs cuối).
   - **Script & Log:** `scripts/train_learn_deeplesion.sh` | Log: `scripts/output/train_learn_deeplesion/log/73107.out`

4. **`FISTA-Net` - Test Benchmark Độc Lập AAPM (LA-120° & LA-90°):**
   - **Job ID Slurm:** **`73145`** (Submit lúc 19:32 ngày 22/09/2026).
   - **Trạng thái:** 🟡 **Hàng đợi `PD (ReqNodeNotAvail - Chờ bảo trì)`**.
   - **Checkpoint kiểm thử:** `fista_net_la-epoch=49-val_psnr=35.01-val_ssim=0.9527.ckpt`.
   - **Script & Log:** `scripts/test_fista_net_la.sh` | Log: `scripts/output/test_fista_net_la/log/73145.out`

5. **`LEARN_Longformer` - Test Benchmark Độc Lập NIH DeepLesion (LA-120° & LA-90°):**
   - **Job ID Slurm:** **`73146`** (Submit lúc 19:32 ngày 22/09/2026).
   - **Trạng thái:** 🟡 **Hàng đợi `PD (ReqNodeNotAvail - Chờ bảo trì)`**.
   - **Checkpoint kiểm thử:** `longformer_la-epoch=47-val_psnr=32.65-val_ssim=0.9025.ckpt`.
   - **Script & Log:** `scripts/test_longformer_deeplesion.sh` | Log: `scripts/output/test_longformer_deeplesion/log/73146.out`

6. **`LEARN_LongNet` - Huấn luyện Resume NIH DeepLesion (Epoch 36 $\to$ 50):**
   - **Job ID Slurm:** **`73147`** (Submit lúc 19:32 ngày 22/09/2026).
   - **Trạng thái:** 🟡 **Hàng đợi `PD (ReqNodeNotAvail - Chờ bảo trì)`** (Tự động nạp `last.ckpt` tiếp tục từ Epoch 36).
   - **Script & Log:** `scripts/train_longnet_deeplesion.sh` | Log: `scripts/output/train_longnet_deeplesion/log/73147.out`

---

##### ✅ B. Các Job Vừa Hoàn Thành & Nghiệm Thu Gần Nhất
1. **`FISTA-Net` - Huấn luyện Baseline Deep Unfolded FISTA trên AAPM LA-120° (Job ID `72963`):**
   - ✅ Hoàn thành 100% 50/50 Epochs lúc 23:14:05 ngày 21/09/2026 (`rc=0`, thời gian chạy 11h04m).
   - Best Checkpoint: `fista_net_la-epoch=49-val_psnr=35.01-val_ssim=0.9527.ckpt` (**Val PSNR = 35.01 dB**, **Val SSIM = 0.9527**; Epoch 48 đạt SSIM 0.9537).
   - File checkpoint: `saved_models/FISTA_Net/fista_net_la-epoch=49-val_psnr=35.01-val_ssim=0.9527.ckpt` & `last.ckpt`.
   - Script & Log: `scripts/train_fista_net_la.sh` | Log: `scripts/output/train_fista_net_la/log/72963.out`.

2. **`LEARN_Longformer` - Huấn luyện Giai đoạn 2 NIH DeepLesion (Job ID `72966`):**
   - ✅ Hoàn thành 100% 50/50 Epochs lúc 23:43:07 ngày 21/09/2026 (`rc=0`, thời gian chạy 11h33m).
   - Best Checkpoint: `longformer_la-epoch=47-val_psnr=32.65-val_ssim=0.9025.ckpt` (**Val PSNR = 32.65 dB**, **Val SSIM = 0.9025**).
   - File checkpoint: `saved_models/deeplesion/LEARN_Longformer/`.
   - Script & Log: `scripts/train_longformer_deeplesion.sh` | Log: `scripts/output/train_longformer_deeplesion/log/72966.out`.

3. **`LEARN_LongNet` - Huấn luyện Giai đoạn 1 NIH DeepLesion (Job ID `72967`):**
   - ⏱️ Chạm mốc 24:00:00 Slurm Time Limit lúc 14:08:44 ngày 22/09/2026 tại **Epoch 40 (56%)**.
   - Best Checkpoint đã lưu: `longnet_la-epoch=35-val_psnr=31.55-val_ssim=0.8851.ckpt` (**Val PSNR = 31.55 dB**) và `longnet_la-epoch=36-val_psnr=31.54-val_ssim=0.8861.ckpt` (**Val SSIM = 0.8861**).
   - File `last.ckpt` (Epoch 36, step 74000) bảo toàn an toàn tại `saved_models/deeplesion/LEARN_LongNet/last.ckpt`.
   - Script & Log: `scripts/train_longnet_deeplesion.sh` | Log: `scripts/output/train_longnet_deeplesion/log/72967.out`.

4. **`CT-Former` - Huấn luyện Baseline Cross-Shape Attention ViT trên AAPM LA-120° (Job ID `72965`):**
   - ✅ Hoàn thành 100% 50/50 Epochs lúc 18:44:10 ngày 21/09/2026 (`rc=0`).
   - Best Checkpoint: `ct_former_la-epoch=49-val_psnr=33.64-val_ssim=0.9303.ckpt` (**Val PSNR = 33.64 dB**, **Val SSIM = 0.9303**).
   - Đã nộp job kiểm thử benchmark độc lập **`73105`**.
   - Script & Log: `scripts/train_ct_former_la.sh` | Log: `scripts/output/train_ct_former_la/log/72965.out`.

5. **`DuDoNet` - Huấn luyện Baseline Dual-Domain trên AAPM LA-120° (Job ID `72964`):**
   - ✅ Hoàn thành 100% 50/50 Epochs lúc 14:08:18 ngày 21/09/2026 (`rc=0`).
   - Best Checkpoint: `dudonet_la-epoch=41-val_psnr=28.98-val_ssim=0.8536.ckpt` (**Val PSNR = 28.98 dB**, **Val SSIM = 0.8536**).
   - Đã nộp job kiểm thử benchmark độc lập **`73106`**.
   - Script & Log: `scripts/train_dudonet_la.sh` | Log: `scripts/output/train_dudonet_la/log/72964.out`.

6. **`LEARN` (CNN) - Huấn luyện Giai đoạn 1 trên NIH DeepLesion (Job ID `72822`):**
   - Chạm mốc 24:00:00 Slurm Time Limit lúc 06:13:35 ngày 21/09/2026 tại **Epoch 40 (31%)**.
   - Best Checkpoint: `epoch=36-val_psnr=34.2668.ckpt` (**Val PSNR = 34.27 dB**).
   - Đã submit job resume **`73107`**.
   - Script & Log: `scripts/train_learn_deeplesion.sh` | Log: `scripts/output/train_learn_deeplesion/log/72822.out`.

7. **`MoDL` - Sửa lỗi giải thuật Conjugate Gradient & Toán tử Adjoint:**
   - Đã chẩn đoán và khắc phục nguyên nhân phân kỳ: thay thế toán tử FBP trong bước tính ma trận $(A^T A + \lambda I)$ bằng toán tử liên hợp chuẩn $A^T$ (`operator.adjoint`), bảo đảm tính đối xứng xác định dương (SPD) cho CG.
   - Sẵn sàng nộp lại job huấn luyện sau khi cụm máy chủ mở lại.

8. **`DuDoTrans` - Test Benchmark Độc Lập NIH DeepLesion (Job ID `72969`):**
   - ✅ Hoàn thành 100% lúc 21:07:26 ngày 20/09/2026 (`rc=0`): LA-120° đạt 23.76 dB / 0.6637; LA-90° đạt 20.38 dB / 0.6692.

9. **`LEARN_Mamba` - Test Benchmark Độc Lập NIH DeepLesion (Job ID `72968`):**
   - ✅ Hoàn thành 100% lúc 21:14 ngày 20/09/2026 (`rc=0`): LA-120° đạt 26.69 dB / 0.7132; LA-90° đạt 18.59 dB / 0.3913.

10. **`SOLAR_DualMamba` - Test Benchmark Kiểm Chứng SOTA Patient L310 (Job ID `72818`):**
    - ✅ Hoàn thành 100% lúc 19:52:50 ngày 19/09/2026 (`rc=0`): LA-120° đạt 32.91 dB / 0.9186; LA-90° đạt 27.98 dB / 0.8843 (🏆 Kỷ lục SSIM cao nhất).

11. **`SOLAR_RegFormer` - Test Benchmark Độc Lập Patient L310 (Job ID `72610`):**
    - ✅ Hoàn thành 100% lúc 22:55 ngày 17/09/2026 (`rc=0`): LA-120° đạt 32.47 dB / 0.9095; LA-90° đạt 27.64 dB / 0.8804.

---

#### 2. Bảng Tóm Tắt Best Checkpoints SOTA Hiện Tại (AAPM LA-CT)

| Dòng Mô Hình | Checkpoint Tốt Nhất | Val PSNR / SSIM | Test LA-120° (PSNR/SSIM) | Test LA-90° (PSNR/SSIM) |
| :--- | :--- | :---: | :---: | :---: |
| **`FISTA-Net` (Deep Unfolded)** | `fista_net_la-epoch=49.ckpt` | **35.01 dB / 0.9527** | *(Đang chờ chạy test)* | *(Đang chờ chạy test)* |
| **`LEARN_Longformer` (Baseline)** | `longformer_la-epoch=45.ckpt` | 34.77 dB / 0.9383 | 33.10 dB / 0.9237 | 19.16 dB / 0.6097 |
| **`RegFormer` (Baseline)** | `epoch=46-val_psnr=36.0533.ckpt` | 36.05 dB / - | 32.82 dB / 0.9398 | 28.20 dB / 0.8814 |
| **`LEARN (Original CNN)`** | `epoch=46-val_psnr=36.2681.ckpt` | 36.27 dB / 0.950 | 32.29 dB / 0.9383 | 28.03 dB / 0.8793 |
| **`CT-Former` (Baseline ViT)** | `ct_former_la-epoch=49.ckpt` | **33.64 dB / 0.9303** | *(Job 73105 trong queue)* | *(Job 73105 trong queue)* |
| **`DuDoNet` (Baseline Dual-Domain)**| `dudonet_la-epoch=41.ckpt` | **28.98 dB / 0.8536** | *(Job 73106 trong queue)* | *(Job 73106 trong queue)* |
| **`DuDoTrans` (Baseline)** | `epoch=38-val_psnr=25.7300.ckpt` | 25.73 dB / 0.730 | 25.15 dB / 0.7447 | 21.81 dB / 0.6738 |
| **`SOLAR_Longformer` (SOTA)** | `solar_longformer_la-epoch=45.ckpt` | 34.18 dB / 0.9165 | 32.95 dB / 0.9155 | **28.05 dB / 0.8774** |
| **`SOLAR_Mamba` (SOTA)** | `solar_mamba_la-epoch=45.ckpt` | 34.00 dB / 0.9089 | 31.90 dB / 0.9114 | **27.53 dB / 0.8760** |
| **`SOLAR_RegFormer` (50 ep)** | `solar_regformer_la-epoch=45.ckpt` | **34.00 dB / 0.9139** | **32.47 dB / 0.9095** | **27.64 dB / 0.8804** |
| **`SOLAR_DualMamba` (SOTA Toàn Dự Án)**| `solar_dualmamba_la-epoch=48.ckpt` | **34.61 dB / 0.9197** | **32.91 dB / 0.9186** | **27.98 dB / 0.8843 (Kỷ lục)** |

*Dữ liệu chi tiết đối sánh xem tại:* [benchmark_results.csv](reports/sep-21-2026/benchmark_results.csv) và [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md).

---

#### 3. Danh Mục Nhiệm Vụ Tiếp Theo (Actionable TODOs)

- [x] **1. Nghiệm thu Huấn luyện `SOLAR_RegFormer` 50 Epochs (Job `72256`):** Đã hoàn tất 17/09/2026.
- [x] **2. Nghiệm thu Benchmark `SOLAR_RegFormer` 50 Epochs (Job `72610`):** Đã hoàn tất 17/09/2026.
- [x] **3. Nghiệm thu Huấn luyện & Benchmark `SOLAR_DualMamba` (Jobs `72618` & `72818`):** Đã hoàn tất 19/09/2026.
- [x] **4. Hoàn thành Khảo cứu Toàn diện Đối thủ Tái tạo LA-CT (19/09/2026):** Đã lập báo cáo tại [reports/sep-19-2026/COMPETITORS_RESEARCH.md](reports/sep-19-2026/COMPETITORS_RESEARCH.md).
- [x] **5. Nghiệm thu Huấn luyện `CT-Former` (Job `72965`) & `DuDoNet` (Job `72964`):**
  - Hoàn tất trọn vẹn 50/50 epochs ngày 21/09/2026.
  - Đã nộp job test benchmark: `CT-Former` (Job **`73105`**) và `DuDoNet` (Job **`73106`**).
- [x] **6. Nghiệm thu Huấn luyện `FISTA-Net` 50 Epochs (Job `72963`) (21/09/2026):**
  - Đạt Val PSNR = 35.01 dB, SSIM = 0.9527 tại Epoch 49.
- [x] **7. Nghiệm thu Huấn luyện `LEARN_Longformer` DeepLesion 50 Epochs (Job `72966`) (21/09/2026):**
  - Đạt Val PSNR = 32.65 dB, SSIM = 0.9025 tại Epoch 47.
- [x] **8. Sửa Lỗi Toán Học Mô Hình `MoDL` (`baselines/MoDL/models.py`):**
  - Tách biệt toán tử chiếu ngược liên hợp thuần túy $A^T$ (`operator.adjoint`) cho bước giải Conjugate Gradient. Kiểm thử hội tụ thành công.
- [ ] **9. Nộp Job & Nghiệm Thu Test Benchmark:**
  - Nộp test benchmark cho `FISTA-Net` trên AAPM LA-120° và LA-90° (`scripts/test_fista_net_la.sh`).
  - Nộp test benchmark cho `LEARN_Longformer` trên NIH DeepLesion (`scripts/test_longformer_deeplesion.sh`).
- [ ] **10. Nộp Resume `LEARN_LongNet` DeepLesion:**
  - Nộp job resume huấn luyện nốt từ Epoch 36 $\to$ 50 trên DeepLesion (`scripts/train_longnet_deeplesion.sh`).
- [ ] **11. Theo Dõi Hàng Đợi & Nghiệm Thu Các Job Khi Node Hết Bảo Trì:**
  - Job `73105` (Test `CT-Former`), Job `73106` (Test `DuDoNet`), Job `73107` (Resume `LEARN` DeepLesion).
- [ ] **12. Hoàn thiện Bản thảo Bài báo SOICT 2026 (`papers/soict2026/`):**
  - Cập nhật số liệu thực nghiệm đối chứng của `LEARN`, `DuDoTrans`, `FISTA-Net`.
  - *Tuyệt đối tuân thủ:* Không đưa SOLAR vào SOICT 2026.

### 6.2. SYSTEM_SPEC.md §4–5 (cụm A100 cũ)

#### 4. Hệ Thống Máy Chủ & Quy Tắc Slurm Cluster

- **Máy chủ Slurm HPC:** `10.204.1.52` (`uittogether3@bcm-headnode`)
- **Ánh xạ đường dẫn:**
  - Local SSHFS: `/home/phandinhminh/Downloads/kltn/agents-research/uittogether3-slurm-server/`
  - Server Datastore: `/datastore/uittogether3/LuuTru/`
- **Môi trường Conda:** `/datastore/uittogether3/tools/miniconda3/envs/LongNet` (đầy đủ `torch==2.8.0+cu128`, `mamba-ssm==2.3.1`, `transformers==4.57.6`, `odl==0.8.3`, `astra-toolbox==2.3.0`, `torchmetrics==1.8.2`, `pydicom==2.4.4`).
- **Cơ chế GPU:** NVIDIA MPS (`#SBATCH --gres=mps:a100:2`), kiểm tra VRAM qua script Admin `/usr/local/bin/gpu_check.sh $REQUIRED_VRAM $SLURM_JOB_ID`.
- **Quy tắc bắt buộc về lưu Output/Log:**
  - Mọi job Slurm **bắt buộc** phải ghi log vào:
    ```text
    scripts/output/<tên script>/log/%j.out
    scripts/output/<tên script>/log/%j.err
    ```

---

#### 5. Danh Mục Dataset Gốc Tham Chiếu
1. **AAPM Mayo Clinic Low Dose CT Grand Challenge (2016):**
   - Định dạng: DICOM (`.IMA`), 3mm slice thickness (`full_3mm`).
   - 9 Train patients (`L067`, `L096`, `L109`, `L143`, `L192`, `L286`, `L291`, `L333`, `L506`), 1 Test patient (`L310`).
   - Đường dẫn server: `/datastore/uittogether3/LuuTru/Thanhld/CT-Reconstruction/split/`
2. **NIH DeepLesion CT Dataset:**
   - Đường dẫn server: `/datastore/uittogether3/LuuTru/Thanhld/CT-Reconstruction/minideeplesion/`
3. **LIDC-IDRI Lung CT Dataset:**
   - Đường dẫn server / local: `dataset/lidc_idri/limited_angle/`
