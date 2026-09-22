# CHECKPOINT DỰ ÁN NCKH: LIMITED-ANGLE CT RECONSTRUCTION
> **TRẠNG THÁI ACTIVE (Cập nhật: 22/09/2026)**
> 
> *Lưu ý cho AI Agent:* File này lưu trữ **trạng thái công việc tức thời, các job đang chạy và nhiệm vụ cần làm tiếp theo**.
> - Thông số toán học/hình học CT & phần cứng: Tham chiếu [SYSTEM_SPEC.md](SYSTEM_SPEC.md).
> - Lịch sử các mốc thử nghiệm cũ từ tháng 8: Tham chiếu [ARCHIVE_LOG.md](ARCHIVE_LOG.md).
> - Quy tắc thực thi và điều cấm kỵ: Tham chiếu [AGENTS.md](AGENTS.md).

---

## 1. Trạng Thái Các Slurm Job Cluster (DGX-A100)

> [!NOTE]
> **Hiện trạng Cluster DGX-A100:** Node tính toán `DGX-A100` đang trong trạng thái `MIXED+DRAIN` (`Reason=Bao tri [root@2026-09-21T14:23:10]`). Toàn bộ các job đang xếp hàng đợi (`PD`) sẽ tự động được cấp phát và chạy ngay khi quản trị viên hoàn tất bảo trì và mở lại node (`scontrol resume`).

### 🚀 A. Các Job Đang Chờ Xử Lý Trong Hàng Đợi (Queue Pending)
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

### ✅ B. Các Job Vừa Hoàn Thành & Nghiệm Thu Gần Nhất
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

## 2. Bảng Tóm Tắt Best Checkpoints SOTA Hiện Tại (AAPM LA-CT)

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

## 3. Danh Mục Nhiệm Vụ Tiếp Theo (Actionable TODOs)

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
