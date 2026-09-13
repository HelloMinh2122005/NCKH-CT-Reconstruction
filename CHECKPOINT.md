# CHECKPOINT DỰ ÁN NCKH: LIMITED-ANGLE CT RECONSTRUCTION

> **Lưu ý dành cho AI Agent:** Đọc kỹ file này ở đầu mỗi session để nắm bắt toàn bộ bối cảnh, kiến trúc, quy tắc hệ thống và tiến độ hiện tại của dự án. Luôn cập nhật file này sau khi hoàn thành các mốc công việc mới.

---

## 1. Thông Tin Chung
- **Dự án:** Nghiên cứu Khoa học (NCKH) - Tái tạo ảnh cắt lớp CT góc giới hạn (Limited-Angle CT Reconstruction)
- **Tác giả / Không gian làm việc:** `MinhPD` (`/datastore/uittogether3/LuuTru/MinhPD` / `uittogether3-slurm-server/MinhPD`)
- **GitHub Repository:** [https://github.com/HelloMinh2122005/NCKH-CT-Reconstruction.git](https://github.com/HelloMinh2122005/NCKH-CT-Reconstruction.git) (Nhánh: `main`)

---

## 2. Kế Thừa Nghiên Cứu & Cơ Sở Khoa Học

### 2.1. Kế thừa công trình nghiên cứu của Thành (Baseline):
- **Bài báo gốc (Paper):** `~/note/MVA___CT_reconstruction_revised-2.pdf` (Mô hình MVA - Multi-View Attention cho bài toán Sparse-view CT).
- **Toàn bộ Source code của Thành:** `uittogether3-slurm-server/Thanhld` (Bao gồm các mô hình `LEARN_LongNet`, `LEARN_Longformer`, `LEARN_Mamba` và các script sinh sinogram gốc).
- **Tài liệu phân tích kiến trúc MVA vs Đề xuất mới:** [ARCHITECTURE_MVA_VS_PROPOSED_SOLAR.md](note/ARCHITECTURE_MVA_VS_PROPOSED_SOLAR.md).

### 2.2. Mục tiêu nghiên cứu cốt lõi:
- **Mục tiêu nhân văn:** Cắt giảm tối đa liều bức xạ tia X chiếu vào người bệnh nhân (**Radiation Dose Reduction $50\% - 75\%$**) tuân theo nguyên tắc y học **ALARA**, đồng thời bảo vệ các cơ quan nhạy cảm với phóng xạ (tuyến giáp, mắt, tuyến vú).
- **Thách thức toán học:** Cung quét bị giới hạn $\Delta \theta < 180^\circ$ dẫn đến vùng khuyết hình nêm trong miền tần số Fourier (**Missing Wedge Problem**). Địa hình Hessian $A^TA$ suy biến, các bước gradient bậc 1 bị dao động (Zigzag). 
- **Hướng đề xuất:** Phát triển kiến trúc tối ưu hóa bậc 2 (Second-Order) kết hợp phân nhánh kép cục bộ - toàn cục (Dual-Branch Local/Nonlocal như mạng **SOLAR**) để vượt qua giới hạn của MVA trên bài toán Limited-Angle CT.

---

## 3. Cấu Hình & Tham Số Toán Học / Vật Lý Đã Thống Nhất

| Tham số | Giá trị | Ý nghĩa & Lý do lựa chọn |
| :--- | :--- | :--- |
| **Hình học chiếu** | Fan-Beam Geometry | Chuẩn máy CT y tế Siemens Somatom (`src_radius=600mm`, `det_radius=290mm`, `512` detectors) |
| **Dải góc quét chính** | $[-60^\circ, +60^\circ]$ ($120^\circ$ span) | Benchmark quốc tế SOTA; đối xứng qua trục $0^\circ$ bảo vệ cơ quan hai bên cơ thể; giảm $66.7\%$ tia X |
| **Số góc chiếu (Views)** | `64` views | Bước góc $\Delta \theta \approx 1.875^\circ$ (Pure Limited-Angle); $64 = 2^6$ tối ưu cho down/upsampling U-Net/ViT/Mamba |
| **Độ phân giải** | $256 \times 256$ | Chiếu & FBP trên lưới gốc $512 \times 512$ trước, sau đó resize về $256 \times 256$ tối ưu VRAM GPU A100 |
| **Bộ lọc FBP** | Ram-Lak filter | High-pass ramp filter chuẩn tái tạo sơ bộ |
| **Mô phỏng nhiễu** | 2 chế độ: `noise_0` và `noise_1e6` | `noise_0`: Đánh giá khôi phục góc khuyết thuần túy; `noise_1e6` ($I_0=10^6, \sigma=0.05$): Thử thách liều siêu thấp thực tế |

---

## 4. Hệ Thống Máy Chủ & Quy Tắc Slurm Cluster

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

## 5. Danh Mục Dataset Gốc Tham Chiếu
1. **AAPM Mayo Clinic Low Dose CT Grand Challenge (2016):**
   - Định dạng: DICOM (`.IMA`), 3mm slice thickness (`full_3mm`).
   - 9 Train patients (`L067`, `L096`, `L109`, `L143`, `L192`, `L286`, `L291`, `L333`, `L506`), 1 Test patient (`L310`).
   - Đường dẫn server: `/datastore/uittogether3/LuuTru/Thanhld/CT-Reconstruction/split/`
2. **NIH DeepLesion CT Dataset:**
   - Đường dẫn server: `/datastore/uittogether3/LuuTru/Thanhld/CT-Reconstruction/minideeplesion/`

---

## 6. Trạng Thái & Tiến Độ Dự Án (Progress Log)

### Đã hoàn thành:
- [x] Tìm kiếm, xác định và đối chiếu dữ liệu CT gốc của Thành (`Thanhld/CT-Reconstruction`).
- [x] Phân tích pipeline Sparse-view của Thành (`prepare_data_sinogram.py`, `CTSlice_Provider_offline.py`).
- [x] Thiết kế và xây dựng module tạo dữ liệu Limited-Angle CT:
  - `data/CTSlice_Provider_LA.py`: Dataset loader Fan-beam góc giới hạn.
  - `data/prepare_data_sinogram_LA.py`: Script sinh hàng loạt sinogram & FBP `.npy`.
  - `data/datamodule_LA.py`: PyTorch Lightning DataModule.
- [x] Biên soạn tài liệu chi tiết: `data/README.md`, `scripts/README.md`, `README.md`, `baselines/README.md`.
- [x] Khởi tạo Git repo, chuẩn hóa `.gitignore` và push lên GitHub `HelloMinh2122005/NCKH-CT-Reconstruction`.
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
  - **Thư mục ảnh PNG đã xuất:** `visualizations/120deg/` và `visualizations/90deg/` (chứa các folder `slice_050`, `slice_100`, `slice_150` với từng file ảnh PNG riêng lẻ của Ground Truth, FBP, LongNet, Mamba, Longformer và panel đối sánh đa khung hình `comparison_summary.png`).

- [x] **Hiện thực hóa 3 Biến thể Kiến trúc Đề xuất SOLAR theo Cấu trúc Baseline Chuẩn Hóa:**
  - **`baselines/SOLAR_LongNet/`:** Tối ưu hóa bậc 2 Newton-CG + Nhánh kép Res-CNN & Dilated Attention (LongNet). Script: `train_solar_longnet_la.py`, `test_solar_longnet_la.py`, `scripts/train_solar_longnet_la.sh`.
  - **`baselines/SOLAR_Longformer/`:** Tối ưu hóa bậc 2 Newton-CG + Nhánh kép Res-CNN & Sliding-Chunks Attention (Longformer). Script: `train_solar_longformer_la.py`, `test_solar_longformer_la.py`, `scripts/train_solar_longformer_la.sh`.
  - **`baselines/SOLAR_Mamba/`:** Tối ưu hóa bậc 2 Newton-CG + Nhánh kép Res-CNN & Selective SSM (Mamba). Script: `train_solar_mamba_la.py`, `test_solar_mamba_la.py`, `scripts/train_solar_mamba_la.sh`.
- [x] **Theo dõi & Resume Tiến độ Huấn luyện 3 Mô hình Đề xuất SOLAR trên Slurm HPC (DGX-A100):**
  - Đã chuẩn hóa cơ chế tự động phát hiện `last.ckpt` (`--resume_ckpt`) trong toàn bộ 3 script sbatch (`train_solar_longformer_la.sh`, `train_solar_longnet_la.sh`, `train_solar_mamba_la.sh`).
  - **`SOLAR_Longformer` (120°):** Job `67505` đạt mốc 24h Time Limit tại Epoch 36. Tiến trình resume (**Job ID `67820`**) đang chạy ở Epoch 48/50 và đã xác lập **ĐỈNH MỚI KỶ LỤC** tại **Epoch 45: Val PSNR = 34.18 dB, Val SSIM = 0.9165** (Checkpoint: `solar_longformer_la-epoch=45-val_psnr=34.18-val_ssim=0.9165.ckpt`). Đánh giá test trước đó ở Epoch 35 (Job `67828`):
    - *Cung quét chuẩn LA-120°:* **PSNR = 32.51 dB**, **SSIM = 0.9101**, **RMSE = 0.0239**.
    - *Stress test LA-90° (góc khuyết 270°):* **PSNR = 27.92 dB**, **SSIM = 0.8736**, **RMSE = 0.0416** (🏆 **SOTA Toàn diện ở góc hẹp 90°: +8.76 dB PSNR và +0.2639 SSIM** so với LEARN_Longformer).
  - **`SOLAR_Mamba` (120°):** Job `67507` đạt mốc 24h Time Limit tại Epoch 29. Tiến trình resume (**Job ID `67821`**) đang chạy ở Epoch 42/50, 100% ổn định số học không NaN, và đã xác lập **ĐỈNH MỚI KỶ LỤC** tại **Epoch 35: Val PSNR = 33.69 dB, Val SSIM = 0.9062** (Checkpoint: `solar_mamba_la-epoch=35-val_psnr=33.69-val_ssim=0.9062.ckpt`). Đánh giá test trước đó ở Epoch 25 (Job `67829`):
    - *Cung quét chuẩn LA-120°:* **PSNR = 31.21 dB**, **SSIM = 0.8982**, **RMSE = 0.0291** (🚀 **+4.89 dB PSNR và +0.1514 SSIM** so với LEARN_Mamba).
    - *Stress test LA-90° (góc khuyết 270°):* **PSNR = 27.16 dB**, **SSIM = 0.8620**, **RMSE = 0.0472** (🚀 **+8.40 dB PSNR và +0.4328 SSIM (+100.8%)** so với LEARN_Mamba).
  - **`SOLAR_LongNet` (120°):** Job `67506` đạt mốc 24h Time Limit tại Epoch 31 (Best Val: **PSNR = 32.26 dB, SSIM = 0.8935** ở Epoch 29, checkpoint `solar_longnet_la-epoch=29-val_psnr=32.26-val_ssim=0.8935.ckpt`). Đang tạm dừng theo yêu cầu của người dùng. ✅ **Đã hoàn thành 100% Test Benchmark trên Patient L310 (214 lát cắt) - Job ID `67823`**:
    - *Cung quét chuẩn LA-120°:* **PSNR = 31.03 dB**, **SSIM = 0.8958**, **RMSE = 0.0294** (áp sát baseline LEARN_LongNet 50 epoch).
    - *Stress test LA-90° (góc khuyết 270°):* **PSNR = 27.19 dB**, **SSIM = 0.8639**, **RMSE = 0.0462** (🚀 **+8.00 dB PSNR và +0.2763 SSIM (+47.0%)** so với LEARN_LongNet).

- [x] **Biên soạn Báo cáo Tiến độ & Benchmark Định lượng (Ngày 03/09/2026):** Lưu tại [`reports/sep-03-2026/MAIN.md`](reports/sep-03-2026/MAIN.md) và bảng dữ liệu [`reports/sep-03-2026/benchmark_results.csv`](reports/sep-03-2026/benchmark_results.csv).
- [x] **Biên soạn Báo cáo Tiến độ & Đột phá SOLAR (Ngày 05/09/2026):** Lưu tại [`reports/sep-05-2026/MAIN.md`](reports/sep-05-2026/MAIN.md), bảng dữ liệu [`reports/sep-05-2026/benchmark_results.csv`](reports/sep-05-2026/benchmark_results.csv) và kiến trúc [`SOLAR_ARCHITECTURE.md`](SOLAR_ARCHITECTURE.md).
- [x] **- [x] **Hoàn thành Tải Dữ liệu CT Lồng ngực LIDC-IDRI từ TCIA REST API:**
  - Đã tải và giải nén thành công 1,975 lát cắt DICOM trên 12 bệnh nhân tiêu biểu.
  - Phân chia tập hợp đã lưu tại: `dataset/lidc_idri/splits_summary.json` (train: 8 bệnh nhân, val: 2 bệnh nhân, test: 2 bệnh nhân).
- [x] **Khởi chạy Sinh Dữ liệu Limited-Angle CT LIDC-IDRI (Job Slurm ID `67943`):**
  - Đã submit job `sbatch scripts/generate_la_dataset_lidc.sh` chạy trên node `DGX-A100`.
  - Sinh toàn bộ sinogram và FBP cho cả 2 cung quét $120^\circ$ và $90^\circ$ (64 views, 512 detectors, $256 \times 256$, `noise_0`) cho Train, Val, Test. Lưu tại: `dataset/lidc_idri/limited_angle/`.
- [x] **Khởi tạo Bản thảo Bài báo Hội nghị SOICT 2026 (`papers/soict2026/`):**
  - **Định hướng chiến lược từ Giáo sư:** Tuyệt đối giữ bí mật và để dành kiến trúc **SOLAR** (Second-Order Preconditioning) cho bài báo Journal đỉnh cao (IEEE TMI / MedIA, Q1, IF > 10). Bài báo SOICT 2026 chỉ tập trung vào nghiên cứu thực nghiệm đánh giá đối sánh các cơ chế chuỗi dài (Longformer vs LongNet vs Mamba) bên trong mạng Deep Unrolling (khung LEARN) dựa trên kết quả đã hoàn thiện tại [`reports/sep-03-2026/benchmark_results.csv`](reports/sep-03-2026/benchmark_results.csv).
  - Đã thiết lập cấu trúc bài báo chuẩn Springer LNCS (`llncs.cls`, `splncs04.bst`) từ gói `_MICCAI_CLIMEM2026__Tran_Minh_Vu___CL_for_survival_analysis`.
  - Đã hoàn thành bản thảo toàn diện [`papers/soict2026/main.tex`](papers/soict2026/main.tex), bao gồm Abstract, Introduction, Formulation toán học, phân tích 3 cơ chế chuỗi dài, tách 2 Bảng thực nghiệm đối sánh (Table 1 cho 120°, Table 2 cho 90° kèm số tham số Params), và thảo luận sâu sắc về sự suy giảm của Mamba ở góc $90^\circ$.
  - Đã tích hợp hoàn chỉnh hình ảnh Kiến trúc đề xuất Fig. 1 ([`papers/soict2026/figures/methodology.png`](papers/soict2026/figures/methodology.png)), Lưới đối sánh trực quan đa chế độ với ô phóng đại ROI màu vàng Fig. 2 ([`papers/soict2026/figures/visual_comparison_roi.png`](papers/soict2026/figures/visual_comparison_roi.png)), và Bản đồ sai số dư Fig. 3 ([`papers/soict2026/figures/err_...png`](papers/soict2026/figures/)).
  - Đã thiết lập trích dẫn đầy đủ tại [`papers/soict2026/refs.bib`](papers/soict2026/refs.bib).
- [x] **Trích xuất Attention Maps 14 Tầng của LEARN_Longformer (Job Slurm ID `67967`):**
  - Đã xuất thành công 2 ảnh panel Attention Evolution cho $120^\circ$ và $90^\circ$ (Slice 050) tại `visualizations/attention_maps/` và đã tích hợp trực tiếp vào Fig. 4 của bài báo SOICT 2026.
- [x] **Hoàn thành 100% Huấn luyện 50 Epochs cho SOLAR_Longformer (Job `67820`) & SOLAR_Mamba (Job `67821`):**
  - Cả hai mô hình đã hoàn tất trọn vẹn 50/50 Epochs trên cụm `DGX-A100` mà không gặp bất kỳ lỗi số học hay NaN nào.
  - Các checkpoint đỉnh mới đã được lưu:
    - `SOLAR_Longformer`: `solar_longformer_la-epoch=45-val_psnr=34.18-val_ssim=0.9165.ckpt`
    - `SOLAR_Mamba`: `solar_mamba_la-epoch=45-val_psnr=34.00-val_ssim=0.9089.ckpt` (và `epoch=46: val_ssim=0.9112`)
- [x] **Tái cấu trúc thư mục Dataset chuẩn hóa & Tạo Symlink an toàn:**
  - Đã chuyển dữ liệu AAPM về thư mục chuẩn: `dataset/aapm/limited_angle/`.
  - Đã tạo liên kết mềm (Symbolic link) `dataset/limited_angle -> dataset/aapm/limited_angle` đảm bảo tương thích ngược 100% cho toàn bộ hệ thống.
  - Xây dựng DataModule Factory thống nhất tại [`data/datamodule_factory.py`](data/datamodule_factory.py) hỗ trợ tự động nạp cả 3 dataset: AAPM, NIH DeepLesion, và LIDC-IDRI.
- [x] **Cập nhật & Khởi chạy Test Benchmark 50 Epochs + Visualization Tự động:**
  - Đã cập nhật 2 checkpoint đỉnh 50-epoch vào `scripts/test_solar_longformer_la.sh` và `scripts/test_solar_mamba_la.sh`.
  - **Job ID `68551`**: `test_solar_longformer_la.sh`
  - **Job ID `68552`**: `test_solar_mamba_la.sh`
  - **Job ID `68553`**: `visualize_solar_la.sh` (cấu hình Slurm `--dependency=afterok:68551:68552` tự động kích hoạt ngay khi 2 job test hoàn thành).
- [x] **Cấu hình & Khởi chạy Huấn luyện 6 Mô hình trên 2 Dataset Mới (NIH DeepLesion & LIDC-IDRI):**
  - Đã tích hợp tham số `--dataset_type` cho cả 6 mô hình (`LEARN_Longformer`, `LEARN_LongNet`, `LEARN_Mamba`, `SOLAR_Longformer`, `SOLAR_LongNet`, `SOLAR_Mamba`).
  - Đã sinh đầy đủ 12 script sbatch chuẩn NVIDIA MPS, kiểm tra VRAM, log và thư mục lưu checkpoint riêng biệt.
  - **NIH DeepLesion (Jobs `68557` - `68562`):** `train_longformer_deeplesion.sh`, `train_longnet_deeplesion.sh`, `train_mamba_deeplesion.sh`, `train_solar_longformer_deeplesion.sh`, `train_solar_longnet_deeplesion.sh`, `train_solar_mamba_deeplesion.sh`.
  - **LIDC-IDRI (Jobs `68563` - `68568`):** `train_longformer_lidc.sh`, `train_longnet_lidc.sh`, `train_mamba_lidc.sh`, `train_solar_longformer_lidc.sh`, `train_solar_longnet_lidc.sh`, `train_solar_mamba_lidc.sh`.

### Quy tắc nghiêm ngặt cho các session sau:
> [!IMPORTANT]
> **Quy tắc Bảo toàn Chú thích & Tính Toàn vẹn Mã nguồn:**
> - Tuyệt đối **KHÔNG ĐƯỢC tự ý xóa, lược bỏ, rút gọn hoặc thay đổi** bất kỳ dòng comment, docstrings tiếng Việt giải thích chi tiết nào trong toàn bộ codebase.
> - Tuyệt đối **KHÔNG ĐƯỢC viết code sai lệch, làm mâu thuẫn hoặc làm hỏng** các logic và giá trị mặc định đã được giải thích trong comment khi người dùng chưa yêu cầu rõ ràng.
> - Tuyệt đối tuân thủ phân tách ranh giới khoa học: **SOLAR** dành riêng cho bài Journal; bài **SOICT** chỉ sử dụng các mô hình Baseline (`LEARN_Longformer`, `LEARN_LongNet`, `LEARN_Mamba`, `FBP`).

### Các bước tiếp theo (Actionable TODOs dành cho các Session LLM tiếp theo):

- [x] **1. THEO DÕI & CẬP NHẬT KẾT QUẢ TEST BENCHMARK 50 EPOCHS CỦA SOLAR:**
  - **Job đã nghiệm thu:** Job `68551` (`test_solar_longformer_la.sh`) và Job `68552` (`test_solar_mamba_la.sh`) ✅ **Hoàn thành 100%**.
  - **Kết quả định lượng đã trích xuất trên Patient L310 (214 lát cắt):**
    - **`SOLAR_Longformer` (50 ep - Job `68551`):**
      - LA-120°: **PSNR = 32.95 dB**, **SSIM = 0.9155**, **RMSE = 0.0228**
      - LA-90°: **PSNR = 28.05 dB**, **SSIM = 0.8774**, **RMSE = 0.0412** (🏆 **SOTA Toàn diện ở góc 90°: +8.89 dB PSNR và +0.2677 SSIM** so với baseline)
    - **`SOLAR_Mamba` (50 ep - Job `68552`):**
      - LA-120°: **PSNR = 31.90 dB**, **SSIM = 0.9114**, **RMSE = 0.0268** (+5.58 dB so với LEARN_Mamba)
      - LA-90°: **PSNR = 27.53 dB**, **SSIM = 0.8760**, **RMSE = 0.0447** (🚀 **+8.77 dB PSNR và +0.4468 SSIM (+104.1%)** so với LEARN_Mamba)
  - Đã cập nhật đầy đủ vào:
    - [`reports/sep-05-2026/benchmark_results.csv`](reports/sep-05-2026/benchmark_results.csv).
    - [`EXPERIMENT_RESULTS.md`](EXPERIMENT_RESULTS.md).

- [x] **2. KIỂM TRA JOB TỰ ĐỘNG HÓA TRỰC QUAN HÓA (VISUALIZATION):**
  - **Job ID:** `68553` (`visualize_solar_la.sh`) ✅ **Hoàn thành 100%**.
  - Đã tự động kích hoạt sau khi 2 job test kết thúc qua `--dependency=afterok:68551:68552`.
  - Đã kết xuất đầy đủ ảnh thành phần và các panel đối sánh chất lượng cao ($300\text{ DPI}$) trên 3 lát cắt y tế độc lập (`slice_050`, `slice_100`, `slice_150`) cho cả 2 cung quét $120^\circ$ và $90^\circ$ tại [`visualizations/`](visualizations/) và [`reports/sep-05-2026/visualizations/`](reports/sep-05-2026/visualizations/).

- [ ] **3. THEO DÕI 12 SLURM JOB HUẤN LUYỆN ĐA DATASET (DEEPLESION & LIDC-IDRI):**
  - **Kết quả nghiệm thu & kiểm tra chi tiết (Thời điểm kiểm tra 13/09/2026):**
    - *NIH DeepLesion (Jobs `69545` - `69550`):*
      - `69545` (`LEARN_Mamba`): ✅ **COMPLETED 100% (50/50 Epochs)** vào ngày 09/09/2026. Checkpoint đỉnh: `mamba_la-epoch=23-val_psnr=26.18-val_ssim=0.7009.ckpt`, `last.ckpt` tại `saved_models/deeplesion/LEARN_Mamba/`.
      - `69546` (`LEARN_Longformer`): ❌ **FAILED tại Epoch 10 (chạy 8h49m)** do lỗi ASTRA Toolbox CUDA OOM (`Error: createTextureObject2D malloc: CUDA error 2: out of memory`). Đã lưu checkpoint đến Epoch 09 (`longformer_la-epoch=09-val_psnr=29.41-val_ssim=0.8387.ckpt`) và `last.ckpt` tại `saved_models/deeplesion/LEARN_Longformer/`.
      - `69547` (`LEARN_LongNet`): ❌ **FAILED ở 00:01:54** do PyTorch CUDA OOM khi cấp phát bộ nhớ.
      - `69548` (`SOLAR_Mamba`): ❌ **FAILED tại Epoch 3 (chạy 6h04m)** do lỗi ASTRA Toolbox CUDA OOM (`createTextureObject2D malloc`). Đã lưu checkpoint đến Epoch 02 (`solar_mamba_la-epoch=02-val_psnr=28.91-val_ssim=0.7645.ckpt`) và `last.ckpt` tại `saved_models/deeplesion/SOLAR_Mamba/`.
      - `69549` (`SOLAR_Longformer`): ⚠️ **CANCELLED ngay khi start** (ExitCode 0:0, 00:00:01) do `gpu_check.sh` báo không đủ 20000MB VRAM khả dụng.
      - `69550` (`SOLAR_LongNet`): ⚠️ **CANCELLED ngay khi start** (ExitCode 0:0, 00:00:01) do `gpu_check.sh` báo không đủ 20000MB VRAM khả dụng.
    - *LIDC-IDRI (Jobs `69551` - `69556`):*
      - Cả 6 jobs `69551` -> `69556` (`LEARN_Mamba`, `LEARN_Longformer`, `LEARN_LongNet`, `SOLAR_Mamba`, `SOLAR_Longformer`, `SOLAR_LongNet`) đều bị ⚠️ **CANCELLED ngay lập tức khi khởi động** (ngày 09/09/2026) do `gpu_check.sh` kiểm tra không đủ ngưỡng 20000MB VRAM tại thời điểm đó. Chưa có tiến trình train nào được thực thi.
  - **Hướng xử lý & Tiến độ mới nhất (13/09/2026):**
    - ✅ **Đánh giá Test Benchmark cho LEARN_Mamba trên DeepLesion:** **Job ID `71392`** (`scripts/test_mamba_deeplesion.sh`) **Hoàn thành 100%**:
      - *Cung quét chuẩn LA-120°:* **PSNR = 26.69 dB**, **SSIM = 0.7132**, **RMSE = 0.0486**, **Loss = 0.0025**.
      - *Stress test LA-90°:* **PSNR = 18.59 dB**, **SSIM = 0.3913**, **RMSE = 0.1211**, **Loss = 0.0150**.
    - 📌 **TODO BẮT BUỘC 1: CẬP NHẬT KẾT QUẢ TEST & TRỰC QUAN HÓA (VISUALIZATION):**
      - [x] Ghi các chỉ số trên của `LEARN_Mamba` trên DeepLesion vào file [`reports/sep-05-2026/benchmark_results.csv`](reports/sep-05-2026/benchmark_results.csv) phục vụ báo cáo.
      - [ ] Chạy pipeline tạo ảnh trực quan hóa (Visualization) đối sánh ảnh tái tạo của LEARN_Mamba trên tập DeepLesion.
    - 📌 **TODO BẮT BUỘC 2: CHẠY LẠI (RESUBMIT) TOÀN BỘ CÁC JOB HUẤN LUYỆN ĐA DATASET:**
      - **A. Tập dữ liệu NIH DeepLesion (5 mô hình cần chạy lại sau khi LEARN_Mamba đã hoàn thành 50 epochs):**
        - [ ] `LEARN_Longformer` (`scripts/train_longformer_deeplesion.sh`): Tự động resume từ Checkpoint Epoch 09 (`last.ckpt`), áp dụng cơ chế dọn cache ASTRA hoặc cấu hình VRAM tối ưu chống OOM.
        - [ ] `LEARN_LongNet` (`scripts/train_longnet_deeplesion.sh`): Tối ưu VRAM và submit lại.
        - [ ] `SOLAR_Mamba` (`scripts/train_solar_mamba_deeplesion.sh`): Tự động resume từ Checkpoint Epoch 02 (`last.ckpt`).
        - [ ] `SOLAR_Longformer` (`scripts/train_solar_longformer_deeplesion.sh`): Hạ `REQUIRED_VRAM=15000` và submit lại.
        - [ ] `SOLAR_LongNet` (`scripts/train_solar_longnet_deeplesion.sh`): Hạ `REQUIRED_VRAM=15000` và submit lại.
      - **B. Tập dữ liệu LIDC-IDRI (Cả 6 mô hình cần submit lại sau khi tối ưu VRAM):**
        - [ ] Cập nhật 6 script sbatch `scripts/train_*_lidc.sh`: Điều chỉnh `REQUIRED_VRAM` từ 20000MB xuống 12000MB - 15000MB (tránh bị hủy job oan tương tự như AAPM).
        - [ ] Submit lần lượt 6 mô hình khi có slot GPU:
          - [ ] `LEARN_Mamba` (`scripts/train_mamba_lidc.sh`)
          - [ ] `LEARN_Longformer` (`scripts/train_longformer_lidc.sh`)
          - [ ] `LEARN_LongNet` (`scripts/train_longnet_lidc.sh`)
          - [ ] `SOLAR_Mamba` (`scripts/train_solar_mamba_lidc.sh`)
          - [ ] `SOLAR_Longformer` (`scripts/train_solar_longformer_lidc.sh`)
          - [ ] `SOLAR_LongNet` (`scripts/train_solar_longnet_lidc.sh`)

- [ ] **4. LƯU Ý ĐẶC BIỆT VỀ BÀI BÁO SOICT 2026 (`papers/soict2026/main.tex`):**
  - > [!WARNING]
    > **TUYỆT ĐỐI KHÔNG ĐƯA KẾT QUẢ CỦA MÔ HÌNH SOLAR VÀO BÀI BÁO SOICT 2026.**
    > - Bài báo SOICT 2026 chỉ so sánh đối chứng các mô hình Baseline unrolling bậc 1 (`LEARN_Longformer`, `LEARN_LongNet`, `LEARN_Mamba`, và `FBP`).
    > - Toàn bộ bảng Table 1 và Table 2 trong `main.tex` đã được chốt và đồng bộ 100% với baseline hoàn chỉnh.
    > - Kiến trúc đề xuất **SOLAR** cùng kết quả test vượt trội của nó được giữ bí mật để phục vụ riêng cho bài báo Journal Q1 đỉnh cao (IEEE TMI / MedIA, IF > 10).
  - Nhiệm vụ còn lại của bài SOICT 2026: Biên dịch PDF trên Overleaf/TeX Live, kiểm tra số trang quy định (12-15 trang) và rà soát định dạng Springer LNCS.

- [ ] **5. THEO DÕI TIẾN ĐỘ & ĐÁNH GIÁ 3 SLURM JOBS BASELINE MỚI (LEARN, REGFORMER, DUDOTRANS):**
  - **Bối cảnh & Mã nguồn:** Kế thừa công trình nghiên cứu của Thành (`Thanhld`) từ paper MVA sang bài toán Limited-Angle CT (LA-120°). Đã triển khai hoàn tất tại:
    - [`baselines/LEARN/`](baselines/LEARN/): Model mở cuộn 14 stages với CNN 3 tầng thuần túy (841,372 params).
    - [`baselines/RegFormer/`](baselines/RegFormer/): Model điều hòa kép Local CNN + Non-local Swin Transformer (1,228,948 params).
    - [`baselines/DuDoTrans/`](baselines/DuDoTrans/): Model biến đổi đa miền Sinogram Transformer + FBP vi phân + Image Refinement (129,554 params).
  - **Danh sách Job ID đang chạy (Đã Resubmit ngày 13/09/2026 sau khi tối ưu REQUIRED_VRAM):**
    - **`LEARN`:** Job ID **`71393`** (`scripts/train_learn_la.sh` - VRAM=12GB) 🟢 **RUNNING trên DGX-A100**.
    - **`RegFormer`:** Job ID **`71394`** (`scripts/train_regformer_la.sh` - VRAM=15GB) 🟢 **RUNNING trên DGX-A100**.
    - **`DuDoTrans`:** Job ID **`71395`** (`scripts/train_dudotrans_la.sh` - VRAM=12GB) 🟢 **RUNNING trên DGX-A100**.
  - **Hướng dẫn & Hành động kế tiếp:**
    1. Theo dõi tiến độ huấn luyện thời gian thực qua log tại `scripts/output/train_<model>_la/log/%j.out`.
    2. Khi job hoàn thành (`COMPLETED`), chạy đánh giá Test trên Patient L310 qua các script:
       ```bash
       python baselines/LEARN/test_learn_la.py --ckpt_path <path_to_best_model>
       python baselines/RegFormer/test_regformer_la.py --ckpt_path <path_to_best_model>
       python baselines/DuDoTrans/test_dudotrans_la.py --ckpt_path <path_to_best_model>
       ```
    3. Cập nhật số liệu định lượng (PSNR, SSIM, RMSE) vào [`reports/sep-05-2026/benchmark_results.csv`](reports/sep-05-2026/benchmark_results.csv) và [`EXPERIMENT_RESULTS.md`](EXPERIMENT_RESULTS.md).


