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
- **Toàn bộ Source code của Thành:** [`uittogether3-slurm-server/Thanhld`](file:///home/phandinhminh/Downloads/kltn/agents-research/uittogether3-slurm-server/Thanhld) (Bao gồm các mô hình `LEARN_LongNet`, `LEARN_Longformer`, `LEARN_Mamba` và các script sinh sinogram gốc).
- **Tài liệu phân tích kiến trúc MVA vs Đề xuất mới:** [ARCHITECTURE_MVA_VS_PROPOSED_SOLAR.md](file:///home/phandinhminh/Downloads/kltn/agents-research/my-research/report-to-proffessor/ARCHITECTURE_MVA_VS_PROPOSED_SOLAR.md).

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
- [x] **Chuẩn hóa cấu hình Requirements:** Xóa các file requirements con thừa, tập trung vào [requirements.txt](file:///home/phandinhminh/Downloads/kltn/agents-research/uittogether3-slurm-server/MinhPD/requirements.txt) ở thư mục gốc.
- [x] **Biên soạn Báo cáo Q&A Tạo sinh Dữ liệu & Vật lý CT (Local File):** Lưu tại [`my-research/report-to-proffessor/DATASET_GENERATION_AND_PHYSICS_QA_REPORT.md`](file:///home/phandinhminh/Downloads/kltn/agents-research/my-research/report-to-proffessor/DATASET_GENERATION_AND_PHYSICS_QA_REPORT.md) phục vụ báo cáo với Giáo sư.
- [x] **Sinh dữ liệu hoàn tất & Nghiệm thu thành công 100% (Job ID `65483`):** Đã tạo đầy đủ cả 2 bộ dữ liệu `120deg` và `90deg` (Train: 1,920 slices, Validation L333: 244 slices, Test L310: 214 slices) tại `/datastore/uittogether3/LuuTru/MinhPD/dataset/limited_angle/`.
- [x] **Nghiệm thu kết quả Huấn luyện 3 Slurm Job Baseline (120-degree LA-CT):**
  - Xem bảng chi tiết và phân tích tại: [EXPERIMENT_RESULTS.md](file:///home/phandinhminh/Downloads/kltn/agents-research/uittogether3-slurm-server/MinhPD/EXPERIMENT_RESULTS.md)
  - **`LEARN_LongNet`:** Job ID `65486` (`scripts/output/train_longnet_la/log/65486.out`) — ✅ **Hoàn thành 100% (50/50 Epochs)**. Best Val: **PSNR = 33.37 dB**, **SSIM = 0.9090** (Checkpoint: `longnet_la-epoch=45-val_psnr=33.37-val_ssim=0.9090.ckpt`).
  - **`LEARN_Longformer`:** Job ID `65485` / Resume Jobs `65892`, `66652` (`scripts/output/train_longformer_la/log/66652.out`) — ✅ **Hoàn thành 100% (50/50 Epochs)** trên A100 GPU (`DGX-A100`). Đạt kết quả SOTA cao nhất toàn bộ baseline: **PSNR = 34.77 dB, SSIM = 0.9383** tại Epoch 45 (Checkpoint: `longformer_la-epoch=45-val_psnr=34.77-val_ssim=0.9383.ckpt`).
  - **`LEARN_Mamba`:** Job ID `65484` (`scripts/output/train_mamba_la/log/65484.out`) — ⏱️ Chạy 41/50 Epochs (đạt Time Limit 24h). Sử dụng Best Checkpoint: `mamba_la-epoch=17-val_psnr=27.66-val_ssim=0.7373.ckpt` (do sau Epoch 20 xuất hiện mất ổn định gradient/NaN theo đúng lý thuyết).
- [x] **Đánh giá Test Benchmark Độc Lập (Patient L310 - 214 Slices) & Trực Quan Hóa:**
  - **Script Test & Batch Slurm:** `test_longnet_la.sh` (Job `65899`), `test_mamba_la.sh` (Job `65900`), `test_longformer_la.sh` (Job `67223` - Hoàn thành 100%).
  - **Kết quả Test Benchmark LEARN_Longformer (Job `67223`):**
    - *Cung quét chuẩn LA-120°:* **PSNR = 33.10 dB**, **SSIM = 0.9237**, **RMSE = 0.0224** (Cao nhất nhóm baseline).
    - *Stress test LA-90°:* **PSNR = 19.16 dB**, **SSIM = 0.6097**, **RMSE = 0.1055** (SSIM cao nhất baseline).
  - **Script Trực quan hóa & Kết xuất ảnh đối sánh:** [visualize_benchmark.py](file:///home/phandinhminh/Downloads/kltn/agents-research/uittogether3-slurm-server/MinhPD/visualize_benchmark.py) (Slurm Scripts: [scripts/visualize_benchmark.sh](file:///home/phandinhminh/Downloads/kltn/agents-research/uittogether3-slurm-server/MinhPD/scripts/visualize_benchmark.sh), [scripts/visualize_longformer_la.sh](file:///home/phandinhminh/Downloads/kltn/agents-research/uittogether3-slurm-server/MinhPD/scripts/visualize_longformer_la.sh) - Job `67225` ✅ **Hoàn thành 100%**).
  - **Thư mục ảnh PNG đã xuất:** `visualizations/120deg/` và `visualizations/90deg/` (chứa các folder `slice_050`, `slice_100`, `slice_150` với từng file ảnh PNG riêng lẻ của Ground Truth, FBP, LongNet, Mamba, Longformer và panel đối sánh đa khung hình `comparison_summary.png`).

- [x] **Hiện thực hóa 3 Biến thể Kiến trúc Đề xuất SOLAR theo Cấu trúc Baseline Chuẩn Hóa:**
  - **`baselines/SOLAR_LongNet/`:** Tối ưu hóa bậc 2 Newton-CG + Nhánh kép Res-CNN & Dilated Attention (LongNet). Script: `train_solar_longnet_la.py`, `test_solar_longnet_la.py`, `scripts/train_solar_longnet_la.sh`.
  - **`baselines/SOLAR_Longformer/`:** Tối ưu hóa bậc 2 Newton-CG + Nhánh kép Res-CNN & Sliding-Chunks Attention (Longformer). Script: `train_solar_longformer_la.py`, `test_solar_longformer_la.py`, `scripts/train_solar_longformer_la.sh`.
  - **`baselines/SOLAR_Mamba/`:** Tối ưu hóa bậc 2 Newton-CG + Nhánh kép Res-CNN & Selective SSM (Mamba). Script: `train_solar_mamba_la.py`, `test_solar_mamba_la.py`, `scripts/train_solar_mamba_la.sh`.
- [x] **Theo dõi Tiến độ Huấn luyện Các Mô hình SOLAR trên Slurm HPC (DGX-A100):**
  - **`SOLAR_LongNet` (120°):** **Job ID `66662`** (`scripts/output/train_solar_longnet_la/log/66662.out`) — 🚀 **Đang chạy ổn định (Epoch 13/50)**. Best Val hiện tại: **PSNR = 31.90 dB, SSIM = 0.8884** tại Epoch 12.
  - **`SOLAR_Mamba` (120°):** **Job ID `66664`** (`scripts/output/train_solar_mamba_la/log/66664.out`) — 🚀 **Đang chạy ổn định (Epoch 12/50)**. Best Val hiện tại: **PSNR = 31.86 dB, SSIM = 0.8775** tại Epoch 11 (khắc phục hoàn toàn lỗi NaN của LEARN_Mamba).
  - **`SOLAR_Longformer` (120°):** **Job ID `66665`** (`scripts/output/train_solar_longformer_la/log/66665.out`) — ⚠️ **Tạm dừng ở Epoch 8/50** do lỗi bộ nhớ ASTRA texture OOM. Best Val: **PSNR = 29.91 dB, SSIM = 0.8348** tại Epoch 4. Checkpoint `last.ckpt` sẵn sàng để resume.

- [x] **Biên soạn Báo cáo Tiến độ & Benchmark Định lượng (Ngày 03/09/2026):** Lưu tại [`reports/sep-03-2026/MAIN.md`](file:///home/phandinhminh/Downloads/kltn/agents-research/uittogether3-slurm-server/MinhPD/reports/sep-03-2026/MAIN.md) và bảng dữ liệu [`reports/sep-03-2026/benchmark_results.csv`](file:///home/phandinhminh/Downloads/kltn/agents-research/uittogether3-slurm-server/MinhPD/reports/sep-03-2026/benchmark_results.csv).

### Quy tắc nghiêm ngặt cho các session sau:
> [!IMPORTANT]
> **Quy tắc Bảo toàn Chú thích & Tính Toàn vẹn Mã nguồn:**
> - Tuyệt đối **KHÔNG ĐƯỢC tự ý xóa, lược bỏ, rút gọn hoặc thay đổi** bất kỳ dòng comment, docstrings tiếng Việt giải thích chi tiết nào trong toàn bộ codebase.
> - Tuyệt đối **KHÔNG ĐƯỢC viết code sai lệch, làm mâu thuẫn hoặc làm hỏng** các logic và giá trị mặc định đã được giải thích trong comment khi người dùng chưa yêu cầu rõ ràng.

### Các bước tiếp theo:
- [ ] Giám sát tiến độ hoàn tất 2 Job đang chạy trên Slurm cluster: `SOLAR_LongNet` (Job `66662`) và `SOLAR_Mamba` (Job `66664`).
- [ ] Resume lại job huấn luyện `SOLAR_Longformer` từ `last.ckpt` (Job `66665`).
- [ ] Chạy đánh giá Test Set (`test_solar_*_la.py`) và Trực quan hóa (`visualize_benchmark.py`) cho các checkpoint của 3 mô hình SOLAR.
- [ ] Tổng hợp ma trận đối sánh 1-1 toàn diện giữa 3 Baseline Bậc 1 (LEARN) và 3 Mô hình Đề xuất Bậc 2 (SOLAR).




