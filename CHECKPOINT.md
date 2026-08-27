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
- [x] **Bổ sung chú thích (Comments & Docstrings) chi tiết 100%:** Cho từng hàm, từng khối logic toán học, từng phép toán reshape/permute và từng tham số trong toàn bộ 3 thư mục baseline.
- [x] **Chuẩn hóa cấu hình Requirements:** Xóa các file requirements con thừa, tập trung vào [requirements.txt](file:///home/phandinhminh/Downloads/kltn/agents-research/uittogether3-slurm-server/MinhPD/requirements.txt) ở thư mục gốc.

- [x] **Sinh dữ liệu hoàn tất & Nghiệm thu thành công (Job ID `64295`):** Đã tạo đầy đủ cả 2 bộ dữ liệu `120deg` và `90deg` (Train: 1,920 slices, Test: 214 slices) tại `/datastore/uittogether3/LuuTru/MinhPD/dataset/limited_angle/`.
- [x] **Chuẩn hóa DataModule & Runner Scripts:** Tách biệt rõ đường dẫn DICOM gốc (`split/`) và Cache Sinogram/FBP (`dataset/limited_angle/`), đồng bộ 100% các đối số CLI giữa bash script và Python code.
- [x] **Submit 3 Slurm Job huấn luyện Baseline (120-degree LA-CT):**
  - **`LEARN_Mamba`:** Job ID `65477` (`scripts/output/train_mamba_la/log/%j.out`)
  - **`LEARN_Longformer`:** Job ID `65478` (`scripts/output/train_longformer_la/log/%j.out`)
  - **`LEARN_LongNet`:** Job ID `65479` (`scripts/output/train_longnet_la/log/%j.out`)

### Các bước tiếp theo:
- [ ] Giám sát tiến độ huấn luyện của 3 Job Baseline (`65477`, `65478`, `65479`) trên Slurm cluster.
- [ ] Thiết kế kiến trúc mô hình mới đề xuất **SOLAR** (Second-Order Dual-Branch Newton-CG Unrolling Network).
- [ ] Đánh giá đối sánh kết quả Benchmark giữa các baseline và mô hình đề xuất (PSNR, SSIM, RMSE, Visual Reconstruction).
