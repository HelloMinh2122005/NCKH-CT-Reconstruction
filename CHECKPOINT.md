# CHECKPOINT DỰ ÁN NCKH: LIMITED-ANGLE CT RECONSTRUCTION

> **Lưu ý dành cho AI Agent:** Đọc kỹ file này ở đầu mỗi session để nắm bắt toàn bộ bối cảnh, kiến trúc, quy tắc hệ thống và tiến độ hiện tại của dự án. Luôn cập nhật file này sau khi hoàn thành các mốc công việc mới.

---

## 1. Thông Tin Chung
- **Dự án:** Nghiên cứu Khoa học (NCKH) - Tái tạo ảnh cắt lớp CT góc giới hạn (Limited-Angle CT Reconstruction)
- **Tác giả / Không gian làm việc:** `MinhPD` (`/datastore/uittogether3/LuuTru/MinhPD` / `uittogether3-slurm-server/MinhPD`)
- **GitHub Repository:** [https://github.com/HelloMinh2122005/NCKH-CT-Reconstruction.git](https://github.com/HelloMinh2122005/NCKH-CT-Reconstruction.git) (Nhánh: `main`)

---

## 2. Mục Tiêu Nghiên Cứu & Cơ Sở Khoa Học
- **Mục tiêu cốt lõi:** Cắt giảm tối đa liều bức xạ tia X chiếu vào người bệnh nhân (**Radiation Dose Reduction $50\% - 75\%$**) tuân theo nguyên tắc y học **ALARA**, đồng thời bảo vệ các cơ quan nhạy cảm với phóng xạ (tuyến giáp, mắt, tuyến vú).
- **Thách thức toán học:** Cung quét bị giới hạn $\Delta \theta < 180^\circ$ dẫn đến vùng khuyết hình nêm trong miền tần số Fourier (**Missing Wedge Problem**). Giải thuật cổ điển FBP cho ảnh bị vệt sọc nặng (heavy streak artifacts) và mất biên cấu trúc giải phẫu. AI / Deep Learning được dùng để khôi phục vùng thiếu góc này.

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
- **Môi trường Conda:** `/datastore/uittogether3/tools/miniconda3/envs/LongNet` (đầy đủ `torch`, `odl`, `astra-toolbox`, `pydicom`, `torchvision`)
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
- [x] Biên soạn tài liệu chi tiết: `data/README.md`, `scripts/README.md`, `README.md`.
- [x] Khởi tạo Git repo, chuẩn hóa `.gitignore` và push lên GitHub `HelloMinh2122005/NCKH-CT-Reconstruction`.
- [x] Submit Job Slurm sinh dữ liệu: **Job ID `64295`** (`generate_la_dataset.sh`), cấu hình log chuẩn `scripts/output/generate_la_dataset/log/`.

### Các bước tiếp theo:
- [ ] Theo dõi và nghiệm thu dữ liệu sinh ra bởi Job `64295` tại `/datastore/uittogether3/LuuTru/MinhPD/dataset/limited_angle/`.
- [ ] Thiết kế kiến trúc mô hình tái tạo Limited-Angle CT (kế thừa / cải tiến từ các baseline như FBPConvNet, LEARN, Mamba/Transformer).
- [ ] Xây dựng pipeline Training, Validation và Testing với các chỉ số đo lường (PSNR, SSIM, RMSE, Visual Comparison).
