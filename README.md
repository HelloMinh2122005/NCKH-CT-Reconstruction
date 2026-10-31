# Nghiên Cứu Khoa Học: Tái Tạo Ảnh Cắt Lớp CT Góc Giới Hạn (Limited-Angle CT Reconstruction)

Không gian làm việc nghiên cứu và tái lập mô hình của **MinhPD** trên cụm **HPC-UIT** (Slurm, GPU chia MIG) - Trường Đại học Công nghệ Thông tin (VNU-HCM UIT).

> [!WARNING]
> **26/10/2026:** cụm DGX-A100 cũ đã mất (checkpoint, dataset, môi trường). Dự án đang khởi động lại trên HPC-UIT với tập test nhiều bệnh nhân. Trạng thái hiện tại: [CHECKPOINT.md](CHECKPOINT.md); quy tắc: [AGENTS.md](AGENTS.md); thông số cụm: [SYSTEM_SPEC.md](SYSTEM_SPEC.md).

---

## 1. Môi Trường Tính Toán (HPC-UIT)

- Code, dữ liệu, môi trường, checkpoint và log nằm trong `/datastore/doanhbc/minhdphan/` (tài khoản dùng chung của thầy; không dùng `$HOME`).
- Môi trường Python sẽ được tạo tại `/datastore/doanhbc/minhdphan/envs/longnet` bằng một job Slurm `small` (chưa tạo). Các phiên bản giữ gần môi trường cũ: `torch==2.8.0+cu128`, `mamba-ssm==2.3.1`, `transformers==4.57.6`, `odl==0.8.3`, `astra-toolbox==2.3.0`, `torchmetrics==1.8.2`, `pydicom==2.4.4` ([requirements.txt](requirements.txt)).
- Không chạy `pip install` hay huấn luyện trên login node: mọi thứ chạy qua Slurm. Chi tiết phân vùng GPU và giới hạn: [SYSTEM_SPEC.md](SYSTEM_SPEC.md) §4.

---

## 2. Danh Mục Dataset Gốc Tham Chiếu

### 1. AAPM Mayo Clinic Low Dose CT (LDCT) Grand Challenge
- **Giới thiệu:** Bộ dữ liệu chụp CT cắt lớp chuẩn lâm sàng từ Mayo Clinic (Grand Challenge 2016). Bao gồm các lát cắt CT ngực/bụng độ dày 3mm (`full_3mm`) định dạng DICOM (`.IMA`) của 10 bệnh nhân (9 train: `L067`, `L096`, `L109`, `L143`, `L192`, `L286`, `L291`, `L333`, `L506` và 1 test: `L310`).
- **Đường dẫn (HPC-UIT):** `/datastore/doanhbc/minhdphan/datasets/aapm/` (tải lại qua TCIA *LDCT-and-Projection-data*; bản cũ đã mất).
- **Phân chia:** phân chia cũ (test chỉ `L310`) đã bị loại bỏ; tập test mới gồm nhiều bệnh nhân.

### 2. NIH DeepLesion CT Dataset
- **Giới thiệu:** Bộ dữ liệu CT quy mô lớn từ Viện Y tế Quốc gia Hoa Kỳ (NIH Clinical Center) với hơn 32.000 lát cắt CT trục (axial slices) kèm metadata `DL_info.csv`.
- **Đường dẫn (HPC-UIT):** `/datastore/doanhbc/minhdphan/datasets/deeplesion/` (mở rộng, Tier 2; bản cũ đã mất).

---

## 3. Cấu Trúc Dự Án & Các Mô Hình Baseline

```text
MinhPD/
├── baselines/                         # 3 mô hình Baseline tái lập từ paper của Thành (MVA)
│   ├── README.md                      # Báo cáo chi tiết & hướng dẫn chạy baseline
│   ├── LEARN_Mamba/                   # Baseline 1: LEARN + Selective SSM (Mamba)
│   ├── LEARN_Longformer/              # Baseline 2: LEARN + Longformer Attention
│   └── LEARN_LongNet/                 # Baseline 3: LEARN + LongNet Dilated Attention
├── data/                              # Module nạp và tiền xử lý dữ liệu Fan-Beam góc giới hạn
│   ├── CTSlice_Provider_LA.py         # Provider nạp DICOM & cache .npy
│   ├── datamodule_LA.py               # PyTorch Lightning DataModule
│   └── prepare_data_sinogram_LA.py    # Script sinh cache offline
├── scripts/                           # Batch script Slurm (bản cũ theo A100/MPS, đang port sang HPC-UIT MIG)
│   ├── generate_la_dataset.sh         # Job sinh dữ liệu Limited-Angle CT
│   ├── train_mamba_la.sh              # Job huấn luyện LEARN_Mamba trên LA-CT
│   ├── train_longformer_la.sh         # Job huấn luyện LEARN_Longformer trên LA-CT
│   └── train_longnet_la.sh            # Job huấn luyện LEARN_LongNet trên LA-CT
├── note/                              # Tài liệu phân tích kiến trúc MVA vs SOLAR
├── requirements.txt                   # Danh mục toàn bộ thư viện kỹ thuật của dự án
├── AGENTS.md                          # Quy ước dành cho AI Agent
└── CHECKPOINT.md                      # Bản theo dõi tiến độ và trạng thái chi tiết của dự án
```

---

## 4. Hướng Dẫn Chạy Job Trên Slurm Cluster (HPC-UIT)

Script trong `scripts/` phải được port sang template HPC-UIT và qua `slurm_lint` trước khi chạy. Job được nộp từ thư mục repo trên server (qua hpc-operator, mỗi lần nộp cách nhau ≥ 1 giờ):

```bash
# Trên server, trong thư mục làm việc
cd /datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction
git pull --ff-only            # cập nhật code từ GitHub (không sửa trực tiếp trên server)
sbatch scripts/<tên_script>.sh # nộp job
squeue -u $USER               # theo dõi
tail -n 60 scripts/output/<tên_script>/log/<job_id>.out
```

Log mới kéo về local nằm trong `scripts/output-hpc-uit/`; `scripts/output/` ở local chỉ chứa log cũ của cụm A100.
