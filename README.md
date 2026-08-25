# Nghiên Cứu Khoa Học: Tái Tạo Ảnh Cắt Lớp CT Góc Giới Hạn (Limited-Angle CT Reconstruction)

Không gian làm việc nghiên cứu và tái lập mô hình của **MinhPD** trên cụm máy chủ tính toán hiệu năng cao Slurm HPC (GPU NVIDIA A100 / L40) - Trường Đại học Công nghệ Thông tin (VNU-HCM UIT).

---

## 1. Môi Trường Tính Toán & Cụm Slurm HPC (Đã Đáp Ứng 100% Đầy Đủ)

> [!IMPORTANT]
> **Môi trường Conda trên Slurm Cluster đã được cấu hình sẵn 100% đầy đủ toàn bộ thư viện.**  
> Khi chạy các job trên máy chủ tính toán, bạn **KHÔNG CẦN** chạy bất kỳ lệnh `pip install` nào. Môi trường Conda chuẩn tại:
> ```bash
> source /datastore/uittogether3/tools/miniconda3/etc/profile.d/conda.sh
> conda activate /datastore/uittogether3/tools/miniconda3/envs/LongNet
> ```
> Môi trường này đã cài đặt sẵn toàn bộ các binary/CUDA C++ extensions tương thích tối đa với phần cứng A100 (`torch==2.8.0+cu128`, `mamba-ssm==2.3.1`, `transformers==4.57.6`, `odl==0.8.3`, `astra-toolbox==2.3.0`, `torchmetrics==1.8.2`, `pydicom==2.4.4`).

---

## 2. Danh Mục Dataset Gốc Tham Chiếu

### 1. AAPM Mayo Clinic Low Dose CT (LDCT) Grand Challenge
- **Giới thiệu:** Bộ dữ liệu chụp CT cắt lớp chuẩn lâm sàng từ Mayo Clinic (Grand Challenge 2016). Bao gồm các lát cắt CT ngực/bụng độ dày 3mm (`full_3mm`) định dạng DICOM (`.IMA`) của 10 bệnh nhân (9 train: `L067`, `L096`, `L109`, `L143`, `L192`, `L286`, `L291`, `L333`, `L506` và 1 test: `L310`).
- **Đường dẫn Slurm Datastore:** `/datastore/uittogether3/LuuTru/Thanhld/CT-Reconstruction/split/`
- **Đường dẫn Local (SSHFS):** `uittogether3-slurm-server/Thanhld/CT-Reconstruction/split/`

### 2. NIH DeepLesion CT Dataset
- **Giới thiệu:** Bộ dữ liệu CT quy mô lớn từ Viện Y tế Quốc gia Hoa Kỳ (NIH Clinical Center) với hơn 32.000 lát cắt CT trục (axial slices) kèm metadata `DL_info.csv`.
- **Đường dẫn Slurm Datastore:** `/datastore/uittogether3/LuuTru/Thanhld/CT-Reconstruction/minideeplesion/`

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
├── scripts/                           # Tập hợp các batch script chạy job Slurm chuẩn MPS
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

## 4. Hướng Dẫn Chạy Job Trên Slurm Cluster

```bash
# 1. Sinh dữ liệu Limited-Angle CT (LA-120° & LA-90°)
sbatch scripts/generate_la_dataset.sh

# 2. Huấn luyện Baseline LEARN_Mamba
sbatch scripts/train_mamba_la.sh

# 3. Huấn luyện Baseline LEARN_Longformer
sbatch scripts/train_longformer_la.sh

# 4. Huấn luyện Baseline LEARN_LongNet
sbatch scripts/train_longnet_la.sh

# Theo dõi log real-time
tail -f scripts/output/<tên_script>/log/<job_id>.out
```
