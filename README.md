# Dataset Paths & Overview

## 1. AAPM Mayo Clinic Low Dose CT (LDCT) Grand Challenge
- **Giới thiệu:** Bộ dữ liệu chụp CT cắt lớp chuẩn lâm sàng từ Mayo Clinic (Grand Challenge 2016). Bao gồm các lát cắt CT ngực/bụng độ dày 3mm (`full_3mm`) định dạng DICOM (`.IMA`) của 10 bệnh nhân (9 train: `L067`, `L096`, `L109`, `L143`, `L192`, `L286`, `L291`, `L333`, `L506` và 1 test: `L310`).
- **Đường dẫn cục bộ (Local Path):**
  `/home/phandinhminh/Downloads/kltn/agents-research/uittogether3-slurm-server/Thanhld/CT-Reconstruction/split/`
- **Đường dẫn Slurm Server (Cluster Datastore Path):**
  `/datastore/uittogether3/LuuTru/Thanhld/CT-Reconstruction/split/`

## 2. NIH DeepLesion CT Dataset
- **Giới thiệu:** Bộ dữ liệu CT quy mô lớn từ Viện Y tế Quốc gia Hoa Kỳ (NIH Clinical Center) với hơn 32.000 lát cắt CT trục (axial slices) kèm thông tin phân vùng tổn thương, định dạng `.png` và metadata `DL_info.csv`.
- **Đường dẫn cục bộ (Local Path):**
  `/home/phandinhminh/Downloads/kltn/agents-research/uittogether3-slurm-server/Thanhld/CT-Reconstruction/minideeplesion/`
- **Đường dẫn Slurm Server (Cluster Datastore Path):**
  `/datastore/uittogether3/LuuTru/Thanhld/CT-Reconstruction/minideeplesion/`
