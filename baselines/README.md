# Các Mô Hình Baseline (Replication of Thanh's Paper on Limited-Angle CT)

Thư mục này chứa toàn bộ các mô hình học sâu tái lập từ công trình nghiên cứu của **Thành (`Thanhld`)**, được chuẩn hóa và chuyển đổi từ bài toán **Sparse-View CT** sang bài toán **Limited-Angle CT (LA-CT)**.

---

## 1. Danh Mục Các Mô Hình Baseline

| Tên Baseline | Thư mục | Động cơ Attention / SSM | Mô tả đặc tính |
| :--- | :--- | :--- | :--- |
| **LEARN_Mamba** | [`LEARN_Mamba/`](./LEARN_Mamba) | **Selective SSM (State Space Model)** | Xử lý chuỗi token $2 \times 2$ ($16.384$ tokens) qua cơ chế quét song song chọn lọc $\mathcal{O}(N)$. Tốc độ huấn luyện và suy luận nhanh, tiêu thụ ít VRAM. |
| **LEARN_Longformer** | [`LEARN_Longformer/`](./LEARN_Longformer) | **Longformer Self-Attention** | Kết hợp Sliding Window Attention cục bộ ($w=256$) và $50$ Global Attention tokens phân bố đều trên chuỗi. |
| **LEARN_LongNet** | [`LEARN_LongNet/`](./LEARN_LongNet) | **Multi-Scale Dilated Attention** | Cơ chế giãn nở đa tỷ lệ (Dilated Attention) cho phép mở rộng khả năng tiếp nhận ngữ cảnh toàn cục trên chuỗi cực dài. |

---

## 2. Điểm Chuyển Đổi Kỹ Thuật (Sparse-View $\to$ Limited-Angle)

1. **Toán tử Hình học Chiếu (Fan-Beam Limited-Angle Operator):**
   - Thay đổi dải góc từ $360^\circ$ ($[0, 2\pi]$) sang dải góc giới hạn $[-60^\circ, +60^\circ]$ ($[-\pi/3, +\pi/3]$).
   - Số góc chiếu cố định $64$ views trong cung quét $120^\circ$.
   - Kích thước ảnh $256 \times 256$, $512$ detectors.
2. **Data Pipeline:**
   - Tích hợp với `LimitedAngleCTDataModule` nạp trực tiếp dữ liệu sinogram và FBP góc giới hạn đã cache dạng `.npy`.
3. **Tiêu chí Đánh giá & Giám sát:**
   - Theo dõi `val_psnr`, `val_ssim`, `val_rmse` qua từng epoch.
   - Lưu ảnh tái tạo vào TensorBoard để so sánh trực quan cấu trúc Missing Wedge.

---

## 3. Báo Cáo Kiểm Tra Môi Trường Trên Cụm Slurm GPU A100 (UIT HPC)

> [!IMPORTANT]
> **Môi trường Slurm A100/L40 của trường đã đáp ứng 100% ĐẦY ĐỦ các thư viện.**  
> Khi chạy các job trên máy chủ tính toán, bạn **KHÔNG CẦN** chạy bất kỳ lệnh `pip install` nào. Môi trường Conda chuẩn tại `/datastore/uittogether3/tools/miniconda3/envs/LongNet` đã cài đặt và cấu hình sẵn toàn bộ các binary/CUDA C++ extensions tương thích tối đa với phần cứng A100.

### Kết quả kiểm tra đối chiếu thực tế từng thư viện trên Server (`10.204.1.52`):

| Gói thư viện | Vai trò trong dự án | Trạng thái | Phiên bản đã cài trên Slurm Cluster |
| :--- | :--- | :---: | :--- |
| **`torch`** | PyTorch Core (CUDA 12.8 cho GPU A100/L40) | ✅ **[OK]** | `2.8.0+cu128` |
| **`torchvision`** | Xử lý ảnh & Grid logger | ✅ **[OK]** | `0.23.0+cu128` |
| **`pytorch-lightning`**| Quản lý Training Loop & Callbacks | ✅ **[OK]** | `2.6.0` |
| **`mamba-ssm`** | CUDA Kernel cho `LEARN_Mamba` | ✅ **[OK]** | `2.3.1` |
| **`transformers`** | `LongformerSelfAttention` cho `LEARN_Longformer` | ✅ **[OK]** | `4.57.6` |
| **`odl`** | Toán tử toán học Tomography | ✅ **[OK]** | `0.8.3` |
| **`astra-toolbox`** | GPU Accelerated CT Projection (`astra_cuda`) | ✅ **[OK]** | `2.3.0` |
| **`torchmetrics`** | Đo lường PSNR & SSIM động | ✅ **[OK]** | `1.8.2` |
| **`pydicom`** | Đọc ảnh y tế DICOM Mayo Clinic | ✅ **[OK]** | `2.4.4` |
| **`scipy` / `numpy`** | Xử lý ma trận dữ liệu | ✅ **[OK]** | `1.11.4` / `1.26.4` |
| **`matplotlib`** | Vẽ biểu đồ và hiển thị lát cắt | ✅ **[OK]** | `3.9.4` |
| **`tensorboard`** | Theo dõi Loss & hiển thị ảnh reconstruct | ✅ **[OK]** | `2.20.0` |
| **`einops`** | Biến đổi chiều tensor | ✅ **[OK]** | `0.8.2` |
| **`tqdm` / `Pillow`** | Thanh tiến trình & xử lý ảnh | ✅ **[OK]** | `4.67.3` / `11.3.0` |

### Kích hoạt môi trường trên Slurm Cluster:
```bash
source /datastore/uittogether3/tools/miniconda3/etc/profile.d/conda.sh
conda activate /datastore/uittogether3/tools/miniconda3/envs/LongNet
```
*(Các script Slurm trong `scripts/train_*.sh` đã tự động tích hợp sẵn các dòng kích hoạt này).*

### Tệp `requirements.txt` dùng khi nào?
File [requirements.txt](file:///home/phandinhminh/Downloads/kltn/agents-research/uittogether3-slurm-server/MinhPD/requirements.txt) ở thư mục gốc được dùng để làm tài liệu chuẩn hóa kỹ thuật và phục vụ cài đặt (`pip install -r requirements.txt`) nếu bạn muốn triển khai dự án trên một máy trạm cá nhân hoặc máy chủ ngoài khác.

---

## 4. Hướng Dẫn Chạy Huấn Luyện & Đánh Giá

### Chạy trực tiếp qua dòng lệnh Python:
```bash
# Huấn luyện LEARN_Mamba trên LA-120° (64 views, không nhiễu)
python baselines/LEARN_Mamba/train_mamba_la.py \
    --dataset_dir /datastore/uittogether3/LuuTru/MinhPD/dataset/limited_angle/ \
    --output_dir /datastore/uittogether3/LuuTru/MinhPD/saved_models/LEARN_Mamba/ \
    --angle_range_deg 120.0 \
    --num_view 64 \
    --batch_size 1 \
    --n_iterations 14 \
    --max_epochs 50

# Đánh giá trên tập Test
python baselines/LEARN_Mamba/test_mamba_la.py \
    --ckpt_path /datastore/uittogether3/LuuTru/MinhPD/saved_models/LEARN_Mamba/.../checkpoints/best.ckpt \
    --dataset_dir /datastore/uittogether3/LuuTru/MinhPD/dataset/limited_angle/
```

### Chạy qua Slurm Job trên Cluster UIT:
```bash
# Submit job huấn luyện LEARN_Mamba
sbatch scripts/train_mamba_la.sh

# Submit job huấn luyện LEARN_Longformer
sbatch scripts/train_longformer_la.sh

# Submit job huấn luyện LEARN_LongNet
sbatch scripts/train_longnet_la.sh
```

