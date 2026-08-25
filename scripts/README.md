# Hướng Dẫn Chạy Job Trên Hệ Thống Slurm Cluster (UIT HPC)

Tài liệu này hướng dẫn quy trình chạy job tiêu chuẩn trên cụm máy chủ tính toán hiệu năng cao (GPU A100 / L40) của trường UIT sử dụng cơ chế **NVIDIA MPS (Multi-Process Service)**.

---

## 1. Quy Ước Bắt Buộc Về Tổ Chức Output / Log của các Job

> [!IMPORTANT]
> **Quy định lưu log:** Tất cả output và error log khi chạy các job qua Slurm **bắt buộc phải được định tuyến vào thư mục:**
> ```text
> scripts/output/<tên file script>/log/
> ```
> Ví dụ:
> - Script `generate_la_dataset.sh` $\rightarrow$ Logs lưu tại: `scripts/output/generate_la_dataset/log/%j.out` và `%j.err`
> - Script `test_gpu.sh` $\rightarrow$ Logs lưu tại: `scripts/output/test_gpu/log/%j.out` và `%j.err`

### Khai báo trong `#SBATCH`:
```bash
#SBATCH --output=/datastore/uittogether3/LuuTru/MinhPD/scripts/output/<tên script>/log/%j.out
#SBATCH --error=/datastore/uittogether3/LuuTru/MinhPD/scripts/output/<tên script>/log/%j.err
```

*Lưu ý: Slurm yêu cầu thư mục cha phải tồn tại trước khi khởi tạo job. Hãy chắc chắn đã chạy `mkdir -p scripts/output/<tên script>/log` trước khi submit.*

---

## 2. Tổng quan về Cơ chế NVIDIA MPS
- **NVIDIA MPS (Multi-Process Service):** Cho phép nhiều job chia sẻ tài nguyên tính toán và bộ nhớ trên cùng một GPU vật lý một cách an toàn và tối ưu hiệu năng.
- **Lưu ý:** Khi làm việc ở Headnode (Login node), bạn **không thể** xem trực tiếp trạng thái GPU bằng lệnh `nvidia-smi`. Toàn bộ tác vụ bắt buộc phải submit qua Slurm script.
- **Tài liệu chính thức từ trường:** [https://link.uit.edu.vn/slurm](https://link.uit.edu.vn/slurm)

---

## 3. Cấu trúc một File Slurm Job Script (`.sh`) Chuẩn

Mỗi file batch script gồm 3 phần chính:
1. Khai báo ràng buộc tài nguyên (`#SBATCH`)
2. Thiết lập môi trường ảo và kiểm tra VRAM (`gpu_check.sh`)
3. Lệnh thực thi chương trình

### Ví dụ Script mẫu `test_gpu.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --output=/datastore/uittogether3/LuuTru/MinhPD/scripts/output/test_gpu/log/%j.out
#SBATCH --error=/datastore/uittogether3/LuuTru/MinhPD/scripts/output/test_gpu/log/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gres=mps:a100:2
#SBATCH --time=00:05:00

# 1. Setup Environment
module clear -f
module load slurm/slurm/24.11
module load cuda12.8/toolkit/12.8.1
source /datastore/uittogether3/tools/miniconda3/etc/profile.d/conda.sh
conda activate LongNet

# 2. Khai báo VRAM dự tính (Tránh OOM)
REQUIRED_VRAM=2048

# 3. Kiểm tra GPU khả dụng (Logic Admin cung cấp)
unset CUDA_VISIBLE_DEVICES
CHECK_OUT=$(/usr/local/bin/gpu_check.sh $REQUIRED_VRAM $SLURM_JOB_ID)
EXIT_CODE=$?

if [ $EXIT_CODE -eq 10 ]; then
    echo "$CHECK_OUT"
    exit 0   # Thoát để Slurm tự động xếp hàng lại (Re-queue)
elif [ $EXIT_CODE -eq 11 ]; then
    echo "$CHECK_OUT"
    exit 1   # Lỗi hệ thống, dừng hẳn
fi

BEST_GPU=$CHECK_OUT
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-gpu$BEST_GPU
export CUDA_VISIBLE_DEVICES=$BEST_GPU

# 4. Chạy tác vụ chính
echo "Job ID: $SLURM_JOB_ID"
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

---

## 4. Giải thích Chi tiết các Chỉ thị `#SBATCH`

| Chỉ thị | Ý nghĩa | Khuyến nghị & Lưu ý |
| :--- | :--- | :--- |
| `#SBATCH --job-name` | Tên của job | Đặt tên ngắn gọn, dễ phân biệt khi xem `squeue` |
| `#SBATCH --output` | Đường dẫn ghi log output chuẩn | Định dạng: `scripts/output/<tên script>/log/%j.out` |
| `#SBATCH --error` | Đường dẫn ghi log lỗi | Định dạng: `scripts/output/<tên script>/log/%j.err` |
| `#SBATCH --nodes=1` | Số lượng node tính toán | Luôn để `1` cho các job đơn node |
| `#SBATCH --ntasks=1` | Số lượng task chạy đồng thời | Thường đặt `1` |
| `#SBATCH --cpus-per-task` | Số lượng CPU cores cấp phát | **Khuyến nghị:** Dùng trong phạm vi `2`, `4`, hoặc `8` cores. `2` core là an toàn nhất |
| `#SBATCH --mem` | Dung lượng RAM hệ thống | Ví dụ: `4G`, `8G`, `16G` |
| `#SBATCH --gres` | Cấu hình tài nguyên MPS GPU | - `mps:2`: Dùng card **NVIDIA L40 (48GB)**<br>- `mps:a100:2`: Dùng card **NVIDIA A100-SXM4-80GB**<br>*(Mức MPS từ 1 -> 4, an toàn nhất là 2)* |
| `#SBATCH --time` | Giới hạn thời gian chạy tối đa (`HH:MM:SS`) | Khai báo sát với thời gian thực tế, thời gian càng ít job càng được ưu tiên điều phối |

---

## 5. Cơ chế Đặt gạch VRAM (Tránh OOM)

Để tránh hiện tượng nhiều job cùng chui vào 1 card gây tràn bộ nhớ GPU (Out-Of-Memory - OOM):
- Khai báo biến `REQUIRED_VRAM` (đơn vị MB).
- Ví dụ: Mô hình huấn luyện dự kiến chiếm 40GB VRAM, nên khai báo dư ra $50\text{GB} = 51200\text{ MB}$.
- Hàm `/usr/local/bin/gpu_check.sh` của Admin sẽ kiểm tra: nếu GPU còn trống $\ge$ dung lượng yêu cầu mới cho phép job chạy; nếu không đủ sẽ trả về exit code `10` để Slurm đưa job về trạng thái chờ (`Pending`/`Re-queue`).

---

## 6. Các Lệnh Quản Lý Job Thông Dụng

### Gửi và kiểm tra Job:
```bash
# 1. Sinh tập dữ liệu Limited-Angle CT
sbatch scripts/generate_la_dataset.sh

# 2. Huấn luyện Baseline LEARN_Mamba trên LA-CT
sbatch scripts/train_mamba_la.sh

# 3. Huấn luyện Baseline LEARN_Longformer trên LA-CT
sbatch scripts/train_longformer_la.sh

# 4. Huấn luyện Baseline LEARN_LongNet trên LA-CT
sbatch scripts/train_longnet_la.sh

# Xem danh sách tất cả các job đang chạy trong cụm
squeue

# Chỉ xem các job của bạn
squeue -u $USER

# Xem kết quả output theo thời gian thực
tail -f scripts/output/<tên script>/log/<jobid>.out

# Xem log lỗi
cat scripts/output/<tên script>/log/<jobid>.err
```

### Hủy Job:
```bash
# Hủy một job cụ thể theo ID
scancel <jobid>       # Ví dụ: scancel 47

# Hủy toàn bộ tất cả các jobs của bạn
scancel -u $USER
```

### Kiểm tra Chi tiết & Lịch sử:
```bash
# Xem thông tin chi tiết các node máy chủ
scontrol show nodes

# Xem lịch sử và trạng thái tiêu thụ tài nguyên của các job
sacct
```

---

## 7. Tài Liệu & Cổng Thông Tin Tham Khảo

- **Theo dõi Job trực quan (Web Portal):** [https://slurmweb.uit.edu.vn:8081/userportal](https://slurmweb.uit.edu.vn:8081/userportal)
- **Tài liệu hướng dẫn Slurm UIT (PDF):** [https://slurmweb.uit.edu.vn:8081/userportal/download/user-manual.pdf](https://slurmweb.uit.edu.vn:8081/userportal/download/user-manual.pdf)
- **Trang hướng dẫn chung:** [https://link.uit.edu.vn/slurm](https://link.uit.edu.vn/slurm)
