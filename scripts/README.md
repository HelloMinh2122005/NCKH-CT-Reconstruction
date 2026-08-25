# Hướng Dẫn Chạy Job Trên Hệ Thống Slurm Cluster (UIT HPC)

Tài liệu này hướng dẫn quy trình chạy job tiêu chuẩn trên cụm máy chủ tính toán hiệu năng cao (GPU A100 / L40) của trường UIT sử dụng cơ chế **NVIDIA MPS (Multi-Process Service)**.

---

## 1. Tổng quan về Cơ chế NVIDIA MPS
- **NVIDIA MPS (Multi-Process Service):** Cho phép nhiều job chia sẻ tài nguyên tính toán và bộ nhớ trên cùng một GPU vật lý một cách an toàn và tối ưu hiệu năng.
- **Lưu ý quan trọng:** Khi làm việc ở Headnode (Login node), bạn **không thể** xem trực tiếp trạng thái GPU bằng lệnh `nvidia-smi`. Toàn bộ tác vụ bắt buộc phải submit qua Slurm script.
- **Tài liệu chính thức từ trường:** [https://link.uit.edu.vn/slurm](https://link.uit.edu.vn/slurm)

---

## 2. Cấu trúc một File Slurm Job Script (`.sh`)

Mỗi file batch script gồm 3 phần chính:
1. Khai báo ràng buộc tài nguyên (`#SBATCH`)
2. Thiết lập môi trường ảo và kiểm tra VRAM (`gpu_check.sh`)
3. Lệnh thực thi chương trình

### Ví dụ Script mẫu `test_gpu.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=check_gpu
#SBATCH --output=check_gpu_%j.out
#SBATCH --error=check_gpu_%j.err
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
REQUIRED_VRAM=4096

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

## 3. Giải thích Chi tiết các Chỉ thị `#SBATCH`

| Chỉ thị | Ý nghĩa | Khuyến nghị & Lưu ý |
| :--- | :--- | :--- |
| `#SBATCH --job-name` | Tên của job | Đặt tên ngắn gọn, dễ phân biệt khi xem `squeue` |
| `#SBATCH --output` | Đường dẫn ghi file log output chuẩn | Sử dụng cú pháp `%j` để tự động gán Job ID (vd: `job_%j.out`) |
| `#SBATCH --error` | Đường dẫn ghi file log lỗi | Sử dụng cú pháp `%j` (vd: `job_%j.err`) |
| `#SBATCH --nodes=1` | Số lượng node tính toán | Luôn để `1` cho các job đơn node |
| `#SBATCH --ntasks=1` | Số lượng task chạy đồng thời | Thường đặt `1` |
| `#SBATCH --cpus-per-task` | Số lượng CPU cores cấp phát | **Khuyến nghị:** Dùng trong phạm vi `2`, `4`, hoặc `8` cores. `2` core là an toàn nhất |
| `#SBATCH --mem` | Dung lượng RAM hệ thống | Ví dụ: `4G`, `8G`, `16G` |
| `#SBATCH --gres` | Cấu hình tài nguyên MPS GPU | - `mps:2`: Dùng card **NVIDIA L40 (48GB)**<br>- `mps:a100:2`: Dùng card **NVIDIA A100-SXM4-80GB**<br>*(Mức MPS từ 1 -> 4, an toàn nhất là 2)* |
| `#SBATCH --time` | Giới hạn thời gian chạy tối đa (`HH:MM:SS`) | Khai báo sát với thời gian thực tế, thời gian càng ít job càng được ưu tiên điều phối |

---

## 4. Cơ chế Đặt gạch VRAM (Tránh OOM)

Để tránh hiện tượng nhiều job cùng chui vào 1 card gây tràn bộ nhớ GPU (Out-Of-Memory - OOM):
- Khai báo biến `REQUIRED_VRAM` (đơn vị MB).
- Ví dụ: Mô hình huấn luyện dự kiến chiếm 40GB VRAM, nên khai báo dư ra $50\text{GB} = 51200\text{ MB}$.
- Hàm `/usr/local/bin/gpu_check.sh` của Admin sẽ kiểm tra: nếu GPU còn trống $\ge$ dung lượng yêu cầu mới cho phép job chạy; nếu không đủ sẽ trả về exit code `10` để Slurm đưa job về trạng thái chờ (`Pending`/`Re-queue`).

---

## 5. Các Lệnh Quản Lý Job Thông Dụng

### Gửi và kiểm tra Job:
```bash
# Submit file script lên hàng đợi
sbatch scripts/test_gpu.sh

# Xem danh sách tất cả các job đang chạy trong cụm
squeue

# Chỉ xem các job của bạn
squeue -u $USER

# Xem kết quả output theo thời gian thực
tail -f check_gpu_<jobid>.out

# Xem log lỗi
cat check_gpu_<jobid>.err
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

## 6. Tài Liệu & Cổng Thông Tin Tham Khảo

- **Theo dõi Job trực quan (Web Portal):** [https://slurmweb.uit.edu.vn:8081/userportal](https://slurmweb.uit.edu.vn:8081/userportal)
- **Tài liệu hướng dẫn Slurm UIT (PDF):** [https://slurmweb.uit.edu.vn:8081/userportal/download/user-manual.pdf](https://slurmweb.uit.edu.vn:8081/userportal/download/user-manual.pdf)
- **Trang hướng dẫn chung:** [https://link.uit.edu.vn/slurm](https://link.uit.edu.vn/slurm)
