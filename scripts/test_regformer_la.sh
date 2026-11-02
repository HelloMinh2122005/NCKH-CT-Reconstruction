#!/bin/bash
# ==============================================================================
# SCRIPT ĐÁNH GIÁ (TEST BENCHMARK) MÔ HÌNH BASELINE REGFORMER (Local CNN + Swin Transformer)
# Đề tài: Tái tạo ảnh cắt lớp CT góc giới hạn (Limited-Angle CT Reconstruction)
# Tác giả: MinhPD - VNU-HCM UIT
# Cụm máy chủ: Slurm HPC GPU A100/L40 với NVIDIA MPS
# Cập nhật HPC-UIT (MIG): Partition small (gpu:1g.18gb:1, 4 CPU, 32GB RAM)
# ==============================================================================
#SBATCH --job-name=test_regformer_la
#SBATCH --output=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output/test_regformer_la/log/%j.out
#SBATCH --error=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output/test_regformer_la/log/%j.err
#SBATCH --account=uit
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1g.18gb:1

set -euo pipefail

# Ngưỡng VRAM yêu cầu tối thiểu (MB) cho quá trình Test suy luận (Inference)
REQUIRED_VRAM=12000

REPO=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction
ENV_DIR=/datastore/doanhbc/minhdphan/envs/longnet
DATASETS=/datastore/doanhbc/minhdphan/datasets

mkdir -p "$REPO/scripts/output/test_regformer_la/log"
mkdir -p "$REPO/reports"

# Hàm dọn dẹp tài nguyên NVIDIA MPS khi kết thúc job hoặc bị ngắt đột ngột
cleanup() {
    local rc=$?
    echo "[INFO] cleanup rc=$rc at $(date)"
}
trap cleanup EXIT

echo "=== HPC-UIT TESTING: RegFormer ==="
echo "[INFO] start at $(date)"
echo "[INFO] hostname=$(hostname)"
echo "[INFO] SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"
echo "[INFO] SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-<unset>}"
echo "[INFO] Git commit: $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo '<git-not-found>')"

# Khởi tạo môi trường Module Slurm
# Trên HPC-UIT Slurm được nạp sẵn qua hệ thống quản lý tác vụ
nvidia-smi

# ================= GPU CHECK (Admin Policy) =================
# Kiểm tra tài nguyên GPU và VRAM khả dụng theo quy định của Quản trị hệ thống
# ================= NVIDIA MPS CONFIGURATION =================
# Cấu hình đường dẫn pipe và log riêng biệt cho tiến trình MPS của Job này
# Lưu ý: Trên HPC-UIT MIG, việc cô lập tài nguyên được đảm bảo qua phần cứng GPU MIG

# Kích hoạt môi trường Conda chuyên dụng của dự án
set +u
source "$ENV_DIR/bin/activate"
set -u

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Chuyển hướng toàn bộ cache về /datastore
export XDG_CACHE_HOME=/datastore/doanhbc/minhdphan/cache
export PIP_CACHE_DIR=$XDG_CACHE_HOME/pip
export TORCH_HOME=$XDG_CACHE_HOME/torch
export HF_HOME=$XDG_CACHE_HOME/hf
export TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton

# ================= CẤU HÌNH GÓC QUÉT & CHECKPOINT =================
# Tham số hóa góc quét: mặc định 90 độ, có thể cấu hình 120 độ qua đối số $1
ANGLE="${1:-90}"
ANGLE_DEG="${ANGLE}.0"
echo "[INFO] Cấu hình góc quét đánh giá: ${ANGLE_DEG}° (Views: 64)"

# Thư mục chứa checkpoint của mô hình
CKPT_DIR="$REPO/saved_models/RegFormer_${ANGLE}"
if [ ! -d "$CKPT_DIR" ]; then
    echo "[ERROR] Thư mục checkpoint không tồn tại: $CKPT_DIR"
    exit 1
fi

# Tập kiểm thử độc lập: L058, L064, L094, L107 (4 bệnh nhân, 757 lát cắt) theo split.md
TEST_PATIENTS=("L058" "L064" "L094" "L107")
CSV_OUTPUT="$REPO/reports/benchmark_per_slice_results.csv"

# ================= TIẾN HÀNH ĐÁNH GIÁ (TESTING) =================
# Xác định Checkpoint tốt nhất (ưu tiên epoch=46 đỉnh val_psnr=36.0533, sau đó đến last.ckpt)
# Quy tắc T5 & Q7: Tuyệt đối chỉ đánh giá checkpoint tốt nhất theo val_psnr trên tập L110, dừng và báo lỗi nếu không có (KHÔNG fallback sang last.ckpt).
echo "[INFO] Launching RegFormer Testing on Limited-Angle CT at $(date)"
echo "[INFO] Checkpoint directory: $CKPT_DIR"
echo "[INFO] Test patients: ${TEST_PATIENTS[*]}"
echo "[INFO] Output CSV: $CSV_OUTPUT"

python -u baselines/RegFormer/test_regformer_la.py \
    --ckpt_path "$CKPT_DIR" \
    --dataset_type aapm \
    --dicom_dir "$DATASETS/aapm/split/" \
    --cache_dir "$DATASETS/aapm/limited_angle/" \
    --angle_range_deg "$ANGLE_DEG" \
    --num_view 64 \
    --num_detectors 512 \
    --input_size 256 \
    --poisson_level 0 \
    --gaussian_level 0 \
    --batch_size 1 \
    --num_workers 4 \
    --test_patients "${TEST_PATIENTS[@]}" \
    --output_csv "$CSV_OUTPUT" \
    --job_id "${SLURM_JOB_ID:-local}" \
    --method "RegFormer"

echo "[INFO] RegFormer Testing finished at $(date)"
