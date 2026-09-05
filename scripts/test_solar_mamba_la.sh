#!/bin/bash
#SBATCH --job-name=test_solar_mamba_la
#SBATCH --output=/datastore/uittogether3/LuuTru/MinhPD/scripts/output/test_solar_mamba_la/log/%j.out
#SBATCH --error=/datastore/uittogether3/LuuTru/MinhPD/scripts/output/test_solar_mamba_la/log/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=mps:a100:2
#SBATCH --time=02:00:00

# ==============================================================================
# SCRIPT ĐÁNH GIÁ (TEST BENCHMARK) MÔ HÌNH ĐỀ XUẤT SOLAR_MAMBA
# Đề tài: Tái tạo ảnh cắt lớp CT góc giới hạn (Limited-Angle CT Reconstruction)
# Tác giả: MinhPD - VNU-HCM UIT
# Cụm máy chủ: Slurm HPC GPU A100/L40 với NVIDIA MPS
# ==============================================================================

set -euo pipefail

# Ngưỡng VRAM yêu cầu tối thiểu (MB) cho quá trình Test suy luận (Inference)
REQUIRED_VRAM=15000

# Hàm dọn dẹp tài nguyên NVIDIA MPS khi kết thúc job hoặc bị ngắt đột ngột
cleanup() {
    local rc=$?
    echo "[INFO] cleanup rc=$rc at $(date)"
    if [ -n "${CUDA_MPS_PIPE_DIRECTORY:-}" ]; then
        rm -rf "${CUDA_MPS_PIPE_DIRECTORY}" 2>/dev/null || true
    fi
    if [ -n "${CUDA_MPS_LOG_DIRECTORY:-}" ]; then
        rm -rf "${CUDA_MPS_LOG_DIRECTORY}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "[INFO] start at $(date)"
echo "[INFO] hostname=$(hostname)"
echo "[INFO] SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"

# Khởi tạo môi trường Module Slurm
module clear -f
module load slurm/slurm/24.11

# Kích hoạt môi trường Conda chuyên dụng của dự án
source /datastore/uittogether3/tools/miniconda3/etc/profile.d/conda.sh

export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"

set +u
conda activate /datastore/uittogether3/tools/miniconda3/envs/LongNet
set -u

# ================= GPU CHECK (Admin Policy) =================
# Kiểm tra tài nguyên GPU và VRAM khả dụng theo quy định của Quản trị hệ thống
unset CUDA_VISIBLE_DEVICES

set +e
CHECK_OUT=$(/usr/local/bin/gpu_check.sh "$REQUIRED_VRAM" "$SLURM_JOB_ID" 2>&1)
EXIT_CODE=$?
set -e

echo "[INFO] gpu_check exit_code=$EXIT_CODE"
echo "[INFO] gpu_check output=$CHECK_OUT"

if [ "$EXIT_CODE" -eq 10 ]; then
    echo "$CHECK_OUT"
    exit 0
elif [ "$EXIT_CODE" -eq 11 ]; then
    echo "$CHECK_OUT"
    exit 1
elif [ "$EXIT_CODE" -ne 0 ]; then
    echo "[ERROR] gpu_check.sh returned unexpected exit code: $EXIT_CODE"
    exit "$EXIT_CODE"
fi

BEST_GPU="$CHECK_OUT"
echo "[INFO] BEST_GPU=$BEST_GPU"

# ================= NVIDIA MPS CONFIGURATION =================
# Cấu hình đường dẫn pipe và log riêng biệt cho tiến trình MPS của Job này
export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-job${SLURM_JOB_ID}"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-mps-log-job${SLURM_JOB_ID}"

rm -rf "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"

export CUDA_VISIBLE_DEVICES="${BEST_GPU}"

# ================= TIẾN HÀNH ĐÁNH GIÁ (TESTING) =================
echo "[INFO] Launching SOLAR_Mamba Testing on Limited-Angle CT at $(date)"

cd /datastore/uittogether3/LuuTru/MinhPD
export PYTHONPATH="/datastore/uittogether3/LuuTru/MinhPD:${PYTHONPATH:-}"

# Xác định Checkpoint tốt nhất (ưu tiên epoch=25 đạt đỉnh PSNR 33.19 dB, sau đó đến last.ckpt)
CKPT_PATH="/datastore/uittogether3/LuuTru/MinhPD/saved_models/SOLAR_Mamba/solar_mamba_la-epoch=25-val_psnr=33.19-val_ssim=0.8975.ckpt"
if [ ! -f "$CKPT_PATH" ]; then
    CKPT_PATH="/datastore/uittogether3/LuuTru/MinhPD/saved_models/SOLAR_Mamba/last.ckpt"
fi
echo "[INFO] Selected Checkpoint: $CKPT_PATH"

echo "================================================================================"
echo "🎯 ĐÁNH GIÁ 1: SOLAR_Mamba trên Cấu hình Chuẩn LA-120° (64 views)"
echo "   (Tập kiểm thử độc lập Patient L310: 214 lát cắt CT)"
echo "================================================================================"
python -u baselines/SOLAR_Mamba/test_solar_mamba_la.py \
    --checkpoint_path "$CKPT_PATH" \
    --dataset_dir /datastore/uittogether3/LuuTru/MinhPD/dataset/limited_angle/ \
    --angle_range_deg 120.0 \
    --num_view 64 \
    --num_detectors 512 \
    --input_size 256 \
    --poisson_level 0 \
    --gaussian_level 0 \
    --batch_size 1 \
    --num_workers 4 \
    --test_patients L310

echo "================================================================================"
echo "🎯 ĐÁNH GIÁ 2: SOLAR_Mamba trên Cấu hình Khắc nghiệt LA-90° (64 views)"
echo "   (Đánh giá khả năng bù đắp góc khuyết mở rộng 270° của Selective SSM & Newton-CG)"
echo "================================================================================"
python -u baselines/SOLAR_Mamba/test_solar_mamba_la.py \
    --checkpoint_path "$CKPT_PATH" \
    --dataset_dir /datastore/uittogether3/LuuTru/MinhPD/dataset/limited_angle/ \
    --angle_range_deg 90.0 \
    --num_view 64 \
    --num_detectors 512 \
    --input_size 256 \
    --poisson_level 0 \
    --gaussian_level 0 \
    --batch_size 1 \
    --num_workers 4 \
    --test_patients L310

echo "[INFO] SOLAR_Mamba Testing finished at $(date)"
