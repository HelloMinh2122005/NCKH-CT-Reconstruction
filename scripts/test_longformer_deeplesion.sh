#!/bin/bash
#SBATCH --job-name=test_longformer_dl
#SBATCH --output=/datastore/uittogether3/LuuTru/MinhPD/scripts/output/test_longformer_deeplesion/log/%j.out
#SBATCH --error=/datastore/uittogether3/LuuTru/MinhPD/scripts/output/test_longformer_deeplesion/log/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=mps:a100:2
#SBATCH --time=01:00:00

# ==============================================================================
# SCRIPT ĐÁNH GIÁ (TEST BENCHMARK): LEARN_Longformer trên NIH DeepLesion CT
# Cung quét chuẩn: LA-120° & Cung khắc nghiệt: LA-90° (64 views, 512 detectors, 256x256, noise_0)
# Tác giả: MinhPD - VNU-HCM UIT
# Cụm máy chủ: Slurm HPC GPU A100/L40 với NVIDIA MPS
# ==============================================================================

set -euo pipefail

# Ngưỡng VRAM yêu cầu tối thiểu (MB)
REQUIRED_VRAM=12000

# Hàm dọn dẹp tài nguyên NVIDIA MPS
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
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

set +u
conda activate /datastore/uittogether3/tools/miniconda3/envs/LongNet
set -u

# ================= GPU CHECK =================
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

# ================= MPS SETUP =================
export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-job${SLURM_JOB_ID}"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-mps-log-job${SLURM_JOB_ID}"

rm -rf "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"

export CUDA_VISIBLE_DEVICES="${BEST_GPU}"

# ================= RUN TESTING =================
echo "[INFO] Launching LEARN_Longformer Testing on NIH DeepLesion CT at $(date)"

cd /datastore/uittogether3/LuuTru/MinhPD
export PYTHONPATH="/datastore/uittogether3/LuuTru/MinhPD:${PYTHONPATH:-}"

CKPT_PATH="${1:-/datastore/uittogether3/LuuTru/MinhPD/saved_models/deeplesion/LEARN_Longformer/longformer_la-epoch=47-val_psnr=32.65-val_ssim=0.9025.ckpt}"
if [ ! -f "$CKPT_PATH" ]; then
    CKPT_PATH="/datastore/uittogether3/LuuTru/MinhPD/saved_models/deeplesion/LEARN_Longformer/last.ckpt"
fi

echo "================================================================================"
echo "🎯 ĐÁNH GIÁ 1: LEARN_Longformer trên DeepLesion LA-120° (64 views)"
echo "   Checkpoint: $CKPT_PATH"
echo "================================================================================"
python -u baselines/LEARN_Longformer/test_longformer_la.py \
    --checkpoint_path "$CKPT_PATH" \
    --dataset_type deeplesion \
    --cache_dir /datastore/uittogether3/LuuTru/MinhPD/dataset/nih_deep_lesion/limited_angle/ \
    --angle_range_deg 120.0 \
    --num_view 64 \
    --num_detectors 512 \
    --input_size 256 \
    --poisson_level 0 \
    --gaussian_level 0 \
    --batch_size 1 \
    --num_workers 4

echo "================================================================================"
echo "🎯 ĐÁNH GIÁ 2: LEARN_Longformer trên DeepLesion LA-90° (64 views)"
echo "   Checkpoint: $CKPT_PATH"
echo "================================================================================"
python -u baselines/LEARN_Longformer/test_longformer_la.py \
    --checkpoint_path "$CKPT_PATH" \
    --dataset_type deeplesion \
    --cache_dir /datastore/uittogether3/LuuTru/MinhPD/dataset/nih_deep_lesion/limited_angle/ \
    --angle_range_deg 90.0 \
    --num_view 64 \
    --num_detectors 512 \
    --input_size 256 \
    --poisson_level 0 \
    --gaussian_level 0 \
    --batch_size 1 \
    --num_workers 4

echo "[INFO] LEARN_Longformer DeepLesion Testing finished at $(date)"
