#!/bin/bash
#SBATCH --job-name=tr_dudo_deeplesion
#SBATCH --output=/datastore/uittogether3/LuuTru/MinhPD/scripts/output/train_dudotrans_deeplesion/log/%j.out
#SBATCH --error=/datastore/uittogether3/LuuTru/MinhPD/scripts/output/train_dudotrans_deeplesion/log/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=mps:a100:2
#SBATCH --time=24:00:00

# ==============================================================================
# SCRIPT HUẤN LUYỆN: DuDoTrans trên NIH DeepLesion CT
# Bài toán: Tái tạo ảnh cắt lớp CT góc giới hạn (Limited-Angle CT Reconstruction)
# Cung quét chuẩn: LA-120° (64 views, 512 detectors, 256x256, noise_0)
# Tác giả: MinhPD - VNU-HCM UIT
# Cụm máy chủ: Slurm HPC GPU A100/L40 với NVIDIA MPS
# ==============================================================================

set -euo pipefail

# Ngưỡng VRAM yêu cầu tối thiểu (MB)
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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ================= GPU CHECK (Admin Policy) =================
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
export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-job${SLURM_JOB_ID}"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-mps-log-job${SLURM_JOB_ID}"

rm -rf "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"

export CUDA_VISIBLE_DEVICES="${BEST_GPU}"

# ================= RUN TRAINING =================
echo "[INFO] Launching DuDoTrans Training on NIH DeepLesion CT at $(date)"

cd /datastore/uittogether3/LuuTru/MinhPD
export PYTHONPATH="/datastore/uittogether3/LuuTru/MinhPD:${PYTHONPATH:-}"

RESUME_CKPT="/datastore/uittogether3/LuuTru/MinhPD/saved_models/deeplesion/DuDoTrans/DuDoTrans_DEEPLESION_LA_120deg_view64/last.ckpt"
EXTRA_ARGS=""
if [ -f "$RESUME_CKPT" ]; then
    echo "[INFO] Tìm thấy checkpoint để resume: $RESUME_CKPT"
    EXTRA_ARGS="--resume_ckpt $RESUME_CKPT"
fi

python -u baselines/DuDoTrans/train_dudotrans_la.py \
    --dataset_type deeplesion \
    --dataset_dir /datastore/uittogether3/LuuTru/MinhPD/dataset/nih_deep_lesion/limited_angle/ \
    --output_dir /datastore/uittogether3/LuuTru/MinhPD/saved_models/deeplesion/DuDoTrans/ \
    --angle_range_deg 120.0 \
    --num_view 64 \
    --num_detectors 512 \
    --input_size 256 \
    --poisson_level 0 \
    --gaussian_level 0 \
    --batch_size 1 \
    --feature_dim 48 \
    --lambda_sino 0.1 \
    --max_epochs 50 \
    --lr 1e-4 \
    --num_workers 4 \
    --use_precomputed \
    $EXTRA_ARGS

echo "[INFO] DuDoTrans Training on NIH DeepLesion CT finished at $(date)"
