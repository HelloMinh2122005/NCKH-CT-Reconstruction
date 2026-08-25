#!/bin/bash
#SBATCH --job-name=train_longnet_la
#SBATCH --output=/datastore/uittogether3/LuuTru/MinhPD/scripts/output/train_longnet_la/log/%j.out
#SBATCH --error=/datastore/uittogether3/LuuTru/MinhPD/scripts/output/train_longnet_la/log/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=mps:a100:2
#SBATCH --time=24:00:00

set -euo pipefail

REQUIRED_VRAM=25000

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

module clear -f
module load slurm/slurm/24.11

source /datastore/uittogether3/tools/miniconda3/etc/profile.d/conda.sh

export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"

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

# ================= RUN TRAINING =================
echo "[INFO] Launching LEARN_LongNet Training on Limited-Angle CT at $(date)"

cd /datastore/uittogether3/LuuTru/MinhPD

python -u baselines/LEARN_LongNet/train_longnet_la.py \
    --dataset_dir /datastore/uittogether3/LuuTru/MinhPD/dataset/limited_angle/ \
    --output_dir /datastore/uittogether3/LuuTru/MinhPD/saved_models/LEARN_LongNet/ \
    --angle_range_deg 120.0 \
    --num_view 64 \
    --num_detectors 512 \
    --input_size 256 \
    --window_size 2 \
    --poisson_level 0 \
    --gaussian_level 0 \
    --batch_size 1 \
    --n_iterations 14 \
    --max_epochs 50 \
    --lr 1e-4 \
    --num_workers 4 \
    --use_precomputed

echo "[INFO] LEARN_LongNet Training finished at $(date)"
