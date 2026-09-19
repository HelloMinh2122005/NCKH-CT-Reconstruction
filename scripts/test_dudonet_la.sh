#!/bin/bash
#SBATCH --job-name=test_dudonet_la
#SBATCH --output=/datastore/uittogether3/LuuTru/MinhPD/scripts/output/test_dudonet_la/log/%j.out
#SBATCH --error=/datastore/uittogether3/LuuTru/MinhPD/scripts/output/test_dudonet_la/log/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=mps:a100:2
#SBATCH --time=02:00:00

set -euo pipefail

REQUIRED_VRAM=12000

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

set +u
conda activate /datastore/uittogether3/tools/miniconda3/envs/LongNet
set -u

# GPU check
unset CUDA_VISIBLE_DEVICES
set +e
CHECK_OUT=$(/usr/local/bin/gpu_check.sh "$REQUIRED_VRAM" "$SLURM_JOB_ID" 2>&1)
EXIT_CODE=$?
set -e

if [ "$EXIT_CODE" -ne 0 ]; then
    echo "$CHECK_OUT"
    exit "$EXIT_CODE"
fi

BEST_GPU="$CHECK_OUT"
export CUDA_VISIBLE_DEVICES="${BEST_GPU}"

cd /datastore/uittogether3/LuuTru/MinhPD
export PYTHONPATH="/datastore/uittogether3/LuuTru/MinhPD:${PYTHONPATH:-}"

CKPT_PATH="${1:-/datastore/uittogether3/LuuTru/MinhPD/saved_models/DuDoNet/last.ckpt}"

echo "================================================================================"
echo "🎯 ĐÁNH GIÁ 1: DuDoNet trên Cấu hình Chuẩn LA-120° (64 views)"
echo "================================================================================"
python -u baselines/DuDoNet/test_dudonet_la.py \
    --ckpt_path "$CKPT_PATH" \
    --dataset_type aapm \
    --angle_range_deg 120.0 \
    --num_view 64

echo "================================================================================"
echo "🎯 ĐÁNH GIÁ 2: DuDoNet trên Cấu hình Khắc nghiệt LA-90° (64 views)"
echo "================================================================================"
python -u baselines/DuDoNet/test_dudonet_la.py \
    --ckpt_path "$CKPT_PATH" \
    --dataset_type aapm \
    --angle_range_deg 90.0 \
    --num_view 64

echo "[INFO] DuDoNet Testing finished at $(date)"
