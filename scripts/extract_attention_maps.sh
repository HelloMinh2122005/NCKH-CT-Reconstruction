#!/bin/bash
#SBATCH --job-name=attn_evolution
#SBATCH --output=/datastore/uittogether3/LuuTru/MinhPD/scripts/output/attn_evolution/log/%j.out
#SBATCH --error=/datastore/uittogether3/LuuTru/MinhPD/scripts/output/attn_evolution/log/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=mps:a100:2
#SBATCH --time=00:30:00

# ==============================================================================
# SCRIPT CHẠY TRÍCH XUẤT ATTENTION MAPS (SLURM BATCH)
# Dự án: Limited-Angle CT Reconstruction (SOICT 2026)
# ==============================================================================

set -euo pipefail

REQUIRED_VRAM=15000

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

# ================= GPU CHECK =================
unset CUDA_VISIBLE_DEVICES

set +e
CHECK_OUT=$(/usr/local/bin/gpu_check.sh "$REQUIRED_VRAM" "$SLURM_JOB_ID" 2>&1)
EXIT_CODE=$?
set -e

if [ "$EXIT_CODE" -eq 10 ]; then
    echo "$CHECK_OUT"
    exit 0
elif [ "$EXIT_CODE" -eq 11 ]; then
    echo "$CHECK_OUT"
    exit 1
elif [ "$EXIT_CODE" -ne 0 ]; then
    echo "[ERROR] gpu_check.sh failed with exit code: $EXIT_CODE"
    exit "$EXIT_CODE"
fi

BEST_GPU="$CHECK_OUT"
echo "[INFO] BEST_GPU=$BEST_GPU"

export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-job${SLURM_JOB_ID}"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-mps-log-job${SLURM_JOB_ID}"

rm -rf "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"

export CUDA_VISIBLE_DEVICES="${BEST_GPU}"

# ================= EXECUTE ATTENTION MAP EXTRACTION =================
cd /datastore/uittogether3/LuuTru/MinhPD
export PYTHONPATH="/datastore/uittogether3/LuuTru/MinhPD:${PYTHONPATH:-}"

mkdir -p /datastore/uittogether3/LuuTru/MinhPD/scripts/output/attn_evolution/log
mkdir -p /datastore/uittogether3/LuuTru/MinhPD/visualizations/attention_maps

echo "================================================================================"
echo "🎯 KẾT XUẤT ATTENTION EVOLUTION PANEL CHO LA-120° (SLICE 050)"
echo "================================================================================"
python -u scripts/extract_attention_maps.py \
    --slice_idx 50 \
    --angle_range_deg 120.0 \
    --num_view 64 \
    --output_dir /datastore/uittogether3/LuuTru/MinhPD/visualizations/attention_maps

echo "================================================================================"
echo "🎯 KẾT XUẤT ATTENTION EVOLUTION PANEL CHO LA-90° (SLICE 050)"
echo "================================================================================"
python -u scripts/extract_attention_maps.py \
    --slice_idx 50 \
    --angle_range_deg 90.0 \
    --num_view 64 \
    --output_dir /datastore/uittogether3/LuuTru/MinhPD/visualizations/attention_maps

echo "[INFO] FINISHED at $(date)"
