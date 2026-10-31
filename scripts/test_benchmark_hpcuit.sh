#!/bin/bash
# ==============================================================================
# BATCHED TEST BENCHMARK CHO CỤM HPC-UIT (MIG)
# Đánh giá tuần tự nhiều checkpoint trong 1 job duy nhất để tiết kiệm slot submit (1h spacing)
# Profile: hpc-uit, Partition: small (gpu:1g.18gb:1)
# ==============================================================================
#SBATCH --job-name=test_benchmark_hpcuit
#SBATCH --output=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output/test_benchmark_hpcuit/log/%j.out
#SBATCH --error=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output/test_benchmark_hpcuit/log/%j.err
#SBATCH --account=uit
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1g.18gb:1

set -euo pipefail

REPO=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction
ENV_DIR=/datastore/doanhbc/minhdphan/envs/longnet
DATASETS=/datastore/doanhbc/minhdphan/datasets

mkdir -p "$REPO/scripts/output/test_benchmark_hpcuit/log"

cleanup() {
    local rc=$?
    echo "[INFO] cleanup rc=$rc at $(date)"
}
trap cleanup EXIT

echo "=== HPC-UIT BATCHED TEST BENCHMARK ==="
echo "[INFO] Start at: $(date)"
echo "[INFO] Hostname: $(hostname)"
echo "[INFO] SLURM_JOB_ID: ${SLURM_JOB_ID:-<unset>}"
echo "[INFO] Git commit: $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo '<git-not-found>')"

nvidia-smi

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

# Cấu hình danh sách bệnh nhân kiểm thử đa trung tâm / multi-patient test cohort
TEST_PATIENTS=("L310")

echo "[INFO] Khởi động đánh giá benchmark hàng loạt trên các checkpoint sẵn có..."
echo "[INFO] Danh sách test patients: ${TEST_PATIENTS[*]}"

# Ví dụ mẫu đánh giá LEARN LA-120°
LEARN_120_CKPT="$REPO/saved_models/LEARN_120/LEARN_AAPM_LA_120deg_view64/last.ckpt"
if [ -f "$LEARN_120_CKPT" ]; then
    echo "================================================================================"
    echo "🎯 ĐÁNH GIÁ: LEARN LA-120° (64 views)"
    echo "================================================================================"
    python -u baselines/LEARN/test_learn_la.py \
        --ckpt_path "$LEARN_120_CKPT" \
        --cache_dir "$DATASETS/aapm/limited_angle/" \
        --angle_range_deg 120.0 \
        --num_view 64 \
        --batch_size 1 \
        --test_patients "${TEST_PATIENTS[@]}"
fi

# Ví dụ mẫu đánh giá SOLAR_DualMamba LA-90°
SOLAR_90_CKPT="$REPO/saved_models/SOLAR_DualMamba_90/SOLAR_DualMamba_AAPM_LA_90deg_view64/last.ckpt"
if [ -f "$SOLAR_90_CKPT" ]; then
    echo "================================================================================"
    echo "🎯 ĐÁNH GIÁ: SOLAR_DualMamba LA-90° (64 views)"
    echo "================================================================================"
    python -u baselines/SOLAR_DualMamba/test_solar_dualmamba_la.py \
        --ckpt_path "$SOLAR_90_CKPT" \
        --cache_dir "$DATASETS/aapm/limited_angle/" \
        --angle_range_deg 90.0 \
        --num_view 64 \
        --batch_size 1 \
        --test_patients "${TEST_PATIENTS[@]}"
fi

echo "[INFO] Batched benchmark hoàn tất thành công lúc $(date)"

