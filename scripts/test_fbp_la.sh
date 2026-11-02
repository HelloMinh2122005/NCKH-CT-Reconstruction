#!/bin/bash
# ==============================================================================
# SCRIPT ĐÁNH GIÁ (TEST BENCHMARK) FBP TRÊN CỤM HPC-UIT (MIG)
# Đề tài: Tái tạo ảnh cắt lớp CT góc giới hạn (Limited-Angle CT Reconstruction)
# Tác vụ: Đánh giá phương pháp Filtered Backprojection (FBP) cho góc 120° và 90°
# Tập kiểm thử: 4 bệnh nhân (L058, L064, L094, L107 - 757 lát cắt) theo split.md
# Metric: Tuân thủ nghiêm ngặt metrics-decision.md (data_range riêng từng lát cắt)
# Partition: small = gpu:1g.18gb:1, 4 CPU, 32 GB RAM
# ==============================================================================
#SBATCH --job-name=test_fbp_la
#SBATCH --output=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output/test_fbp_la/log/%j.out
#SBATCH --error=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output/test_fbp_la/log/%j.err
#SBATCH --account=uit
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1g.18gb:1

set -euo pipefail

REPO=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction
ENV_DIR=/datastore/doanhbc/minhdphan/envs/longnet
DATASETS=/datastore/doanhbc/minhdphan/datasets

# Tạo thư mục log trước khi thực thi lệnh Slurm
mkdir -p "$REPO/scripts/output/test_fbp_la/log"

cleanup() {
    local rc=$?
    echo "[INFO] cleanup rc=$rc at $(date)"
}
trap cleanup EXIT

echo "[INFO] start at $(date)"
echo "[INFO] hostname=$(hostname)"
echo "[INFO] SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"
echo "[INFO] partition=${SLURM_JOB_PARTITION:-<unset>}"

# Ghi nhận cấu hình GPU MIG được cấp phát
nvidia-smi

# Kích hoạt môi trường Python trên /datastore (tránh tràn hạn ngạch 10GB của $HOME)
set +u
source "$ENV_DIR/bin/activate"
set -u

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Chuyển toàn bộ cache ra phân vùng /datastore
export XDG_CACHE_HOME=/datastore/doanhbc/minhdphan/cache
export PIP_CACHE_DIR=$XDG_CACHE_HOME/pip
export TORCH_HOME=$XDG_CACHE_HOME/torch
export HF_HOME=$XDG_CACHE_HOME/hf
export TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton

# Ghi lại commit git đang thực thi để bảo đảm tính truy nguyên nguồn gốc
echo "[INFO] git HEAD commit: $(git rev-parse HEAD 2>/dev/null || echo 'unknown')"

# ================= TIẾN HÀNH ĐÁNH GIÁ FBP (120° VÀ 90°) =================
# Chạy đánh giá FBP trên tập test cho cả hai góc 120 độ và 90 độ
python -u scripts/evaluate_fbp.py \
    --angles 120.0 90.0 \
    --test_patients L058 L064 L094 L107 \
    --dicom_dir "$DATASETS/aapm/split/" \
    --cache_dir "$DATASETS/aapm/limited_angle/" \
    --output_base_dir "$REPO/saved_models" \
    --job_id "${SLURM_JOB_ID:-local}" \
    --device cuda

echo "[INFO] FINISHED at $(date)"
