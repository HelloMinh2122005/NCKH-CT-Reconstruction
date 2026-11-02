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
#SBATCH --nodes=1
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
mkdir -p "$REPO/reports"

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
# Theo split.md: L058, L064, L094, L107 (4 bệnh nhân, tổng cộng 757 lát cắt)
TEST_PATIENTS=("L058" "L064" "L094" "L107")
CSV_OUTPUT="$REPO/reports/benchmark_per_slice_results.csv"

echo "[INFO] Khởi động đánh giá benchmark hàng loạt trên các checkpoint Wave A sẵn có..."
echo "[INFO] Danh sách test patients: ${TEST_PATIENTS[*]}"
echo "[INFO] Tệp CSV kết quả benchmark: $CSV_OUTPUT"

# Hàm phụ trợ đánh giá từng cấu hình mô hình và góc quét
run_model_eval() {
    local MODEL="$1"
    local SCRIPT_NAME="$2"
    local ANGLE="$3"
    local CKPT_DIR="$REPO/saved_models/${MODEL}_${ANGLE}"

    echo "================================================================================"
    echo "🎯 ĐÁNH GIÁ: ${MODEL} LA-${ANGLE}° (64 views)"
    echo "================================================================================"

    if [ ! -d "$CKPT_DIR" ]; then
        echo "[WARN] Bỏ qua ${MODEL}_${ANGLE}: Thư mục không tồn tại ($CKPT_DIR)"
        return 0
    fi

    # Kiểm tra xem có file checkpoint nào trong thư mục hay không
    local NUM_CKPT
    NUM_CKPT=$(find "$CKPT_DIR" -name "*.ckpt" 2>/dev/null | wc -l)
    if [ "$NUM_CKPT" -eq 0 ]; then
        echo "[WARN] Bỏ qua ${MODEL}_${ANGLE}: Không tìm thấy checkpoint nào trong $CKPT_DIR"
        return 0
    fi

    python -u "$SCRIPT_NAME" \
        --ckpt_path "$CKPT_DIR" \
        --dataset_type aapm \
        --dicom_dir "$DATASETS/aapm/split/" \
        --cache_dir "$DATASETS/aapm/limited_angle/" \
        --angle_range_deg "${ANGLE}.0" \
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
        --method "${MODEL}"
}

# Ví dụ mẫu đánh giá LEARN LA-120°
# Đánh giá LEARN cho cả 2 góc quét 90° và 120°
run_model_eval "LEARN" "baselines/LEARN/test_learn_la.py" "90"
run_model_eval "LEARN" "baselines/LEARN/test_learn_la.py" "120"

# Đánh giá RegFormer cho cả 2 góc quét 90° và 120°
run_model_eval "RegFormer" "baselines/RegFormer/test_regformer_la.py" "90"
run_model_eval "RegFormer" "baselines/RegFormer/test_regformer_la.py" "120"

# Đánh giá DuDoTrans cho cả 2 góc quét 90° và 120°
run_model_eval "DuDoTrans" "baselines/DuDoTrans/test_dudotrans_la.py" "90"
run_model_eval "DuDoTrans" "baselines/DuDoTrans/test_dudotrans_la.py" "120"

# Ví dụ mẫu đánh giá SOLAR_DualMamba LA-90°
# Đánh giá mô hình đề xuất SOLAR_DualMamba cho cả 2 góc quét 90° và 120°
run_model_eval "SOLAR_DualMamba" "baselines/SOLAR_DualMamba/test_solar_dualmamba_la.py" "90"
run_model_eval "SOLAR_DualMamba" "baselines/SOLAR_DualMamba/test_solar_dualmamba_la.py" "120"

echo "[INFO] Batched benchmark hoàn tất thành công lúc $(date)"
