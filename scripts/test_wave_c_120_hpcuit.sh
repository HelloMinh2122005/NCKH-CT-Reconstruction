#!/bin/bash
# ==============================================================================
# SCRIPT ĐÁNH GIÁ BATCHED WAVE C (LA-120° AAPM) TRÊN CỤM HPC-UIT (MIG)
# Đề tài: Tái tạo ảnh cắt lớp CT góc giới hạn (Limited-Angle CT Reconstruction)
# Tác giả: MinhPD - VNU-HCM UIT
# Mục tiêu: Đánh giá tuần tự toàn bộ 4 mô hình Wave C trên góc quét rộng LA-120° (64 views):
#   1. LEARN (Baseline unrolling bậc 1)
#   2. RegFormer (Baseline Transformer điều hòa miền ảnh)
#   3. DuDoTrans (Baseline Transformer miền kép)
#   4. SOLAR_DualMamba (Mô hình đề xuất đột phá miền kép bậc 2)
# Profile: hpc-uit, Partition: normal (gpu:2g.35gb:1, 8 CPU, 16GB RAM, Time: 04:00:00)
# Tuân thủ: spec-1.md, spec-1-decisions.md, split.md (Change 29)
# ==============================================================================
#SBATCH --job-name=test_wave_c_120_hpcuit
#SBATCH --output=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output-hpc-uit/test_wave_c_120_hpcuit/log/%j.out
#SBATCH --error=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output-hpc-uit/test_wave_c_120_hpcuit/log/%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:2g.35gb:1

set -euo pipefail

# Hàm dọn dẹp tài nguyên khi kết thúc job hoặc bị ngắt đột ngột
cleanup() {
    local rc=$?
    echo "[INFO] cleanup rc=$rc at $(date)"
}
trap cleanup EXIT

REPO=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction
ENV_DIR=/datastore/doanhbc/minhdphan/envs/longnet
DATASETS=/datastore/doanhbc/minhdphan/datasets

mkdir -p "$REPO/scripts/output-hpc-uit/test_wave_c_120_hpcuit/log"
mkdir -p "$REPO/reports"

echo "=== HPC-UIT BATCHED TEST: WAVE C (LA-120°) EVALUATION ==="
echo "[INFO] Start at: $(date)"
echo "[INFO] Hostname: $(hostname)"
echo "[INFO] SLURM_JOB_ID: ${SLURM_JOB_ID:-<unset>}"
echo "[INFO] Git commit: $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo '<git-not-found>')"

# Ghi lại cấu hình GPU MIG được cấp phát
nvidia-smi

# Kích hoạt môi trường Python trên /datastore (tránh tràn quota 10GB của $HOME)
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

# Cấu hình danh sách bệnh nhân kiểm thử đóng băng theo split.md
# Gồm 4 bệnh nhân kiểm thử độc lập: L058, L064, L094, L107 (tổng cộng 757 lát cắt)
TEST_PATIENTS=("L058" "L064" "L094" "L107")
CSV_OUTPUT="$REPO/reports/wave_c_120_per_slice_results.csv"

# Xóa file CSV cũ nếu có để đảm bảo kết quả ghi mới hoàn toàn sạch sẽ
rm -f "$CSV_OUTPUT"

echo "[INFO] Khởi động đánh giá benchmark Wave C (LA-120°, 64 views)..."
echo "[INFO] Danh sách test patients: ${TEST_PATIENTS[*]}"
echo "[INFO] Tệp CSV kết quả per-slice: $CSV_OUTPUT"

# Hàm phụ trợ đánh giá từng mô hình Wave C trên góc quét 120 độ
# Khớp chính xác với các thư mục được sinh ra bởi train_*_120.sh:
#   - saved_models/LEARN_120
#   - saved_models/RegFormer_120
#   - saved_models/DuDoTrans_120
#   - saved_models/SOLAR_DualMamba_120
run_model_eval() {
    local MODEL="${1:-}"
    local SCRIPT_NAME="${2:-}"
    local ANGLE="120"
    local CKPT_DIR="$REPO/saved_models/${MODEL}_${ANGLE}"

    echo "================================================================================"
    echo "🎯 ĐÁNH GIÁ WAVE C: ${MODEL} LA-${ANGLE}° (64 views)"
    echo "================================================================================"

    # Kiểm tra nghiêm ngặt: Fail loudly nếu thư mục checkpoint hoặc tệp checkpoint không tồn tại
    if [ ! -d "$CKPT_DIR" ]; then
        echo "[ERROR] Thư mục checkpoint không tồn tại: $CKPT_DIR" >&2
        exit 1
    fi

    # Kiểm tra xem có file checkpoint nào trong thư mục hay không
    local NUM_CKPT
    NUM_CKPT=$(find "$CKPT_DIR" -name "*.ckpt" 2>/dev/null | wc -l)
    if [ "$NUM_CKPT" -eq 0 ]; then
        echo "[ERROR] Không tìm thấy checkpoint nào trong $CKPT_DIR" >&2
        exit 1
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

# 1. Đánh giá Baseline LEARN LA-120°
run_model_eval "LEARN" "baselines/LEARN/test_learn_la.py"

# 2. Đánh giá Baseline RegFormer LA-120°
run_model_eval "RegFormer" "baselines/RegFormer/test_regformer_la.py"

# 3. Đánh giá Baseline DuDoTrans LA-120°
run_model_eval "DuDoTrans" "baselines/DuDoTrans/test_dudotrans_la.py"

# 4. Đánh giá mô hình đề xuất SOLAR_DualMamba LA-120°
run_model_eval "SOLAR_DualMamba" "baselines/SOLAR_DualMamba/test_solar_dualmamba_la.py"

echo "[INFO] Batched test Wave C (LA-120°) hoàn tất thành công lúc $(date)"
