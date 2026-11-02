#!/bin/bash
# ==============================================================================
# TIỀN XỬ LÝ VÀ TẠO SINH CACHE SINOGRAM / FBP CHO LIMITED-ANGLE CT (AAPM)
# Profile: hpc-uit, Partition: small (gpu:1g.18gb:1)
# Hỗ trợ phân chia đa bệnh nhân: 55 train, 1 val (L110), 4 test (L058, L064, L094, L107)
# Tạo cache đồng thời cho 120° và 90° trong 1 job, sau đó chuỗi gọi đánh giá T3 FBP
# ==============================================================================
#SBATCH --job-name=generate_la_dataset
#SBATCH --output=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output/generate_la_dataset/log/%j.out
#SBATCH --error=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output/generate_la_dataset/log/%j.err
#SBATCH --account=uit
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1g.18gb:1

set -euo pipefail

# Đường dẫn không gian làm việc trên cụm HPC-UIT
REPO=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction
ENV_DIR=/datastore/doanhbc/minhdphan/envs/longnet
DATASETS=/datastore/doanhbc/minhdphan/datasets

# Đảm bảo thư mục log tồn tại trước khi sbatch thực thi
mkdir -p "$REPO/scripts/output/generate_la_dataset/log"

cleanup() {
    local rc=$?
    echo "[INFO] cleanup rc=$rc at $(date)"
}
trap cleanup EXIT

echo "=== KHỞI ĐỘNG TIỀN XỬ LÝ SINH DỮ LIỆU LIMITED-ANGLE CT ==="
echo "[INFO] start at $(date)"
echo "[INFO] hostname=$(hostname)"
echo "[INFO] SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"
echo "[INFO] SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-<unset>}"
echo "[INFO] Git commit: $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo '<git-not-found>')"

# ================= GPU CHECK =================
# Ghi lại cấu hình phần cứng GPU MIG phân bổ bởi Slurm
nvidia-smi

# ================= MPS SETUP =================
# Nạp module CUDA toolkit nếu khả dụng trên node tính toán HPC-UIT
if command -v module &> /dev/null; then
    module load cuda12.9/toolkit/12.9.1 2>/dev/null || true
elif [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
    module load cuda12.9/toolkit/12.9.1 2>/dev/null || true
fi

# Kích hoạt môi trường Python ảo trong /datastore
set +u
source "$ENV_DIR/bin/activate"
set -u

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Chuyển hướng toàn bộ bộ nhớ đệm (cache) về /datastore để bảo vệ hạn ngạch $HOME 10 GB
export XDG_CACHE_HOME=/datastore/doanhbc/minhdphan/cache
export PIP_CACHE_DIR=$XDG_CACHE_HOME/pip
export TORCH_HOME=$XDG_CACHE_HOME/torch
export HF_HOME=$XDG_CACHE_HOME/hf
export TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton
mkdir -p "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TORCH_HOME" "$HF_HOME" "$TRITON_CACHE_DIR"

# Danh sách phân chia bệnh nhân chuẩn theo experiments/split.md (FROZEN 55 train / 1 val / 4 test)
TRAIN_PATIENTS=(
    "L004" "L006" "L012" "L014" "L019" "L024" "L027" "L030" "L033" "L035"
    "L036" "L043" "L044" "L045" "L048" "L049" "L056" "L057" "L071" "L072"
    "L075" "L077" "L079" "L081" "L082" "L111" "L113" "L114" "L116" "L121"
    "L123" "L125" "L127" "L129" "L131" "L133" "L134" "L136" "L138" "L143"
    "L144" "L145" "L147" "L148" "L150" "L154" "L160" "L163" "L166" "L170"
    "L171" "L172" "L175" "L178" "L179"
)
VAL_PATIENTS=("L110")
TEST_PATIENTS=("L058" "L064" "L094" "L107")

echo "[INFO] Cấu hình phân chia: Train=${#TRAIN_PATIENTS[@]} BN, Val=${#VAL_PATIENTS[@]} BN, Test=${#TEST_PATIENTS[@]} BN"

# ================= RUN DATA GENERATION =================
echo "[INFO] Launching Limited-Angle CT Data Preparation at $(date)"

# 1. Benchmark 120-degree Limited Angle (64 views, No Noise)
echo "--------------------------------------------------------------------------------"
echo "🚀 [1/2] Đang tạo dữ liệu góc giới hạn LA-120° (64 views, 512 detectors, 256x256)"
echo "--------------------------------------------------------------------------------"
python -u data/aapm/prepare_data_sinogram_LA.py \
    --data_dir "$DATASETS/aapm/split/" \
    --output_dir "$DATASETS/aapm/limited_angle/" \
    --angle_range_deg 120.0 \
    --num_view 64 \
    --num_detectors 512 \
    --input_size 256 \
    --poisson_level 0 \
    --gaussian_level 0 \
    --train_patients "${TRAIN_PATIENTS[@]}" \
    --val_patients "${VAL_PATIENTS[@]}" \
    --test_patients "${TEST_PATIENTS[@]}"

# 2. Benchmark 90-degree Limited Angle (64 views, No Noise)
echo "--------------------------------------------------------------------------------"
echo "🚀 [2/2] Đang tạo dữ liệu góc giới hạn LA-90° (64 views, 512 detectors, 256x256)"
echo "--------------------------------------------------------------------------------"
python -u data/aapm/prepare_data_sinogram_LA.py \
    --data_dir "$DATASETS/aapm/split/" \
    --output_dir "$DATASETS/aapm/limited_angle/" \
    --angle_range_deg 90.0 \
    --num_view 64 \
    --num_detectors 512 \
    --input_size 256 \
    --poisson_level 0 \
    --gaussian_level 0 \
    --train_patients "${TRAIN_PATIENTS[@]}" \
    --val_patients "${VAL_PATIENTS[@]}" \
    --test_patients "${TEST_PATIENTS[@]}"

echo "[INFO] All Limited-Angle CT data preparation finished at $(date)"

# ================= 3. CHUỖI GỌI T3: ĐÁNH GIÁ BASELINE FBP =================
# Tự động kích hoạt đánh giá FBP trên tập test cho cả 120° và 90° ngay sau khi sinh cache
if [ -f "$REPO/scripts/evaluate_fbp_la.py" ]; then
    echo "================================================================================"
    echo "🎯 KÍCH HOẠT T3: ĐÁNH GIÁ BASELINE FBP (TEST COHORT: 120° & 90°)"
    echo "================================================================================"
    python -u "$REPO/scripts/evaluate_fbp_la.py" \
        --cache_dir "$DATASETS/aapm/limited_angle/" \
        --dicom_dir "$DATASETS/aapm/split/" \
        --output_dir "$REPO/reports/fbp_benchmark/" \
        --test_patients "${TEST_PATIENTS[@]}"
elif [ -f "$REPO/scripts/evaluate_fbp.py" ]; then
    echo "================================================================================"
    echo "🎯 KÍCH HOẠT T3: ĐÁNH GIÁ BASELINE FBP (TEST COHORT: 120° & 90°)"
    echo "================================================================================"
    python -u "$REPO/scripts/evaluate_fbp.py" \
        --cache_dir "$DATASETS/aapm/limited_angle/" \
        --dicom_dir "$DATASETS/aapm/split/" \
        --output_dir "$REPO/reports/fbp_benchmark/" \
        --test_patients "${TEST_PATIENTS[@]}"
else
    echo "[INFO] Script đánh giá FBP (evaluate_fbp_la.py) chưa được thêm, bỏ qua chuỗi gọi T3."
fi

echo "[INFO] Hoàn tất toàn bộ công việc của generate_la_dataset lúc $(date)"
