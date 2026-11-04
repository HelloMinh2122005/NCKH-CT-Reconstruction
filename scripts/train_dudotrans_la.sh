#!/bin/bash
# ==============================================================================
# SCRIPT HUẤN LUYỆN MÔ HÌNH BASELINE DUDOTRANS TRÊN SLURM HPC
# Đề tài: Tái tạo ảnh cắt lớp CT góc giới hạn (Limited-Angle CT Reconstruction)
# Tác giả: MinhPD - VNU-HCM UIT
# Cụm máy chủ: Slurm HPC GPU (HPC-UIT MIG)
# Cập nhật HPC-UIT (MIG): Partition normal (gpu:2g.35gb:1, 8 CPU, 32GB RAM)
# ==============================================================================
#SBATCH --job-name=train_dudotrans_la
#SBATCH --output=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output/train_dudotrans_la/log/%j.out
#SBATCH --error=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output/train_dudotrans_la/log/%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2g.35gb:1

set -euo pipefail

# Ngưỡng VRAM yêu cầu tối thiểu (MB)
REQUIRED_VRAM=12000

REPO=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction
ENV_DIR=/datastore/doanhbc/minhdphan/envs/longnet
DATASETS=/datastore/doanhbc/minhdphan/datasets

# Đảm bảo thư mục log tồn tại trước khi Slurm ghi nhận
mkdir -p "$REPO/scripts/output/train_dudotrans_la/log"

# Hàm dọn dẹp tài nguyên khi kết thúc job hoặc bị ngắt đột ngột
cleanup() {
    local rc=$?
    echo "[INFO] cleanup rc=$rc at $(date)"
}
trap cleanup EXIT

echo "=== HPC-UIT TRAINING: DuDoTrans ==="
echo "[INFO] start at $(date)"
echo "[INFO] hostname=$(hostname)"
echo "[INFO] SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"
echo "[INFO] SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-<unset>}"
echo "[INFO] Git commit: $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo '<git-not-found>')"

# Ghi lại cấu hình GPU MIG được cấp phát
nvidia-smi

# ================= GPU CHECK =================
# Trên HPC-UIT, phân vùng MIG quản lý tài nguyên GPU trực tiếp qua Slurm GRES
# ================= MPS SETUP =================
# Cấu hình MIG độc lập phần cứng thay thế cơ chế chia sẻ NVIDIA MPS cũ

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

# ================= CẤU HÌNH GÓC QUÉT & CHECKPOINT =================
# Tham số hóa góc quét: mặc định 90 độ, có thể cấu hình 120 độ qua đối số $1
ANGLE="${1:-90}"
ANGLE_DEG="${ANGLE}.0"
echo "[INFO] Cấu hình góc quét: ${ANGLE_DEG}° (Views: 64)"

# Thư mục lưu checkpoint và log theo quy chuẩn saved_models/<Model>_<angle>/
OUTPUT_DIR="$REPO/saved_models/DuDoTrans_${ANGLE}"
LOG_DIR="$REPO/lightning_logs/DuDoTrans_${ANGLE}"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# ================= PHÂN CHIA BỆNH NHÂN (SPLIT.MD) =================
# Tập kiểm định cố định: L110 (1 bệnh nhân, 133 lát cắt)
VAL_PATIENTS=("L110")
# Tập kiểm thử độc lập: L058, L064, L094, L107 (4 bệnh nhân, 757 lát cắt)
TEST_PATIENTS=("L058" "L064" "L094" "L107")

# Lấy danh sách 55 bệnh nhân tập train từ thư mục split/train/ trên cụm
TRAIN_PATIENT_ARGS=()
if [ -d "$DATASETS/aapm/split/train" ]; then
    TRAIN_PATIENTS=($(find "$DATASETS/aapm/split/train" -mindepth 1 -maxdepth 1 -type d -exec basename {} + 2>/dev/null | grep -v "L110" | sort || true))
    if [ ${#TRAIN_PATIENTS[@]} -gt 0 ]; then
        echo "[INFO] Đã tìm thấy ${#TRAIN_PATIENTS[@]} bệnh nhân tập train trên cụm."
        TRAIN_PATIENT_ARGS=(--train_patients "${TRAIN_PATIENTS[@]}")
    fi
fi

# ================= CƠ CHẾ RESUME TỪ LAST.CKPT =================
RESUME_CKPT=$(find "$OUTPUT_DIR" -name "last.ckpt" 2>/dev/null | head -n 1 || true)
EXTRA_ARGS=()
if [ -n "$RESUME_CKPT" ] && [ -f "$RESUME_CKPT" ]; then
    echo "[INFO] Tìm thấy checkpoint để resume: $RESUME_CKPT"
    EXTRA_ARGS=(--resume_ckpt "$RESUME_CKPT")
fi

# ================= RUN TRAINING =================
echo "[INFO] Launching DuDoTrans Training on Limited-Angle CT at $(date)"
echo "[INFO] Hạt giống tái lập thực nghiệm: --seed 42 (spec-1-decisions.md Q5)"

python -u baselines/DuDoTrans/train_dudotrans_la.py \
    --dataset_type aapm \
    --dicom_dir "$DATASETS/aapm/split/" \
    --dataset_dir "$DATASETS/aapm/limited_angle/" \
    --output_dir "$OUTPUT_DIR" \
    --log_dir "$LOG_DIR" \
    --angle_range_deg "$ANGLE_DEG" \
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
    --seed 42 \
    --val_patients "${VAL_PATIENTS[@]}" \
    "${TRAIN_PATIENT_ARGS[@]}" \
    --num_workers 4 \
    --use_precomputed \
    "${EXTRA_ARGS[@]}"

echo "[INFO] DuDoTrans Training finished at $(date)"
