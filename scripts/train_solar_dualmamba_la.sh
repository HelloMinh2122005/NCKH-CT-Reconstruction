#!/bin/bash
# ==============================================================================
# SCRIPT HUẤN LUYỆN MÔ HÌNH ĐỘT PHÁ MIỀN KÉP SOLAR_DualMamba TRÊN SLURM HPC
# Đề tài: Tái tạo ảnh cắt lớp CT góc giới hạn (Limited-Angle CT Reconstruction)
# Tác giả: MinhPD - VNU-HCM UIT
# Cụm máy chủ: Slurm HPC GPU A100/L40 với NVIDIA MPS
# Cập nhật HPC-UIT (MIG): Partition normal (gpu:2g.35gb:1, 8 CPU, 32GB RAM)
# ==============================================================================
#SBATCH --job-name=train_solar_dualmamba_la
#SBATCH --output=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output-hpc-uit/train_solar_dualmamba_la/log/%j.out
#SBATCH --error=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output-hpc-uit/train_solar_dualmamba_la/log/%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2g.35gb:1

set -euo pipefail

# Hàm dọn dẹp tài nguyên khi kết thúc job hoặc bị ngắt đột ngột
cleanup() {
    local rc=$?
    echo "[INFO] cleanup rc=$rc at $(date)"
}
trap cleanup EXIT

# Ngưỡng VRAM yêu cầu tối thiểu (MB)
REQUIRED_VRAM=16000

REPO=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction
ENV_DIR=/datastore/doanhbc/minhdphan/envs/longnet
DATASETS=/datastore/doanhbc/minhdphan/datasets

# Đảm bảo thư mục log tồn tại trước khi Slurm ghi nhận
mkdir -p "$REPO/scripts/output-hpc-uit/train_solar_dualmamba_la/log"

echo "=== HPC-UIT TRAINING: SOLAR_DualMamba ==="
echo "[INFO] start at $(date)"
echo "[INFO] hostname=$(hostname)"
echo "[INFO] SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"
echo "[INFO] SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-<unset>}"
echo "[INFO] Git commit: $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo '<git-not-found>')"

# Ghi lại cấu hình GPU MIG được cấp phát
nvidia-smi

# ================= GPU CHECK =================
# Trên HPC-UIT, phân vùng MIG quản lý tài nguyên GPU trực tiếp qua Slurm GRES
# ================= NVIDIA MPS SETUP =================
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
# [2026-11-20 T1.7] Tham số hóa số epoch: mặc định 50
# Tham số hóa góc quét: mặc định 90 độ, có thể cấu hình 120 độ qua đối số $1
ANGLE="${1:-90}"
ANGLE_DEG="${ANGLE}.0"
# Tham số hóa số epoch: mặc định 50 (T1.7)
MAX_EPOCHS="${MAX_EPOCHS:-50}"
echo "[INFO] Cấu hình góc quét: ${ANGLE_DEG}° (Views: 64)"
echo "[INFO] MAX_EPOCHS: $MAX_EPOCHS"

# Thư mục lưu checkpoint và log theo quy chuẩn saved_models/<Model>_<angle>/
# [2026-11-20 T1.6] Tách triệt để Checkpoint 55 BN: lưu vào saved_models_55p/<Model>_<angle>/
OUTPUT_DIR="$REPO/saved_models_55p/SOLAR_DualMamba_${ANGLE}"
LOG_DIR="$REPO/lightning_logs/SOLAR_DualMamba_${ANGLE}"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# ================= PHÂN CHIA BỆNH NHÂN (SPLIT.MD) =================
# Tập kiểm định cố định: L110 (1 bệnh nhân, 133 lát cắt)
VAL_PATIENTS=("L110")
# Tập kiểm thử độc lập: L058, L064, L094, L107 (4 bệnh nhân, 757 lát cắt)
TEST_PATIENTS=("L058" "L064" "L094" "L107")

# Lấy danh sách 55 bệnh nhân tập train từ thư mục split/train/ trên cụm
# [2026-11-19 GT-FIX-55PATIENTS-CHANGE35] Dùng danh sách 55 bệnh nhân đóng băng theo experiments/split.md và generate_la_dataset_120.sh (tránh find -exec basename bỏ qua symlink)
TRAIN_PATIENTS=(
    "L004" "L006" "L012" "L014" "L019" "L024" "L027" "L030" "L033" "L035"
    "L036" "L043" "L044" "L045" "L048" "L049" "L056" "L057" "L071" "L072"
    "L075" "L077" "L079" "L081" "L082" "L111" "L113" "L114" "L116" "L121"
    "L123" "L125" "L127" "L129" "L131" "L133" "L134" "L136" "L138" "L143"
    "L144" "L145" "L147" "L148" "L150" "L154" "L160" "L163" "L166" "L170"
    "L171" "L172" "L175" "L178" "L179"
)
[ "${#TRAIN_PATIENTS[@]}" -eq 55 ] || { echo "[ERROR] TRAIN_PATIENTS phải có đúng 55 bệnh nhân, thực tế: ${#TRAIN_PATIENTS[@]}" >&2; exit 1; }
echo "[INFO] Phân chia cố định: Train=${#TRAIN_PATIENTS[@]} BN, Val=${#VAL_PATIENTS[@]} BN, Test=${#TEST_PATIENTS[@]} BN"

# ================= CƠ CHẾ RESUME TỪ LAST.CKPT =================
# [2026-11-20 T1.6] Tách triệt để Checkpoint 55 BN (Chặn T2): Không tự động resume đè trong saved_models_55p.
# Nếu thư mục đích $OUTPUT_DIR đã tồn tại và chứa bất kỳ tệp .ckpt nào, dừng ngay để bảo vệ checkpoint.
if [ -d "$OUTPUT_DIR" ] && find "$OUTPUT_DIR" -name "*.ckpt" 2>/dev/null | grep -q .; then
    echo "[ERROR] Thư mục đích $OUTPUT_DIR đã tồn tại và chứa checkpoint. Dừng ngay để bảo vệ checkpoint 55 BN (T1.6)." >&2
    exit 1
fi
RESUME_CKPT=$(find "$OUTPUT_DIR" -name "last.ckpt" 2>/dev/null | head -n 1 || true)
EXTRA_ARGS=()
if [ -n "$RESUME_CKPT" ] && [ -f "$RESUME_CKPT" ]; then
    echo "[INFO] Tìm thấy checkpoint để resume: $RESUME_CKPT"
    EXTRA_ARGS=(--resume_ckpt "$RESUME_CKPT")
fi

# ================= RUN TRAINING =================
echo "[INFO] Launching SOLAR_DualMamba Training on Limited-Angle CT (AAPM) at $(date)"
echo "[INFO] Hạt giống tái lập thực nghiệm: --seed 42 (spec-1-decisions.md Q5)"

python -u baselines/SOLAR_DualMamba/train_solar_dualmamba_la.py \
    --dataset_type aapm \
    --dicom_dir "$DATASETS/aapm/split/" \
    --dataset_dir "$DATASETS/aapm/limited_angle/" \
    --output_dir "$OUTPUT_DIR" \
    --log_dir "$LOG_DIR" \
    --angle_range_deg "$ANGLE_DEG" \
    --num_view 64 \
    --num_detectors 512 \
    --input_size 256 \
    --feature_dim 48 \
    --window_size 8 \
    --sino_loss_weight 0.1 \
    --poisson_level 0 \
    --gaussian_level 0 \
    --batch_size 1 \
    --n_iterations 8 \
    --cg_iters 4 \
    --max_epochs "$MAX_EPOCHS" \
    --lr 1e-4 \
    --final_lr 1e-5 \
    --seed 42 \
    --val_patients "${VAL_PATIENTS[@]}" \
    --train_patients "${TRAIN_PATIENTS[@]}" \
    --num_workers 4 \
    --use_precomputed \
    "${EXTRA_ARGS[@]}"

echo "[INFO] SOLAR_DualMamba Training finished at $(date)"
