#!/bin/bash
# ==============================================================================
# SCRIPT ĐÁNH GIÁ CHECKPOINT CUỐI CÙNG (LAST.CKPT) CỦA SOLAR_DualMamba TRÊN HPC-UIT
# Đề tài: Tái tạo ảnh cắt lớp CT góc giới hạn (Limited-Angle CT Reconstruction)
# Tác giả: MinhPD - VNU-HCM UIT
# Mục tiêu: Đánh giá trọng số epoch cuối cùng (last.ckpt) của SOLAR_DualMamba LA-90°
#   trên 4 bệnh nhân kiểm thử độc lập (L058, L064, L094, L107) để xác minh xem việc
#   chọn checkpoint sớm tại epoch=05 trong Wave A có gây sụt giảm hiệu năng so với
#   biến thể abl-c trong Wave B hay không.
# Profile: hpc-uit, Partition: normal (gpu:2g.35gb:1, 8 CPU, 16GB RAM, Time: 01:00:00)
# Tuân thủ: spec-1.md, spec-1-decisions.md, split.md (Change 29)
# ==============================================================================
#SBATCH --job-name=test_solar_last_ckpt_hpcuit
#SBATCH --output=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output-hpc-uit/test_solar_last_ckpt_hpcuit/log/%j.out
#SBATCH --error=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output-hpc-uit/test_solar_last_ckpt_hpcuit/log/%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
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

mkdir -p "$REPO/scripts/output-hpc-uit/test_solar_last_ckpt_hpcuit/log"
mkdir -p "$REPO/reports"
# [2026-11-20 T1.6] Thư mục báo cáo đánh giá 55 BN độc lập
mkdir -p "$REPO/reports/hpc-uit-55p"

echo "=== HPC-UIT TEST: SOLAR_DualMamba LAST.CKPT EVALUATION ==="
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
# [2026-11-20 T1.6] Tách triệt để kết quả 55 BN sang reports/hpc-uit-55p/
CSV_OUTPUT="$REPO/reports/hpc-uit-55p/solar_last_ckpt_per_slice_results.csv"

# Xóa file CSV cũ nếu có để đảm bảo kết quả ghi mới hoàn toàn sạch sẽ
rm -f "$CSV_OUTPUT"

echo "[INFO] Khởi động đánh giá SOLAR_DualMamba trên checkpoint last.ckpt (LA-90°, 64 views)..."
echo "[INFO] Danh sách test patients: ${TEST_PATIENTS[*]}"
echo "[INFO] Tệp CSV kết quả per-slice: $CSV_OUTPUT"

# Đường dẫn tệp checkpoint last.ckpt của mô hình SOLAR LA-90°
# [2026-11-20 T1.6] Đọc checkpoint từ saved_models_55p/
CKPT_FILE="$REPO/saved_models_55p/SOLAR_DualMamba_90/last.ckpt"

echo "================================================================================"
echo "🎯 ĐÁNH GIÁ: SOLAR_DualMamba (last.ckpt) LA-90° (64 views)"
echo "================================================================================"

# Kiểm tra nghiêm ngặt: Fail loudly nếu tệp last.ckpt không tồn tại
if [ ! -f "$CKPT_FILE" ]; then
    echo "[ERROR] Checkpoint last.ckpt không tồn tại: $CKPT_FILE" >&2
    exit 1
fi

python -u baselines/SOLAR_DualMamba/test_solar_dualmamba_la.py \
    --ckpt_path "$CKPT_FILE" \
    --dataset_type aapm \
    --dicom_dir "$DATASETS/aapm/split/" \
    --cache_dir "$DATASETS/aapm/limited_angle/" \
    --angle_range_deg 90.0 \
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
    --method "SOLAR_DualMamba_last_ckpt"

echo "[INFO] Đánh giá SOLAR_DualMamba last.ckpt hoàn tất thành công lúc $(date)"
