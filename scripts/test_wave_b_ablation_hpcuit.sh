#!/bin/bash
# ==============================================================================
# SCRIPT ĐÁNH GIÁ BATCHED WAVE B ABLATION TRÊN CỤM HPC-UIT (MIG)
# Đề tài: Tái tạo ảnh cắt lớp CT góc giới hạn (Limited-Angle CT Reconstruction)
# Tác giả: MinhPD - VNU-HCM UIT
# Mục tiêu: Đánh giá tuần tự toàn bộ 4 biến thể ablation Wave B của SOLAR_DualMamba LA-90°:
#   1. Ablation a (Loại bỏ Sinogram Refinement - no sinogram restoration)
#   2. Ablation b (Gradient Descent Data Consistency - GD DC stage-matched K=8)
#   3. Ablation c (Thay Bi-Mamba bằng Conv2D trong Sinogram Restoration)
#   4. Ablation d (Thay Swin Transformer bằng nhánh CNN trong Image Regularizer)
# Profile: hpc-uit, Partition: normal (gpu:2g.35gb:1, 8 CPU, 16GB RAM)
# Tuân thủ: spec-1.md, spec-1-decisions.md, split.md (GT-03-wave-b-test)
# ==============================================================================
#SBATCH --job-name=test_wave_b_ablation_hpcuit
#SBATCH --output=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output-hpc-uit/test_wave_b_ablation_hpcuit/log/%j.out
#SBATCH --error=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output-hpc-uit/test_wave_b_ablation_hpcuit/log/%j.err
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

mkdir -p "$REPO/scripts/output-hpc-uit/test_wave_b_ablation_hpcuit/log"
mkdir -p "$REPO/reports"
# [2026-11-20 T1.6] Thư mục báo cáo đánh giá 55 BN độc lập
mkdir -p "$REPO/reports/hpc-uit-55p"

echo "=== HPC-UIT BATCHED TEST: WAVE B ABLATION EVALUATION ==="
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

# [2026-11-21 T1.11 C3] Tham số hóa ANGLE và VARIANTS
ANGLE="${1:-${ANGLE:-90}}"
VARIANTS="${2:-${VARIANTS:-a b c d}}"

# [2026-11-20 T1.6] Tách triệt để kết quả 55 BN sang reports/hpc-uit-55p/
# [2026-11-21 T1.11 P6] Phân tách CSV theo ANGLE để tránh ghi đè giữa 90° và 120°
CSV_OUTPUT="$REPO/reports/hpc-uit-55p/wave_b_ablation_${ANGLE}_per_slice_results.csv"

echo "[INFO] Khởi động đánh giá hàng loạt 4 biến thể ablation Wave B trên LA-90°..."
echo "[INFO] Cấu hình: LA-${ANGLE}°, danh sách biến thể: ${VARIANTS}"
echo "[INFO] Danh sách test patients: ${TEST_PATIENTS[*]}"
echo "[INFO] Tệp CSV kết quả per-slice: $CSV_OUTPUT"

# Hàm phụ trợ đánh giá từng biến thể ablation
run_ablation_eval() {
    local VARIANT="$1"
    local METHOD="$2"
    local SCRIPT_NAME="baselines/SOLAR_DualMamba/test_solar_dualmamba_la.py"
    local ANGLE="$ANGLE"
    # [2026-11-20 T1.6] Đọc checkpoint từ saved_models_55p/
    # [2026-11-21 T1.11 C3] Đường dẫn checkpoint theo ${ANGLE} và ${VARIANT}
    local CKPT_DIR="$REPO/saved_models_55p/SOLAR_DualMamba_${ANGLE}_abl-${VARIANT}"
    local EXTRA_ARGS=("${@:3}")

    echo "================================================================================"
    echo "🎯 ĐÁNH GIÁ BIẾN THỂ WAVE B: ${METHOD} LA-${ANGLE}° (64 views)"
    echo "================================================================================"

    if [ ! -d "$CKPT_DIR" ]; then
        echo "[WARN] Bỏ qua ${METHOD}: Thư mục không tồn tại ($CKPT_DIR)"
        return 0
    fi

    # Kiểm tra xem có file checkpoint nào trong thư mục hay không
    local NUM_CKPT
    NUM_CKPT=$(find "$CKPT_DIR" -name "*.ckpt" 2>/dev/null | wc -l)
    if [ "$NUM_CKPT" -eq 0 ]; then
        echo "[WARN] Bỏ qua ${METHOD}: Không tìm thấy checkpoint nào trong $CKPT_DIR"
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
        --method "${METHOD}" \
        "${EXTRA_ARGS[@]}"
}

# Lặp qua các biến thể được yêu cầu trong $VARIANTS
for V in $VARIANTS; do
    case "$V" in
        a)
            # 1. Đánh giá Ablation a: SOLAR_DualMamba_90_abl-a (Loại bỏ Sinogram Refinement - ŷ = y, loss_sino = 0)
            run_ablation_eval "a" "SOLAR_DualMamba_abl-a" --ablate a
            ;;
        b)
            # 2. Đánh giá Ablation b: SOLAR_DualMamba_90_abl-b (Gradient Descent Data Consistency - stage-matched K=8)
            run_ablation_eval "b" "SOLAR_DualMamba_abl-b" --ablate b --n_iterations 8
            ;;
        c)
            # 3. Đánh giá Ablation c: SOLAR_DualMamba_90_abl-c (Bi-Mamba -> Conv2D trong Sinogram Restoration)
            run_ablation_eval "c" "SOLAR_DualMamba_abl-c" --ablate c
            ;;
        d)
            # 4. Đánh giá Ablation d: SOLAR_DualMamba_90_abl-d (Swin Transformer -> CNN trong Image Regularizer)
            run_ablation_eval "d" "SOLAR_DualMamba_abl-d" --ablate d
            ;;
        *)
            echo "[WARN] Bỏ qua biến thể không xác định: $V"
            ;;
    esac
done

echo "[INFO] Batched test Wave B ablation hoàn tất thành công lúc $(date)"
