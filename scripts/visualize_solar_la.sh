#!/bin/bash
#SBATCH --job-name=visualize_solar_la
#SBATCH --output=/datastore/uittogether3/LuuTru/MinhPD/scripts/output/visualize_solar_la/log/%j.out
#SBATCH --error=/datastore/uittogether3/LuuTru/MinhPD/scripts/output/visualize_solar_la/log/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=mps:a100:2
#SBATCH --time=00:45:00

# ==============================================================================
# SCRIPT TRỰC QUAN HÓA & KẾT XUẤT ẢNH CHO 3 BASELINE MÔ HÌNH SOLAR
# Đề tài: Tái tạo ảnh cắt lớp CT góc giới hạn (Limited-Angle CT Reconstruction)
# Tác giả: MinhPD - VNU-HCM UIT
# Cụm máy chủ: Slurm HPC GPU A100/L40 với NVIDIA MPS
# ==============================================================================

set -euo pipefail

# Ngưỡng VRAM yêu cầu tối thiểu (MB)
REQUIRED_VRAM=15000

# Hàm dọn dẹp tài nguyên NVIDIA MPS
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

# Khởi tạo môi trường Module Slurm
module clear -f
module load slurm/slurm/24.11

# Kích hoạt môi trường Conda chuyên dụng của dự án
source /datastore/uittogether3/tools/miniconda3/etc/profile.d/conda.sh

export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"

set +u
conda activate /datastore/uittogether3/tools/miniconda3/envs/LongNet
set -u

# ================= GPU CHECK (Admin Policy) =================
unset CUDA_VISIBLE_DEVICES

set +e
CHECK_OUT=$(/usr/local/bin/gpu_check.sh "$REQUIRED_VRAM" "$SLURM_JOB_ID" 2>&1)
EXIT_CODE=$?
set -e

echo "[INFO] gpu_check exit_code=$EXIT_CODE"
echo "[INFO] gpu_check output=$CHECK_OUT"

if [ "$EXIT_CODE" -eq 10 ]; then
    echo "$CHECK_OUT"
    exit 0
elif [ "$EXIT_CODE" -eq 11 ]; then
    echo "$CHECK_OUT"
    exit 1
elif [ "$EXIT_CODE" -ne 0 ]; then
    echo "[ERROR] gpu_check.sh returned unexpected exit code: $EXIT_CODE"
    exit "$EXIT_CODE"
fi

BEST_GPU="$CHECK_OUT"
echo "[INFO] BEST_GPU=$BEST_GPU"

# ================= NVIDIA MPS CONFIGURATION =================
export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-job${SLURM_JOB_ID}"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-mps-log-job${SLURM_JOB_ID}"

rm -rf "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"

export CUDA_VISIBLE_DEVICES="${BEST_GPU}"

# ================= RUN VISUALIZATION PIPELINE =================
echo "[INFO] Launching SOLAR Benchmark Visualization Pipeline at $(date)"

cd /datastore/uittogether3/LuuTru/MinhPD
export PYTHONPATH="/datastore/uittogether3/LuuTru/MinhPD:${PYTHONPATH:-}"

echo "================================================================================"
echo "🎯 KẾT XUẤT ẢNH TRỰC QUAN HÓA CẤU HÌNH LA-120° (Slices 50, 100, 150)"
echo "   (Bao gồm 3 biến thể SOLAR: SOLAR_LongNet, SOLAR_Mamba, SOLAR_Longformer)"
echo "   (Và các Panel đối sánh trực tiếp với Baseline LEARN)"
echo "================================================================================"
python -u visualize_solar_benchmark.py \
    --slices 50 100 150 \
    --angle_range_deg 120.0 \
    --num_view 64 \
    --num_detectors 512 \
    --input_size 256 \
    --output_dir /datastore/uittogether3/LuuTru/MinhPD/visualizations/ \
    --report_dir /datastore/uittogether3/LuuTru/MinhPD/reports/sep-05-2026/visualizations/

echo "================================================================================"
echo "🎯 KẾT XUẤT ẢNH TRỰC QUAN HÓA CẤU HÌNH LA-90° (Slices 50, 100, 150)"
echo "   (Minh chứng khả năng tái tạo vượt trội của SOLAR khi nêm khuyết mở rộng 270°)"
echo "================================================================================"
python -u visualize_solar_benchmark.py \
    --slices 50 100 150 \
    --angle_range_deg 90.0 \
    --num_view 64 \
    --num_detectors 512 \
    --input_size 256 \
    --output_dir /datastore/uittogether3/LuuTru/MinhPD/visualizations/ \
    --report_dir /datastore/uittogether3/LuuTru/MinhPD/reports/sep-05-2026/visualizations/

echo "[INFO] SOLAR Benchmark Visualization completed successfully at $(date)"
