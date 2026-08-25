#!/bin/bash
#SBATCH --job-name=check_gpu
#SBATCH --output=check_gpu_%j.out
#SBATCH --error=check_gpu_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gres=mps:a100:2
#SBATCH --time=00:05:00

set -euo pipefail

# Khai báo dung lượng VRAM tối thiểu cần (MB)
REQUIRED_VRAM=2048

echo "[INFO] Start at $(date)"
echo "[INFO] Hostname: $(hostname)"
echo "[INFO] SLURM_JOB_ID: ${SLURM_JOB_ID:-<unset>}"

# Setup environment
module clear -f
module load slurm/slurm/24.11
module load cuda12.8/toolkit/12.8.1

source /datastore/uittogether3/tools/miniconda3/etc/profile.d/conda.sh
conda activate LongNet || conda activate /datastore/uittogether3/tools/miniconda3/envs/LongNet || true

# ================= GPU CHECK & MPS SETUP =================
unset CUDA_VISIBLE_DEVICES

set +e
CHECK_OUT=$(/usr/local/bin/gpu_check.sh "$REQUIRED_VRAM" "$SLURM_JOB_ID" 2>&1)
EXIT_CODE=$?
set -e

if [ "$EXIT_CODE" -eq 10 ]; then
    echo "$CHECK_OUT"
    exit 0 # Thoát để Slurm xếp hàng lại (Re-queue)
elif [ "$EXIT_CODE" -eq 11 ]; then
    echo "$CHECK_OUT"
    exit 1 # Lỗi hệ thống, dừng hẳn
fi

BEST_GPU="$CHECK_OUT"
echo "[INFO] Allocated BEST_GPU: $BEST_GPU"

export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-gpu${BEST_GPU}"
export CUDA_VISIBLE_DEVICES="${BEST_GPU}"

# ================= RUN TEST =================
echo "[INFO] Job ID: $SLURM_JOB_ID"
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
