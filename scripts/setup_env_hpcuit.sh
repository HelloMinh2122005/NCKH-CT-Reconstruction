#!/bin/bash
# ==============================================================================
# SETUP PYTHON ENVIRONMENT CHO CỤM HPC-UIT
# Profile: hpc-uit, Partition: small (gpu:1g.18gb:1)
# Đích: /datastore/doanhbc/minhdphan/envs/longnet
# ==============================================================================
#SBATCH --job-name=setup_env_hpcuit
#SBATCH --output=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output/setup_env_hpcuit/log/%j.out
#SBATCH --error=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output/setup_env_hpcuit/log/%j.err
#SBATCH --account=uit
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1g.18gb:1

set -euo pipefail

REPO=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction
ENV_DIR=/datastore/doanhbc/minhdphan/envs/longnet

mkdir -p "$REPO/scripts/output/setup_env_hpcuit/log"
mkdir -p "/datastore/doanhbc/minhdphan/envs"

cleanup() {
    local rc=$?
    echo "[INFO] cleanup rc=$rc at $(date)"
}
trap cleanup EXIT

echo "=== HPC-UIT ENVIRONMENT SETUP ==="
echo "[INFO] Start at: $(date)"
echo "[INFO] Hostname: $(hostname)"
echo "[INFO] SLURM_JOB_ID: ${SLURM_JOB_ID:-<unset>}"
echo "[INFO] Git commit: $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo '<git-not-found>')"

# BẮT BUỘC: Đưa mọi cache vào /datastore (Bảo vệ $HOME không vượt quá 10 GB)
export XDG_CACHE_HOME=/datastore/doanhbc/minhdphan/cache
export PIP_CACHE_DIR=$XDG_CACHE_HOME/pip
export TORCH_HOME=$XDG_CACHE_HOME/torch
export HF_HOME=$XDG_CACHE_HOME/hf
export TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton
export CONDA_PKGS_DIRS=$XDG_CACHE_HOME/conda-pkgs
mkdir -p "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TORCH_HOME" "$HF_HOME" "$TRITON_CACHE_DIR" "$CONDA_PKGS_DIRS"

nvidia-smi

echo "=== 1. Tạo Virtual Environment tại $ENV_DIR ==="
if [ ! -d "$ENV_DIR" ]; then
    python3 -m venv "$ENV_DIR"
fi

set +u
source "$ENV_DIR/bin/activate"
set -u

echo "=== 2. Nâng cấp pip và wheel ==="
pip install --upgrade pip setuptools wheel

echo "=== 3. Cài đặt PyTorch và CUDA ops ==="
# Cài đặt PyTorch tương thích CUDA 12
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 || \
pip install torch torchvision torchaudio

echo "=== 4. Cài đặt các gói lõi từ requirements.txt ==="
pip install \
    "pytorch-lightning>=2.0.0" \
    "torchmetrics>=1.0.0,<=1.8.2" \
    "pydicom>=2.4.0,<=2.4.4" \
    "transformers>=4.30.0,<=4.57.6" \
    "scipy>=1.10.0" \
    "numpy>=1.23.0" \
    "Pillow>=9.5.0" \
    "tqdm>=4.65.0" \
    "matplotlib>=3.7.0" \
    "tensorboard>=2.12.0" \
    "einops>=0.6.0"

echo "=== 5. Cài đặt Tomography & Mamba Packages ==="
# Cài đặt ASTRA-Toolbox và ODL
pip install astra-toolbox odl || echo "[WARN] ODL/Astra pip install needs verification from probe output"

# Cài đặt Mamba-SSM (yêu cầu torch đã sẵn sàng)
pip install "mamba-ssm>=2.0.0,<=2.3.1" --no-build-isolation || echo "[WARN] mamba-ssm install needs probe info"

echo "=== 6. Kiểm tra Import và CUDA Invariants ==="
python -c "
import torch
print('PyTorch Version:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device Name:', torch.cuda.get_device_name(0))
    x = torch.randn(10, 10, device='cuda')
    print('Tensor CUDA test:', (x @ x).norm().item())
"

echo "=== 7. Danh sách gói (pip freeze) ==="
pip freeze

echo "=== SETUP HOÀN TẤT THÀNH CÔNG TẠI $(date) ==="

