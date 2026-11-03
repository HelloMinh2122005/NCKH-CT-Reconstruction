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
#SBATCH --time=02:00:00
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
# Xóa môi trường cũ nếu có để đảm bảo tạo lại venv sạch hoàn toàn
rm -rf "$ENV_DIR"
python3 -m venv "$ENV_DIR"

set +u
source "$ENV_DIR/bin/activate"
set -u

echo "=== 2. Nâng cấp pip và wheel ==="
# Nâng cấp công cụ cài đặt pip, setuptools và wheel lên phiên bản mới nhất
pip install --upgrade pip setuptools wheel

echo "=== 3. Cài đặt PyTorch và CUDA ops ==="
# Cài đặt PyTorch tương thích CUDA 12
# Ghim chính xác PyTorch 2.8.0+cu128 cùng torchvision và torchaudio từ index CUDA 12.8 của PyTorch
pip install \
    "torch==2.8.0+cu128" \
    "torchvision==0.23.0+cu128" \
    "torchaudio==2.8.0+cu128" \
    --index-url https://download.pytorch.org/whl/cu128

echo "=== 4. Cài đặt các gói lõi từ requirements.txt ==="
# Cài đặt các gói phụ thuộc lõi với phiên bản được ghim chính xác tuyệt đối (fail-fast, không dùng range)
pip install \
    "pytorch-lightning==2.5.0" \
    "torchmetrics==1.8.2" \
    "pydicom==2.4.4" \
    "transformers==4.57.6" \
    "scipy==1.13.1" \
    "numpy==1.26.4" \
    "Pillow==10.4.0" \
    "tqdm==4.67.1" \
    "matplotlib==3.9.2" \
    "tensorboard==2.18.0" \
    "einops==0.8.0" \
    "future==1.0.0"

echo "=== 5. Cài đặt Tomography & Mamba Packages ==="
# Cài đặt ASTRA-Toolbox và ODL
# Ghim chính xác astra-toolbox 2.3.0 và odl 0.8.3, loại bỏ fallback để đảm bảo fail-fast
# Cài đặt với cờ --no-deps để tránh pip resolver tự động hạ cấp nvidia-cuda-runtime-cu12
pip install --no-deps \
    "astra-toolbox==2.3.0" \
    "odl==0.8.3"

# Tái xác nhận và cài đặt lại nvidia-cuda-runtime-cu12 và nvidia-cufft-cu12 tương thích CUDA 12.8 với --no-deps
pip install --no-deps \
    "nvidia-cuda-runtime-cu12==12.8.90" \
    "nvidia-cufft-cu12==11.3.3.83" \
    --index-url https://download.pytorch.org/whl/cu128

# Nạp module CUDA toolkit 12.9 trước khi cài đặt mamba-ssm để cung cấp nvcc và CUDA headers biên dịch
module load cuda12.9/toolkit/12.9.1
nvcc --version

# Cài đặt Mamba-SSM (yêu cầu torch đã sẵn sàng)
# Biên dịch mamba-ssm 2.3.1 từ nguồn với --no-build-isolation bằng nvcc từ module cuda12.9
pip install "mamba-ssm==2.3.1" --no-build-isolation

echo "=== 6. Kiểm tra Import và CUDA Invariants ==="
# Khối kiểm thử cuối cùng: kiểm tra tính toàn vẹn của PyTorch CUDA op, mamba_ssm selective scan và ODL+ASTRA FanBeam
python -c "
import sys
import numpy as np
import torch

# 1. Kiểm tra toán tử CUDA qua PyTorch
print('[CHECK 1/3] Kiểm tra PyTorch và CUDA...')
print('PyTorch Version:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
if not torch.cuda.is_available():
    print('LỖI: CUDA không khả dụng trong PyTorch!', file=sys.stderr)
    sys.exit(1)
print('Device Name:', torch.cuda.get_device_name(0))
x = torch.randn(10, 10, device='cuda')
matmul_norm = (x @ x).norm().item()
print('Tensor CUDA matmul norm:', matmul_norm)
assert matmul_norm > 0, 'Toán tử matmul trên GPU trả về chuẩn bằng 0'
print('PyTorch CUDA op test: PASSED')

# 2. Kiểm tra mamba_ssm và selective scan CUDA kernel trên GPU
print('[CHECK 2/3] Kiểm tra mamba_ssm và selective scan CUDA kernel...')
import mamba_ssm
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba_ssm import Mamba
print('Mamba-SSM Version:', mamba_ssm.__version__)
mamba_layer = Mamba(d_model=64, d_state=16, d_conv=4, expand=2).to('cuda')
x_mamba = torch.randn(2, 16, 64, device='cuda')
y_mamba = mamba_layer(x_mamba)
print('Mamba forward output shape:', y_mamba.shape)
assert y_mamba.shape == x_mamba.shape, 'Đầu ra Mamba forward không khớp kích thước đầu vào'
print('mamba_ssm selective scan test: PASSED')

# 3. Kiểm tra ASTRA và ODL với FanBeamGeometry (src_radius=600, det_radius=290)
print('[CHECK 3/3] Kiểm tra ODL và ASTRA-Toolbox FanBeam forward projection...')
import odl
import astra
print('ODL Version:', odl.__version__)
print('ASTRA Version:', astra.__version__)
xx = 200
space = odl.uniform_discr([-xx, -xx], [xx, xx], [256, 256], dtype='float32')
angle_partition = odl.uniform_partition(0, np.radians(120), 64)
detector_partition = odl.uniform_partition(-480, 480, 512)
geometry = odl.tomo.FanBeamGeometry(
    angle_partition,
    detector_partition,
    src_radius=600,
    det_radius=290
)
impl = 'astra_cuda' if torch.cuda.is_available() else 'astra_cpu'
print('Sử dụng toán tử Radon ODL với backend:', impl)
operator = odl.tomo.RayTransform(space, geometry, impl=impl)
phantom = space.one()
sino = operator(phantom)
sino_norm = float(np.linalg.norm(sino))
print('FanBeam forward projection sino norm:', sino_norm)
assert sino_norm > 0, 'Sinogram forward projection norm phải lớn hơn 0'
print('ODL và ASTRA FanBeam test: PASSED')
"

echo "=== 7. Danh sách gói (pip freeze) ==="
# Xuất toàn bộ danh sách gói đã cài đặt để lưu vết phục vụ kiểm tra provenance
pip freeze

echo "=== SETUP HOÀN TẤT THÀNH CÔNG TẠI $(date) ==="
