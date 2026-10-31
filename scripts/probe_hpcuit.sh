#!/bin/bash
# ==============================================================================
# PROBE JOB CHO CỤM HPC-UIT (MIG) - Kiểm tra môi trường node tính toán
# Profile: hpc-uit, Partition: small (gpu:1g.18gb:1)
# ==============================================================================
#SBATCH --job-name=probe_hpcuit
#SBATCH --output=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output/probe_hpcuit/log/%j.out
#SBATCH --error=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output/probe_hpcuit/log/%j.err
#SBATCH --account=uit
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1g.18gb:1

set -euo pipefail

REPO=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction
mkdir -p "$REPO/scripts/output/probe_hpcuit/log"

cleanup() {
    local rc=$?
    echo "[INFO] cleanup rc=$rc at $(date)"
}
trap cleanup EXIT

echo "=== HPC-UIT COMPUTE NODE PROBE ==="
echo "[INFO] Start at: $(date)"
echo "[INFO] Hostname: $(hostname)"
echo "[INFO] SLURM_JOB_ID: ${SLURM_JOB_ID:-<unset>}"
echo "[INFO] SLURM_JOB_PARTITION: ${SLURM_JOB_PARTITION:-<unset>}"
echo "[INFO] Git commit: $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo '<git-not-found>')"

# Cache redirects vào /datastore (bảo vệ $HOME khỏi bị tràn quá 10 GB)
export XDG_CACHE_HOME=/datastore/doanhbc/minhdphan/cache
export PIP_CACHE_DIR=$XDG_CACHE_HOME/pip
export TORCH_HOME=$XDG_CACHE_HOME/torch
export HF_HOME=$XDG_CACHE_HOME/hf
export TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton
export CONDA_PKGS_DIRS=$XDG_CACHE_HOME/conda-pkgs
mkdir -p "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TORCH_HOME" "$HF_HOME" "$TRITON_CACHE_DIR" "$CONDA_PKGS_DIRS"

echo ""
echo "--- 1. GPU Information ---"
nvidia-smi

echo ""
echo "--- 2. CPU and RAM ---"
lscpu | grep -E "Model name|CPU\(s\):|Thread|Core" || true
free -h

echo ""
echo "--- 3. Storage and Filesystem ---"
df -h /datastore/doanhbc/minhdphan
ls -ld /datastore/doanhbc/minhdphan

echo ""
echo "--- 4. System Python and Compilers ---"
echo "python3: $(which python3 2>&1 || true)"
python3 --version || true
echo "gcc: $(which gcc 2>&1 || true)"
gcc --version | head -n 1 || true
echo "nvcc: $(which nvcc 2>&1 || true)"
nvcc --version 2>&1 | head -n 4 || true
echo "conda: $(which conda 2>&1 || true)"
conda --version 2>&1 || true

echo ""
echo "--- 5. Environment Modules ---"
if command -v module &> /dev/null; then
    module list 2>&1 || true
    echo "Available modules:"
    module avail 2>&1 | head -n 40 || true
elif [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
    module list 2>&1 || true
    module avail 2>&1 | head -n 40 || true
else
    echo "module command not directly available in non-interactive subshell"
fi

echo ""
echo "--- 6. Internet Access from Compute Node ---"
echo -n "PyPI (https://pypi.org): "
curl -sI --connect-timeout 5 https://pypi.org | head -n 1 || echo "FAILED"

echo -n "GitHub (https://github.com): "
curl -sI --connect-timeout 5 https://github.com | head -n 1 || echo "FAILED"

echo -n "HuggingFace (https://huggingface.co): "
curl -sI --connect-timeout 5 https://huggingface.co | head -n 1 || echo "FAILED"

echo -n "TCIA (https://services.cancerimagingarchive.net): "
curl -sI --connect-timeout 5 https://services.cancerimagingarchive.net | head -n 1 || echo "FAILED"

echo ""
echo "[INFO] Probe completed successfully at $(date)"

