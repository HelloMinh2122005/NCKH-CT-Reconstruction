#!/bin/bash
# ==============================================================================
# HELLO WORLD TEST JOB FOR HPC-UIT SLURM DISPATCHER
# Partition: small (gpu:1g.18gb:1), max time: 10 minutes
# ==============================================================================
#SBATCH --job-name=hello_world
#SBATCH --output=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output/hello_world/log/%j.out
#SBATCH --error=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output/hello_world/log/%j.err
#SBATCH --account=uit
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1g.18gb:1

set -euo pipefail

REPO=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction
mkdir -p "$REPO/scripts/output/hello_world/log"

echo "=========================================================="
echo "      HELLO WORLD FROM HPC-UIT CLUSTER (MIG H200)        "
echo "=========================================================="
echo "[INFO] Timestamp: $(date)"
echo "[INFO] Hostname:  $(hostname)"
echo "[INFO] Slurm Job: ${SLURM_JOB_ID:-<local>}"
echo "[INFO] Partition: ${SLURM_JOB_PARTITION:-<local>}"
echo ""
echo "--- GPU Allocation ---"
nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv || nvidia-smi
echo ""
echo "[INFO] Successful execution test. Finished at: $(date)"
echo "=========================================================="

