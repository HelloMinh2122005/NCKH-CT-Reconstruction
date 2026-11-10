#!/bin/bash
# ==============================================================================
# SCRIPT SINH DỮ LIỆU SINOGRAM & FBP CHO LIDC-IDRI TRÊN CỤM HPC-UIT
# Profile: hpc-uit, Partition: small (gpu:1g.18gb:1), max time: 04:00:00, mem: 16G
# ==============================================================================
#SBATCH --job-name=gen_la_lidc
#SBATCH --output=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output/generate_la_dataset_lidc/log/%j.out
#SBATCH --error=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output/generate_la_dataset_lidc/log/%j.err
#SBATCH --account=uit
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1g.18gb:1

set -euo pipefail

cleanup() {
    local rc=$?
    echo "[INFO] cleanup rc=$rc at $(date)"
}
trap cleanup EXIT

REPO=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction
ENV_DIR=/datastore/doanhbc/minhdphan/envs/longnet
DICOM_DIR=/datastore/doanhbc/minhdphan/datasets/lidc_idri/dicom/
OUTPUT_DIR=/datastore/doanhbc/minhdphan/datasets/lidc_idri/limited_angle/

mkdir -p "$REPO/scripts/output/generate_la_dataset_lidc/log"
mkdir -p "$OUTPUT_DIR"

echo "=========================================================="
echo "    LIDC-IDRI LIMITED-ANGLE CT DATA PREPARATION           "
echo "=========================================================="
echo "[INFO] start at:         $(date)"
echo "[INFO] hostname:         $(hostname)"
echo "[INFO] SLURM_JOB_ID:     ${SLURM_JOB_ID:-<unset>}"
echo "[INFO] Partition:        ${SLURM_JOB_PARTITION:-<unset>}"
echo "[INFO] Git commit:       $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo '<git-not-found>')"
echo "[INFO] Dicom source:     $DICOM_DIR"
echo "[INFO] Cache target:     $OUTPUT_DIR"
echo ""

# GPU allocation check
nvidia-smi

# Nạp môi trường Python venv
if [ -f "$ENV_DIR/bin/activate" ]; then
    source "$ENV_DIR/bin/activate"
    echo "[INFO] Activated env at: $ENV_DIR"
else
    echo "[ERROR] Environment not found at $ENV_DIR"
    exit 1
fi

python3 --version

# ================= RUN DATA GENERATION =================
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

python3 -u data/lidc_idri/prepare_data_sinogram_LA_LIDC.py \
    --dicom_dir "$DICOM_DIR" \
    --output_base_dir "$OUTPUT_DIR" \
    --num_view 64 \
    --num_detectors 512 \
    --input_size 256 \
    --poisson_level 0 \
    --gaussian_level 0 \
    --angles 120.0 90.0 \
    --splits train val test

echo ""
echo "[INFO] LIDC-IDRI Limited-Angle CT data preparation finished at $(date)"
echo "=========================================================="
