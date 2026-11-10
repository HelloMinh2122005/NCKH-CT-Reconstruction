#!/bin/bash
# ==============================================================================
# SCRIPT TẢI TẬP CON LIDC-IDRI TRÊN CỤM HPC-UIT
# Partition: small (gpu:1g.18gb:1), max time: 01:00:00, mem: 8G
# ==============================================================================
#SBATCH --job-name=download_lidc
#SBATCH --output=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output/download_lidc_hpcuit/log/%j.out
#SBATCH --error=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output/download_lidc_hpcuit/log/%j.err
#SBATCH --account=uit
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1g.18gb:1

set -euo pipefail

REPO=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction
ENV_DIR=/datastore/doanhbc/minhdphan/envs/longnet
OUTPUT_DIR=/datastore/doanhbc/minhdphan/datasets/lidc_idri/dicom

mkdir -p "$REPO/scripts/output/download_lidc_hpcuit/log"
mkdir -p "$OUTPUT_DIR"

cleanup() {
    local rc=$?
    echo "[INFO] cleanup rc=$rc at $(date)"
}
trap cleanup EXIT

echo "=========================================================="
echo "    DOWNLOADING LIDC-IDRI DATASET FROM TCIA REST API     "
echo "=========================================================="
echo "[INFO] Start at:      $(date)"
echo "[INFO] Hostname:      $(hostname)"
echo "[INFO] Slurm Job:     ${SLURM_JOB_ID:-<local>}"
echo "[INFO] Partition:     ${SLURM_JOB_PARTITION:-<local>}"
echo "[INFO] Git commit:    $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo '<git-not-found>')"
echo "[INFO] Target dir:    $OUTPUT_DIR"
echo ""

# Nạp môi trường Python venv
if [ -f "$ENV_DIR/bin/activate" ]; then
    source "$ENV_DIR/bin/activate"
    echo "[INFO] Activated env at: $ENV_DIR"
else
    echo "[ERROR] Environment not found at $ENV_DIR"
    exit 1
fi

python3 --version
nvidia-smi

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "[INFO] Launching download_lidc_subset.py..."
python3 -u data/lidc_idri/download_lidc_subset.py --output_dir "$OUTPUT_DIR"

echo ""
echo "[INFO] Verifying downloaded DICOM files..."
PATIENT_COUNT=$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
FILE_COUNT=$(find "$OUTPUT_DIR" -type f -name "*.dcm" | wc -l)
echo "[INFO] Total patients: $PATIENT_COUNT / 12"
echo "[INFO] Total DICOM files: $FILE_COUNT"
echo "[INFO] LIDC-IDRI download finished successfully at $(date)"
echo "=========================================================="
