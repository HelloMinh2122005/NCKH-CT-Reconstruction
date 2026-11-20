#!/bin/bash
# ==============================================================================
# SCRIPT SMOKE BENCHMARK & KIỂM TRA TÍCH HỢP DỮ LIỆU 55 BỆNH NHÂN (WAVE C LA-120°)
# Profile: hpc-uit, Partition: normal (gpu:2g.35gb:1, 8 CPU, 12GB RAM, 01:00:00)
#
# Mục đích:
# 1. Preflight Python trên phân chia thực AAPM (55 train / 1 val L110 / 4 test):
#    - Khẳng định len(train_dataset) == 8218, len(val_dataset) == 133, len(test_dataset) == 757.
#    - Đọc thử chỉ số 0 (__getitem__(0)) trên cả 3 tập và kiểm tra toàn bộ symlink không gãy.
# 2. Chạy huấn luyện smoke benchmark 1 epoch cho SOLAR_DualMamba trên LA-120° với batch_size=4:
#    - Lưu checkpoint và log tách biệt hoàn toàn tại saved_models/SOLAR_DualMamba_120_smoke
#      và lightning_logs/SOLAR_DualMamba_120_smoke (không đè checkpoint chính thức).
#    - Đo và in thời gian thực thi 1 epoch (giây) và Peak GPU VRAM (MB) để căn chỉnh
#      --batch_size, --max_epochs và --time cho chiến dịch huấn luyện Wave A/B/C đầy đủ.
#
# LƯU Ý KHOA HỌC: Số liệu từ job smoke 1-epoch này CHỈ dùng để kiểm chứng dữ liệu,
# đo thông lượng (wall-clock) và VRAM phần cứng; KHÔNG dùng làm kết quả thực nghiệm khoa học.
# ==============================================================================
#SBATCH --job-name=smoke_wave_c_55p_hpcuit
#SBATCH --output=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output-hpc-uit/smoke_wave_c_55p_hpcuit/log/%j.out
#SBATCH --error=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output-hpc-uit/smoke_wave_c_55p_hpcuit/log/%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:2g.35gb:1

set -euo pipefail

# Hàm dọn dẹp tài nguyên khi kết thúc job hoặc gặp sự cố
cleanup() {
    local rc=$?
    echo "[INFO] cleanup rc=$rc at $(date)"
}
trap cleanup EXIT

REPO=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction
ENV_DIR=/datastore/doanhbc/minhdphan/envs/longnet
DATASETS=/datastore/doanhbc/minhdphan/datasets

# Thư mục log bắt buộc phải được tạo trước khi ghi nhận đầu ra
LOG_RUN_DIR="$REPO/scripts/output-hpc-uit/smoke_wave_c_55p_hpcuit/log"
mkdir -p "$LOG_RUN_DIR"

echo "================================================================================"
echo "🚀 SMOKE BENCHMARK 55 BỆNH NHÂN (8,218 LÁT): SOLAR_DualMamba LA-120° (batch_size=4)"
echo "[INFO] start at $(date)"
echo "[INFO] hostname=$(hostname)"
echo "[INFO] SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"
echo "[INFO] SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-<unset>}"
echo "[INFO] Git commit: $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo '<git-not-found>')"
echo "================================================================================"

# Ghi lại cấu hình GPU MIG được phân bổ bởi Slurm
nvidia-smi

# Nạp module CUDA toolkit nếu khả dụng trên node tính toán HPC-UIT
if command -v module &> /dev/null; then
    module load cuda12.9/toolkit/12.9.1 2>/dev/null || true
elif [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
    module load cuda12.9/toolkit/12.9.1 2>/dev/null || true
fi

# Kích hoạt môi trường Python trên /datastore (bảo vệ hạn ngạch $HOME 10 GB)
set +u
source "$ENV_DIR/bin/activate"
set -u

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Chuyển hướng toàn bộ bộ nhớ đệm về /datastore
export XDG_CACHE_HOME=/datastore/doanhbc/minhdphan/cache
export PIP_CACHE_DIR=$XDG_CACHE_HOME/pip
export TORCH_HOME=$XDG_CACHE_HOME/torch
export HF_HOME=$XDG_CACHE_HOME/hf
export TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton
mkdir -p "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TORCH_HOME" "$HF_HOME" "$TRITON_CACHE_DIR"

# ================= PHÂN CHIA BỆNH NHÂN ĐÓNG BĂNG (SPLIT.MD) =================
# Danh sách 55 bệnh nhân tập train đóng băng theo experiments/split.md và generate_la_dataset_120.sh
TRAIN_PATIENTS=(
    "L004" "L006" "L012" "L014" "L019" "L024" "L027" "L030" "L033" "L035"
    "L036" "L043" "L044" "L045" "L048" "L049" "L056" "L057" "L071" "L072"
    "L075" "L077" "L079" "L081" "L082" "L111" "L113" "L114" "L116" "L121"
    "L123" "L125" "L127" "L129" "L131" "L133" "L134" "L136" "L138" "L143"
    "L144" "L145" "L147" "L148" "L150" "L154" "L160" "L163" "L166" "L170"
    "L171" "L172" "L175" "L178" "L179"
)
[ "${#TRAIN_PATIENTS[@]}" -eq 55 ] || { echo "[ERROR] TRAIN_PATIENTS phải có đúng 55 bệnh nhân, thực tế: ${#TRAIN_PATIENTS[@]}" >&2; exit 1; }

# Tập kiểm định cố định: L110 (1 bệnh nhân, 133 lát cắt)
VAL_PATIENTS=("L110")
# Tập kiểm thử độc lập: L058, L064, L094, L107 (4 bệnh nhân, 757 lát cắt)
TEST_PATIENTS=("L058" "L064" "L094" "L107")

echo "[INFO] Cấu hình phân chia: Train=${#TRAIN_PATIENTS[@]} BN, Val=${#VAL_PATIENTS[@]} BN, Test=${#TEST_PATIENTS[@]} BN"

# ================= BƯỚC 1: PREFLIGHT DỮ LIỆU & SYMLINK =================
echo ""
echo "================================================================================"
echo "🔍 [BƯỚC 1/2] PREFLIGHT KIỂM TRA TÍCH HỢP DỮ LIỆU (8,218 / 133 / 757) & SYMLINK"
echo "================================================================================"

export SMOKE_DICOM_DIR="$DATASETS/aapm/split/"
export SMOKE_CACHE_DIR="$DATASETS/aapm/limited_angle/"
export SMOKE_TRAIN_PATIENTS="${TRAIN_PATIENTS[*]}"
export SMOKE_VAL_PATIENTS="${VAL_PATIENTS[*]}"
export SMOKE_TEST_PATIENTS="${TEST_PATIENTS[*]}"

python -u - <<'PYEOF'
import os
import numpy as np
import torch
from data.datamodule_factory import get_datamodule

train_patients = os.environ["SMOKE_TRAIN_PATIENTS"].split()
val_patients = os.environ["SMOKE_VAL_PATIENTS"].split()
test_patients = os.environ["SMOKE_TEST_PATIENTS"].split()

assert len(train_patients) == 55, f"Kỳ vọng 55 bệnh nhân train, thực tế: {len(train_patients)}"
assert val_patients == ["L110"], f"Kỳ vọng val=['L110'], thực tế: {val_patients}"
assert test_patients == ["L058", "L064", "L094", "L107"], f"Sai danh sách test: {test_patients}"

dm = get_datamodule(
    dataset_type="aapm",
    dicom_dir=os.environ["SMOKE_DICOM_DIR"],
    cache_dir=os.environ["SMOKE_CACHE_DIR"],
    setting_tag="limited_ang_120deg_numview_64_size_256_noise_0",
    start_ang=float(np.deg2rad(-60.0)),
    end_ang=float(np.deg2rad(60.0)),
    num_view=64,
    num_detectors=512,
    input_size=256,
    poisson_level=0.0,
    gaussian_level=0.0,
    use_precomputed=True,
    batch_size=4,
    num_workers=4,
    train_patients=train_patients,
    val_patients=val_patients,
    test_patients=test_patients,
)
dm.setup(None)

assert len(dm.train_dataset) == 8218, f"Kỳ vọng len(train_dataset) == 8218, thực tế: {len(dm.train_dataset)}"
assert len(dm.val_dataset) == 133, f"Kỳ vọng len(val_dataset) == 133, thực tế: {len(dm.val_dataset)}"
assert len(dm.test_dataset) == 757, f"Kỳ vọng len(test_dataset) == 757, thực tế: {len(dm.test_dataset)}"

def _slice_stem(path: str) -> str:
    base = os.path.basename(path)
    return base.split(".IMA")[0] if ".IMA" in base else os.path.splitext(base)[0]

MIN_SLICE_BYTES = 1000
for split_name, ds in [
    ("train", dm.train_dataset),
    ("val", dm.val_dataset),
    ("test", dm.test_dataset),
]:
    broken_paths = [p for p in ds.slices_path if not os.path.exists(p)]
    assert not broken_paths, f"Phát hiện {len(broken_paths)} symlink gãy trong tập {split_name}: {broken_paths[:5]}"
    # [2026-11-20 T0.7/T0.5] Chặn tệp 0-byte / <1000B ngay từ preflight cho cả DICOM lẫn cache_only .npy
    undersized_paths = [(p, os.path.getsize(p)) for p in ds.slices_path if os.path.getsize(p) < MIN_SLICE_BYTES]
    assert not undersized_paths, (
        f"Phát hiện {len(undersized_paths)} tệp lát cắt 0-byte/quá nhỏ (<{MIN_SLICE_BYTES}B) trong tập {split_name}: {undersized_paths[:5]}"
    )
    if getattr(ds, "use_precomputed", False) and getattr(ds, "sino_dir", None) and getattr(ds, "fbp_dir", None):
        bad_cache = []
        for p in ds.slices_path:
            stem = _slice_stem(p)
            for cache_file in (
                os.path.join(ds.sino_dir, f"{stem}.npy"),
                os.path.join(ds.fbp_dir, f"{stem}.npy"),
            ):
                if (not os.path.exists(cache_file)) or os.path.getsize(cache_file) < MIN_SLICE_BYTES:
                    sz = os.path.getsize(cache_file) if os.path.exists(cache_file) else -1
                    bad_cache.append((cache_file, sz))
        assert not bad_cache, (
            f"Phát hiện {len(bad_cache)} tệp cache .npy thiếu hoặc <{MIN_SLICE_BYTES}B trong tập {split_name}: {bad_cache[:5]}"
        )
    min_slice_bytes = min(os.path.getsize(p) for p in ds.slices_path) if ds.slices_path else 0
    phantom, fbp_u, sino = ds.__getitem__(0)
    assert tuple(phantom.shape) == (1, 256, 256), f"{split_name} phantom shape={tuple(phantom.shape)}"
    assert tuple(fbp_u.shape) == (1, 256, 256), f"{split_name} fbp_u shape={tuple(fbp_u.shape)}"
    assert tuple(sino.shape) == (1, 64, 512), f"{split_name} sino shape={tuple(sino.shape)}"
    assert torch.isfinite(phantom).all() and torch.isfinite(fbp_u).all() and torch.isfinite(sino).all(), (
        f"Phát hiện NaN/Inf ở mẫu 0 tập {split_name}"
    )
    print(
        f"[PREFLIGHT OK] {split_name}: len={len(ds)}, broken_symlinks=0, min_bytes={min_slice_bytes}, "
        f"item0_shapes=({tuple(phantom.shape)}, {tuple(fbp_u.shape)}, {tuple(sino.shape)})"
    )
PYEOF

# ================= BƯỚC 2: HUẤN LUYỆN SMOKE 1-EPOCH (BATCH_SIZE=4) =================
echo ""
echo "================================================================================"
echo "🎯 [BƯỚC 2/2] CHẠY 1-EPOCH SMOKE BENCHMARK: SOLAR_DualMamba LA-120° (batch_size=4)"
echo "================================================================================"

OUTPUT_DIR="$REPO/saved_models/SOLAR_DualMamba_120_smoke"
LOG_DIR="$REPO/lightning_logs/SOLAR_DualMamba_120_smoke"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

TRAIN_STDOUT_LOG="$LOG_RUN_DIR/${SLURM_JOB_ID:-local}_train_stdout.log"
EPOCH_START_TS=$(date +%s)

python -u baselines/SOLAR_DualMamba/train_solar_dualmamba_la.py \
    --dataset_type aapm \
    --dicom_dir "$DATASETS/aapm/split/" \
    --dataset_dir "$DATASETS/aapm/limited_angle/" \
    --output_dir "$OUTPUT_DIR" \
    --log_dir "$LOG_DIR" \
    --angle_range_deg 120.0 \
    --num_view 64 \
    --num_detectors 512 \
    --input_size 256 \
    --feature_dim 48 \
    --window_size 8 \
    --sino_loss_weight 0.1 \
    --poisson_level 0 \
    --gaussian_level 0 \
    --batch_size 4 \
    --n_iterations 8 \
    --cg_iters 4 \
    --max_epochs 1 \
    --lr 1e-4 \
    --final_lr 1e-5 \
    --seed 42 \
    --val_patients "${VAL_PATIENTS[@]}" \
    --train_patients "${TRAIN_PATIENTS[@]}" \
    --num_workers 4 \
    --use_precomputed | tee "$TRAIN_STDOUT_LOG"

EPOCH_END_TS=$(date +%s)
EPOCH_ELAPSED_SEC=$((EPOCH_END_TS - EPOCH_START_TS))
PEAK_VRAM_LINE=$(grep "Peak VRAM allocated" "$TRAIN_STDOUT_LOG" | tail -n 1 || true)

echo ""
echo "================================================================================"
echo "📊 [BENCHMARK SUMMARY] SOLAR_DualMamba LA-120° (55 train patients, batch_size=4, 1 epoch)"
echo "[BENCHMARK] Elapsed wall-clock seconds (1 epoch + val): ${EPOCH_ELAPSED_SEC}s"
echo "[BENCHMARK] ${PEAK_VRAM_LINE:-Peak VRAM allocated: <not-printed>}"
echo "[INFO] Kết thúc lúc $(date)"
echo "================================================================================"
