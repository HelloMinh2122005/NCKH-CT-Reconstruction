#!/bin/bash
# ==============================================================================
# SCRIPT SMOKE TEST HỆ THỐNG TRÊN CỤM HPC-UIT (MIG GPU)
# Profile: hpc-uit, Partition: small (gpu:1g.18gb:1), QOS: mig1-18, Thời gian: 30 phút
#
# Mục đích:
# 1. Kiểm tra tính đúng đắn toàn diện của pipeline huấn luyện trên node tính toán HPC-UIT
#    cho cả 2 mô hình: LEARN (Original) và SOLAR_DualMamba trong MỘT job Slurm duy nhất.
# 2. Xác minh tính tương thích của CUDA kernels: ODL RayTransform, ASTRA-toolbox GPU,
#    Mamba selective scan (mamba_ssm), và PyTorch autograd forward/backward passes.
# 3. Chạy chế độ on-the-fly sinogram (--use_precomputed False), hoàn toàn độc lập với
#    dữ liệu cache .npy (chưa cần sinh cache trước).
# 4. Giới hạn số lượng mini-batch (limit_train_batches=3, limit_val_batches=2, max_epochs=1)
#    để hoàn thành nhanh trong vòng vài phút.
# 5. Ghi nhận Git commit HEAD và lượng VRAM thực tế đỉnh điểm (Peak VRAM allocated).
#
# LƯU Ý QUAN TRỌNG VỀ TÍNH KHOA HỌC:
# Các số liệu đo được từ job smoke test này (PSNR, SSIM, loss, thời gian epoch) CHỈ nhằm
# mục đích kiểm chứng kỹ thuật rằng pipeline code, dữ liệu và CUDA kernels vận hành thông suốt.
# Đây TUYỆT ĐỐI KHÔNG PHẢI là kết quả nghiên cứu hay số liệu thực nghiệm khoa học.
# ==============================================================================
#SBATCH --job-name=smoke_hpcuit
#SBATCH --output=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output/smoke_hpcuit/log/%j.out
#SBATCH --error=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output/smoke_hpcuit/log/%j.err
#SBATCH --account=uit
#SBATCH --qos=mig1-18
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1g.18gb:1

set -euo pipefail

# ------------------------------------------------------------------------------
# 1. ĐỊNH NGHĨA ĐƯỜNG DẪN KHÔNG GIAN LÀM VIỆC TẠI /datastore (BẢO VỆ $HOME 10 GB)
# ------------------------------------------------------------------------------
REPO=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction
ENV_DIR=/datastore/doanhbc/minhdphan/envs/longnet
DATASETS=/datastore/doanhbc/minhdphan/datasets

# Thư mục chứa log bắt buộc phải được tạo trước khi Slurm xuất log
mkdir -p "$REPO/scripts/output/smoke_hpcuit/log"

cleanup() {
    local rc=$?
    echo "[INFO] cleanup kết thúc với mã thoát rc=$rc lúc $(date)"
}
trap cleanup EXIT

echo "================================================================================"
echo "🚀 BẮT ĐẦU SMOKE TEST HPC-UIT: LEARN & SOLAR_DualMamba"
echo "[INFO] Thời điểm bắt đầu: $(date)"
echo "[INFO] Máy chủ thực thi: $(hostname)"
echo "[INFO] SLURM_JOB_ID: ${SLURM_JOB_ID:-<unset>}"
echo "[INFO] SLURM_JOB_PARTITION: ${SLURM_JOB_PARTITION:-<unset>}"
echo "[INFO] SLURM_JOB_QOS: ${SLURM_JOB_QOS:-<unset>}"
echo "[INFO] Git commit HEAD: $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo '<git-not-found>')"
echo "================================================================================"

# Hiển thị cấu hình GPU MIG được phân bổ để theo dõi tài nguyên phần cứng
nvidia-smi

# ------------------------------------------------------------------------------
# 2. THIẾT LẬP MÔI TRƯỜNG PYTHON VÀ CẤU HÌNH CACHE
# ------------------------------------------------------------------------------
# Kích hoạt môi trường Python ảo venv từ /datastore
set +u
source "$ENV_DIR/bin/activate"
set -u

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Chuyển hướng toàn bộ bộ nhớ đệm (cache) về /datastore để ngăn ngừa tràn hạn mức $HOME 10 GB
export XDG_CACHE_HOME=/datastore/doanhbc/minhdphan/cache
export PIP_CACHE_DIR=$XDG_CACHE_HOME/pip
export TORCH_HOME=$XDG_CACHE_HOME/torch
export HF_HOME=$XDG_CACHE_HOME/hf
export TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton
export CONDA_PKGS_DIRS=$XDG_CACHE_HOME/conda-pkgs
mkdir -p "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TORCH_HOME" "$HF_HOME" "$TRITON_CACHE_DIR" "$CONDA_PKGS_DIRS"

# Nạp module CUDA toolkit 12.9 nếu có để hỗ trợ biên dịch JIT kernel
if command -v module &> /dev/null; then
    module load cuda12.9/toolkit/12.9.1 2>/dev/null || true
elif [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
    module load cuda12.9/toolkit/12.9.1 2>/dev/null || true
fi

# ------------------------------------------------------------------------------
# 3. THIẾT LẬP THƯ MỤC KIỂM THỬ VÀ DỮ LIỆU ĐẦU VÀO
# ------------------------------------------------------------------------------
# Thư mục lưu checkpoint và log tạm thời cho smoke test (tách biệt hoàn toàn với saved_models chính)
SMOKE_OUTPUT_DIR="$REPO/saved_models/smoke_test"
SMOKE_LOG_DIR="$REPO/scripts/output/smoke_hpcuit/lightning_logs"
mkdir -p "$SMOKE_OUTPUT_DIR" "$SMOKE_LOG_DIR"

# Thư mục chứa dữ liệu CT AAPM gốc phục vụ tạo sinogram trực tiếp on-the-fly
DICOM_DIR="$DATASETS/aapm/split"

# Chọn mẫu bệnh nhân từ phân chia tập (L004 thuộc 55 bệnh nhân train, L110 thuộc val)
# Nếu symlink L110 chưa sẵn sàng trong split/train, fallback nạp kèm L006 để luôn có dữ liệu
TRAIN_PATIENT="L004"
VAL_PATIENTS=("L110" "L006")

# ------------------------------------------------------------------------------
# 4. TIẾN TRÌNH SMOKE TEST TUẦN TỰ 1: MÔ HÌNH BASELINE LEARN
# ------------------------------------------------------------------------------
echo ""
echo "================================================================================"
echo "🎯 [PHẦN 1/2] KIỂM TRẢ SMOKE TEST: LEARN (Unrolled CNN)"
echo "   - Cung quét góc giới hạn: 120.0 độ, 64 views, 512 detectors, 256x256"
echo "   - Chế độ sinh dữ liệu: ON-THE-FLY (--use_precomputed False), không cần file cache .npy"
echo "   - Số lượng bước lặp: max_epochs=1, limit_train_batches=3, limit_val_batches=2"
echo "   - Ghi chú: Kết quả chỉ để kiểm tra pipeline kỹ thuật, không dùng làm kết quả nghiên cứu"
echo "================================================================================"

python -u baselines/LEARN/train_learn_la.py \
    --dataset_type aapm \
    --dicom_dir "$DICOM_DIR" \
    --output_dir "$SMOKE_OUTPUT_DIR/LEARN/" \
    --log_dir "$SMOKE_LOG_DIR" \
    --train_patients "$TRAIN_PATIENT" \
    --val_patients "${VAL_PATIENTS[@]}" \
    --angle_range_deg 120.0 \
    --num_view 64 \
    --num_detectors 512 \
    --input_size 256 \
    --batch_size 1 \
    --max_epochs 1 \
    --limit_train_batches 3 \
    --limit_val_batches 2 \
    --num_workers 2 \
    --use_precomputed False

echo ""
echo "[INFO] Đo kiểm bộ nhớ GPU sau khi hoàn tất LEARN smoke test:"
python -c "import torch; print(f'🔥 LEARN Peak VRAM Allocated: {torch.cuda.max_memory_allocated() / (1024**2):.2f} MB') if torch.cuda.is_available() else print('CUDA not available')"

# ------------------------------------------------------------------------------
# 5. TIẾN TRÌNH SMOKE TEST TUẦN TỰ 2: MÔ HÌNH ĐỘT PHÁ SOLAR_DualMamba
# ------------------------------------------------------------------------------
echo ""
echo "================================================================================"
echo "🎯 [PHẦN 2/2] KIỂM TRA SMOKE TEST: SOLAR_DualMamba (Khối điều hòa kép & Mamba SSM)"
echo "   - Cung quét góc giới hạn: 120.0 độ, 64 views, 512 detectors, 256x256"
echo "   - Chế độ sinh dữ liệu: ON-THE-FLY (--use_precomputed False), không cần file cache .npy"
echo "   - Số lượng bước lặp: max_epochs=1, limit_train_batches=3, limit_val_batches=2"
echo "   - Ghi chú: Kết quả chỉ để kiểm tra pipeline kỹ thuật, không dùng làm kết quả nghiên cứu"
echo "================================================================================"

python -u baselines/SOLAR_DualMamba/train_solar_dualmamba_la.py \
    --dataset_type aapm \
    --dicom_dir "$DICOM_DIR" \
    --output_dir "$SMOKE_OUTPUT_DIR/SOLAR_DualMamba/" \
    --log_dir "$SMOKE_LOG_DIR" \
    --train_patients "$TRAIN_PATIENT" \
    --val_patients "${VAL_PATIENTS[@]}" \
    --angle_range_deg 120.0 \
    --num_view 64 \
    --num_detectors 512 \
    --input_size 256 \
    --batch_size 1 \
    --max_epochs 1 \
    --limit_train_batches 3 \
    --limit_val_batches 2 \
    --num_workers 2 \
    --use_precomputed False

echo ""
echo "[INFO] Đo kiểm bộ nhớ GPU sau khi hoàn tất SOLAR_DualMamba smoke test:"
python -c "import torch; print(f'🔥 SOLAR_DualMamba Peak VRAM Allocated: {torch.cuda.max_memory_allocated() / (1024**2):.2f} MB') if torch.cuda.is_available() else print('CUDA not available')"

# ------------------------------------------------------------------------------
# 6. TỔNG KẾT VÀ XÁC THỰC HOÀN TẤT
# ------------------------------------------------------------------------------
echo ""
echo "================================================================================"
echo "✅ HOÀN TẤT TOÀN BỘ SMOKE TEST CHO LEARN VÀ SOLAR_DualMamba TRÊN HPC-UIT!"
echo "[INFO] Thời điểm kết thúc: $(date)"
echo "[INFO] Git commit HEAD: $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo '<git-not-found>')"
echo "================================================================================"
