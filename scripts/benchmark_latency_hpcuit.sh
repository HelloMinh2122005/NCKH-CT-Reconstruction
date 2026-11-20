#!/bin/bash
# ==============================================================================
# SCRIPT BENCHMARK ĐỘ TRỄ SUY LUẬN & ĐỈNH BỘ NHỚ VRAM TRÊN HPC-UIT (WAVE E)
# Đề tài: Tái tạo ảnh cắt lớp CT góc giới hạn (Limited-Angle CT Reconstruction)
# Kịch bản: scripts/benchmark_latency_hpcuit.sh
# Mục tiêu: Thực thi scripts/benchmark_latency_hpcuit.py để đo thời gian suy luận
#   (mean/std ms/slice, FPS) và đỉnh bộ nhớ VRAM cho các mô hình Wave E (SOLAR_DualMamba
#   cùng các baseline LEARN, RegFormer, DuDoTrans, Mamba, Longformer, LongNet)
#   trên cùng một phân vùng GPU MIG của HPC-UIT như Wave A/B.
# Profile: hpc-uit, Partition: normal (gpu:2g.35gb:1, 8 CPU, 16GB RAM, Time: 00:30:00)
# Tuân thủ nghiêm ngặt giao thức khoa học GT-03 / Wave E:
#   - batch_size = 1 (kích thước batch suy luận thực tế từng lát cắt)
#   - Kiểu dữ liệu FP32 (float32) trên tensor fbp [1, 1, 256, 256] và sino [1, 1, 64, 512]
#   - warmup_runs >= 20 lượt khởi động bỏ qua đo lường
#   - test_runs >= 100 lượt đo lường chính thức
#   - torch.cuda.synchronize() trước và sau mỗi forward pass
#   - Cùng phân vùng phần hardware MIG gpu:2g.35gb:1
#   - Báo cáo Mean ± Std latency (ms) và FPS vào CSV chuẩn 9 cột
# ==============================================================================
#SBATCH --job-name=benchmark_latency_hpcuit
#SBATCH --output=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output-hpc-uit/benchmark_latency_hpcuit/log/%j.out
#SBATCH --error=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/scripts/output-hpc-uit/benchmark_latency_hpcuit/log/%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:2g.35gb:1

set -euo pipefail

# Hàm dọn dẹp tài nguyên và ghi nhận trạng thái kết thúc job
cleanup() {
    local rc=$?
    echo "[INFO] cleanup rc=$rc at $(date)"
}
trap cleanup EXIT

# Đường dẫn không gian làm việc cô lập trên cụm HPC-UIT (cluster.json)
REPO=/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction
ENV_DIR=/datastore/doanhbc/minhdphan/envs/longnet
DATASETS=/datastore/doanhbc/minhdphan/datasets

# Đảm bảo thư mục log Slurm và thư mục reports tồn tại trước khi chạy
mkdir -p "$REPO/scripts/output-hpc-uit/benchmark_latency_hpcuit/log"
mkdir -p "$REPO/reports"

echo "=== HPC-UIT WAVE E: INFERENCE LATENCY & VRAM PROFILING BENCHMARK ==="
echo "[INFO] Start at: $(date)"
echo "[INFO] Hostname: $(hostname)"
echo "[INFO] SLURM_JOB_ID: ${SLURM_JOB_ID:-<unset>}"
echo "[INFO] SLURM_JOB_PARTITION: ${SLURM_JOB_PARTITION:-normal}"
echo "[INFO] Git commit: $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo '<git-not-found>')"

# Ghi lại cấu hình phần cứng GPU MIG được cấp phát để bảo đảm kiểm chứng nguồn gốc
nvidia-smi

# Kích hoạt môi trường Python trên /datastore (tránh tràn hạn ngạch 10GB của $HOME)
set +u
source "$ENV_DIR/bin/activate"
set -u

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Chuyển hướng toàn bộ bộ nhớ đệm (cache) về /datastore để bảo vệ quota $HOME
export XDG_CACHE_HOME=/datastore/doanhbc/minhdphan/cache
export PIP_CACHE_DIR=$XDG_CACHE_HOME/pip
export TORCH_HOME=$XDG_CACHE_HOME/torch
export HF_HOME=$XDG_CACHE_HOME/hf
export TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton

# Tệp CSV lưu trữ kết quả benchmark độ trễ suy luận chuẩn hóa theo Wave E
CSV_OUTPUT="$REPO/reports/benchmark_latency_results.csv"

# Nếu người dùng truyền danh sách mô hình từ dòng lệnh thì dùng danh sách đó,
# ngược lại mặc định đo toàn bộ 4 mô hình cốt lõi của Wave A/E và 3 baseline mở rộng
if [ "$#" -gt 0 ]; then
    TARGET_MODELS=("$@")
else
    # 4 mô hình cốt lõi Wave A: SOLAR_DualMamba, LEARN, RegFormer, DuDoTrans
    # 3 baseline mở rộng: Mamba, Longformer, LongNet
    TARGET_MODELS=("solar" "learn" "regformer" "dudotrans" "mamba" "longformer" "longnet")
fi

echo "[INFO] Danh sách mô hình đo lường: ${TARGET_MODELS[*]}"
echo "[INFO] Đường dẫn tệp CSV kết quả: $CSV_OUTPUT"

# Hàm phụ trợ kiểm tra và đối soát đường dẫn checkpoint Wave A 90°
# Giúp xác nhận nguồn gốc mô hình tương ứng trong saved_models/
check_wave_a_checkpoint() {
    local MODEL_KEY="$1"
    local CKPT_DIR=""

    case "$MODEL_KEY" in
        "solar")
            CKPT_DIR="$REPO/saved_models/SOLAR_DualMamba_90"
            ;;
        "learn")
            CKPT_DIR="$REPO/saved_models/LEARN_90"
            ;;
        "regformer")
            CKPT_DIR="$REPO/saved_models/RegFormer_90"
            ;;
        "dudotrans")
            CKPT_DIR="$REPO/saved_models/DuDoTrans_90"
            ;;
        "mamba")
            CKPT_DIR="$REPO/saved_models/LEARN_Mamba_90"
            ;;
        "longformer")
            CKPT_DIR="$REPO/saved_models/LEARN_Longformer_90"
            ;;
        "longnet")
            CKPT_DIR="$REPO/saved_models/LEARN_LongNet_90"
            ;;
    esac

    if [ -n "$CKPT_DIR" ] && [ -d "$CKPT_DIR" ]; then
        local FOUND_CKPT
        FOUND_CKPT=$(find "$CKPT_DIR" -name "*.ckpt" 2>/dev/null | head -n 1)
        if [ -n "$FOUND_CKPT" ]; then
            echo "[INFO] Khớp checkpoint Wave A (90°): $FOUND_CKPT"
        else
            echo "[INFO] Thư mục checkpoint tồn tại nhưng chưa có tệp .ckpt: $CKPT_DIR"
        fi
    else
        echo "[INFO] Chưa có thư mục checkpoint Wave A cục bộ cho ${MODEL_KEY} (${CKPT_DIR:-N/A}), tiến hành benchmark trực tiếp kiến trúc mô hình."
    fi
}

# Lặp qua từng mô hình để đo lường độ trễ suy luận và VRAM theo đúng giao thức khoa học
for MODEL in "${TARGET_MODELS[@]}"; do
    echo "================================================================================"
    echo "🎯 ĐO LƯỜNG ĐỘ TRỄ SUY LUẬN & VRAM: ${MODEL^^} (LA-90°, 64 views)"
    echo "================================================================================"

    # Kiểm tra sự tồn tại của checkpoint Wave A (90°) tương ứng
    check_wave_a_checkpoint "$MODEL"

    # Thực thi benchmark với batch_size=1, FP32, warmup=20, test_runs=100
    python -u scripts/benchmark_latency_hpcuit.py \
        --model "$MODEL" \
        --input_size 256 \
        --num_view 64 \
        --num_detectors 512 \
        --warmup_runs 20 \
        --test_runs 100 \
        --angle_range_deg 90.0 \
        --hardware_profile "HPC-UIT MIG Slice (gpu:2g.35gb:1)" \
        --output_csv "$CSV_OUTPUT"

    # Ghi nhận trạng thái bộ nhớ VRAM sau khi benchmark từng mô hình
    if command -v nvidia-smi &>/dev/null; then
        echo "[INFO] VRAM sử dụng sau khi benchmark ${MODEL^^}:"
        nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader
    fi
    echo ""
done

echo "================================================================================"
echo "📊 TỔNG HỢP KẾT QUẢ WAVE E BENCHMARK:"
if [ -f "$CSV_OUTPUT" ]; then
    echo "[INFO] Nội dung tệp $CSV_OUTPUT:"
    cat "$CSV_OUTPUT"
else
    echo "[WARN] Tệp CSV kết quả chưa được tạo: $CSV_OUTPUT"
fi
echo "================================================================================"

echo "[INFO] Wave E latency benchmark hoàn tất thành công lúc $(date)"
