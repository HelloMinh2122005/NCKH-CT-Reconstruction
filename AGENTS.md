# AGENT INSTRUCTIONS & CONTEXT

Khi bắt đầu một session mới trong dự án này:

1. **Đọc Checkpoint đầu tiên:**
   Luôn mở và đọc file `CHECKPOINT.md` để nắm bắt:
   - Mục tiêu nghiên cứu (Limited-Angle CT nhằm giảm liều tia X cho bệnh nhân).
   - Các tham số hình học/vật lý đã chốt (Fan-beam, $120^\circ$ span, $64$ views, $256 \times 256$, ODL + ASTRA).
   - Trạng thái các job Slurm và tiến độ các đầu việc đã hoàn thành/cần làm tiếp theo.

2. **Quy tắc Slurm Cluster:**
   - Mọi script chạy job phải tuân thủ chuẩn NVIDIA MPS trên GPU A100/L40.
   - Luôn sử dụng logic kiểm tra VRAM của Admin: `/usr/local/bin/gpu_check.sh $REQUIRED_VRAM $SLURM_JOB_ID`.
   - **Bắt buộc lưu log tại:** `scripts/output/<tên script>/log/%j.out` và `%j.err`.

3. **Cập nhật Checkpoint:**
   Sau khi hoàn thành hoặc có thay đổi quan trọng trong session (tạo mô hình mới, chạy thí nghiệm, sửa lỗi), luôn cập nhật lại mục *6. Trạng Thái & Tiến Độ Dự Án* trong file `CHECKPOINT.md`.
