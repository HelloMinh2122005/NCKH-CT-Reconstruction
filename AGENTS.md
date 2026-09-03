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

3. **Quy tắc Bảo Toàn Chú Thích & Tính Toàn Vẹn Mã Nguồn (BẮT BUỘC - NGHIÊM NGẶT):**
   - **Tuyệt đối KHÔNG ĐƯỢC tự ý xóa, lược bỏ, rút gọn hoặc sửa đổi bất kỳ dòng comment, docstrings tiếng Việt giải thích chi tiết nào trong toàn bộ codebase.**
   - **Tuyệt đối KHÔNG ĐƯỢC viết code sai lệch, làm mâu thuẫn hoặc làm hỏng các logic và giá trị mặc định đã được giải thích trong comment khi người dùng chưa có yêu cầu rõ ràng.**
   - Mọi file mới được tạo hoặc chỉnh sửa phải duy trì 100% chú thích chi tiết từng dòng, kích thước tensor và ý nghĩa toán học/vật lý để phục vụ đọc hiểu và review code.

4. **Cập nhật Checkpoint:**
   Sau khi hoàn thành hoặc có thay đổi quan trọng trong session (tạo mô hình mới, chạy thí nghiệm, sửa lỗi), luôn cập nhật lại mục *6. Trạng Thái & Tiến Độ Dự Án* trong file `CHECKPOINT.md`.

5. **Quy tắc Viết Báo Cáo Tiến Độ (Reports):**
   - Báo cáo định kỳ trong `reports/<thời-gian>/MAIN.md` phải viết **ngắn gọn, trực diện, súc tích**.
   - **Tuyệt đối KHÔNG lặp lại** các thông tin đã được báo cáo trong các phiên trước (như mô tả chi tiết lại dataset, kiến trúc đã chạy, lý thuyết cũ...).
   - Chỉ tập trung báo cáo những mốc mới hoàn thành trong ngày (mô hình nào vừa train xong, checkpoint đạt đỉnh, kết quả test và visualize mới).
   - Mọi số liệu đo lường định lượng chi tiết phải đưa vào file `benchmark_results.csv` và dẫn link trực tiếp (`[benchmark_results.csv](benchmark_results.csv)`), không trình bày bảng biểu dài dòng trùng lặp trong `MAIN.md`.
