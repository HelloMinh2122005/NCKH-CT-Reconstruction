# AGENT INSTRUCTIONS & CONTEXT ROUTING
# DỰ ÁN: LIMITED-ANGLE CT RECONSTRUCTION (MINHPD)

> **Mục đích:** File này quy định các **nguyên tắc bất biến, quy trình khởi động session và bản đồ điều hướng ngữ cảnh** cho AI Agent. Mọi Agent tham gia dự án bắt buộc phải tuân thủ nghiêm ngặt các điều khoản trong tài liệu này.

---

## 🛑 CÁC ĐIỀU CẤM KỴ TUYỆT ĐỐI (CRITICAL CONSTRAINTS)

1. **Bảo Toàn 100% Chú Thích & Tính Toàn Vẹn Mã Nguồn (NGHIÊM NGẶT - BẮT BUỘC):**
   - **Tuyệt đối KHÔNG ĐƯỢC tự ý xóa, lược bỏ, rút gọn hoặc thay đổi bất kỳ dòng comment, docstrings tiếng Việt giải thích chi tiết nào trong toàn bộ codebase.**
   - **Tuyệt đối KHÔNG ĐƯỢC viết code sai lệch, làm mâu thuẫn hoặc làm hỏng các logic và giá trị mặc định đã được giải thích trong comment khi người dùng chưa yêu cầu rõ ràng.**
   - Mọi file mới được tạo hoặc chỉnh sửa phải duy trì 100% chú thích chi tiết từng dòng, kích thước tensor và ý nghĩa toán học/vật lý.

2. **Thao Tác Git & GitHub (BẮT BUỘC TRÊN LOCAL MOUNT - KHÔNG SSH):**
   - **Bắt buộc `cd` trên local** vào thư mục mount:
     ```bash
     cd /home/phandinhminh/Downloads/kltn/agents-research/uittogether3-slurm-server/MinhPD
     ```
   - **Tuyệt đối KHÔNG SSH lên server để thao tác Git.**
   - Sử dụng `git status -uno` để kiểm tra nhanh mà không tốn thời gian duyệt các file ảnh dataset lớn qua SSHFS.

3. **BẢO MẬT CÔNG NGHỆ BÀI BÁO SOICT 2026 (`papers/soict2026/main.tex`):**
   - **TUYỆT ĐỐI KHÔNG ĐƯỢC ĐƯA KẾT QUẢ / KIẾN TRÚC CỦA MÔ HÌNH SOLAR VÀO BÀI BÁO SOICT 2026.**
   - Bài báo SOICT 2026 chỉ so sánh đối chứng các mô hình Baseline unrolling bậc 1 (`LEARN_Longformer`, `LEARN_LongNet`, `LEARN_Mamba`, và `FBP`).
   - Kiến trúc đề xuất **SOLAR** cùng kết quả test vượt trội được giữ bí mật để phục vụ riêng cho bài báo Journal Q1 đỉnh cao (IEEE TMI / MedIA, IF > 10).

---

## 🧭 BẢN ĐỒ ĐIỀU HƯỚNG NGỮ CẢNH (CONTEXT ROUTING TABLE)

Để tránh hiện tượng **Context Rot** và lãng phí token, Agent **chỉ mở đúng tài liệu liên quan đến tác vụ hiện tại**, không quét đọc tràn lan:

| Nhiệm vụ cần thực hiện | File tài liệu duy nhất cần đọc |
| :--- | :--- |
| **Khởi động session / Nắm bắt tiến độ** | [CHECKPOINT.md](CHECKPOINT.md) + file log trong `scripts/output/` |
| **Tra cứu tham số CT / Slurm / Dataset gốc** | [SYSTEM_SPEC.md](SYSTEM_SPEC.md) |
| **Tra cứu lịch sử các thử nghiệm cũ (tháng 8)** | [ARCHIVE_LOG.md](ARCHIVE_LOG.md) |
| **Chỉnh sửa / Phát triển kiến trúc SOLAR** | [SOLAR_ARCHITECTURE.md](SOLAR_ARCHITECTURE.md) + `baselines/SOLAR_*/models.py` |
| **Kiểm tra / Phân tích số liệu Benchmark** | [benchmark_results.csv](reports/sep-14-2026/benchmark_results.csv) hoặc [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md) |
| **Biên tập bài báo Hội nghị SOICT** | `papers/soict2026/main.tex` *(Nhớ tuân thủ điều cấm kỵ SOLAR)* |

---

## ⚡ QUY TRÌNH KHỞI ĐỘNG PHIÊN LÀM VIỆC (SESSION INIT WORKFLOW)

Khi bắt đầu một session mới, Agent phải thực hiện tuần tự 3 bước:
1. **Đọc [CHECKPOINT.md](CHECKPOINT.md):** Xác định các job Slurm đang chạy hoặc vừa được giao và các TODO ưu tiên.
2. **Kiểm tra thực tế:**
   - Đọc đuôi file log tương ứng tại `scripts/output/<tên_job>/log/%j.out`.
   - Kiểm tra các file `.ckpt` mới nhất trong `saved_models/<tên_mô_hình>/`.
3. **Báo cáo tình trạng:** Trình bày bảng tóm tắt trạng thái job cho người dùng và đề xuất bước hành động tiếp theo trước khi thay đổi mã nguồn.

---

## 📋 QUY TẮC CẬP NHẬT TIẾN ĐỘ & BÁO CÁO

1. **Cập nhật Checkpoint:**
   Sau khi hoàn thành hoặc có thay đổi quan trọng trong session (job chạy xong, test benchmark xong, sửa code), luôn cập nhật lại mục *1 & 2* trong file [CHECKPOINT.md](CHECKPOINT.md).
2. **Quy tắc Báo cáo (Reports):**
   - Báo cáo định kỳ trong `reports/<thời-gian>/MAIN.md` phải viết **ngắn gọn, trực diện, súc tích**.
   - **Tuyệt đối KHÔNG lặp lại** thông tin cũ (dataset, lý thuyết nền). Chỉ báo cáo kết quả mới hoàn thành trong ngày.
   - Mọi số liệu đo lường định lượng chi tiết phải đưa vào `benchmark_results.csv`.
