# AGENT INSTRUCTIONS & CONTEXT ROUTING
# DỰ ÁN: LIMITED-ANGLE CT RECONSTRUCTION (MINHPD)

> **Mục đích:** File này quy định các **nguyên tắc bất biến, quy trình khởi động session và bản đồ điều hướng ngữ cảnh** cho AI Agent. Mọi Agent tham gia dự án bắt buộc phải tuân thủ nghiêm ngặt các điều khoản trong tài liệu này.
>
> **Cập nhật 26/10/2026:** Cụm DGX-A100 cũ đã mất (toàn bộ checkpoint, dataset, môi trường). Dự án chạy lại từ đầu trên **HPC-UIT** với tập test **nhiều bệnh nhân** (bài SOICT 2026 bị từ chối vì chỉ test 1 bệnh nhân). Nếu đang làm việc trong workspace `agents-research/`, đọc thêm `.lab/MISSION_BRIEF.md` (ngoài repo).

---

## 🛑 CÁC ĐIỀU CẤM KỴ TUYỆT ĐỐI (CRITICAL CONSTRAINTS)

1. **Bảo Toàn 100% Chú Thích & Tính Toàn Vẹn Mã Nguồn (NGHIÊM NGẶT - BẮT BUỘC):**
   - **Tuyệt đối KHÔNG ĐƯỢC tự ý xóa, lược bỏ, rút gọn hoặc thay đổi bất kỳ dòng comment, docstrings tiếng Việt giải thích chi tiết nào trong toàn bộ codebase.**
   - **Tuyệt đối KHÔNG ĐƯỢC viết code sai lệch, làm mâu thuẫn hoặc làm hỏng các logic và giá trị mặc định đã được giải thích trong comment khi người dùng chưa yêu cầu rõ ràng.**
   - Mọi file mới được tạo hoặc chỉnh sửa phải duy trì 100% chú thích chi tiết từng dòng, kích thước tensor và ý nghĩa toán học/vật lý.

2. **Thao Tác Git & GitHub (Cập nhật 26/10/2026):**
   - **Sửa code và commit/push chỉ trên máy local**, trong thư mục:
     ```bash
     cd /home/phandinhminh/Downloads/kltn/agents-research/temp-github/NCKH-CT-Reconstruction
     ```
     Dùng `git status -uno` để kiểm tra nhanh. Chỉ commit/push khi người dùng yêu cầu.
   - **Không sửa file qua thư mục mount `hpc-uit-server/`.** Bản trên server chỉ được cập nhật bằng `git pull`.
   - **Trên server HPC-UIT chỉ clone/pull** (không commit), trong `/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction`:
     - Bản clone là sparse checkout **loại trừ `scripts/output/`** (log cũ của cụm A100), để log cũ không bị kéo về nhầm thành kết quả `hpc-uit:`.
     - **Không bao giờ** chạy `git sparse-checkout disable` trên server.
     - Cập nhật bằng: `git -C /datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction pull --ff-only`. Không rsync chồng lên bản clone.

3. **BẢO MẬT CÔNG NGHỆ BÀI BÁO SOICT 2026 (`papers/soict2026/main.tex`):**
   - **TUYỆT ĐỐI KHÔNG ĐƯỢC ĐƯA KẾT QUẢ / KIẾN TRÚC CỦA MÔ HÌNH SOLAR VÀO BÀI BÁO SOICT 2026.**
   - Bài báo SOICT 2026 chỉ so sánh đối chứng các mô hình Baseline unrolling bậc 1 (`LEARN_Longformer`, `LEARN_LongNet`, `LEARN_Mamba`, và `FBP`).
   - Kiến trúc đề xuất **SOLAR** cùng kết quả được giữ riêng cho khóa luận và bài báo tạp chí.

4. **Liêm Chính Số Liệu (Provenance):**
   - **Tuyệt đối không bịa số** (metric, độ lệch chuẩn, seed, job ID, trích dẫn). "Chưa chạy" là trạng thái hợp lệ.
   - Mọi số liệu phải gắn nhãn nguồn:
     - `hpc-uit:<jobid>` cho job mới, log nằm trong `scripts/output-hpc-uit/`.
     - `legacy:<jobid>` cho cụm A100 cũ, log nằm trong `scripts/output/`.
     - `legacy-unverifiable` khi không còn log.
   - Số liệu cũ được đo trên **1 bệnh nhân (`L310`)**, trên cụm đã mất, và checkpoint đã mất. **Không bao giờ đặt chung bảng với số liệu `hpc-uit`.**

5. **Cụm HPC-UIT (Tài Khoản Dùng Chung):**
   - Tài khoản `doanhbc` là của thầy Bùi Cao Doanh, **dùng chung với các sinh viên khác**. Chỉ làm việc trong `/datastore/doanhbc/minhdphan`. Không đọc/ghi/xóa thư mục của người khác, không `scancel` job của người khác, không để dữ liệu trong `$HOME` (10 GB dùng chung).
   - **Không tính toán trên login node** (python, pip, conda, huấn luyện): mọi thứ chạy qua `sbatch`.
   - Giới hạn chung cho cả tài khoản: ≤ 3 job chạy song song, các lần submit cách nhau ≥ 1 giờ, mỗi job ≤ 72 giờ.
   - Hướng dẫn chính thức: `Huong_Dan_Su_Dung_Tai_Nguyen_HPC.docx.pdf` (trong workspace).

---

## 🧭 BẢN ĐỒ ĐIỀU HƯỚNG NGỮ CẢNH (CONTEXT ROUTING TABLE)

Để tránh hiện tượng **Context Rot** và lãng phí token, Agent **chỉ mở đúng tài liệu liên quan đến tác vụ hiện tại**, không quét đọc tràn lan:

| Nhiệm vụ cần thực hiện | File tài liệu cần đọc |
| :--- | :--- |
| **Khởi động session / Nắm bắt tiến độ** | [CHECKPOINT.md](CHECKPOINT.md). Trong workspace: thêm `.lab/MISSION_BRIEF.md` |
| **Từng bước khôi phục trên HPC-UIT** | `.lab/RECOVERY_CHECKLIST.md` + `.lab/recovery/state.json` (workspace) |
| **Vì sao phải chạy lại / lỗi cần tránh khi viết bài** | [papers/soict2026/review-soict.md](papers/soict2026/review-soict.md) |
| **Tra cứu tham số CT / cụm HPC-UIT / Dataset** | [SYSTEM_SPEC.md](SYSTEM_SPEC.md) |
| **Lịch sử thử nghiệm trên cụm A100 cũ (08–09/2026)** | [ARCHIVE_LOG.md](ARCHIVE_LOG.md) |
| **Chỉnh sửa / Phát triển kiến trúc SOLAR** | [SOLAR_ARCHITECTURE.md](SOLAR_ARCHITECTURE.md) + `baselines/SOLAR_DualMamba/models.py` |
| **Số liệu cũ (legacy, 1 bệnh nhân)** | [reports/sep-21-2026/benchmark_results.csv](reports/sep-21-2026/benchmark_results.csv), [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md), log trong `scripts/output/` |
| **Số liệu mới (HPC-UIT)** | `reports/<ngày>/benchmark_results.csv` (cột job ID), log trong `scripts/output-hpc-uit/` |
| **Viết / sửa script Slurm** | Template `.claude/skills/slurm-jobs/references/sbatch-template-hpc-uit.sh` (workspace), kiểm tra bằng `.claude/tools/slurm_lint.py` |
| **Biên tập bài báo Hội nghị SOICT** | `papers/soict2026/main.tex` + `review-soict.md` *(Nhớ tuân thủ điều cấm kỵ SOLAR)* |

---

## ⚡ QUY TRÌNH KHỞI ĐỘNG PHIÊN LÀM VIỆC (SESSION INIT WORKFLOW)

Khi bắt đầu một session mới, Agent phải thực hiện tuần tự 3 bước:
1. **Đọc [CHECKPOINT.md](CHECKPOINT.md) §1–3:** tình hình hiện tại, trạng thái cụm HPC-UIT, việc cần làm tiếp.
2. **Kiểm tra thực tế:**
   - Job HPC-UIT: `squeue`/`sacct` (qua hpc-operator) và `.lab/submit-log.jsonl`.
   - Log mới đã kéo về: `scripts/output-hpc-uit/<tên script>/log/<jobid>.out`.
   - Checkpoint mới chỉ nằm trên server: `/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction/saved_models/`. Local không có `saved_models/`.
3. **Báo cáo tình trạng:** Trình bày bảng tóm tắt cho người dùng và đề xuất bước tiếp theo **trước khi** sửa mã nguồn hay nộp job.

---

## 📋 QUY TẮC CẬP NHẬT TIẾN ĐỘ & BÁO CÁO

1. **Cập nhật Checkpoint:**
   Sau khi hoàn thành hoặc có thay đổi quan trọng trong session (job chạy xong, test benchmark xong, sửa code), luôn cập nhật lại mục *1–3* trong file [CHECKPOINT.md](CHECKPOINT.md). Lịch sử cũ chuyển sang [ARCHIVE_LOG.md](ARCHIVE_LOG.md), không xóa.
2. **Quy tắc Báo cáo (Reports):**
   - Báo cáo định kỳ trong `reports/<thời-gian>/MAIN.md` phải viết **ngắn gọn, trực diện, súc tích**.
   - **Tuyệt đối KHÔNG lặp lại** thông tin cũ (dataset, lý thuyết nền). Chỉ báo cáo kết quả mới hoàn thành trong ngày.
   - Mọi số liệu đo lường định lượng chi tiết phải đưa vào `benchmark_results.csv` kèm job ID.
3. **Log job:** Mọi job ghi log vào `scripts/output/<tên script>/log/%j.out|.err` **trên server**. Khi kéo về local, log nằm trong `scripts/output-hpc-uit/` (không ghi đè), không bao giờ vào `scripts/output/`, nơi chỉ chứa log cũ.
