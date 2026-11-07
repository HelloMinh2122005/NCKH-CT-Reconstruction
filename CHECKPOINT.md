# CHECKPOINT DỰ ÁN NCKH: LIMITED-ANGLE CT RECONSTRUCTION
> **TRẠNG THÁI ACTIVE (Cập nhật: 26/10/2026)**: khởi động lại trên HPC-UIT sau khi mất cụm DGX-A100.
>
> *Lưu ý cho AI Agent:* File này chỉ lưu **trạng thái tức thời, job đang chạy và việc cần làm tiếp**. Giữ nó ngắn.
> - Quy tắc bắt buộc và bản đồ tài liệu: [AGENTS.md](AGENTS.md).
> - Thông số CT, cụm HPC-UIT, dataset: [SYSTEM_SPEC.md](SYSTEM_SPEC.md).
> - Lịch sử, gồm trạng thái cuối cùng trên cụm A100 ngày 22/09/2026: [ARCHIVE_LOG.md](ARCHIVE_LOG.md) §6.

---

## 1. Tình Hình Hiện Tại

- **Cụm DGX-A100 cũ đã mất (sau 22/09/2026).**
  - Mất toàn bộ checkpoint (`saved_models/`), cache dữ liệu LA, dataset gốc và môi trường conda.
  - 6 job đang chờ (`73105`, `73106`, `73107`, `73145`, `73146`, `73147`) **chưa từng chạy**, nên không có số liệu.
- **Còn lại:**
  - Mã nguồn.
  - 114 log job cũ trong `scripts/output/` (chỉ đọc, bằng chứng `legacy:<jobid>`).
  - Báo cáo trong `reports/` và các bản thảo.
- **Bài báo SOICT 2026 (#9041) bị từ chối.** Lý do quyết định: chỉ test trên **1 bệnh nhân (`L310`)**, không có khoảng tin cậy hay kiểm định thống kê. Phân tích phản biện: [papers/soict2026/review-soict.md](papers/soict2026/review-soict.md).
- **Nhiệm vụ:** huấn luyện và đánh giá lại trên HPC-UIT với **tập test nhiều bệnh nhân** (báo cáo theo bệnh nhân: mean ± std, Wilcoxon). Các mô hình:
  - FBP;
  - LEARN, RegFormer, DuDoTrans;
  - LEARN_Longformer, LEARN_LongNet, LEARN_Mamba;
  - **SOLAR_DualMamba** và ablation (a)–(d).
- **Số liệu cũ** (bảng tổng hợp trong [reports/sep-21-2026/benchmark_results.csv](reports/sep-21-2026/benchmark_results.csv) và [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md)) chỉ là lịch sử. **Không trộn** với số liệu mới.

---

## 2. Trạng Thái Cụm HPC-UIT (`/datastore/doanhbc/minhdphan`, tài khoản dùng chung)

| Hạng mục | Trạng thái | Ghi chú / Bằng chứng |
| :--- | :---: | :--- |
| Mount SSHFS `hpc-uit-server/` | ✅ | 26/10/2026 |
| SSH bằng key | ✅ | Login node `login01`. `$HOME` dùng chung: 3.9/10 GB |
| Mã nguồn trên server | ✅ | Sparse clone commit `9129290`, **không có** `scripts/output/`. git 2.43.0, 230 MB |
| Lệnh Slurm qua ssh | ✅ | Đã giải quyết: đường dẫn /cm/local/apps/slurm/current/bin và wrapper bash -lc |
| Môi trường Python (`envs/longnet`) | ✅ | Job 9610: hoàn tất 100% (PyTorch 2.8+cu128, mamba-ssm, astra, odl) |
| Dataset AAPM (TCIA) | ✅ | Đã hoàn tất: 60 bệnh nhân, 9,108 lát cắt tại /datastore/doanhbc/minhdphan/datasets/aapm/raw/ |
| Chia tập train/val/test nhiều bệnh nhân | ✅ | Đã hoàn tất: 100% relative symlinks (test: 4 BN / 757 lát, val: 1 BN / 133 lát, train: 55 BN / 8,218 lát) |
| Cache dữ liệu LA-90° (sinogram/FBP) | ✅ | Job 10432: 100% (8,218 train, 133 val, 757 test). FBP baseline test: PSNR 12.27 dB |
| Huấn luyện Wave A (LA-90°, 50 epochs) | ✅ | 4/4 mô hình hoàn thành 100% (Jobs 10556, 10561, 10563, 10565) |
| - SOLAR_DualMamba (Proposed) | ✅ | Job 10556: Best Val PSNR = 25.89 dB, SSIM = 0.6240, Peak VRAM = 5.26 GB |
| - RegFormer (Baseline) | ✅ | Job 10565: Best Val PSNR = 23.80 dB, SSIM = 0.693, Peak VRAM = 4.20 GB |
| - LEARN (Baseline) | ✅ | Job 10561: Best Val PSNR = 21.38 dB, SSIM = 0.676, Peak VRAM = 0.41 GB |
| - DuDoTrans (Baseline) | ✅ | Job 10563: Best Val PSNR = 19.31 dB, SSIM = 0.503, Peak VRAM = 0.72 GB |
| Đánh giá Benchmark Test Cohort (4 BN) | ✅ | Job 10667: COMPLETED (4 BN, 757 lát test 90°). SOLAR: 24.47±1.09 dB, SSIM 0.7214 (vượt RegFormer +2.23 dB, LEARN +3.36 dB, DuDoTrans +6.12 dB) |

**Checkpoints đã lưu (`saved_models/`):**
- SOLAR_DualMamba: `saved_models/SOLAR_DualMamba_90/solar_dualmamba_la-epoch=05-val_psnr=25.8919-val_ssim=0.6240.ckpt`
- RegFormer: `saved_models/RegFormer_90/RegFormer_AAPM_LA_90deg_view64/epoch=00-val_psnr=23.7951.ckpt`
- LEARN: `saved_models/LEARN_90/LEARN_AAPM_LA_90deg_view64/epoch=01-val_psnr=21.3799.ckpt`
- DuDoTrans: `saved_models/DuDoTrans_90/DuDoTrans_AAPM_LA_90deg_view64/epoch=11-val_psnr=19.3073.ckpt`

---

## 3. Danh Mục Nhiệm Vụ Tiếp Theo (Actionable TODOs)

Checklist đầy đủ, theo thứ tự: `.lab/RECOVERY_CHECKLIST.md` (trong workspace `agents-research`, ngoài repo).

- [x] **1. [Người dùng] Đường dẫn Slurm:** đã giải quyết qua `/cm/local/apps/slurm/current/bin` và `bash -lc`.
- [x] **2. [Người dùng] Dữ liệu AAPM:** 60 bệnh nhân, 9,108 lát cắt đã chuyển và giải nén xong trên HPC-UIT.
- [ ] **3. [ml-engineer → code-qa] Port script sang MIG:**
  - Port đợt đầu theo template HPC-UIT.
  - Mỗi script phải resume từ `last.ckpt`, ghi `git rev-parse HEAD` vào log, và xuất metric **theo lát cắt và theo bệnh nhân**.
- [ ] **4. [hpc-operator] Môi trường và smoke test:** job tạo môi trường (`small`), rồi smoke test LEARN, rồi đo VRAM và thời gian mỗi epoch.
- [x] **5. [phd-experiment-analyst] Chia tập:** đã hoàn tất test (4 BN, 757 lát), val (1 BN, 133 lát), train (55 BN, 8,218 lát) với 100% relative symlinks.
- [ ] **6. [phd-theory-architect] Hình học chiếu:** thống nhất hình học ($D_{SO}/D_{SD}$, xem SYSTEM_SPEC §3) trước khi tạo lại cache.
- [ ] **7. Huấn luyện lại:** theo thứ tự trong checklist R6.
- [ ] **8. Sửa bài SOICT không cần số liệu mới:** bib, "270°", mâu thuẫn ở 90°, weight sharing, $A^T$ so với $PA^T$.
- *Tuyệt đối tuân thủ:* không đưa SOLAR vào SOICT 2026.
