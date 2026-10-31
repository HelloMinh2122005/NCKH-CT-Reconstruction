# ĐẶC TẢ HỆ THỐNG & CẤU HÌNH KHOA HỌC (SYSTEM SPECIFICATIONS)
# DỰ ÁN NCKH: LIMITED-ANGLE CT RECONSTRUCTION

> **Mục đích:** Tài liệu này chứa toàn bộ thông số tĩnh về phần cứng, cụm Slurm HPC, hình học chiếu CT, bài toán toán học và danh mục dataset gốc. Đây là tài liệu tham chiếu tĩnh, chỉ cần đọc khi cấu hình môi trường, thiết lập tham số mô hình hoặc kiểm tra tính đúng đắn của dữ liệu.
>
> **Cập nhật 26/10/2026:** Cụm DGX-A100 cũ (`uittogether3`) đã mất, kèm toàn bộ dữ liệu trên đó. §4–§5 mô tả cụm **HPC-UIT** hiện tại. Thông tin cụm cũ chỉ còn trong [ARCHIVE_LOG.md](ARCHIVE_LOG.md).

---

## 1. Thông Tin Chung
- **Dự án:** Nghiên cứu Khoa học (NCKH) - Tái tạo ảnh cắt lớp CT góc giới hạn (Limited-Angle CT Reconstruction)
- **Tác giả:** `MinhPD`
- **Không gian làm việc:**
  - Local (sửa code ở đây): `/home/phandinhminh/Downloads/kltn/agents-research/temp-github/NCKH-CT-Reconstruction`
  - Server HPC-UIT: `/datastore/doanhbc/minhdphan/NCKH-CT-Reconstruction` (sparse clone, chỉ `git pull`)
  - Mount SSHFS để xem: `/home/phandinhminh/Downloads/kltn/agents-research/hpc-uit-server/`
- **GitHub Repository:** [https://github.com/HelloMinh2122005/NCKH-CT-Reconstruction.git](https://github.com/HelloMinh2122005/NCKH-CT-Reconstruction.git) (Nhánh: `main`)

---

## 2. Kế Thừa Nghiên Cứu & Cơ Sở Khoa Học

### 2.1. Kế thừa công trình nghiên cứu của Thành (Baseline):
- **Bài báo gốc (Paper):** `~/note/MVA___CT_reconstruction_revised-2.pdf` (Mô hình MVA - Multi-View Attention cho bài toán Sparse-view CT).
- **Source code của Thành:** bản gốc nằm trên cụm A100 cũ (đã mất). Các mô hình `LEARN_LongNet`, `LEARN_Longformer`, `LEARN_Mamba` và script sinh sinogram đã được sao chép và chuẩn hóa trong `baselines/` và `data/` của repo này.
- **Tài liệu phân tích kiến trúc MVA vs Đề xuất mới:** [ARCHITECTURE_MVA_VS_PROPOSED_SOLAR.md](note/ARCHITECTURE_MVA_VS_PROPOSED_SOLAR.md).

### 2.2. Mục tiêu nghiên cứu cốt lõi:
- **Mục tiêu nhân văn:** Cắt giảm tối đa liều bức xạ tia X chiếu vào người bệnh nhân (**Radiation Dose Reduction 50% - 75%**) tuân theo nguyên tắc y học **ALARA**, đồng thời bảo vệ các cơ quan nhạy cảm với phóng xạ (tuyến giáp, mắt, tuyến vú).
- **Thách thức toán học:** Cung quét bị giới hạn $\Delta \theta < 180^\circ$ dẫn đến vùng khuyết hình nêm trong miền tần số Fourier (**Missing Wedge Problem**). Địa hình Hessian $A^TA$ suy biến, các bước gradient bậc 1 bị dao động (Zigzag). 
- **Hướng đề xuất:** Phát triển kiến trúc tối ưu hóa bậc 2 (Second-Order) kết hợp phân nhánh kép cục bộ - toàn cục (Dual-Branch Local/Nonlocal như mạng **SOLAR**) và mô hình đa miền Sinogram/Image (**SOLAR-DualMamba**) để vượt qua giới hạn của MVA trên bài toán Limited-Angle CT.

---

## 3. Cấu Hình & Tham Số Toán Học / Vật Lý Đã Thống Nhất

| Tham số | Giá trị | Ý nghĩa & Lý do lựa chọn |
| :--- | :--- | :--- |
| **Hình học chiếu** | Fan-Beam Geometry | Chuẩn máy CT y tế Siemens Somatom (`src_radius=600mm`, `det_radius=290mm`, `512` detectors) |
| **Dải góc quét chính** | $[-60^\circ, +60^\circ]$ ($120^\circ$ span) | Benchmark quốc tế SOTA; đối xứng qua trục $0^\circ$ bảo vệ cơ quan hai bên cơ thể; giảm $66.7\%$ tia X |
| **Số góc chiếu (Views)** | `64` views | Bước góc $\Delta \theta \approx 1.875^\circ$ (Pure Limited-Angle); $64 = 2^6$ tối ưu cho down/upsampling U-Net/ViT/Mamba |
| **Độ phân giải** | $256 \times 256$ | Chiếu & FBP trên lưới gốc $512 \times 512$ trước, sau đó resize về $256 \times 256$ để tiết kiệm VRAM GPU |
| **Bộ lọc FBP** | Ram-Lak filter | High-pass ramp filter chuẩn tái tạo sơ bộ |
| **Mô phỏng nhiễu** | 2 chế độ: `noise_0` và `noise_1e6` | `noise_0`: Đánh giá khôi phục góc khuyết thuần túy; `noise_1e6` ($I_0=10^6, \sigma=0.05$): Thử thách liều siêu thấp thực tế |

> ⚠️ **Chưa thống nhất:** `proposal.md` §4.1 ghi $D_{SO}=1075$ mm, $D_{SD}=1550$ mm, khác với bảng trên (600/290 mm, khớp `main.tex`). Giá trị đúng phải lấy từ code sinh sinogram (`data/*/prepare_data_sinogram_LA*.py`) trước khi tạo lại cache trên HPC-UIT.

---

## 4. Hệ Thống Máy Chủ & Quy Tắc Slurm Cluster (HPC-UIT)

- **Cụm:** HPC-UIT (Slurm, GPU chia MIG). Host và user được lưu trong `.claude/cluster.json` của workspace, không ghi trong repo. Hướng dẫn chính thức: `Huong_Dan_Su_Dung_Tai_Nguyen_HPC.docx.pdf` (trong workspace). Ngoài mạng trường phải dùng VPN UIT.
- **Tài khoản dùng chung:** tài khoản của thầy Bùi Cao Doanh, dùng chung với các sinh viên khác. **Chỉ làm việc trong `/datastore/doanhbc/minhdphan`**. Không đụng thư mục hay job của người khác.
- **Lưu trữ:**
  - `$HOME` ≤ 10 GB dùng chung (vượt sẽ bị khóa quyền chạy job): không để gì ở đây.
  - `/datastore/doanhbc/minhdphan/`: code (`NCKH-CT-Reconstruction/`), `envs/`, `datasets/`, checkpoint, log.
  - `/datastore/shared/`: dữ liệu dùng chung, chỉ đọc (26/10/2026: không có dữ liệu CT).
- **Phân vùng GPU** (luôn `--account=uit`; chọn phân vùng nhỏ nhất đủ dùng):

  | Partition | `--gres` | VRAM | CPU / RAM tối đa | Dùng cho |
  | :--- | :--- | :--- | :--- | :--- |
  | `small` | `gpu:1g.18gb:1` | < 20 GB | 4 / 56 GB | smoke test, debug, build môi trường |
  | `normal` | `gpu:2g.35gb:1` | 20–40 GB | 8 / 64 GB | mặc định cho huấn luyện |
  | `large` | `gpu:3g.71gb:1` | 40–80 GB | 16 / 128 GB | chỉ khi đo được nhu cầu > 35 GB |

- **Giới hạn (chung cho cả tài khoản):** ≤ 50 job xếp hàng, ≤ 3 job chạy song song, hai lần submit cách nhau ≥ 1 giờ, mỗi job ≤ 72 giờ. Không tính toán trên login node.
- **Môi trường Python:** sẽ tạo tại `/datastore/doanhbc/minhdphan/envs/longnet` bằng một job `small`. Phiên bản giữ gần môi trường cũ: `torch==2.8.0+cu128`, `mamba-ssm==2.3.1`, `transformers==4.57.6`, `odl==0.8.3`, `astra-toolbox==2.3.0`, `torchmetrics==1.8.2`, `pydicom==2.4.4`.
- **Mẫu script:** `.claude/skills/slurm-jobs/references/sbatch-template-hpc-uit.sh` (workspace). Kiểm tra bằng `python3 .claude/tools/slurm_lint.py --profile hpc-uit scripts/<tên>.sh`. Các script cũ trong `scripts/` vẫn theo định dạng A100/MPS và phải port trước khi chạy.
- **Quy tắc bắt buộc về lưu Output/Log:**
  - Mọi job Slurm **bắt buộc** phải ghi log (trên server) vào:
    ```text
    scripts/output/<tên script>/log/%j.out
    scripts/output/<tên script>/log/%j.err
    ```
  - Khi kéo về local, log mới nằm trong `scripts/output-hpc-uit/<tên script>/log/`. `scripts/output/` ở local chỉ chứa log cũ của cụm A100 (chỉ đọc).

---

## 5. Danh Mục Dataset Gốc Tham Chiếu
Tất cả dataset trên cụm cũ đã mất và phải tải lại vào `/datastore/doanhbc/minhdphan/datasets/`.

1. **AAPM Mayo Clinic Low Dose CT (TCIA *LDCT-and-Projection-data*):**
   - Định dạng: DICOM (`.IMA`), 3mm slice thickness (`full_3mm`).
   - Tải lại qua TCIA. Người dùng phải tự chấp nhận thỏa thuận sử dụng dữ liệu và tải manifest `.tcia`.
   - Đích: `/datastore/doanhbc/minhdphan/datasets/aapm/`.
   - **Phân chia cũ** (8 train, `L333` validation, chỉ `L310` test) **đã bị loại bỏ**. Tập test mới gồm **nhiều bệnh nhân**, được chốt sau khi biết số bệnh nhân tải về.
2. **NIH DeepLesion CT Dataset** (mở rộng, Tier 2): `/datastore/doanhbc/minhdphan/datasets/deeplesion/`.
3. **LIDC-IDRI Lung CT Dataset** (mở rộng, Tier 2): `/datastore/doanhbc/minhdphan/datasets/lidc_idri/`.
