# ĐỀ CƯƠNG NGHIÊN CỨU KHOA HỌC & BÁO CÁO KỸ THUẬT CHUYÊN SÂU
# MỞ RỘNG MÔ HÌNH HÓA CHUỖI DÀI VÀ TỐI ƯU HÓA BIẾN PHÂN BẬC HAI CHO TÁI TẠO ẢNH CT GÓC GIỚI HẠN (LIMITED-ANGLE CT RECONSTRUCTION)

**Dự án:** Nghiên cứu Khoa học & Khóa luận Tốt nghiệp Kỹ sư AI  
**Học viên thực hiện:** Phan Đình Minh (MSSV: 23520949)  
**Đơn vị:** Khoa Khoa học Máy tính — Trường Đại học Công nghệ Thông tin, ĐHQG-HCM (VNU-HCM UIT)  
**Giảng viên Hướng dẫn / Cố vấn:** TS. Bùi Cao Doanh  
**Ngày cập nhật:** Tháng 10/2026 (Bản Hiệu đính Toàn diện theo Phản biện Học thuật)  

---

## MỤC LỤC CHI TIẾT

- [PHẦN 1: TỔNG QUAN ĐỀ TÀI & TÍNH CẤP THIẾT LÂM SÀNG](#phần-1-tổng-quan-đề-tài--tính-cấp-thiết-lâm-sàng)
  - [1.1. Bối cảnh bài toán Limited-Angle CT (LA-CT) & Điều kiện Đủ Parker](#11-bối-cảnh-bài-toán-limited-angle-ct-la-ct--điều-kiện-đủ-parker)
  - [1.2. Mục tiêu nghiên cứu cốt lõi & Định vị đề tài](#12-mục-tiêu-nghiên-cứu-cốt-lõi--định-vị-đề-tài)
- [PHẦN 2: BÁO CÁO TRUNG THỰC VẤN ĐỀ HIỆN TẠI: VÌ SAO HOẠT ĐỘNG TỐT TRÊN SPARSE-VIEW NHƯNG THẤT BẠI TRÊN LIMITED-ANGLE?](#phần-2-báo-cáo-trung-thực-vấn-đề-hiện-tại-vì-sao-hoạt-động-tốt-trên-sparse-view-nhưng-thất-bại-trên-limited-angle)
  - [2.1. Đánh giá khách quan công trình nền tảng MVA (Sparse-View CT)](#21-đánh-giá-khách-quan-công-trình-nền-tảng-mva-sparse-view-ct)
  - [2.2. Phân tích bản chất vật lý & toán học: Vì sao Limited-Angle CT khó hơn Sparse-View CT rất nhiều?](#22-phân-tích-bản-chất-vật-lý--toán-học-vì-sao-limited-angle-ct-khó-hơn-sparse-view-ct-rất-nhiều)
    - [📌 Lý do 1: Định lý Lát cắt Fourier & Đối xứng Hermitian của Nêm khuyết (Missing Wedge)](#-lý-do-1-định-lý-lát-cắt-fourier--đối-xứng-hermitian-của-nêm-khuyết-missing-wedge)
    - [📌 Lý do 2: Giải tích vi cục bộ & Tập mặt sóng vô hình (Microlocal Analysis & Invisible Wavefront Set)](#-lý-do-2-giải-tích-vi-cục-bộ--tập-mặt-sóng-vô-hình-microlocal-analysis--invisible-wavefront-set)
    - [📌 Lý do 3: Toán tử chuẩn $A^T A$ và Sự suy biến trên Không gian Hạt nhân (Null-space $\mathcal{N}(A)$)](#-lý-do-3-toán-tử-chuẩn-at-a-và-sự-suy-biến-trên-không-gian-hạt-nhân-null-space-mathcalna)
  - [2.3. Hiện tượng "Sụp đổ" khi áp dụng nguyên xi mô hình MVA sang LA-CT](#23-hiện-tượng-sụp-đổ-khi-áp-dụng-nguyên-xi-mô-hình-mva-sang-la-ct)
    - [💥 1. Sự bất ổn định số học và lỗi `NaN` của Mamba-1 (Causal 1D vs 2D Void)](#-1-sự-bất-ổn-định-số-học-và-lỗi-nan-của-mamba-1-causal-1d-vs-2d-void)
    - [💥 2. Dao động Zigzag và sự mất phương hướng của Gradient Descent bậc 1](#-2-dao-động-zigzag-và-sự-mất-phương-hướng-của-gradient-descent-bậc-1)
    - [💥 3. Sự mù quáng của thuật toán chọn Landmark sMEANS trong Nyströmformer](#-3-sự-mù-quáng-của-thuật-toán-chọn-landmark-smeans-trong-nyströmformer)
- [PHẦN 3: PHƯƠNG PHÁP ĐỀ XUẤT CỦA CHÚNG TA: KIẾN TRÚC SOLAR](#phần-3-phương-pháp-đề-xuất-của-chúng-ta-kiến-trúc-solar)
  - [3.1. Triết lý kiến trúc: Kết hợp Đa miền (Dual-Domain) và Mở cuộn Biến phân Bậc hai Gauss-Newton](#31-triết-lý-kiến-trúc-kết-hợp-đa-miền-dual-domain-và-mở-cuộn-biến-phân-bậc-hai-gauss-newton)
  - [3.2. Miền Sinogram: Mạng Angular Bi-Mamba Net & Cơ chế Mặt nạ Bảo toàn Dữ liệu (Consistency Masking)](#32-miền-sinogram-mạng-angular-bi-mamba-net--cơ-chế-mặt-nạ-bảo-toàn-dữ-liệu-consistency-masking)
  - [3.3. Miền Ảnh: Động cơ Mở cuộn Gauss-Newton Proximal CG Solver](#33-miền-ảnh-động-cơ-mở-cuộn-gauss-newton-proximal-cg-solver)
    - [📐 Dẫn xuất Toán học Biến phân (Local Quadratic Surrogate & Gauss-Newton Approximation)](#-dẫn-xuất-toán-học-biến-phân-local-quadratic-surrogate--gauss-newton-approximation)
    - [🛡️ Cơ chế Giảm chấn SPD Damping ($\mu_t I$) — Triệt tiêu 100% lỗi NaN](#️-cơ-chế-giảm-chấn-spd-damping-mu_t-i--triệt-tiêu-100-lỗi-nan)
    - [⚡ Giải tích Toán học trên Không gian Hạt nhân: Sự suy biến về Bậc 1 trên $\mathcal{N}(A)$](#-giải-tích-toán-học-trên-không-gian-hạt-nhân-sự-suy-biến-về-bậc-1-trên-mathcalna)
  - [3.4. Khối Tiên nghiệm Điều hòa Kép Swin-CNN (Dual-Branch Prior)](#34-khối-tiên-nghiệm-điều-hòa-kép-swin-cnn-dual-branch-prior)
  - [3.5. Cầu Nối Vật Lý & Khởi Tạo Nghiệm Hạt Giống (Physical Seeding)](#35-cầu-nối-vật-lý--khởi-tạo-nghiệm-hạt-giống-physical-seeding)
  - [3.6. Chi Tiết Thuật Toán Safe Matrix-Free CG Solver & Đạo Hàm Ngược qua CG](#36-chi-tiết-thuật-toán-safe-matrix-free-cg-solver--đạo-hàm-ngược-qua-cg)
  - [3.7. Hàm Mất Mát Huấn Luyện Liên Kết Đa Miền & Giải Pháp Khắc Phục Suy Giảm SSIM](#37-hàm-mất-mát-huấn-luyện-liên-kết-đa-miền--giải-pháp-khắc-phục-suy-giảm-ssim)
- [PHẦN 4: BẰNG CHỨNG THỰC NGHIỆM ĐỊNH LƯỢNG & ĐỐI SÁNH KHÁCH QUAN](#phần-4-bằng-chứng-thực-nghiệm-định-lượng--đối-sánh-khách-quan)
  - [4.1. Thiết lập thực nghiệm kiểm thử chuẩn hóa](#41-thiết-lập-thực-nghiệm-kiểm-thử-chuẩn-hóa)
  - [4.2. Bảng đối sánh trực diện trên Tập Test Bệnh nhân `L310` (LA-120°)](#42-bảng-đối-sánh-trực-diện-trên-tập-test-bệnh-nhân-l310-la-120)
  - [4.3. Đánh giá trên Cung Quét Khắc Nghiệt LA-90° (Stress Test Khuyết $180^\circ$ Không gian Tần số)](#43-đánh-giá-trên-cung-quét-khắc-nghiệt-la-90-stress-test-khuyết-180circ-không-gian-tần-số)
    - [💡 Phân Tích Thực Nghiệm Khách Quan & Đính Chính So Sánh:](#-phân-tích-thực-nghiệm-khách-quan--đính-chính-so-sánh)
  - [4.4. Kiểm chứng tổng quát hóa độc lập trên bộ dữ liệu lớn NIH DeepLesion](#44-kiểm-chứng-tổng-quát-hóa-độc-lập-trên-bộ-dữ-liệu-lớn-nih-deeplesion)
  - [4.5. Phân tích chi phí tính toán, bộ nhớ VRAM và số lượng tham số](#45-phân-tích-chi-phí-tính-toán-bộ-nhớ-vram-và-số-lượng-tham-số)
  - [4.6. Bảng Nghiệm Thu Toàn Diện Các Baseline Mới Nhất Trên Tập Validation AAPM LA-120°](#46-bảng-nghiệm-thu-toàn-diện-các-baseline-mới-nhất-trên-tập-validation-aapm-la-120)
  - [4.7. Đánh Giá Độ An Toàn Lâm Sàng: Sai Số Vùng Tổn Thương (ROI HU) & Bản Đồ Ảo Giác Không Gian Hạt Nhân](#47-đánh-giá-độ-an-toàn-lâm-sàng-sai-số-vùng-tổn-thương-roi-hu--bản-đồ-ảo-giác-không-gian-hạt-nhân)
- [PHẦN 5: KẾ HOẠCH ĐÁNH GIÁ ĐÓNG GÓP TỪNG THÀNH PHẦN (ABLATION STUDIES FRAMEWORK)](#phần-5-kế-hoạch-đánh-giá-đóng-góp-từng-thành-phần-ablation-studies-framework)
  - [5.1. Khảo sát 1: Tách bạch Đóng góp của Khối Ngoại suy Sinogram (Seeding vs CG RHS)](#51-khảo-sát-1-tách-bạch-đóng-góp-của-khối-ngoại-suy-sinogram-seeding-vs-cg-rhs)
  - [5.2. Khảo sát 2: Đóng góp của Thuật toán Tối ưu Hóa (Gradient Descent vs QN-Mixer vs Newton-CG)](#52-khảo-sát-2-đóng-góp-của-thuật-toán-tối-ưu-hóa-gradient-descent-vs-qn-mixer-vs-newton-cg)
  - [5.3. Khảo sát 3: Vai trò của Hệ số Damping $\mu_t I$ đối với Ổn định Số học](#53-khảo-sát-3-vai-trò-của-hệ-số-damping-mu_t-i-đối-với-ổn-định-số-học)
  - [5.4. Khảo sát 4: Phân rã Đóng góp của Nhánh Local Res-CNN và Non-local Swin](#54-khảo-sát-4-phân-rã-đóng-góp-của-nhánh-local-res-cnn-và-non-local-swin)
  - [5.5. Khảo sát 5: Khảo sát Công bằng Tham số (Parameter-Matched Ablation: Shared vs Unshared)](#55-khảo-sát-5-khảo-sát-công-bằng-tham-số-parameter-matched-ablation-shared-vs-unshared)
- [PHẦN 6: LỘ TRÌNH TRIỂN KHAI, QUẢN TRỊ RỦI RO & KẾT QUẢ MONG ĐỢI](#phần-6-lộ-trình-triển-khai-quản-trị-rủi-ro--kết-quả-mong-đợi)
  - [6.1. Xác định Kết quả Bắt buộc (Core Mandatory Deliverables) vs Kết quả Mở rộng (Extended Deliverables)](#61-xác-định-kết-quả-bắt-buộc-core-mandatory-deliverables-vs-kết-quả-mở-rộng-extended-deliverables)
  - [6.2. Kế hoạch dự phòng hạ tầng tính toán Slurm HPC A100 và kiểm thử cục bộ](#62-kế-hoạch-dự-phòng-hạ-tầng-tính-toán-slurm-hpc-a100-và-kiểm-thử-cục-bộ)
- [PHẦN 7: DANH MỤC TÀI LIỆU THAM KHẢO CHỌN LỌC (SELECTED ACADEMIC REFERENCES)](#phần-7-danh-mục-tài-liệu-tham-khảo-chọn-lọc-selected-academic-references)
- [KẾT LUẬN & ĐỀ NGHỊ](#kết-luận--đề-nghị)

---

# PHẦN 1: TỔNG QUAN ĐỀ TÀI & TÍNH CẤP THIẾT LÂM SÀNG

### 1.1. Bối cảnh bài toán Limited-Angle CT (LA-CT) & Điều kiện Đủ Parker
Trong chụp cắt lớp vi tính y tế (Computed Tomography — CT), để tái tạo bức ảnh cắt lớp hoàn chỉnh và trung thực bằng các thuật toán giải tích cổ điển như Chiếu ngược có lọc (Filtered Backprojection — FBP), dữ liệu hình chiếu bắt buộc phải thỏa mãn điều kiện đầy đủ về mặt thông tin hình học:
* Đối với hình học chùm tia song song (Parallel-Beam), góc quét tối thiểu phải đạt $\Theta_{\text{scan}} \ge 180^\circ$ ($\pi$ rad).
* Đối với hình học chùm tia quạt (Fan-Beam), điều kiện đủ Parker (Parker's Short-Scan Sufficiency Condition, Parker 1982) yêu cầu góc quay tối thiểu của đầu dò phải thỏa mãn:
  $$\Theta_{\text{short-scan}} = 180^\circ + 2\gamma_m$$
  trong đó $\gamma_m = \arcsin(R_{\text{FOV}} / D_{\text{so}})$ là nửa góc quạt chùm tia ($R_{\text{FOV}}$ là bán kính trường quan sát, $D_{\text{so}}$ là khoảng cách từ nguồn X-ray đến tâm quay).

Tuy nhiên, trong các tình huống can thiệp ngoại khoa và chẩn đoán chuyên biệt, **việc quay đầu dò thỏa mãn điều kiện Parker là bất khả thi do rào cản vật lý**:
1. **Chụp C-arm trong phẫu thuật (Intraoperative C-arm CT):** Bàn mổ, hệ thống gây mê và trang thiết bị hồi sức bao quanh bệnh nhân gây va chạm cơ học, giới hạn cánh tay chữ C chỉ quay được trong khoảng $100^\circ - 140^\circ$.
2. **Chụp nhũ ảnh cắt lớp (Digital Breast Tomosynthesis — DBT):** Để hạn chế liều chiếu xạ tích lũy và giảm biến dạng mô ngực ép cơ học, góc quét đầu phát thường chỉ đạt $40^\circ - 90^\circ$.
3. **Xạ trị định vị hướng dẫn bằng hình ảnh (Image-Guided Radiation Therapy — IGRT):** Cần theo dõi chuyển động khối u tức thời với tốc độ cao trong cung quét hẹp để tối ưu hóa chùm tia xạ trị.

Khi góc quét bị giới hạn (ví dụ chỉ quét được $\Theta_{\text{scan}} = 120^\circ$ hoặc $90^\circ$), bài toán trở thành **Limited-Angle CT (LA-CT)** — một bài toán nghịch đảo phi chỉnh cực kỳ nghiêm trọng (severely ill-posed inverse problem). Ảnh FBP tái tạo trực tiếp bị phá hủy bởi các vệt sọc nhiễu năng lượng cao (streak artifacts), ranh giới mô bị kéo giãn biến dạng (geometric stretching), và các cấu trúc tổn thương có phương tiếp tuyến rơi vào vùng khuyết bị biến mất hoàn toàn.

---

### 1.2. Mục tiêu nghiên cứu cốt lõi & Định vị đề tài
Tiếp thu ý kiến phản biện của Giảng viên hướng dẫn (TS. Bùi Cao Doanh), đề tài kiên quyết tránh dàn trải mục tiêu kỹ thuật, tập trung giải quyết một vấn đề khoa học then chốt:

> **MỤC TIÊU CỐT LÕI:** Xây dựng, chuẩn hóa toán học và kiểm chứng thực nghiệm khung kiến trúc Mở cuộn Biến phân Đa miền **SOLAR**, kết hợp ngoại suy sinogram góc quét (Angular Bi-Mamba) và bộ giải Conjugate Gradient tiền điều kiện hóa xác định dương nghiêm ngặt, nhằm triệt tiêu hiện tượng phân kỳ số học `NaN` của Mamba-1 và khắc phục sự suy thoái nghiệm của các mô hình unrolling bậc 1 trong bài toán Limited-Angle CT.

---

# PHẦN 2: BÁO CÁO TRUNG THỰC VẤN ĐỀ HIỆN TẠI: VÌ SAO HOẠT ĐỘNG TỐT TRÊN SPARSE-VIEW NHƯNG THẤT BẠI TRÊN LIMITED-ANGLE?

### 2.1. Đánh giá khách quan công trình nền tảng MVA (Sparse-View CT)
Công trình nền tảng của Lab: *"Long-Sequence Modeling for Sparse-View Computed Tomography Reconstruction"* (MVA 2026) do TS. Bùi Cao Doanh và cộng sự công bố đã giải quyết xuất sắc mâu thuẫn giữa *"locality bias"* của CNN (LEARN 3 lớp Conv $5 \times 5$, không xóa được sọc toàn cục) và *"kích thước patch quá thô"* của Vision Transformer tiêu chuẩn ($16 \times 16$ làm mờ nốt tổn thương nhỏ $< 5\text{ mm}$). MVA sử dụng token siêu mịn **$2 \times 2$ pixels** ($N = 16.384$ tokens trên ảnh $256 \times 256$) tích hợp các cơ chế chú ý tuyến tính $\mathcal{O}(N)$ (Nyströmformer, LongNet, Mamba-1) vào khung mở cuộn LEARN.

* **Thành tựu trên Sparse-View CT:** Trên tập AAPM 32 góc chiếu thưa trải đều $360^\circ$ ở mức nhiễu cao ($5 \times 10^5$), `LEARN + Nyströmformer` đạt **38.8960 dB PSNR / 0.9607 SSIM** (vượt LEARN gốc 38.73 dB / 0.9594). Ở kịch bản cực thưa 18 góc, LEARN gốc sụp đổ (PSNR âm) trong khi Nyströmformer vẫn duy trì 32.75 dB / 0.9205.

Tuy nhiên, **khi chuyển giao nguyên xi thiết kế của MVA sang bài toán Limited-Angle CT (LA-CT), toàn bộ hệ thống đã gặp sự cố nghiêm trọng**.

---

### 2.2. Phân tích bản chất vật lý & toán học: Vì sao Limited-Angle CT khó hơn Sparse-View CT rất nhiều?

Bảng so sánh đối đầu được chuẩn hóa lại chính xác theo lý thuyết giải tích Fourier và điều kiện Parker:

| Tiêu chí So sánh | Sparse-View CT (Góc thưa — SV-CT) | Limited-Angle CT (Góc giới hạn — LA-CT) |
| :--- | :--- | :--- |
| **Phân bố góc chiếu** | Phân bố **rời rạc nhưng đều đặn trên toàn bộ cung quét $180^\circ$ hoặc $360^\circ$** (ví dụ: cách nhau mỗi $5.6^\circ$ hoặc $11.25^\circ$). | Bị chặn cứng trong **một cung liên tục $\Theta_{\text{scan}} < 180^\circ$** (ví dụ: $[-60^\circ, +60^\circ]$ hoặc $[-45^\circ, +45^\circ]$). |
| **Bản chất Không gian Tần số Fourier 2D** | **Lấy mẫu dưới mức (Sub-sampling):** Các tia phổ Fourier đi qua gốc tọa độ xoay đều $360^\circ$ theo hình nan hoa bánh xe. | **Khuyết trắng theo hình nêm (Fourier Missing Wedge):** Xuất hiện góc nêm đối đỉnh hoàn toàn không có dữ liệu đo đạc. |
| **Góc nêm khuyết trên Mặt phẳng Tần số 2D** | **Không có nêm khuyết:** Tần số thấp và biên ở mọi hướng đều có dữ liệu mẫu. | **Góc quét $120^\circ$:** Khuyết **$60^\circ$** trên nửa mặt phẳng (tạo 2 nêm đối đỉnh tổng cộng **$120^\circ / 360^\circ$**).<br>**Góc quét $90^\circ$:** Khuyết **$90^\circ$** trên nửa mặt phẳng (tạo 2 nêm đối đỉnh tổng cộng **$180^\circ / 360^\circ$**). |
| **Giải tích Vi cục bộ (Microlocal Analysis)** | **Khả kiến toàn bộ:** Mọi đường biên giải phẫu đều có tia tiếp xúc ($WF(x) \subset \mathcal{E}_{\text{visible}}$). | **Xuất hiện Tập mặt sóng vô hình ($WF_{\text{invisible}}$):** Các đường biên có vector pháp tuyến rơi vào nêm khuyết bị xóa sổ. |
| **Trạng thái Toán tử Hessian $A^T A$** | **Khả nghịch cục bộ (Well-conditioned):** $A^T A$ có phổ trị riêng phân bổ trên mọi hướng không gian. | **Suy biến hạng nghiêm trọng (Rank-Deficient):** Vô số trị riêng $\lambda_i = 0$ ứng với không gian hạt nhân $\mathcal{N}(A)$. |
| **Bản chất toán học bài toán nghịch đảo** | Bài toán **Nội suy dữ liệu (Interpolation):** Điền khuyết giữa các nan hoa lân cận. | Bài toán **Ngoại suy tần số (Extrapolation):** Khôi phục thông tin từ số $0$ tuyệt đối. |

#### 📌 Lý do 1: Định lý Lát cắt Fourier & Đối xứng Hermitian của Nêm khuyết (Missing Wedge)
Hệ số suy giảm tuyến tính của mô sinh học $x(u, v) \in \mathbb{R}$ là một hàm thực. Theo tính chất **đối xứng liên hợp Hermitian** của biến đổi Fourier 2D:
$$\mathcal{F}_{2D}\{x\}(-\boldsymbol{\omega}) = \overline{\mathcal{F}_{2D}\{x\}(\boldsymbol{\omega})}$$
Do đó, phổ Fourier tại tần số $-\boldsymbol{\omega}$ hoàn toàn suy ra được từ phổ tại $+\boldsymbol{\omega}$. 

Theo Định lý Lát cắt Fourier (Fourier Slice Theorem, Kak & Slaney 2001), biến đổi Fourier 1D của hình chiếu tại góc $\theta$ lấp đầy một đường thẳng đi qua gốc tọa độ tần số 2D tại góc $\theta$:
$$\mathcal{F}_{1D}\{y(\theta, \cdot)\}(\omega) = \mathcal{F}_{2D}\{x\}(\omega \cos\theta, \omega \sin\theta)$$
* Vì đường thẳng quét qua cả 2 hướng $\theta$ và $\theta + 180^\circ$, một cung quét $180^\circ$ đã phủ kín toàn bộ $360^\circ$ không gian tần số 2D.
* **Đính chính góc nêm khuyết:** Khi hệ thống chỉ quét được một cung đối xứng $\Theta_{\text{scan}} = 120^\circ$ (từ $-60^\circ$ đến $+60^\circ$), miền tần số đo được trong nửa mặt phẳng góc $[0^\circ, 180^\circ)$ là $[0^\circ, 60^\circ] \cup [120^\circ, 180^\circ)$. Miền khuyết thiếu dữ liệu trong nửa mặt phẳng chỉ rộng:
  $$\Delta\theta_{\text{missing}} = 180^\circ - 120^\circ = 60^\circ$$
  Xét trên toàn bộ mặt phẳng tần số 2D ($360^\circ$), nêm khuyết tạo thành hai góc đối đỉnh bằng nhau, mỗi góc rộng $60^\circ$, tổng diện tích góc khuyết là $120^\circ$ (chiếm đúng $1/3$ diện tích mặt phẳng Fourier, không phải $240^\circ$).
* Tương tự, khi góc quét thu hẹp xuống $\Theta_{\text{scan}} = 90^\circ$ (từ $-45^\circ$ đến $+45^\circ$), phần nêm khuyết trong nửa mặt phẳng rộng $180^\circ - 90^\circ = 90^\circ$. Hai góc đối đỉnh nêm khuyết chiếm tổng cộng **$180^\circ / 360^\circ$** (đúng $50\%$ không gian tần số 2D, không phải $270^\circ$).

```
        ĐỐI XỨNG HERMITIAN & NÊM KHUYẾT FOURIER 2D (LA-120°)
                           ω_v
                            │        / Nêm khuyết đối đỉnh 1 (60°)
                     \      │       /  [KHÔNG CÓ DỮ LIỆU ĐO]
                      \     │      /  
                       \    │     /   
        ────────────────\───┼────/──────────────── ω_u
                         \  │   /     
                          \ │  /      
                           \│ /       \ Nêm khuyết đối đỉnh 2 (60°)
                            │          \ [KHÔNG CÓ DỮ LIỆU ĐO]
                            
        -> Cung quét 120° phủ: [-60°, +60°] và đối xứng [+120°, +240°]
        -> Tổng góc nêm khuyết 2D: 60° + 60° = 120° (chiếm 1/3 diện tích phổ)
```

#### 📌 Lý do 2: Giải tích vi cục bộ & Tập mặt sóng vô hình (Microlocal Analysis & Invisible Wavefront Set)
Theo lý thuyết giải tích vi cục bộ của Quinto et al. (Inverse Problems 2013):
Một điểm kỳ dị (đường biên giải phẫu) của ảnh $x$ tại tọa độ $(u, v)$ có vector pháp tuyến $\mathbf{n} = (\cos\phi, \sin\phi)$ chỉ có thể được dựng lại một cách ổn định nếu tồn tại một tia X trong tập quét tiếp xúc với đường biên đó (tức góc quét $\theta = \phi$).
* **Tập mặt sóng khả kiến (Visible Wavefront Set):** $\mathcal{WF}_{\text{visible}}(x) = \{ (u, v, \phi) \in \mathcal{WF}(x) \mid \phi \in [-\Theta/2, +\Theta/2] \}$.
* **Tập mặt sóng vô hình (Invisible Wavefront Set):** $\mathcal{WF}_{\text{invisible}}(x) = \{ (u, v, \phi) \in \mathcal{WF}(x) \mid \phi \notin [-\Theta/2, +\Theta/2] \}$.

Hệ quả vật lý: Đối với cung quét $120^\circ$ đối xứng qua trục ngang, các đường biên nằm ngang (vector pháp tuyến hướng dọc) thuộc hoàn toàn vào $\mathcal{WF}_{\text{invisible}}$. Không một photon nào ghi nhận được sự tồn tại của các bề mặt này, dẫn đến hiện tượng FBP kéo giãn hình học vô cực theo phương thẳng đứng và xóa sạch ranh giới trên/dưới của các cơ quan nội tạng.

#### 📌 Lý do 3: Toán tử chuẩn $A^T A$ và Sự suy biến trên Không gian Hạt nhân (Null-space $\mathcal{N}(A)$)
Trong bài toán cực tiểu hóa bình phương tối thiểu $\min_x \frac{1}{2}\|Ax - y\|_2^2$, toán tử chuẩn là $A^T A$.
Trong Limited-Angle CT, toán tử $A$ có một **không gian hạt nhân khổng lồ $\mathcal{N}(A) = \{ v \in \mathbb{R}^N \mid A v = 0 \}$** tương ứng với các thành phần tần số nằm trọn trong nêm khuyết.
* Mọi vector $v \in \mathcal{N}(A)$ đều có trị riêng $\lambda = 0$ đối với $A^T A$.
* Toán tử $A^T A$ suy biến hạng trầm trọng. Tỷ số điều kiện $\kappa(A^T A) = \frac{\lambda_{\max}}{\lambda_{\min}} \to \infty$. Địa hình hàm mất mát bị kéo giãn thành hình "hẻm núi phi đẳng hướng hẹp dài": dốc đứng ở các hướng quét khả kiến $\mathcal{R}(A^T)$ và phẳng lì ở không gian hạt nhân $\mathcal{N}(A)$.

---

### 2.3. Hiện tượng "Sụp đổ" khi áp dụng nguyên xi mô hình MVA sang LA-CT

#### 💥 1. Sự bất ổn định số học và lỗi `NaN` của Mamba-1 (Causal 1D vs 2D Void)
Mamba-1 là mô hình Không gian trạng thái chọn lọc (Selective SSM) được thiết kế cho chuỗi 1D nhân quả. Trong MVA, ảnh 2D được làm phẳng thành chuỗi 1D $16.384$ tokens theo thứ tự raster scan.
* Trên Sparse-View CT, thông tin phân bố đều, gradient không bị lệch hướng cực đoan.
* Trên Limited-Angle CT, nêm khuyết và vệt sọc mang tính định hướng không gian 2D rõ rệt. Khi quét dòng 1D, các pixel lân cận theo trục dọc bị đẩy cách xa nhau 128 bước nhảy, làm đứt gãy tương quan không gian hai chiều.
* **Bằng chứng thực nghiệm trung thực:** Trong quá trình huấn luyện trên cụm DGX-A100 (Job `65474`, `65477`), mô hình `LEARN_Mamba` chỉ chạy ổn định được đến Epoch 17 (Val PSNR đạt đỉnh 27.66 dB). **Từ sau Epoch 20, ma trận trạng thái ẩn $h_t = \bar{A} h_{t-1} + \bar{B} x_t$ bị cộng dồn gradient mất kiểm soát qua 14 vòng mở cuộn, dẫn đến tràn số mũ và xuất hiện lỗi `NaN` đồng loạt**, làm sập quá trình huấn luyện. Khi test trên cung $90^\circ$, `LEARN_Mamba` sụt giảm thảm hại xuống **18.76 dB PSNR / 0.4292 SSIM**.

#### 💥 2. Dao động Zigzag và sự mất phương hướng của Gradient Descent bậc 1
Thuật toán hạ gradient bậc 1 trong LEARN: $x_{k+1} = x_k - \alpha [A^T(Ax_k - y) + \mathcal{R}_\theta(x_k)]$.
Trong nêm khuyết, $A^T(Ax_k - y) \equiv 0$, gradient vật lý biến mất. Trên địa hình hẻm núi suy biến, vector gradient luôn trực giao với mặt đẳng mức, khiến các bước nhảy bị nảy qua nảy lại giữa hai vách hẻm (dao động Zigzag), không thể tiến sâu vào lòng hẻm núi để phục hồi tần số khuyết thiếu.

#### 💥 3. Sự mù quáng của thuật toán chọn Landmark sMEANS trong Nyströmformer
Thuật toán `sMEANS` chia đều chuỗi $16.384$ tokens thành các phân đoạn bằng nhau để lấy trung bình làm landmark. Cách tiếp cận đồng đều này phân bổ tài nguyên ngang nhau giữa vùng đã quét và vùng nêm khuyết, không ưu tiên tài nguyên cho vùng bị tổn thương nặng nhất.

---

# PHẦN 3: PHƯƠNG PHÁP ĐỀ XUẤT CỦA CHÚNG TA: KIẾN TRÚC SOLAR

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                            KIẾN TRÚC TỔNG THỂ SOLAR (DUAL-DOMAIN PROXIMAL CG UNROLLING)                     │
│                                                                                                             │
│  [1. MIỀN SINOGRAM: NGOẠI SUY VÀ BẢO TOÀN DỮ LIỆU ĐO]                                                       │
│                                                                                                             │
│   Sinogram khuyết y_pad ──► [ Angular Bi-Mamba Net ] ──► Δy                                                 │
│   (V_full x D, đệm 0)        - Forward SSM (-60° -> +60°)   │                                               │
│                              - Backward SSM (+60° -> -60°)  ▼                                               │
│                              - Gated SiLU Fusion            [ MẶT NẠ BẢO TOÀN DỮ LIỆU ĐO ]                  │
│                                                             ŷ = M ⊙ y_pad + (1 - M) ⊙ Δy                    │
│                                                                     │                                       │
│  [2. MIỀN ẢNH: MỞ CUỘN BIẾN PHÂN BẬC HAI PROXIMAL CG (T = 8 STAGES)] ▼                                       │
│                                                     ┌──► x_t (Trạng thái ảnh hiện tại)                      │
│                                                     │    │                                                  │
│                                                     │    ▼                                                  │
│                                                     │  [ Khối Tiên nghiệm Điều hòa Kép Swin-CNN ]           │
│                                                     │  - Nhánh Cục bộ: Multi-scale Res-CNN 3x3              │
│                                                     │  - Nhánh Toàn cục: Non-local Swin W-MSA 8x8           │
│                                                     │    │                                                  │
│                                                     │    ▼                                                  │
│                                                     │  Vector Tiên nghiệm: ∇R_θ(x_t)                        │
│                                                     │    │                                                  │
│                                                     │    ▼                                                  │
│                                                     │  [ Bộ tổng hợp Vector Vế Phải (RHS Synthesizer) ]     │
│                                                     │  b_t = λ_t A^T ŷ + μ_t x_t - ∇R_θ(x_t)                │
│                                                     │    │                                                  │
│                                                     │    ▼                                                  │
│                                                     │  [ ĐỘNG CƠ GIẢI MA TRẬN MATRIX-FREE SAFE-CG ]         │
│                                                     │  (λ_t A^T A + μ_t I) x_{t+1} = b_t                    │
│                                                     │  * Damping SPD: μ_t = Softplus + 1e-4 > 0 (Zero-NaN)  │
│                                                     │  * 4 bước CG trong không gian Krylov K_4 (float32)    │
│                                                     │    │                                                  │
│                                                     └───┴─── x_{t+1} (Cập nhật ảnh)                         │
│                                                              │                                              │
│                                                              ▼ (Sau 8 Stages)                               │
│                                                     ẢNH TÁI TẠO HOÀN HẢO x_T                                │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.1. Triết lý kiến trúc: Kết hợp Đa miền (Dual-Domain) và Mở cuộn Biến phân Bậc hai Gauss-Newton
SOLAR phối hợp đồng bộ hai cơ chế:
1. **Miền Sinogram:** Ngoại suy dải góc khuyết bằng mô hình chuỗi Angular Bi-Mamba kết hợp cơ chế mặt nạ bảo toàn dữ liệu đo (Data Consistency Masking).
2. **Miền Ảnh:** Khung mở cuộn biến phân giải bài toán Data Consistency bằng bộ giải Conjugate Gradient (CG) có cản dịu đối xứng xác định dương nghiêm ngặt ($\mu_t I$).

---

### 3.2. Miền Sinogram: Mạng Angular Bi-Mamba Net & Cơ chế Mặt nạ Bảo toàn Dữ liệu (Consistency Masking)
Để giải quyết triệt để mâu thuẫn kích thước tensor được chỉ ra trong phản biện học thuật:
* Giả sử hình học quét yêu cầu $V_{\text{full}}$ góc chiếu để thỏa mãn điều kiện Parker (ví dụ $V_{\text{full}} = 128$ views trên cung $180^\circ + 2\gamma_m$). Dữ liệu đo đạc thực tế chỉ thu được $V_{\text{measured}} = 64$ views trong cung giới hạn $\Theta_{\text{scan}}$.
* Ta định nghĩa sinogram đệm $y_{\text{pad}} \in \mathbb{R}^{1 \times V_{\text{full}} \times D}$, trong đó các góc đo được giữ nguyên giá trị và các góc khuyết được gán bằng $0$.
* Ta định nghĩa mặt nạ nhị phân cố định $M \in \{0, 1\}^{1 \times V_{\text{full}} \times D}$:
  $$M(\theta, s) = \begin{cases} 1, & \text{nếu } \theta \in [-\Theta_{\text{scan}}/2, +\Theta_{\text{scan}}/2] \\ 0, & \text{nếu } \theta \in \text{Góc khuyết} \end{cases}$$
* **Cơ chế Mặt nạ Bảo toàn Dữ liệu Đo (Data Consistency Masking, Lin et al. CVPR 2019):**
  Mạng Angular Bi-Mamba tiếp nhận $y_{\text{pad}}$ và dự đoán phần bù góc chiếu $\Delta y \in \mathbb{R}^{1 \times V_{\text{full}} \times D}$. Sinogram hoàn chỉnh $\hat{y}$ được tạo ra bằng công thức:
  $$\hat{y} = M \odot y_{\text{pad}} + (1 - M) \odot \Delta y$$
  Công thức này **bảo đảm toán học $100\%$ rằng dữ liệu tại các góc đo thực tế không bị thay đổi dù chỉ một sai số nhỏ từ mạng nơ-ron**, trong khi vùng góc khuyết được điền đầy bởi giá trị ngoại suy trơn tru của Bi-Mamba.

---

### 3.3. Miền Ảnh: Động cơ Mở cuộn Gauss-Newton Proximal CG Solver

#### 📐 Dẫn xuất Toán học Biến phân (Local Quadratic Surrogate & Gauss-Newton Approximation)
Chuẩn hóa thuật ngữ toán học chính xác theo Nocedal & Wright (2006) và Aggarwal et al. (MoDL 2018):
Xét bài toán tối ưu hóa biến phân với hàm năng lượng:
$$\min_x F(x) = \min_x \left[ \frac{\lambda_t}{2} \| A x - \hat{y} \|_2^2 + \mathcal{R}_\theta(x) \right]$$
Tại mỗi stage mở cuộn $t$, ta xấp xỉ hàm mục tiêu bằng một mặt bậc hai địa phương (Local Quadratic Surrogate):
1. Số hạng khớp dữ liệu $\frac{1}{2}\|Ax - \hat{y}\|_2^2$ vốn đã là hàm bậc hai chính xác với Hessian là $\lambda_t A^T A$ (xấp xỉ Gauss-Newton).
2. Số hạng điều hòa học sâu $\mathcal{R}_\theta(x)$ được tuyến tính hóa qua khai triển Taylor bậc một quanh trạng thái hiện tại $x_t$:
   $$\mathcal{R}_\theta(x) \approx \mathcal{R}_\theta(x_t) + \langle \nabla\mathcal{R}_\theta(x_t), x - x_t \rangle$$
3. Để kiểm soát bán kính tin cậy (Trust-region / Proximal point), ta bổ sung số hạng cản dịu bậc hai $\frac{\mu_t}{2}\|x - x_t\|_2^2$.

Hàm năng lượng xấp xỉ bậc hai tại stage $t$ có dạng:
$$E(x) = \frac{\lambda_t}{2} \| A x - \hat{y} \|_2^2 + \frac{\mu_t}{2} \| x - x_t \|_2^2 + \langle \nabla\mathcal{R}_\theta(x_t), x \rangle + \text{const}$$

Lấy đạo hàm Fréchet theo $x$ và cho triệt tiêu $\nabla_x E(x) = \mathbf{0}$:
$$\lambda_t A^T(Ax - \hat{y}) + \mu_t(x - x_t) + \nabla\mathcal{R}_\theta(x_t) = \mathbf{0}$$
$$\iff \underbrace{(\lambda_t A^T A + \mu_t I)}_{\text{Toán tử } \mathcal{H}_t} x_{t+1} = \underbrace{\lambda_t A^T \hat{y} + \mu_t x_t - \nabla\mathcal{R}_\theta(x_t)}_{\text{Vector Vế Phải } b_t}$$

#### 🛡️ Cơ chế Giảm chấn SPD Damping ($\mu_t I$) — Triệt tiêu 100% lỗi NaN
Hệ số cản dịu $\mu_t$ được tham số hóa nghiêm ngặt:
$$\mu_t = \text{Softplus}(\tilde{\mu}_t) + \epsilon = \ln(1 + e^{\tilde{\mu}_t}) + 10^{-4} > 0 \quad (\forall t)$$
Vì toán tử chuẩn $A^T A \succeq 0$ (bán xác định dương), ta có:
$$\lambda_i(\mathcal{H}_t) = \lambda_t \lambda_i(A^T A) + \mu_t \ge \mu_t \ge 10^{-4} > 0 \quad (\forall i)$$
Ma trận $\mathcal{H}_t$ luôn luôn **đối xứng xác định dương nghiêm ngặt (Strictly SPD)**. Tỷ số điều kiện bị chặn trên:
$$\kappa(\mathcal{H}_t) \le \frac{\lambda_t \|A\|^2 + \mu_t}{\mu_t} < \infty$$
Điều này loại bỏ hoàn toàn khả năng mẫu số bằng 0 trong bước tính bước nhảy CG $\alpha = \frac{r^T r}{p^T \mathcal{H}_t p + 10^{-7}}$, triệt tiêu $100\%$ nguy cơ sinh lỗi `NaN`.

#### ⚡ Giải tích Toán học trên Không gian Hạt nhân: Sự suy biến về Bậc 1 trên $\mathcal{N}(A)$
Bản đề cương xin báo cáo minh bạch bản chất toán học của toán tử $\mathcal{H}_t^{-1}$ theo phản biện chuyên gia:
* Phân rã không gian ảnh $\mathbb{R}^N = \mathcal{R}(A^T) \oplus \mathcal{N}(A)$, với toán tử chiếu trực giao lên không gian hạt nhân là $P_{\mathcal{N}(A)}$ thỏa mãn $P_{\mathcal{N}(A)} A^T = 0$ và $A P_{\mathcal{N}(A)} = 0$.
* Chiếu phương trình $\mathcal{H}_t x_{t+1} = b_t$ lên không gian hạt nhân $\mathcal{N}(A)$:
  $$P_{\mathcal{N}(A)} (\lambda_t A^T A + \mu_t I) x_{t+1} = P_{\mathcal{N}(A)} (\lambda_t A^T \hat{y} + \mu_t x_t - \nabla\mathcal{R}_\theta(x_t))$$
  $$\iff \mu_t P_{\mathcal{N}(A)} x_{t+1} = \mu_t P_{\mathcal{N}(A)} x_t - P_{\mathcal{N}(A)} \nabla\mathcal{R}_\theta(x_t)$$
  $$\iff x_{t+1}^{\text{null}} = x_t^{\text{null}} - \frac{1}{\mu_t} \nabla\mathcal{R}_\theta^{\text{null}}(x_t)$$

> **KẾT LUẬN TOÁN HỌC QUAN TRỌNG:**  
> Trong không gian hạt nhân $\mathcal{N}(A)$ (vùng nêm khuyết Fourier), toán tử $\mathcal{H}_t^{-1}$ **không cung cấp bất kỳ thông tin độ cong bậc hai nào từ dữ liệu đo**, mà suy biến chính xác thành một bước **Hạ Gradient Bậc 1 (First-Order Gradient Descent)** với tốc độ học $\eta_t = 1/\mu_t$ được điều khiển hoàn toàn bởi mạng tiên nghiệm $\nabla\mathcal{R}_\theta$.  
> **Tác dụng thực sự của bộ giải CG tiền điều kiện:** $\mathcal{H}_t^{-1}$ thực hiện tiền điều kiện hóa (preconditioning) mạnh mẽ trên không gian ảnh $\mathcal{R}(A^T)$ (các hướng đã quét), ngăn không cho các bước cập nhật của mạng nơ-ron phá vỡ tính khớp với dữ liệu đo đạc thực tế và dập tắt dao động zigzag.

---

### 3.4. Khối Tiên nghiệm Điều hòa Kép Swin-CNN (Dual-Branch Prior)
Vector tiên nghiệm $\nabla\mathcal{R}_\theta(x_t)$ được tổng hợp từ hai nhánh song song:
1. **Nhánh Cục bộ (Local Res-CNN $3 \times 3$):** Bảo toàn chi tiết tần số cao, bè xương và vi vôi hóa.
2. **Nhánh Không cục bộ (Non-local Swin W-MSA $8 \times 8$):** Sử dụng $1.024$ cửa sổ chú ý song song kết hợp Relative Position Bias ($B_{\text{rel}}$), tạo tường lửa dập tắt vệt sọc toàn cục.
Khung mở cuộn rút gọn xuống **$T = 8$ stages** với cơ chế chia sẻ trọng số, giúp mô hình chỉ có **$\approx 150.000$ tham số (150K)**.

---

### 3.5. Cầu Nối Vật Lý & Khởi Tạo Nghiệm Hạt Giống (Physical Seeding)
Thay vì khởi tạo $x_0 = \text{FBP}(y_{\text{pad}})$ chứa sọc nhiễu năng lượng cao, SOLAR sử dụng cầu nối vật lý:
$$x_0 = \text{FBP}(\hat{y})$$
Sinogram đã vá $\hat{y}$ giúp triệt tiêu phần lớn năng lượng vệt sọc ngay tại nghiệm xuất phát $x_0$, giảm tải gánh nặng điều hòa cho các stage mở cuộn tiếp theo.

---

### 3.6. Chi Tiết Thuật Toán Safe Matrix-Free CG Solver & Đạo Hàm Ngược qua CG
Với $K = 4$ bước lặp, thuật toán CG tìm nghiệm xấp xỉ trong không gian con Krylov 4 chiều:
$$x^{(4)} \in x_0 + \mathcal{K}_4(\mathcal{H}_t, r_0) = x_0 + \text{span}\{r_0, \mathcal{H}_t r_0, \mathcal{H}_t^2 r_0, \mathcal{H}_t^3 r_0\}$$
Về mặt bản chất, đây là một **bộ lọc đa thức bậc 4 (4th-order polynomial preconditioner)** của toán tử $A^T A$.

```python
# Thuật toán Safe Matrix-Free CG Solver (Chạy bắt buộc ở float32)
def SafeCGSolver(x_init, b_t, lambda_t, mu_t, forward_op, backward_op, max_iters=4):
    # Tắt ép kiểu tự động AMP để chống underflow mẫu số trong CG
    with torch.cuda.amp.autocast(enabled=False):
        x = x_init.float().clone()
        b_t = b_t.float()
        
        def H_op(p):
            return lambda_t * backward_op(forward_op(p)) + mu_t * p

        r = b_t - H_op(x)
        p = r.clone()
        # Tính chuẩn phần dư trên từng mẫu batch
        rs_old = torch.sum(r * r, dim=(1, 2, 3), keepdim=True)

        for i in range(max_iters):
            Ap = H_op(p)
            pAp = torch.sum(p * Ap, dim=(1, 2, 3), keepdim=True)
            alpha = rs_old / (pAp + 1e-7)   # float32 an toàn tuyệt đối
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = torch.sum(r * r, dim=(1, 2, 3), keepdim=True)
            
            # Điều kiện dừng kiểm tra trên batch
            if torch.max(rs_new) < 1e-6:
                break
            beta = rs_new / (rs_old + 1e-7)
            p = r + beta * p
            rs_old = rs_new
        return x
```

* **Đạo hàm ngược (Unrolled Autograd vs IFT):** Việc dùng Unrolled CG với $K = 4$ bước lặp lưu đồ thị tính toán có độ phức tạp bộ nhớ $\mathcal{O}(K)$ (rất nhẹ với $K=4$). Nó cho phép lan truyền gradient chính xác tuyệt đối mà không phụ thuộc vào giả định hội tụ vô hạn của Định lý Hàm ẩn (IFT).

---

### 3.7. Hàm Mất Mát Huấn Luyện Liên Kết Đa Miền & Giải Pháp Khắc Phục Suy Giảm SSIM
* **Hàm mất mát cơ sở:**
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{img}}(x_T, x^*) + \gamma \cdot \mathcal{L}_{\text{sino}}(\hat{y}, y^*)$$
* **Phân tích nguyên nhân suy giảm SSIM của SOLAR:**
  Việc sử dụng hàm mất mát thuần $\ell_2^2$ (MSE) cực tiểu hóa kỳ vọng hậu nghiệm (MMSE), dẫn đến xu hướng lấy trung bình các nghiệm khả dĩ trong không gian hạt nhân $\mathcal{N}(A)$ và gây hiệu ứng làm trơn (over-smoothing) biên mô mềm. Kết hợp cùng số hạng cản dịu $\mu_t \|x - x_t\|_2^2$, chi tiết tần số cao bị giảm độ tương phản hạt.
* **Nâng cấp Hàm mất mát đề xuất:** Để khôi phục SSIM cho SOLAR, hàm mất mát miền ảnh được mở rộng kết hợp $\ell_1$ và $\text{SSIM Loss}$ (Zhao et al., IEEE TCI 2017):
  $$\mathcal{L}_{\text{img}}(x_T, x^*) = \alpha \mathcal{L}_{\ell_1}(x_T, x^*) + (1 - \alpha) \mathcal{L}_{\text{SSIM}}(x_T, x^*)$$

---

# PHẦN 4: BẰNG CHỨNG THỰC NGHIỆM ĐỊNH LƯỢNG & ĐỐI SÁNH KHÁCH QUAN

### 4.1. Thiết lập thực nghiệm kiểm thử chuẩn hóa
* **Bộ dữ liệu:** AAPM Mayo Clinic LDCT (Tập test độc lập Patient `L310` gồm 214 lát cắt liên tục) và NIH DeepLesion CT (300 lát cắt test).
* **Hình học:** Fan-Beam CT ($D_{\text{so}} = 1075\text{ mm}$, $D_{\text{sd}} = 1550\text{ mm}$, $512$ kênh detector, ảnh $256 \times 256$).
* **Cung quét:** Chuẩn **LA-120°** ($64$ views trong $[-60^\circ, +60^\circ]$) và Khắc nghiệt **LA-90°** ($64$ views trong $[-45^\circ, +45^\circ]$).

---

### 4.2. Bảng đối sánh trực diện trên Tập Test Bệnh nhân `L310` (LA-120°)

```
========================================================================================================================
BẢNG 1: ĐỐI SÁNH ĐỊNH LƯỢNG TRÊN CUNG QUÉT CHUẨN LA-120° (64 VIEWS, PATIENT L310 - 214 SLICES)
========================================================================================================================
Mô hình                    Bản chất Kiến trúc             Số Stages    Test PSNR (dB)   Test SSIM   Test RMSE    Throughput
------------------------------------------------------------------------------------------------------------------------
FBP Thô (Ram-Lak)          Giải tích (Analytical)         N/A          17.89 dB         0.4984      0.1145       < 5 ms
LEARN_Mamba (MVA)          Bậc 1 + Mamba-1 1D             14           26.32 dB         0.7468      0.0493       ~12 ms
DuDoTrans                  Transformer Đa miền            N/A          25.15 dB         0.7447      0.0650       ~15 ms
LEARN_LongNet (MVA)        Bậc 1 + Dilated Attn           14           31.62 dB         0.8991      0.0270       ~28 ms
SOLAR_LongNet (Đề xuất)    Bậc 2 Proximal CG + LongNet    8            31.03 dB         0.8958      0.0294       ~25 ms
SOLAR_Mamba (Đề xuất)      Bậc 2 Proximal CG + Mamba-1    8            31.90 dB         0.9114      0.0268       ~18 ms
SOLAR_RegFormer (Đề xuất)  Bậc 2 Proximal CG + Swin-CNN   8            32.47 dB         0.9095      0.0247       ~22 ms
SOLAR_Longformer (Đề xuất) Bậc 2 Proximal CG + Longformer 8            32.95 dB         0.9155      0.0228       ~35 ms
SOLAR_DualMamba (Đề xuất)  Bậc 2 Proximal CG + Bi-Mamba   8            32.91 dB         0.9186      0.0228       ~16 ms
------------------------------------------------------------------------------------------------------------------------
* CÁC BASELINE ĐẠT SSIM CAO HƠN TRÊN TẬP TEST 120°:
LEARN (Original CNN)       Bậc 1 + CNN 3 tầng (14 stg)    14           32.29 dB         0.9383      0.0246       ~20 ms
RegFormer (Baseline gốc)   Bậc 1 + Swin (10 stg, 1.82M)   10           32.82 dB         0.9398      0.0234       ~30 ms
LEARN_Longformer (MVA)     Bậc 1 + Sliding Chunks         14           33.10 dB         0.9237      0.0224       ~45 ms
========================================================================================================================
```

> **Đánh giá Trung thực Khoa học:**  
> Ở góc quét $120^\circ$, các mô hình unrolling bậc 1 thuần cục bộ như `LEARN CNN` (SSIM 0.9383) và `RegFormer` (SSIM 0.9398) duy trì chỉ số SSIM cao hơn họ mô hình SOLAR ($0.9095 - 0.9186$). Điều này xuất phát từ việc SOLAR sử dụng hàm mất mát thuần MSE ($\ell_2^2$) kết hợp với số hạng cản dịu $\mu_t$, làm trơn nhẹ các vi cấu trúc tần số cao. Tuy nhiên, `SOLAR_DualMamba` đạt PSNR cạnh tranh cao (**32.91 dB**) với số tham số siêu nhẹ (chỉ $180\text{K}$ so với $1.82\text{M}$ của RegFormer) và tốc độ suy luận nhanh gần gấp đôi.

---

### 4.3. Đánh giá trên Cung Quét Khắc Nghiệt LA-90° (Stress Test Khuyết $180^\circ$ Không gian Tần số)

```
========================================================================================================================
BẢNG 2: STRESS TEST TRÊN CUNG QUÉT KHẮC NGHIỆT LA-90° (PATIENT L310 - 214 SLICES)
========================================================================================================================
Mô hình                    Kiến trúc                     Test PSNR (dB)   Test SSIM   So sánh với Phiên bản Bậc 1 Tương ứng
------------------------------------------------------------------------------------------------------------------------
FBP Thô (Ram-Lak)          Giải tích                     15.20 dB         0.4120      -
LEARN_Mamba                Bậc 1 (MVA)                   18.76 dB         0.4292      Sụp đổ do lỗi tuần tự 1D
LEARN_LongNet              Bậc 1 (MVA)                   19.19 dB         0.5876      Suy thoái
LEARN_Longformer           Bậc 1 (MVA)                   19.16 dB         0.6097      Suy thoái
DuDoTrans                  Transformer Đa miền           21.81 dB         0.6738      -
------------------------------------------------------------------------------------------------------------------------
SOLAR_LongNet (Đề xuất)    Bậc 2 Proximal CG             27.19 dB         0.8639      🚀 +8.00 dB PSNR / +0.2763 SSIM (vs LEARN_LongNet)
SOLAR_Mamba (Đề xuất)      Bậc 2 Proximal CG             27.53 dB         0.8760      🚀 +8.77 dB PSNR / +0.4468 SSIM (vs LEARN_Mamba)
SOLAR_Longformer (Đề xuất) Bậc 2 Proximal CG             28.05 dB         0.8774      🚀 +8.89 dB PSNR / +0.2677 SSIM (vs LEARN_Longformer)
SOLAR_RegFormer (Đề xuất)  Bậc 2 Proximal CG             27.64 dB         0.8804      ⚠️ -0.56 dB PSNR / -0.0010 SSIM (vs RegFormer gốc)
SOLAR_DualMamba (Đề xuất)  Bậc 2 Proximal CG + Bi-Mamba  27.98 dB         0.8843      🏆 SSIM CAO NHẤT TOÀN DỰ ÁN Ở GÓC 90°
------------------------------------------------------------------------------------------------------------------------
* BASELINE ĐẠT PSNR CAO Ở GÓC 90°:
LEARN (Original CNN)       Bậc 1 CNN (1.25M params)      28.03 dB         0.8793      -
RegFormer (Baseline gốc)   Bậc 1 Swin (1.82M params)     28.20 dB         0.8814      PSNR cao nhất ở góc 90°
========================================================================================================================
```

#### 💡 Phân Tích Thực Nghiệm Khách Quan & Đính Chính So Sánh:
1. **Đính chính phép so sánh của `SOLAR_RegFormer`:**
   Mức chênh lệch thực tế giữa `SOLAR_RegFormer` ($27.64\text{ dB} / 0.8804$) và baseline bậc 1 tương ứng của nó (`RegFormer gốc` đạt $28.20\text{ dB} / 0.8814$) là **$-0.56\text{ dB}$ PSNR và $-0.0010$ SSIM**. Con số $+8.48\text{ dB}$ ở bản thảo trước là do so sánh chéo với `LEARN_Longformer` ($19.16\text{ dB}$).
2. **Nguyên nhân sâu xa của hiện tượng phân kỳ bậc 1 trên các mô hình chuỗi dài:**
   Các mô hình chuỗi dài (Mamba-1, Longformer, LongNet) khi kết hợp với unrolling bậc 1 bị sụp đổ nặng ở $90^\circ$ ($\sim 18.7 - 19.1\text{ dB}$) vì cơ chế tự chú ý chuỗi dài rất nhạy cảm với việc mất đối xứng không gian 2D, dẫn đến gradient bị thổi bùng. Động cơ Proximal CG của SOLAR đã **cứu vãn ngoạn mục các mô hình này, tăng từ $+8.00\text{ dB}$ đến $+8.89\text{ dB}$ PSNR**.
3. **Ưu thế của `SOLAR_DualMamba`:**
   Nhờ có khối ngoại suy Sinogram Angular Bi-Mamba vá lành dải sin trước khi chiếu, `SOLAR_DualMamba` đạt **SSIM = 0.8843** — cao nhất trong toàn bộ các mô hình tại góc $90^\circ$, vượt qua cả RegFormer ($0.8814$) và LEARN ($0.8793$).

---

### 4.4. Kiểm chứng tổng quát hóa độc lập trên bộ dữ liệu lớn NIH DeepLesion
* `DuDoTrans`: LA-120° đạt 23.76 dB / 0.6637; LA-90° đạt 20.38 dB / 0.6692.
* `LEARN_Mamba`: LA-120° đạt 26.69 dB / 0.7132; LA-90° sụp đổ xuống 18.59 dB / 0.3913.
* `LEARN_Longformer`: Đạt **32.65 dB / 0.9025** ở LA-120° (Job `72966` hoàn tất 50 epochs).

---

### 4.5. Phân tích chi phí tính toán, bộ nhớ VRAM và số lượng tham số

| Mô hình | Số Stages Mở Cuộn | Số lượng Tham số (Params) | Mức chiếm dụng GPU VRAM | Tốc độ Tái tạo (Throughput) |
| :--- | :---: | :---: | :---: | :---: |
| **LEARN Gốc (CNN)** | 14 stages | $1.25\text{ M}$ | $\approx 2.4\text{ GB}$ | $\approx 50\text{ fps}$ |
| **DuDoTrans** | N/A (Đa miền) | $2.84\text{ M}$ | $\approx 9.5\text{ GB}$ | $\approx 0.32\text{ fps}$ *(Rất chậm)* |
| **RegFormer Gốc** | 10 stages | $1.82\text{ M}$ | $\approx 9.2\text{ GB}$ | $\approx 33\text{ fps}$ |
| **SOLAR_RegFormer (Đề xuất)** | **8 stages** | **$0.15\text{ M}$ (150K)** | **$\approx 5.8\text{ GB}$** | **$\approx 45\text{ fps}$** |
| **SOLAR_DualMamba (Đề xuất)** | **8 stages** | **$0.18\text{ M}$ (180K)** | **$\approx 6.1\text{ GB}$** | **$\approx 62\text{ fps}$** |

---

### 4.6. Bảng Nghiệm Thu Toàn Diện Các Baseline Mới Nhất Trên Tập Validation AAPM LA-120°
(Cập nhật trực tiếp từ hệ thống Slurm HPC DGX-A100 theo [CHECKPOINT.md](file:///home/phandinhminh/Downloads/kltn/agents-research/temp-github/NCKH-CT-Reconstruction/CHECKPOINT.md)):

| Dòng Mô Hình | Job ID Slurm | Số Epochs | Best Val PSNR (dB) | Best Val SSIM | Ghi Chú Kỹ Thuật & Tình Trạng Nghiệm Thu |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`FISTA-Net`** (Deep Unfolded) | `72963` | 50/50 | **35.01 dB** | **0.9527** | ✅ Hoàn thành trọn vẹn (`rc=0`, 11h04m). Ckpt: `fista_net_la-epoch=49.ckpt`. Khung mở cuộn FISTA bậc 1 hội tụ rất cao trên tập Val. |
| **`CT-Former`** (Cross Attention ViT) | `72965` | 50/50 | **33.64 dB** | **0.9303** | ✅ Hoàn thành trọn vẹn (`rc=0`). Ckpt: `ct_former_la-epoch=49.ckpt`. Attention đa hướng bắt tương quan tốt. |
| **`LEARN_Longformer`** (DeepLesion) | `72966` | 50/50 | **32.65 dB** | **0.9025** | ✅ Hoàn thành trọn vẹn trên NIH DeepLesion CT (`rc=0`). Ckpt: `longformer_la-epoch=47.ckpt`. |
| **`LEARN_LongNet`** (DeepLesion) | `72967` | 40/50 | **31.55 dB** | **0.8861** | ⏱️ Chạm mốc 24h Slurm limit tại Epoch 40. Checkpoint an toàn tại Epoch 36 (`last.ckpt`). |
| **`DuDoNet`** (Dual-Domain CNN) | `72964` | 50/50 | **28.98 dB** | **0.8536** | ✅ Hoàn thành trọn vẹn (`rc=0`). Ckpt: `dudonet_la-epoch=41.ckpt`. |
| **`LEARN` (CNN Gốc)** (DeepLesion) | `72822` | 40/50 | **34.27 dB** | - | ⏱️ Chạm mốc 24h Slurm limit tại Epoch 40. Best checkpoint đạt tại Epoch 36. |
| **`MoDL`** (Conjugate Gradient) | `72962` | - | - | - | 🔧 **Đã khắc phục lỗi phân kỳ:** Đã thay thế toán tử FBP trong bước tính ma trận $(A^T A + \lambda I)$ bằng toán tử liên hợp chuẩn $A^T$ (`operator.adjoint`), bảo đảm tính đối xứng xác định dương (SPD) cho CG. |

---

### 4.7. Đánh Giá Độ An Toàn Lâm Sàng: Sai Số Vùng Tổn Thương (ROI HU) & Bản Đồ Ảo Giác Không Gian Hạt Nhân
Để tránh bẫy đánh giá "Inverse Crime" (Kaipio & Somersalo 2007) và kiểm soát rủi ro ảo giác giải phẫu trong không gian hạt nhân $\mathcal{N}(A)$:
1. **Phân tích sai số Hounsfield Unit (HU) trên vùng tổn thương (Lesion ROI):** Đánh giá sai số trung bình tuyệt đối MAE(HU) riêng biệt bên trong các nốt phổi và khối u trên tập NIH DeepLesion, thay vì chỉ đo PSNR trên toàn bộ khung ảnh đen bên ngoài.
2. **Bản đồ Ảo giác Không gian Hạt nhân (Null-space Hallucination Map, Bhadra et al. IEEE TMI 2021):** Phân rã sai số tái tạo thành 2 thành phần độc lập:
   $$e = (x^* - x_{\text{recon}}) = \underbrace{P_{\mathcal{R}(A^T)}(x^* - x_{\text{recon}})}_{\text{Sai số do dữ liệu đo đạc}} + \underbrace{P_{\mathcal{N}(A)}(x^* - x_{\text{recon}})}_{\text{Sai số Ảo giác Không gian Hạt nhân}}$$
   Thành phần $P_{\mathcal{N}(A)}$ phản ánh chính xác mức độ mạng nơ-ron tự "vẽ" ra các chi tiết giải phẫu không có thực trong vùng nêm khuyết.

---

# PHẦN 5: KẾ HOẠCH ĐÁNH GIÁ ĐÓNG GÓP TỪNG THÀNH PHẦN (ABLATION STUDIES FRAMEWORK)

Tiếp thu ý kiến phản biện học thuật, khung khảo sát triệt tiêu được nâng cấp thành **5 nghiên cứu độc lập**:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                             MA TRẬN ĐÁNH GIÁ ĐÓNG GÓP TỪNG THÀNH PHẦN (ABLATION MATRIX)                     │
│                                                                                                             │
│  [Ablation 1: Khối Ngoại suy Sinogram]      [Ablation 2: Động cơ Tối ưu Bậc 2]                              │
│  - Cấu hình 1A: Không có Sinogram (FBP thô) - Cấu hình 2A: Hạ Gradient Bậc 1 (GD)                           │
│  - Cấu hình 1B: Chỉ dùng ŷ khởi tạo x_0     - Cấu hình 2B: Quasi-Newton L-BFGS / QN-Mixer                   │
│  - Cấu hình 1C: Dùng ŷ cả x_0 và CG RHS     - Cấu hình 2C: Gauss-Newton Proximal CG (SOLAR đề xuất)         │
│                                                                                                             │
│  [Ablation 3: Cơ chế Damping μ_t I]         [Ablation 4: Khối Điều hòa Tiên nghiệm]                         │
│  - Cấu hình 3A: μ_t = 0 (Không Damping)     - Cấu hình 4A: Thuần Local Res-CNN 3x3                          │
│  - Cấu hình 3B: μ_t hằng số cố định         - Cấu hình 4B: Thuần Non-local Swin W-MSA 8x8                   │
│  - Cấu hình 3C: Softplus + 1e-4 (SPD Damping)- Cấu hình 4C: Nhánh kép Swin-CNN (SOLAR đề xuất)              │
│                                                                                                             │
│  [Ablation 5: Khảo sát Công bằng Tham số (Parameter-Matched)]                                               │
│  - Cấu hình 5A: SOLAR Chia sẻ Trọng số (Weight-Sharing, 8 stages, 150K params)                              │
│  - Cấu hình 5B: SOLAR Không Chia sẻ Trọng số (Unshared-Weights, 8 stages, ~1.2M params)                     │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 5.1. Khảo sát 1: Tách bạch Đóng góp của Khối Ngoại suy Sinogram (Seeding vs CG RHS)
Tách rõ hai tác động của Sinogram vá lành $\hat{y}$:
* *Cấu hình 1A:* Không dùng mạng sinogram, $x_0 = \text{FBP}(y_{\text{pad}})$, CG khớp với $y_{\text{pad}}$.
* *Cấu hình 1B:* Chỉ dùng $\hat{y}$ để khởi tạo nghiệm hạt giống $x_0 = \text{FBP}(\hat{y})$, còn trong bộ giải CG chỉ khớp với dữ liệu đo thực tế $y_{\text{measured}}$ (ngăn ngừa lan truyền sai số ngoại suy vào CG).
* *Cấu hình 1C:* Dùng $\hat{y}$ cho cả khởi tạo $x_0$ và vế phải CG ($\lambda_t A^T \hat{y}$) ở mọi stage.

### 5.2. Khảo sát 2: Đóng góp của Thuật toán Tối ưu Hóa (Gradient Descent vs QN-Mixer vs Newton-CG)
* So sánh trực diện trên cùng mạng điều hòa Swin-CNN:
  1. *Gradient Descent bậc 1:* $x_{t+1} = x_t - \alpha [A^T(Ax_t - \hat{y}) + \nabla\mathcal{R}(x_t)]$.
  2. *Quasi-Newton L-BFGS / QN-Mixer:* Sử dụng xấp xỉ Hessian đường chéo/Kronecker không lưu ma trận đầy đủ (Ayad et al., CVPR 2024).
  3. *Gauss-Newton Proximal CG (SOLAR):* Giải hệ $(\lambda_t A^T A + \mu_t I) x_{t+1} = b_t$.

### 5.3. Khảo sát 3: Vai trò của Hệ số Damping $\mu_t I$ đối với Ổn định Số học
* Đánh giá 3 mức: $\mu_t = 0$ (phân kỳ chia cho 0 trong CG), $\mu_t = \text{const}$ (không thích nghi theo stage), và $\mu_t = \text{Softplus} + 10^{-4}$ (SPD Damping). Đo lường tỷ lệ lỗi `NaN` và phổ trị riêng $\lambda_{\min}$.

### 5.4. Khảo sát 4: Phân rã Đóng góp của Nhánh Local Res-CNN và Non-local Swin
* Đo độ sắc nét biên (Edge Gradient Difference) cho nhánh Local CNN và chỉ số triệt tiêu sọc (Streak Artifact Indicator — SAI) cho nhánh Non-local Swin.

### 5.5. Khảo sát 5: Khảo sát Công bằng Tham số (Parameter-Matched Ablation: Shared vs Unshared)
* Huấn luyện phiên bản **`SOLAR_RegFormer (Unshared Weights)`** với trọng số riêng biệt ở mỗi stage ($1.2\text{M}$ tham số, tương đương kích thước của RegFormer $1.82\text{M}$) để xác minh một cách khoa học: Liệu sự sụt giảm nhẹ SSIM của SOLAR ở góc $120^\circ$ ($0.9095$ vs $0.9398$) là do thuật toán tối ưu hay do việc nén số lượng tham số xuống 12 lần ($150\text{K}$ params).

---

# PHẦN 6: LỘ TRÌNH TRIỂN KHAI, QUẢN TRỊ RỦI RO & KẾT QUẢ MONG ĐỢI

### 6.1. Xác định Kết quả Bắt buộc (Core Mandatory Deliverables) vs Kết quả Mở rộng (Extended Deliverables)

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   PHÂN ĐỊNH RÕ RÀNG HAI NHÓM KẾT QUẢ ĐẦU RA                                 │
│                                                                                                             │
│  [NHÓM 1: KẾT QUẢ BẮT BUỘC ĐỂ BẢO VỆ KLTN]      [NHÓM 2: KẾT QUẢ MỞ RỘNG - NÂNG TẦM CÔNG TRÌNH]             │
│  (Mandatory Core - Quyết định kết quả khóa luận)   (Extended Goals - Thực hiện sau khi hoàn tất Nhóm 1)     │
│                                                                                                             │
│  ✅ 1. Mô hình SOLAR_DualMamba hoạt động ổn định,   🌟 1. Nâng cấp hàm mất mát L1 + MS-SSIM để đẩy SSIM      │
│        100% Zero-NaN, giải quyết nêm khuyết.             của SOLAR vượt mốc 0.940 ở góc quét 120°.          │
│  ✅ 2. Bảng thực nghiệm đối chứng định lượng trung  🌟 2. Hoàn thiện ứng dụng Web Demo tương tác thời gian   │
│        thực với 7 Baselines (FISTA-Net, RegFormer,       thực (FastAPI Backend + React UI).                 │
│        LEARN, Mamba, LongNet, Longformer, DuDoTrans).🌟 3. Hoàn thiện bản thảo bài báo khoa học đỉnh cao     │
│  ✅ 3. Hoàn thành 5 khảo sát Ablation Study định         nhắm tới Tạp chí Q1 (IEEE TMI / MedIA, IF > 10).   │
│        lượng phân rã đóng góp từng thành phần.      🌟 4. Mở rộng kiểm thử trên LIDC-IDRI và đánh giá mù đôi │
│  ✅ 4. Quyển Khóa luận Tốt nghiệp hoàn chỉnh             bởi Bác sĩ Chẩn đoán Hình ảnh.                     │
│        với phân tích toán học giải tích chặt chẽ.                                                           │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 6.2. Kế hoạch dự phòng hạ tầng tính toán Slurm HPC A100 và kiểm thử cục bộ
1. **Kiểm thử Cục bộ (Local Environment):** Tận dụng GPU máy cá nhân (NVIDIA RTX 3050 Laptop 6GB VRAM) để kiểm tra cú pháp, kích thước tensor và tính ổn định số học của `SafeCGSolver` trên các batch nhỏ ($1 - 2$ slices) mà không cần chờ GPU server.
2. **Kịch bản Tự động hóa khi Server Slurm A100 Kích hoạt Lại:** Toàn bộ script slurm (`scripts/test_*.sh`) đã sẵn sàng. Khi node A100 mở lại (`scontrol resume`), các job nghiệm thu test benchmark cho `FISTA-Net` (`73145`), `CT-Former` (`73105`), `DuDoNet` (`73106`) sẽ chạy ngay lập tức và đồng bộ vào `benchmark_results.csv`.
3. **Bảo mật Học thuật:** Bản thảo bài báo Hội nghị SOICT 2026 (`papers/soict2026/main.tex`) tuyệt đối tuân thủ [AGENTS.md](file:///home/phandinhminh/Downloads/kltn/agents-research/temp-github/NCKH-CT-Reconstruction/AGENTS.md): chỉ so sánh các baseline unrolling bậc 1, giữ kín $100\%$ kiến trúc và số liệu của **SOLAR** phục vụ riêng cho Khóa luận Tốt nghiệp và bài báo Tạp chí Q1.

---

# PHẦN 7: DANH MỤC TÀI LIỆU THAM KHẢO CHỌN LỌC (SELECTED ACADEMIC REFERENCES)

1. **[SIAM 2001 - Giáo trình CT Kinh điển]** Kak, A. C., & Slaney, M. (2001). *Principles of Computerized Tomographic Imaging*. SIAM Classics in Applied Mathematics. [https://epubs.siam.org/doi/pdf/10.1137/1.9780898719277.fm](https://epubs.siam.org/doi/pdf/10.1137/1.9780898719277.fm)
2. **[Med. Phys. 1982 - Điều kiện Đủ Parker]** Parker, D. L. (1982). *Optimal short scan convolution reconstruction for fanbeam CT*. Medical Physics, 9(2), 254–257. [https://pubmed.ncbi.nlm.nih.gov/7087912/](https://pubmed.ncbi.nlm.nih.gov/7087912/)
3. **[Inverse Problems 2013 - Microlocal Analysis]** Frikel, J., & Quinto, E. T. (2013). *Characterization and reduction of artifacts in limited-angle tomography using microlocal analysis*. Inverse Problems, 29(12), 125007. [https://doi.org/10.1088/0266-5611/29/12/125007](https://doi.org/10.1088/0266-5611/29/12/125007)
4. **[MVA 2026 - UIT]** Bui, D. C., Le, D. T., & Nguyen, K. (2026). *Long-Sequence Modeling for Sparse-View Computed Tomography Reconstruction*. Machine Vision and Applications / VNU-HCM UIT.
5. **[IEEE TMI 2018 - LEARN Gốc]** Chen, H., Zhang, Y., Chen, Y., et al. (2018). *LEARN: Learned Experts' Assessment-based Reconstruction Network for sparse-data CT*. IEEE Transactions on Medical Imaging, 37(6), 1333-1347.
6. **[IEEE TRPMS 2023 - RegFormer]** Xia, W., Wang, Z., & Zhang, Y. (2023). *RegFormer: A Local–Nonlocal Regularization-Based Deep Learning Model for Limited-Angle CT Reconstruction*. IEEE Transactions on Radiation and Plasma Medical Sciences, 8(2), 173-183.
7. **[CVPR 2019 - DuDoNet]** Lin, W.-A., Liao, H., Peng, C., et al. (2019). *DuDoNet: Dual Domain Network for CT Metal Artifact Reduction*. Proceedings of the IEEE/CVF CVPR 2019, pp. 10512-10521.
8. **[IEEE TRPMS 2023 - DuDoTrans]** Wang, C., Shang, K., Zhang, H., et al. (2023). *DuDoTrans: Dual-Domain Transformer for Incomplete-Scan CT Reconstruction*. IEEE TRPMS, 7(5), 487-499.
9. **[IEEE TMI 2018 - MoDL]** Aggarwal, H. K., Mani, M. P., & Jacob, M. (2018). *MoDL: Model-Based Deep Learning Architecture for Inverse Problems*. IEEE Transactions on Medical Imaging, 38(2), 394–405.
10. **[CVPR 2024 - QN-Mixer]** Ayad, I., Larue, N., & Nguyen, M. K. (2024). *QN-Mixer: A Quasi-Newton MLP-Mixer Model for Sparse-View CT Reconstruction*. IEEE/CVF CVPR 2024, pp. 11452-11461.
11. **[Springer 2006 - Numerical Optimization]** Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.). Springer Series in Operations Research and Financial Engineering.
12. **[IEEE TCI 2017 - Loss Functions for Restoration]** Zhao, H., Gallo, O., Frosio, I., & Kautz, J. (2017). *Loss Functions for Image Restoration With Neural Networks*. IEEE Transactions on Computational Imaging, 3(1), 47–57.
13. **[IEEE TMI 2021 - Hallucinations in CT]** Bhadra, S., Kelkar, V. A., Brooks, F. J., & Anastasio, M. A. (2021). *On Hallucinations in Tomographic Image Reconstruction*. IEEE Transactions on Medical Imaging, 40(11), 3249–3260.
14. **[JCAM 2007 - Inverse Crimes]** Kaipio, J., & Somersalo, E. (2007). *Statistical inverse problems: Discretization, model reduction and inverse crimes*. Journal of Computational and Applied Mathematics, 198(2), 493–504.
15. **[NeurIPS 2023 - Mamba]** Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv:2312.00752.
16. **[IEEE TMI 2021 - FISTA-Net]** Xiang, J., Dong, Y., & Yang, Y. (2021). *FISTA-Net: Learning a Rapid and Wavelet-Based Inversion for Sparse-Data CT Reconstruction*. IEEE Transactions on Medical Imaging, 40(11), 3290-3302.

---

### KẾT LUẬN & ĐỀ NGHỊ
Bản đề cương nghiên cứu này đã được hiệu đính toàn diện theo các chuẩn mực học thuật quốc tế khắt khe nhất: từ việc đính chính giải tích đối xứng Hermitian của nêm khuyết Fourier, chuẩn hóa toán học của bộ giải biến phân Gauss-Newton Proximal CG và cơ chế mặt nạ bảo toàn dữ liệu sinogram, cho đến việc đính chính và phân tích khách quan các số liệu thực nghiệm đối chứng.

Kính trình Giảng viên Hướng dẫn xem xét, đánh giá và cho ý kiến chỉ đạo để nhóm nghiên cứu tự tin bảo vệ thành công Khóa luận Tốt nghiệp.

*Kính chúc Thầy nhiều sức khỏe và công tác tốt!*
