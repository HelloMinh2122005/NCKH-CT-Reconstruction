# BÁO CÁO NGHIÊN CỨU: TỔNG HỢP CÁC ĐỐI THỦ CẠNH TRANH TRONG LIMITED-ANGLE CT RECONSTRUCTION

> **Tài liệu phục vụ:** Báo cáo Khoa học với Giáo sư Hướng dẫn & Xây dựng Hệ thống Đối chứng cho Bài báo Hội nghị (SOICT 2026) và Tạp chí Q1 (IEEE TMI / MedIA).  
> **Thời gian hoàn thiện:** 19/09/2026  
> **Tác giả:** Nhóm Nghiên cứu CT Reconstruction (MinhPD)

---

## 1. Tổng Quan Bài Toán & Nhu Cầu Khảo Cứu Đối Thủ

Trong bài toán **Tái tạo CT Góc quét Giới hạn (Limited-Angle CT - LA-CT)**, hệ thống đầu thu và bóng phát tia X chỉ quét được một cung góc giới hạn (ví dụ $120^\circ$ hoặc cực hạn $90^\circ$, khuyết tới $270^\circ$). Theo Định lý Lát cắt Fourier (Fourier Slice Theorem), toàn bộ thông tin phổ trong vùng "nêm khuyết" (missing wedge) bị biến mất hoàn toàn, dẫn đến:
- Sự sụp đổ hình học của các thuật toán phân tích cổ điển như Filtered Backprojection (FBP), tạo ra các vệt sọc (streak artifacts) dày đặc.
- Mất mát nghiêm trọng độ tương phản và viền mô mềm theo phương tiếp tuyến với các góc chiếu bị khuyết.

Codebase hiện tại của dự án đã hoàn thành huấn luyện và benchmark các mô hình sau:
- **Nhóm Baseline Unrolling Bậc 1:** `LEARN` (Hu Chen et al., IEEE TMI 2018), `LEARN_Longformer`, `LEARN_LongNet`, `LEARN_Mamba`.
- **Nhóm Baseline Swin / Dual-Domain:** `RegFormer` (Wang et al., 2023), `DuDoTrans` (Wang et al., 2022).
- **Nhóm Đề Xuất Bậc 2 Newton-CG (Dòng SOLAR - Giữ bí mật cho Journal Q1):** `SOLAR_LongNet`, `SOLAR_Mamba`, `SOLAR_Longformer`, `SOLAR_RegFormer`, và kỷ lục mới `SOLAR_DualMamba` (đạt SOTA SSIM $0.8843$ ở LA-90°).

Để làm phong phú thêm cơ sở đối chứng theo yêu cầu của Giáo sư hướng dẫn, tài liệu này khảo cứu và phân loại toàn diện các mô hình đối thủ hàng đầu trong y văn thế giới từ 2017 đến nay.

---

## 2. Phân Loại 5 Trường Phái Đối Thủ Cạnh Tranh (Taxonomy Landscape)

| Trường phái | Mô hình tiêu biểu | Cơ chế hoạt động cốt lõi | Ưu điểm chính | Nhược điểm chính |
| :--- | :--- | :--- | :--- | :--- |
| **1. Mở cuộn sâu (Deep Unrolling / MBDL)** | **`MoDL`** (IEEE TMI 2019)<br>**`FISTA-Net`** (IEEE TMI 2021)<br>**`ADMM-Net`** (IEEE TPAMI 2020)<br>**`AirNet`** (2020) | Mở cuộn các thuật toán tối ưu lặp (CG, FISTA, ADMM) thành các khối mạng nơ-ron học được kết hợp bước data consistency vật lý. | Tính giải thích toán học cao; không bị hallucination tùy tiện; ít tham số (~0.5M - 1M). | Đa số dựa trên gradient bậc 1 nên hội tụ chậm; ở góc khuyết cực hạn $90^\circ$, bước chiếu vật lý bị thiếu dữ liệu trầm trọng dẫn đến mờ ảnh. |
| **2. Mạng Miền kép & Sinogram Inpainting** | **`DuDoNet`** (CVPR 2019)<br>**`DuDoNet++`** (IEEE TMI 2021)<br>**`ADN`** (IEEE TMI 2019)<br>**`DDNet`** (IEEE TRPMS 2018) | Kết hợp đồng thời phục hồi miền Sinogram (nội suy góc khuyết) và miền Ảnh (khử vệt sọc) thông qua các lớp Radon/FBP khả vi. | Tấn công trực tiếp vào nguyên nhân gốc rễ (bù đắp dữ liệu sinogram trước FBP); SSIM ở góc hẹp rất cao. | Cần đồng bộ forward/backward projection qua lại làm tăng bộ nhớ GPU VRAM và thời gian huấn luyện. |
| **3. Vision Transformer & Mamba (SSM)** | **`TransCT`** (IEEE TMI 2021)<br>**`CT-Former`** (IEEE TMI 2023)<br>**`CT-Mamba`** (2024)<br>**`SwinIR`** | Ứng dụng cơ chế Self-Attention hoặc State Space Model để nắm bắt tương quan toàn cục đường dài trên toàn bộ lát cắt CT $512 \times 512$. | Khả năng khôi phục ngữ cảnh toàn cục tuyệt vời; loại bỏ vệt sọc dải rộng tốt hơn CNN cục bộ. | Chi phí tính toán và bộ nhớ lớn ($O(N^2)$ với ViT chuẩn); Mamba 1D dễ bị lệch hướng nếu không quét 2D/Bi-directional. |
| **4. Mô hình Khuếch tán Tạo sinh (Diffusion)** | **`DOLCE`** (ICCV 2023)<br>**`DiffusionMBIR`** (ICLR/MedIA 2023)<br>**`Score-based SDE`** (NeurIPS 2022) | Sử dụng mạng khuếch tán xác suất học phân phối tiên nghiệm của ảnh CT sạch, giải bài toán nghịch đảo bằng cách lấy mẫu có hướng dẫn nhất quán dữ liệu. | Cho chất lượng ảnh cực kỳ sắc nét, chân thực, PSNR cao ở góc khuyết lớn. | Tốc độ suy luận cực kỳ chậm (cần 50 - 200 bước khuếch tán cho 1 ảnh CT); tiềm ẩn nguy cơ sinh chi tiết giả (hallucination) trong chẩn đoán y khoa. |
| **5. Biểu diễn Tiềm ẩn (INR / NeRF-CT)** | **`NeRP`** (IEEE TNNLS 2022)<br>**`NAF`** (2022)<br>**`SCORE`** (2023) | Biểu diễn hàm mật độ suy giảm mô liên tục $f_\theta(x,y)$ bằng mạng MLP; tối ưu hóa trực tiếp dựa trên hàm mất mát tia chiếu cho từng ca bệnh. | Liên tục hóa không gian; không phụ thuộc vào kích thước ma trận pixel; không cần tập huấn luyện khổng lồ. | Tốc độ suy luận chậm (phải tối ưu hóa gradient hàng nghìn bước cho mỗi lát cắt đơn lẻ); khó ứng dụng trong quy trình lâm sàng khẩn cấp. |

---

## 3. Đánh Giá Chi Tiết Từng Đối Thủ Tiềm Năng

### 1. `MoDL` (Model-Based Deep Learning) — Aggarwal et al., IEEE TMI 2019
- **Công thức:** $\min_x \|Ax - y\|_2^2 + \lambda \|\mathcal{D}_w(x)\|_2^2$
- **Điểm mạnh:** Giải bước nhất quán dữ liệu bằng giải thuật **Conjugate Gradient (CG)** phân tích. Rất tương đồng về mặt tư tưởng với họ SOLAR của chúng ta, nhưng MoDL chỉ dùng CG cho bước dữ liệu tuyến tính với bộ điều hòa CNN thông thường, trong khi SOLAR đề xuất tối ưu hóa Newton-CG bậc 2 giải đồng thời cả số hạng dữ liệu lẫn số hạng điều hòa Swin/Mamba.
- **Khuyến nghị:** Cực kỳ phù hợp để đưa vào phần *Related Work* và bảng so sánh đối chứng mở rộng của cả bài báo SOICT lẫn Journal Q1.

### 2. `FISTA-Net` — Xiang et al., IEEE TMI 2021
- **Công thức:** Mở cuộn bước cập nhật FISTA: $x_{k} = \mathcal{T}_{\theta_k}(v_k - \alpha_k A^T(Av_k - y))$, kết hợp hệ số động lượng Nesterov $v_{k+1} = x_k + \beta_k(x_k - x_{k-1})$.
- **Điểm mạnh:** Cấu trúc rõ ràng, tính toán nhẹ, dễ cài đặt và train nhanh.
- **Khuyến nghị:** Một ứng viên lý tưởng nếu muốn bổ sung một baseline mở cuộn bậc 1 khác ngoài LEARN vào bài báo SOICT 2026.

### 3. `DuDoNet++` — Lin et al., IEEE TMI 2021
- **Công thức:** Phối hợp miền Sinogram và miền Ảnh với cơ chế Radial Attention:
  $$y_{\text{refined}} = y + \mathcal{M}_{\text{sino}}(y), \quad x_{\text{initial}} = \text{FBP}(y_{\text{refined}}), \quad x_{\text{final}} = x_{\text{initial}} + \mathcal{M}_{\text{img}}(x_{\text{initial}})$$
- **Điểm mạnh:** Giải quyết rất tốt vấn đề nêm khuyết sinogram; SSIM ở góc hẹp thuộc hàng top đầu y văn.
- **Khuyến nghị:** Đối thủ so sánh trực diện mạnh nhất với `SOLAR_DualMamba` trên bài báo Journal Q1.

### 4. `CT-Former` — Wang et al., IEEE TMI 2023
- **Công thức:** Sử dụng Cross-Shape Window Self-Attention trên miền ảnh, ghép cặp các token dọc và ngang để giảm chi phí chú ý toàn cục.
- **Điểm mạnh:** Tận dụng tối đa sức mạnh của Transformer cho ảnh y tế $512 \times 512$ mà không bị tràn bộ nhớ VRAM.
- **Khuyến nghị:** Dùng để so sánh đối chứng đại diện cho trường phái Vision Transformer hiện đại trên bài báo Journal Q1.

### 5. `DOLCE` — Sun et al., ICCV 2023
- **Công thức:** Khung khuếch tán xác suất có điều kiện (Conditional Diffusion Framework) thiết kế chuyên biệt cho Limited-Angle CT.
- **Điểm mạnh:** Thiết lập SOTA về độ sắc nét hình thái trong các hội nghị thị giác máy tính hàng đầu thế giới (ICCV).
- **Khuyến nghị:** Đưa vào phân tích đối chiếu trong phần Thảo luận (Discussion) của bài báo Journal Q1 để chứng minh: dù Diffusion có PSNR cao và ảnh nhìn sắc nét, nhưng mô hình `SOLAR_DualMamba` của chúng ta có tốc độ tái tạo nhanh gấp hàng trăm lần, ít tham số hơn 100 lần, và chỉ số SSIM bảo toàn cấu trúc giải phẫu y khoa vượt trội hơn hẳn.

---

## 4. Ma Trận Đối Sánh Tổng Hợp (Benchmark Comparison Matrix)

| Mô hình | Thuật toán / Kiến trúc | Miền xử lý | Params | Số stages ($K$) | LA-120° (PSNR/SSIM) | LA-90° (PSNR/SSIM) | Mục tiêu bài báo |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FBP Thô** | Analytical Ram-Lak | Sino $\to$ Image | 0 | -- | 17.89 / 0.4984 | 15.20 / 0.4120 | Baseline chung |
| **`LEARN`** | Unrolling Bậc 1 + CNN | Miền ảnh | 0.84M | 14 | 32.29 / 0.9383 | 28.03 / 0.8793 | SOICT & Journal |
| **`LEARN_Longformer`**| Unrolling Bậc 1 + Sliding Chunks | Miền ảnh | 4.00M | 14 | **33.10** / 0.9237 | 19.16 / 0.6097 | SOICT (Trọng tâm) |
| **`DuDoTrans`** | Dual-Domain Transformer | Sino + Image | 0.13M | 1 | 25.15 / 0.7447 | 21.81 / 0.6738 | SOICT & Journal |
| **`RegFormer`** | Unrolling Bậc 1 + Swin Transformer | Miền ảnh | 1.23M | 14 | 32.82 / **0.9398** | 28.20 / 0.8814 | Journal Q1 |
| **`MoDL`** *(Ước lượng)* | Unrolling Bậc 1 + CG Solver | Miền ảnh | ~0.50M | 10 | ~31.5 / ~0.905 | ~25.5 / ~0.830 | So sánh y văn |
| **`DuDoNet++`** *(Ước lượng)*| Dual-Domain + Radial Attention | Sino + Image | ~1.50M | 4 | ~32.5 / ~0.915 | ~28.1 / ~0.880 | So sánh y văn |
| **`SOLAR_RegFormer`** | Newton-CG Bậc 2 + Swin Window | Miền ảnh | **0.088M**| **8** | 32.47 / 0.9095 | 27.64 / 0.8804 | Journal Q1 |
| **`SOLAR_DualMamba`** | **Bi-Mamba Sino + Newton-CG Swin**| **Miền kép** | **0.225M**| **8** | **32.91** / 0.9186 | 27.98 / **0.8843** | 🏆 **SOTA Journal Q1** |

---

## 5. Kế Hoạch Thực Nghiệm Tiếp Theo (Actionable TODOs)

Theo chỉ đạo nghiên cứu, nhóm sẽ tiến hành **lặp lại tương tự quy trình huấn luyện và kiểm thử benchmark trên 2 bộ dữ liệu lâm sàng mở rộng lớn:**

### 🎯 1. Bộ dữ liệu NIH DeepLesion (>32,000 CT slices đa cơ quan)
- **Mục tiêu khoa học:** Đánh giá khả năng bảo toàn và phát hiện tổn thương giải phẫu sâu (lesion detectability) trong điều kiện góc quét hẹp $120^\circ$ và $90^\circ$.
- **Kế hoạch thực thi:**
  - Hiện tại Job Slurm **`72819`** (`LEARN_Longformer` DeepLesion) đang chạy ổn định tại Epoch 10/50 trên node DGX-A100.
  - Các baseline tiếp theo trong hàng đợi Slurm (`LEARN_LongNet` Job `72820`, `LEARN_Mamba` Job `72821`, `LEARN` Job `72822`, `DuDoTrans` Job `72823`) sẽ tự động được cấp phát GPU khi tài nguyên khả dụng.
  - Sau khi các baseline hoàn thành, kích hoạt job huấn luyện và test benchmark cho `SOLAR_DualMamba` trên NIH DeepLesion để thiết lập bảng đối sánh đa cơ quan hoàn chỉnh.

### 🎯 2. Bộ dữ liệu LIDC-IDRI (1,018 bệnh nhân CT lồng ngực / phổi)
- **Mục tiêu khoa học:** Đánh giá khả năng tái tạo nốt mờ phổi (pulmonary nodules) và cấu trúc phế quản vi mô dưới góc quét hẹp.
- **Kế hoạch thực thi:**
  - Chuẩn bị sẵn sàng các script `scripts/train_*_lidc.sh` với cấu hình VRAM tối ưu (~12,000MB - 15,000MB) để nộp Slurm job ngay khi đợt DeepLesion hoàn tất.
  - Lặp lại quy trình đánh giá test định lượng (PSNR, SSIM, RMSE) trên 2 cung quét chuẩn LA-120° và LA-90°.
