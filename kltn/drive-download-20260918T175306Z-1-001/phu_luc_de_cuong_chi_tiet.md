# PHỤ LỤC: ĐỀ CƯƠNG CHI TIẾT KHÓA LUẬN TỐT NGHIỆP

**Tên đề tài:**  
*Nghiên cứu và phát triển mạng nơ-ron mở cuộn tối ưu bậc hai miền kép (SOLAR_DualMamba) trong bài toán tái tạo ảnh chụp cắt lớp vi tính góc giới hạn (Limited-Angle CT Reconstruction)*  
**Tên tiếng Anh:**  
*Second-Order Dual-Domain Deep Unrolling Network with Angular Sinogram Bi-Mamba for Limited-Angle CT Reconstruction*

---

## 1. Nội dung đề tài

### 1.1. Tổng quan đề tài
*(Các đề tài, sản phẩm liên quan đã được nghiên cứu hoặc có trên thị trường trước thời điểm hiện tại; thực trạng và lý do thực hiện nghiên cứu)*

Chụp cắt lớp vi tính (Computed Tomography — CT) là phương pháp chẩn đoán hình ảnh trụ cột trong y khoa hiện đại. Tuy nhiên, việc chiếu tia X liên tục ở toàn bộ các góc quay ($180^\circ + \text{fan}$) làm gia tăng đáng kể nguy cơ ung thư do tích lũy liều bức xạ ion hóa. Nhằm tuân thủ nguyên tắc an toàn bức xạ **ALARA** (*As Low As Reasonably Achievable*), cũng như trong các tình huống thực tế đặc biệt (bệnh nhân chấn thương nặng không thể quay vòng toàn phần, chụp cắt lớp X-quang tuyến vú dạng số hóa Tomosynthesis — DBT, hoặc phẫu thuật có dẫn đường bằng cánh tay quay C-arm), cung quét chùm tia bị thu hẹp đáng kể (ví dụ chỉ quét trong dải $[-60^\circ, +60^\circ]$ thay vì đủ $360^\circ$). Bài toán này được gọi là **Tái tạo ảnh CT góc giới hạn (Limited-Angle CT — LA-CT Reconstruction)**.

Về mặt toán học và vật lý, bài toán LA-CT là một bài toán nghịch đảo phi chỉnh nghiêm trọng (severely ill-posed inverse problem). Việc thiếu hụt góc chiếu tạo ra một vùng khuyết hình nêm (**Missing Wedge**) cực lớn lên tới $240^\circ$ trong không gian tần số Fourier (theo Định lý Fourier Slice). Thực trạng của các phương pháp hiện hành như sau:
1. **Phương pháp giải tích cổ điển (Filtered Backprojection — FBP):** Áp dụng bộ lọc Ram-Lak và chiếu ngược đơn thuần. Do thiếu dữ liệu trong nêm khuyết, ảnh FBP chứa đựng các vệt sọc nhiễu giả (**Streak Artifacts**) dày đặc, làm suy giảm tương phản nghiêm trọng và che khuất hoàn toàn các thương tổn mô mềm.
2. **Phương pháp lặp truyền thống (Iterative Reconstruction — TV, SART, CS):** Sử dụng các hàm điều hòa giải tích như Total Variation. Tuy nhiên, phương pháp lặp đòi hỏi thời gian tính toán kéo dài (hàng trăm vòng lặp), phụ thuộc nặng nề vào việc tinh chỉnh siêu tham số thủ công và thường gây ra hiệu ứng giả tạo dạng khối (patchy/cartoon-like artifacts).
3. **Các mô hình Học sâu mở cuộn bậc một (1st-Order Deep Unrolling — LEARN, RegFormer):**
   - Mở cuộn thuật toán hạ bậc một (Gradient Descent). Địa hình hàm mất mát của bài toán LA-CT bị suy biến dị hướng nặng nề do ma trận Hessian $A^T A$ có tỷ số điều kiện rất lớn. Các phương pháp bậc một dễ bị dao động zigzag, hội tụ rất chậm và đòi hỏi số lượng stage mở cuộn lớn ($T \ge 14$ stages).
   - Khi tăng số stage, mô hình trở nên cồng kềnh ($> 1.2\text{M} - 2.9\text{M}$ tham số), tiêu tốn nhiều bộ nhớ GPU và dễ dẫn đến hiện tượng quá khớp (overfitting).
4. **Hạn chế cốt lõi của hướng tiếp cận đơn miền (Single-Domain):**
   - Đa số các nghiên cứu hiện nay chỉ hoạt động thuần túy trong **miền ảnh (Image Domain)**. Khi đó, mạng nơ-ron bị ép phải tự "bịa" ra thông tin bị mất từ con số không, dễ dẫn đến hiện tượng ảo giác (hallucination) làm sai lệch cấu trúc giải phẫu y tế.
   - Một số ít mô hình xử lý đa miền (như DuDoTrans) lại sử dụng cơ chế Cross-Attention của Transformer với chi phí tính toán bậc hai $\mathcal{O}(N^2)$ cực kỳ đắt đỏ, dễ gây quá tải VRAM và làm rò rỉ vệt sọc nêm khuyết xuyên qua các cơ quan lành.

**Lý do thực hiện đề tài:**  
Cần thiết phải nghiên cứu một hướng tiếp cận đột phá kết hợp hài hòa hai miền:
- Khai thác bản chất liên tục của phép biến đổi Radon trên **Miền Chiếu (Sinogram Domain)** để ngoại suy và vá lành dải khuyết góc quét từ gốc rễ hình học trước khi đưa qua toán tử chiếu.
- Ứng dụng động cơ tối ưu hóa **Bậc hai (Second-Order Newton-CG)** trên **Miền Ảnh (Image Domain)** nhằm tăng tốc độ hội tụ, giảm thiểu số stage mở cuộn và đảm bảo tính ổn định số học tuyệt đối ($100\%$ không phát sinh lỗi phân kỳ NaN).

---

### 1.2. Mục tiêu của đề tài
*(Mục tiêu cụ thể của KLTN và những cải tiến vượt bậc so với thực trạng)*

Mục tiêu tổng quát của khóa luận là thiết kế, xây dựng và đánh giá thực nghiệm một mô hình mạng nơ-ron tích hợp vật lý y tế mang tên **`SOLAR_DualMamba`** (*Second-Order Dual-Domain Newton-CG Deep Unrolling with Angular Sinogram Bi-Mamba and Swin-CNN Regularizer*), nhằm tái tạo ảnh CT góc giới hạn với độ chính xác lâm sàng cao, bảo tồn chi tiết giải phẫu và tối ưu hóa tài nguyên phần cứng.

**Các mục tiêu cải tiến cụ thể:**
1. **Đột phá ngoại suy miền chiếu bằng Angular Bi-Mamba:**
   - Đề xuất module **Angular Sinogram Bi-Mamba Extrapolator** coi trục góc chiếu $V = 64$ là chiều chuỗi thời gian của mô hình không gian trạng thái (State Space Model — SSM).
   - Quét hai chiều đối xứng (Forward: $-60^\circ \to +60^\circ$ và Backward: $+60^\circ \to -60^\circ$) để học quy luật quỹ đạo hình sin của phép biến đổi Radon, tự động vá lành dải khuyết góc quét với **độ phức tạp tuyến tính $\mathcal{O}(V)$** thay vì $\mathcal{O}(V^2)$ như Transformer.
2. **Động cơ tối ưu hóa lặp Bậc hai Newton-CG (`SafeCGSolver`):**
   - Khắc phục sự dao động của phương pháp bậc một bằng cách giải phương trình đạo hàm bậc hai tại mỗi stage mở cuộn:
     $$(\lambda_t A^T A + \mu_t I) x_{t+1} = \lambda_t A^T \hat{y} + \mu_t x_t - \nabla\mathcal{R}_\theta(x_t)$$
   - Tối ưu hóa không lưu trữ ma trận (**Matrix-Free Conjugate Gradient**) trực tiếp trên GPU qua thư viện ASTRA CUDA, đạt tốc độ hội tụ siêu tuyến tính chỉ trong $K = 4$ bước lặp Krylov.
   - **Giảm số stages mở cuộn từ 14 xuống 8 stages** ($43\%$ ít vòng lặp hơn so với baseline bậc một).
3. **Bảo đảm ổn định số học tuyệt đối (Strictly SPD & Zero-NaN Guarantee):**
   - Tham số hóa trọng số giảm chấn $\mu_t = \text{Softplus}(\text{raw\_mu}_t) + 10^{-4} > 0$, chứng minh phổ giá trị riêng ma trận Hessian luôn thỏa mãn $\sigma(\mathcal{H}_t) \subset [10^{-4}, +\infty)$, triệt tiêu hoàn toàn nguy cơ chia cho 0 và lỗi tràn số NaN.
4. **Mạng điều hòa kép Swin-RegFormer đóng vai trò "Tường lửa" giam hãm nhiễu:**
   - Kết hợp nhánh **Local Res-CNN $3\times 3$** (Receptive field $7\times 7$ bảo toàn cạnh biên xương và cấu trúc vi thể tần số cao) và nhánh **Non-local Swin Window Attention $8\times 8$** (gồm $1,024$ cửa sổ độc lập kèm ma trận vị trí tương đối 2D $B_{\text{rel}}$).
   - Giới hạn tính toán tương quan trong các cửa sổ cục bộ, tạo thành bức tường lửa ngăn chặn vệt sọc nêm khuyết lan truyền xuyên qua các mô mềm.
5. **Tối ưu hóa kiến trúc siêu gọn nhẹ:**
   - Ứng dụng cơ chế chia sẻ trọng số tuần hoàn (**Recurrent Weight Sharing**) cho toàn bộ 8 stages, đưa tổng số tham số mô hình về mức siêu nhẹ **$\sim 150\text{ K}$ tham số** ($\sim 0.60\text{ MB}$, nhỏ hơn 8 lần so với RegFormer 1.23M và nhỏ hơn 19 lần so với LEARN_Mamba 2.90M).
6. **Công bố khoa học và xây dựng phần mềm ứng dụng:**
   - Viết và nộp (Submit) 01 bài báo khoa học tại hội nghị/tạp chí quốc tế uy tín.
   - Xây dựng 01 ứng dụng phần mềm Demo trực quan hỗ trợ bác sĩ/chuyên gia tải dữ liệu sinogram và thực hiện tái tạo ảnh CT thời gian thực.

---

### 1.3. Phương pháp thực hiện
*(Tổng quan phương pháp tiếp cận, công cụ và quy trình kỹ thuật)*

Đề tài áp dụng phương pháp nghiên cứu kết hợp giữa toán lý thuyết bài toán nghịch đảo, thiết kế mô hình học sâu hiện đại và thực nghiệm trên hệ thống tính toán hiệu năng cao (HPC Slurm):

1. **Cơ sở Lý thuyết và Toán học:**
   - Nghiên cứu phép biến đổi Radon chùm tia quạt (Fan-Beam Geometry), toán tử chiếu thuận $A$ và toán tử chiếu ngược $A^T$ theo chuẩn máy CT y tế Siemens Somatom (`src_radius=600`, `det_radius=290`).
   - Nghiên cứu lý thuyết Mô hình Không gian Trạng thái chọn lọc (Selective State Space Model — Mamba/S6) và phương pháp rời rạc hóa Zero-Order Hold (ZOH).
   - Nghiên cứu phương pháp tối ưu hóa biến phân bậc hai (Second-Order Variational Optimization) và giải thuật Gradient Liên hợp (Conjugate Gradient) trên không gian con Krylov.
2. **Quy trình Xây dựng và Huấn luyện Mô hình:**
   - Môi trường phát triển: Python 3.10, PyTorch, PyTorch Lightning, ODL (Operator Discretization Library), ASTRA Toolbox CUDA.
   - Cụm máy chủ thực nghiệm: Slurm HPC GPU NVIDIA A100 (40GB/80GB VRAM) với hệ thống quản lý NVIDIA Multi-Process Service (MPS).
   - Hàm mất mát: Kết hợp mất mát sai số toàn cục Normalized $L_2$ Loss và mất mát bảo toàn cấu trúc SSIM Loss.
3. **Dữ liệu Thực nghiệm:**
   - **(1) AAPM Mayo Clinic Low-Dose CT Grand Challenge:** Tập dữ liệu chuẩn quốc tế với 10 bệnh nhân ($2,164$ lát cắt CT ngực và bụng), phân chia 8 bệnh nhân huấn luyện ($1,706$ lát cắt), 1 bệnh nhân kiểm định (Patient L333: 244 lát cắt) và 1 bệnh nhân kiểm thử độc lập (Patient L310: 214 lát cắt).
   - **(2) NIH DeepLesion CT Dataset:** Bộ dữ liệu tổn thương đa tạng lâm sàng quy mô lớn với 173 bệnh nhân (tổng cộng $1,096$ lát cắt CT tổn thương mô mềm, gan, thận, phổi và hạch bạch huyết), phân chia 114 bệnh nhân huấn luyện ($812$ lát cắt), 32 bệnh nhân kiểm định ($124$ lát cắt) và 27 bệnh nhân kiểm thử độc lập ($160$ lát cắt) nhằm khảo sát năng lực bù đắp và bảo toàn cấu trúc thương tổn bệnh lý.
   - **(3) LIDC-IDRI (Lung Image Database Consortium):** Bộ dữ liệu CT lồng ngực quốc tế chuyên sâu về chẩn đoán nốt u phổi với 11 bệnh nhân (tổng cộng $1,975$ lát cắt CT lồng ngực độ phân giải cao), phân chia 7 bệnh nhân huấn luyện (LIDC-IDRI-0001 đến 0008, trừ ca 0006: $1,186$ lát cắt), 2 bệnh nhân kiểm định (LIDC-IDRI-0009 và 0010: 533 lát cắt) và 2 bệnh nhân kiểm thử độc lập (LIDC-IDRI-0011 và 0012: 256 lát cắt) nhằm đánh giá khả năng bảo tồn các chi tiết nốt u vi mô tần số cao trong điều kiện nêm khuyết lớn.
4. **Phương pháp Đánh giá và Đối chứng:**
   - Chỉ số định lượng: Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM), Root Mean Squared Error (RMSE).
   - Kiểm thử 2 cấu hình quét: Cung quét chuẩn $\text{LA-}120^\circ$ (khuyết $240^\circ$) và cấu hình khắc nghiệt $\text{LA-}90^\circ$ (khuyết $270^\circ$).
   - Hệ thống Baseline đối chứng: FBP (Ram-Lak), LEARN (CNN 3 tầng), DuDoTrans (Dual-Domain Transformer), RegFormer (Bậc 1 Swin), LEARN_Mamba, LEARN_Longformer, SOLAR_LongNet, SOLAR_Mamba, SOLAR_RegFormer.

---

### 1.4. Các nội dung chính và giới hạn của đề tài
*(Nội dung triển khai, phương pháp đánh giá và hệ thống demo)*

#### A. Các nội dung chính:
1. **Thu thập, khảo sát và tiền xử lý dữ liệu CT y tế:**
   - Giải mã ảnh DICOM 16-bit, chuyển đổi đơn vị Hounsfield Unit (HU), cắt ngưỡng cửa sổ mô mềm $[-1000, 1000]\text{ HU}$.
   - Thiết lập mô phỏng hình học chiếu Fan-beam, tạo cache offline sinogram và ảnh FBP thô cho dải góc $120^\circ$ và $90^\circ$.
2. **Nghiên cứu, thiết kế và tối ưu kiến trúc `SOLAR_DualMamba`:**
   - Hiện thực hóa module `AngularSinogramMambaBlock` quét hai chiều dọc theo trục góc chiếu.
   - Tích hợp bộ giải `SafeCGSolver` matrix-free không gian con Krylov với điều kiện giảm chấn bảo đảm SPD.
   - Xây dựng khối điều hòa `RegFormerDualBranchRegularizer` kết hợp Local CNN $3\times 3$ và Swin Window Attention $8\times 8$.
3. **Thực nghiệm diện rộng trên cụm máy chủ Slurm HPC:**
   - Huấn luyện mô hình trong 50 Epochs; đối soát checkpoint tốt nhất qua từng epoch.
   - Đánh giá toàn diện trên tập test độc lập Patient L310 và các tập dữ liệu mở rộng.
4. **Trực quan hóa và Nghiên cứu chuyên sâu:**
   - Trích xuất Attention Maps, Error Maps khuếch đại $\times 5$ ở chuẩn in ấn 300 DPI.
   - Phân tích hiện tượng giam hãm vệt sọc nêm khuyết giữa các cơ quan giải phẫu.
5. **Xây dựng Ứng dụng Phần mềm Demo (Application Demo):**
   - Xây dựng ứng dụng web tương tác (dựa trên Python Web Framework như Streamlit hoặc FastAPI + Modern HTML5/CSS Dashboard).
   - **Chức năng Demo:**
     + Cho phép người dùng tải lên ảnh Sinogram thô hoặc chọn mẫu ca bệnh từ kho dữ liệu y tế.
     + Cho phép lựa chọn cấu hình góc quét giới hạn ($\text{LA-}120^\circ$ hoặc $\text{LA-}90^\circ$).
     + Hiển thị đồng thời: Ảnh Sinogram trước và sau khi được vá lành bởi Angular Bi-Mamba; Ảnh tái tạo FBP thô đầu vào; Ảnh tái tạo chuẩn đoán của `SOLAR_DualMamba` và bản đồ sai số vi mô (Residual Error Map).
     + Đo đạc và hiển thị tức thời các chỉ số chất lượng ảnh (PSNR, SSIM, RMSE) cùng thời gian suy luận (Inference Time).
6. **Công bố bài báo khoa học và Viết luận văn tốt nghiệp:**
   - Hoàn thiện bản thảo và nộp (Submit) bài báo khoa học đến hội nghị/tạp chí quốc tế.
   - Hoàn tất toàn văn báo cáo Khóa luận tốt nghiệp theo quy chuẩn của trường Đại học Công nghệ Thông tin (VNU-HCM UIT).

#### B. Giới hạn của đề tài:
- **Hình học quét:** Tập trung khảo sát trên hình học chùm tia quạt 2D Fan-Beam (chuẩn quét phổ biến của các máy CT bệnh viện); chưa mở rộng sang chùm tia nón 3D Cone-Beam CT (CBCT).
- **Phạm vi góc quét:** Tập trung nghiên cứu hai dải góc giới hạn chuẩn trong nghiên cứu học thuật quốc tế là $120^\circ$ (khuyết $240^\circ$) và $90^\circ$ (khuyết $270^\circ$).
- **Dữ liệu thử nghiệm:** Sử dụng các bộ dữ liệu ảnh CT y tế công khai chuẩn quốc tế (AAPM Low-Dose CT, NIH DeepLesion, LIDC-IDRI), chưa tiến hành chụp thử nghiệm lâm sàng trực tiếp trên cơ thể người thật do các rào cản đạo đức sinh học y tế.

---

## 2. Kế hoạch thực hiện
*(Phân bổ tiến độ 5 tháng, phân công chi tiết công việc cho sinh viên)*

- **Sinh viên thực hiện:** Phan Đình Minh – MSSV: 23520949  
- **Cán bộ hướng dẫn:** TS. Bùi Cao Doanh – Khoa Công nghệ Phần mềm  

### Kế hoạch làm việc chi tiết từng tháng (Ưu tiên định dạng Bullet Points):

1. **Tháng 1 (01/01/2026 – 31/01/2026): Khảo sát tài liệu, nghiên cứu lý thuyết & Tiền xử lý dữ liệu**
   - **Nội dung công việc chi tiết:**
     - Tìm kiếm, đọc và tổng hợp các bài báo khoa học đỉnh cao về CT Reconstruction, Deep Unrolling (LEARN, RegFormer, DuDoTrans) và State Space Models (Mamba, Vision Mamba).
     - Nghiên cứu cơ sở toán học và vật lý: Biến đổi Radon chùm tia quạt (Fan-beam), bài toán nghịch đảo phi chỉnh, tối ưu hóa biến phân bậc hai và giải thuật Newton-CG.
     - Thu thập và tiền xử lý dữ liệu AAPM Mayo Clinic: giải mã DICOM 16-bit, chuẩn hóa HU, sinh cache offline Sinogram và FBP chùm tia Fan-beam ($120^\circ$ và $90^\circ$).
   - **Sản phẩm / Kết quả đạt được:** Báo cáo tổng quan tài liệu (Literature Review); Bộ dữ liệu cache tiền xử lý (.npy) hoàn chỉnh sẵn sàng cho huấn luyện trên cụm HPC.
   - **Phân công thực hiện:** Phan Đình Minh (100%).

2. **Tháng 2 (01/02/2026 – 28/02/2026): Nghiên cứu các hướng tiếp cận & Thực nghiệm các biến thể đơn miền**
   - **Nội dung công việc chi tiết:**
     - Triển khai và huấn luyện các mô hình Baseline đối chứng (FBP, LEARN CNN, DuDoTrans, RegFormer).
     - Nghiên cứu và thử nghiệm các hướng mở cuộn bậc hai đơn miền: `SOLAR_LongNet`, `SOLAR_Mamba`, `SOLAR_RegFormer`.
     - Phân tích điểm nghẽn của hướng tiếp cận đơn miền (Image Domain) khi nêm khuyết mở rộng $270^\circ$, từ đó hình thành và đề xuất ý tưởng kiến trúc miền kép (Dual-Domain).
   - **Sản phẩm / Kết quả đạt được:** Kết quả huấn luyện và nghiệm thu các mô hình baseline; Báo cáo phân tích so sánh các hướng mở cuộn đơn miền; Thiết kế sơ đồ nguyên lý kiến trúc `SOLAR_DualMamba`.
   - **Phân công thực hiện:** Phan Đình Minh (100%).

3. **Tháng 3 (01/03/2026 – 31/03/2026): Cài đặt, hiện thực hóa và huấn luyện SOLAR_DualMamba**
   - **Nội dung công việc chi tiết:**
     - Lập trình module `AngularSinogramMambaBlock` quét hai chiều Forward/Backward dọc trục góc chiếu.
     - Lập trình module `SafeCGSolver` bậc hai tối ưu không ma trận (Matrix-Free) với điều kiện bảo đảm strictly SPD.
     - Tích hợp khối điều hòa kép Swin-RegFormer và cơ chế Recurrent Weight Sharing.
     - Xây dựng kịch bản Slurm HPC và tiến hành huấn luyện mô hình 50 Epochs trên GPU NVIDIA A100.
   - **Sản phẩm / Kết quả đạt được:** Mã nguồn hoàn chỉnh của `SOLAR_DualMamba` (PyTorch Lightning); Checkpoints mô hình được lưu trữ và tối ưu qua từng epoch trên cụm Slurm HPC.
   - **Phân công thực hiện:** Phan Đình Minh (100%).

4. **Tháng 4 (01/04/2026 – 30/04/2026): Đánh giá thực nghiệm diện rộng, Stress Test & Phân tích giải phẫu**
   - **Nội dung công việc chi tiết:**
     - Thực hiện kiểm thử độc lập (Independent Test Benchmark) trên 214 lát cắt Patient L310 cho cả hai cấu hình $\text{LA-}120^\circ$ và $\text{LA-}90^\circ$.
     - Mở rộng thực nghiệm huấn luyện và kiểm thử trên bộ dữ liệu NIH DeepLesion và LIDC-IDRI để đánh giá khả năng tổng quát hóa.
     - Trích xuất bản đồ chú ý (Attention Maps), bản đồ sai số (Error Maps), phân tích chất lượng trực quan ở độ phân giải 300 DPI.
     - Chuẩn bị số liệu, biểu đồ và hoàn thiện các bảng đối sánh khoa học.
   - **Sản phẩm / Kết quả đạt được:** Bộ số liệu thực nghiệm hoàn chỉnh (PSNR, SSIM, RMSE); Bộ ảnh trực quan hóa chất lượng cao đạt chuẩn công bố quốc tế; Báo cáo kết quả thực nghiệm chi tiết.
   - **Phân công thực hiện:** Phan Đình Minh (100%).

5. **Tháng 5 (01/05/2026 – 31/05/2026 - Tháng cuối): Submit Paper, Xây dựng Ứng dụng Demo & Hoàn thiện Luận văn**
   - **Nội dung công việc chi tiết:**
     - **Viết và nộp (Submit) bài báo khoa học:** Soạn thảo bản thảo paper (LaTeX) theo định dạng hội nghị/tạp chí quốc tế uy tín, hoàn tất thủ tục nộp bài báo khoa học.
     - **Phát triển Ứng dụng Demo (Application Demo):** Xây dựng phần mềm ứng dụng web/GUI tương tác trực quan cho phép nạp dữ liệu Sinogram, chọn góc quét, hiển thị ảnh tái tạo thời gian thực và so sánh sai số.
     - **Hoàn thiện Báo cáo Khóa luận (Finalize KLTN Thesis):** Viết toàn văn cuốn báo cáo khóa luận tốt nghiệp theo đúng quy chuẩn định dạng của nhà trường, in ấn, chuẩn bị slide thuyết trình và bảo vệ trước Hội đồng chấm khóa luận tốt nghiệp.
   - **Sản phẩm / Kết quả đạt được:** Bài báo khoa học đã được Submit chính thức; Sản phẩm phần mềm ứng dụng Demo chạy ổn định; Toàn văn cuốn Khóa luận tốt nghiệp và slide báo cáo bảo vệ.
   - **Phân công thực hiện:** Phan Đình Minh (100%).

---

## 3. Ý kiến và Phê duyệt của Cán bộ Hướng dẫn
*(Cán bộ hướng dẫn ghi nhận xét và ký duyệt đề cương)*

- **Nhận xét của Cán bộ hướng dẫn:**  
  ............................................................................................................................................................  
  ............................................................................................................................................................  
  ............................................................................................................................................................  

- **Kết luận:**  
  $\square$ Đồng ý cho thực hiện theo đề cương  
  $\square$ Yêu cầu chỉnh sửa, bổ sung trước khi thực hiện  

*TP. Hồ Chí Minh, ngày ...... tháng ...... năm 2026*  
**CÁN BỘ HƯỚNG DẪN**  
*(Ký và ghi rõ họ tên)*  

<br><br><br>

**TS. Bùi Cao Doanh**
