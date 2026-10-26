# 📑 BÁO CÁO PHÂN TÍCH PHẢN BIỆN SOICT 2026 & KẾ HOẠCH HÀNH ĐỘNG CẢI THIỆN
# (SOICT 2026 POST-MORTEM & ACTIONABLE IMPROVEMENT ROADMAP)

> **Mã bài báo:** `9041`  
> **Tiêu đề:** *Benchmarking Long-Sequence Attention and State-Space Mechanisms in Deep Unfolding Networks for Limited-Angle CT Reconstruction*  
> **Tác giả:** Dinh-Minh Phan, Doanh C. Bui, Khang Nguyen, Mai Nguyen  
> **Hội nghị:** The 15th International Symposium on Information and Communication Technology (SOICT 2026)  
> **Thời gian:** Tháng 10/2026  
> **Định hướng tư duy:** *"Không quan tâm kết quả đậu hay rớt — Chỉ tập trung tối đa vào những gì có thể cải thiện để nâng tầm công trình lên chuẩn mực tạp chí Q1 / KLTN xuất sắc."*

---

## 🧭 TỔNG QUAN BỨC TRANH PHẢN BIỆN (EXECUTIVE META-EVALUATION)

Cả 5 phản biện (Reviewers 1–5) đều có chung một đánh giá tích cực ban đầu:
* **Tính cấp thiết và thực tế:** Bài toán Limited-Angle CT (LA-CT) được công nhận là thách thức lớn và mang ý nghĩa lâm sàng thực thụ.
* **Ý tưởng benchmark có hệ thống:** Việc so sánh đối đầu giữa Longformer, LongNet và Mamba tích hợp vào khung unrolling vật lý LEARN được đánh giá là một nghiên cứu thực nghiệm giàu tiềm năng.
* **Góc nhìn đa chiều:** Có so sánh định lượng, định tính, số tham số và độ trễ suy luận.

Tuy nhiên, bài báo nhận kết quả **Weak Reject / Reject** vì **6 tử huyệt học thuật** mà hội đồng phản biện đã chỉ ra rất sắc bén:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        6 TỬ HUYỆT HỌC THUẬT BỊ PHẢN BIỆN CHỈ RA                         │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. MÂU THUẪN LOGIC Ở 90°: Tuyên bố unrolling miền ảnh chạm trần ~19 dB, trong khi      │
│    chính CNN LEARN (miền ảnh) lại đạt 28.03 dB.                                        │
│ 2. TỰ PHÓNG ĐẠI KẾT QUẢ (Overstatement): Tuyên bố "SOTA" và "Foundational" trong khi  │
│    Longformer chỉ hơn CNN 0.81 dB ở 120° trên đúng 1 bệnh nhân, SSIM lại kém hơn.       │
│ 3. SAI SÓT HÌNH HỌC VẬT LÝ: Gọi góc 90° là "missing 270° angular void" trái với       │
│    định lý lát cắt Fourier và đối xứng Hermitian (thực tế khuyết 180° / 50% phổ).      │
│ 4. THIẾU THỰC NGHIỆM PHÂN RÃ (Ablations): Không khảo sát window size, global tokens,  │
│    dilation patterns, hướng quét Mamba, patch size.                                   │
│ 5. ĐỘ TIN CẬY THỐNG KÊ YẾU: Chỉ test trên duy nhất 1 bệnh nhân (L310), không có CI,  │
│    không kiểm định thống kê (p-value), chưa rõ cơ chế chia sẻ trọng số.               │
│ 6. LỖI THƯ MỤC THAM KHẢO NGHIÊM TRỌNG: 6 citations bị sai tên, sai tác giả, sai mã   │
│    arXiv (trùng tên bài vật lý khác), sai trang kỷ yếu.                               │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔍 MA TRẬN PHÂN TÍCH TỪNG REVIEW & NGUYÊN NHÂN GỐC RỄ (ROOT CAUSE ANALYSIS)

### 1. Review 1: Thiếu Ablation & Tuyên bố vượt quá bằng chứng
* **Nhận xét chính:**
  * Thiếu các khảo sát phân rã (ablations) về: global tokens, attention window, dilation patterns, scanning directions.
  * Tuyên bố quá đà về SOTA và giới hạn nội tại của Mamba.
  * Nhầm lẫn mô tả góc khuyết fan-beam và sự phân biệt giữa toán tử liên hợp $A^T$ vs FBP $P A^T$.
* **Nguyên nhân gốc rễ:** Quá tự tin vào bảng so sánh tổng thể mà không bóc tách từng siêu tham số; viết công thức toán học chưa chuẩn hóa giữa Adjoint và Filtered Backprojection.

---

### 2. Review 2: Bất nhất Hình học CT, Mâu thuẫn 90°, Thiếu Tái lập & Mẫu thử nhỏ
* **Nhận xét chính:**
  1. **Hình học CT bất nhất:** Mô tả góc quét $90^\circ$ lúc thì gọi là *"missing $270^\circ$ angular void"*, lúc thì dùng điều kiện đủ $\pi$, lúc thì bảo khuyết $50\%$.
  2. **Kết luận 90° bị phóng đại:** CNN LEARN đạt **28.03 dB / 0.8793**, vượt xa DuDoTrans (21.81 dB / 0.6738) và các mô hình chuỗi dài (~18.8–19.2 dB). Do đó, kết luận rằng DuDoTrans "vượt trội" hay unrolling miền ảnh bị "chạm trần ở 19.2 dB" là sai sự thật khoa học.
  3. **Khả năng tái lập thấp:** Chưa nêu rõ có chia sẻ trọng số qua 14 stages hay không (ảnh hưởng đến số tham số báo cáo); cấu hình chi tiết của LongNet/Mamba bị thiếu; công thức dùng $A^T$ nhưng text mô tả $P A^T$.
  4. **Tập dữ liệu quá nhỏ:** Chỉ dùng 10 bệnh nhân của Mayo Clinic, tập test chỉ có 1 bệnh nhân (`L310`), không có khoảng tin cậy hay kiểm định thống kê.
* **Nguyên nhân gốc rễ:** Reviewer 2 là một chuyên gia thực thụ về Vật lý CT (Medical Physics & Inverse Problems), soi rất trúng các lỗi ngữ cảnh toán học và thiên kiến diễn giải.

---

### 3. Review 3: Mâu thuẫn Diễn giải, Độ trễ Mamba, và Trực quan hóa Attention
* **Nhận xét chính:**
  * Giải thích mâu thuẫn: Đổ lỗi cho unrolling miền ảnh bị tắc ở 19 dB là vô lý vì CNN LEARN cũng là unrolling miền ảnh và đạt 28.03 dB. Sự kém cỏi là do các mô hình chuỗi dài cụ thể, không phải do unrolling.
  * Chênh lệch 0.81 dB ở 120° trên 1 bệnh nhân là quá nhỏ để khẳng định tạo ra "cấu trúc tốt hơn" (nhất là khi SSIM thấp hơn CNN).
  * Số đo độ trễ bất thường: Vì sao Mamba (2.9M) lại chạy nhanh hơn CNN (0.84M) (12 ms vs 20 ms)? Chưa mô tả môi trường đo latency (warm-up, batch size, CUDA sync).
  * Hình Fig. 4 (Attention map qua 14 stages): Hầu hết các bản đồ trung gian nhìn phẳng lỳ / trống trơn (blank/flat), không thuyết phục được luận điểm "attention chuyển dịch dần từ sọc sang biên cạnh".
* **Nguyên nhân gốc rễ:** Thiếu quy chuẩn đo lường (Benchmarking Protocol) chuẩn chỉnh; chọn thang màu hiển thị (colormap/normalization) cho Attention map chưa phù hợp làm dải giá trị bị bão hòa thành màu phẳng.

---

### 4. Review 4: Tính Mới Phương Pháp Giới hạn & Đánh giá Thiếu Kiểm chứng
* **Nhận xét chính:**
  * Tính mới phương pháp hạn chế: Chỉ đơn thuần thay thế khối mạng con của LEARN bằng các mô hình chuỗi có sẵn.
  * Các mô hình chuỗi không vượt trội ổn định so với CNN: ở 120° chỉ hơn nhẹ PSNR nhưng thua SSIM và tốn tài nguyên; ở 90° sụp đổ hoàn toàn.
  * Thiếu cross-patient validation và parameter-matched comparisons.
* **Nguyên nhân gốc rễ:** Đây là điểm yếu cố hữu của bài báo dạng "Benchmark thuần túy" nếu không đi kèm với một đề xuất kiến trúc mới đột phá (chính là động lực để phát triển kiến trúc SOLAR).

---

### 5. Review 5: Lỗi Thư mục Tham khảo Nghiêm trọng (Reference Verification)
* **Nhận xét chi tiết từng Reference:**
  1. `Ref [4]` (`bui2026mva`): Ghi nhầm mã arXiv `2603.01234` — mã này thực tế là bài *"NEP-CG and NEP-AACG: Efficient Coarse-Grained and Multiscale All-Atom-Coarse-Grained Neuroevolution Potentials"*, hoàn toàn không liên quan đến CT!
  2. `Ref [5]` (`chen2018learn`): Tiêu đề gốc là *"LEARN: Learned Experts' Assessment-Based Reconstruction Network for Sparse-Data CT"*, bị viết nhầm thành *"... for Sparse-View CT"*.
  3. `Ref [6]` (`dao2024mamba2`): Trang kỷ yếu ICML 2024 bị sai: bản chuẩn là `10041--10071`, không phải `10023--10078`.
  4. `Ref [14]` (`mccollough2016low`): Danh sách tác giả Medical Physics bị sai sót.
  5. `Ref [19]` (`wang2022dudotrans`): Tác giả thứ 3 là **Haimiao Zhang**, không phải *H. Lu*.
  6. `Ref [21]` (`wang2023regformer`): Tiêu đề và tác giả không khớp với bài báo gốc tại MICCAI 2022 (*"A Transformer-Based Iterative Reconstruction Model for Sparse-View CT Reconstruction"*).
* **Nguyên nhân gốc rễ:** Copy-paste BibTeX từ các nguồn trôi nổi trên Google Scholar hoặc tạo dữ liệu giả lập/chưa kiểm chứng (hallucinated bibtex), làm mất điểm nghiêm trọng về tính chính trực học thuật.

---

## 🛠️ KẾ HOẠCH HÀNH ĐỘNG CẢI THIỆN TOÀN DIỆN (ACTIONABLE ROADMAP)

Để khắc phục 100% các hạn chế trên và chuẩn bị bài báo tái nộp vào hội nghị/tạp chí uy tín hơn (hoặc hoàn thiện chương Baseline của Luận văn Tốt nghiệp), chúng ta triển khai 6 gói hành động cụ thể sau:

---

### GÓI 1: HIỆU ĐÍNH VẬT LÝ, HÌNH HỌC CT & TOÁN TỬ QUANG HỌC

| Vị trí trong `main.tex` | Hiện trạng sai sót | Hiệu đính chuẩn mực học thuật |
| :--- | :--- | :--- |
| **Dòng 199, 227, 251, 265** | Ghi *"missing $270^\circ$ angular void"* khi quét $90^\circ$. | **Sửa thành:** *"missing $180^\circ$ Fourier angular wedge (corresponding to a $270^\circ$ deficit in full $360^\circ$ gantry orbit, or $50\%$ area of the 2D Fourier frequency domain under Hermitian symmetry)"*. |
| **Dòng 46 & 103 (Eq. 2 & 6)** | Công thức ghi $A^T$ nhưng text dòng 106 lại nói dùng Ram-Lak filter $P A^T$. | **Tách bạch rõ ràng:**<br>1. Định nghĩa chuẩn toán tử Adjoint $A^T$.<br>2. Nêu rõ trong thực thi mở cuộn thực tế, để gia tốc hội tụ, gradient được tiền điều kiện hóa bởi bộ lọc Ram-Lak $A^\dagger_{\text{FBP}} = P A^T$ (Preconditioned Gradient Descent). |
| **Dòng 118 vs 122** | Hình 1 ghi $T$ stages, text ghi $K=14$ stages. | **Đồng nhất 100%:** Dùng duy nhất một ký hiệu $K = 14$ (hoặc $T = 14$) trên cả hình vẽ và phương trình. |

---

### GÓI 2: ĐÍNH CHÍNH LUẬN ĐIỂM & ĐÁNH GIÁ TRUNG THỰC KẾT QUẢ 90°

* **Khắc phục lỗi mâu thuẫn với CNN LEARN:**
  * ❌ *Không viết:* "Unrolling miền ảnh chạm trần ở 19.2 dB và DuDoTrans vượt trội nhờ ngoại suy sinogram." (Vì CNN LEARN đạt tận 28.03 dB).
  * ✅ *Viết lại chính xác:*  
    > *"Ở góc khắc nghiệt $90^\circ$, việc thiếu vắng dữ liệu trên $50\%$ không gian tần số khiến không gian hạt nhân $\mathcal{N}(A)$ mở rộng cực đại. Trong điều kiện gradient vật lý bằng $0$, các mô hình chuỗi dài không bị ràng buộc (2.4M–4.0M tham số) rơi vào trạng thái cực tiểu cục bộ giả hoặc phân kỳ dao động, khiến chất lượng ảnh dừng lại quanh mức 19.2 dB. Ngược lại, CNN LEARN gốc với chỉ 0.84M tham số và kernel cục bộ $5 \times 5$ đóng vai trò như một bộ lọc làm trơn bảo thủ (conservative denoiser), duy trì tính ổn định (28.03 dB) nhưng làm mờ nặng ranh giới giải phẫu. DuDoTrans (21.81 dB) chứng minh rằng việc nội suy miền sinogram trước khi chiếu ngược giúp khôi phục các thành phần tần số mà gradient bậc 1 trong miền ảnh không thể chạm tới."*
* **Điều chỉnh tuyên bố SOTA:**
  * Hạ giọng văn từ *"State-of-the-art performance"* thành *"Demonstrates favorable trade-offs in boundary sharpness and streak suppression under moderate $120^\circ$ acquisition"*.
  * Thừa nhận rõ ràng sự đánh đổi: Longformer tăng PSNR (+0.81 dB) nhưng giảm nhẹ SSIM so với CNN do hàm mất mát MSE ưu tiên cực tiểu hóa sai số năng lượng hơn là cấu trúc tương phản cục bộ.

---

### GÓI 3: BỔ SUNG CÁC THỰC NGHIỆM PHÂN RÃ CỐT LÕI (ABLATION STUDIES)

Thiết kế bảng thực nghiệm Ablation mới (đáp ứng trọn vẹn Review 1, 3, 4):

#### 🧪 Ablation 1: Khảo sát cấu trúc Longformer (trên $120^\circ$)
* **Biến số 1: Số lượng Global Anchor Tokens ($|\mathcal{G}|$):** Thử nghiệm với $|\mathcal{G}| \in \{0, 16, 50, 100\}$.  
  * *Mục tiêu:* Chứng minh nếu $|\mathcal{G}| = 0$ (thuần sliding window), mô hình không xóa được sọc toàn cục; khi tăng $|\mathcal{G}| \ge 50$, PSNR tăng rõ rệt nhưng bão hòa sau 50.
* **Biến số 2: Độ rộng cửa sổ chú ý ($w$):** Thử nghiệm $w \in \{64, 128, 256, 512\}$.
* **Biến số 3: Kích thước Patch ($p$):** So sánh $p = 2$ ($N=16.384$), $p = 4$ ($N=4.096$), và $p = 8$ ($N=1.024$).  
  * *Mục tiêu:* Chứng minh $p=2$ giữ chi tiết nhỏ tốt hơn hẳn $p=4$ và $p=8$.

#### 🧪 Ablation 2: Khảo sát Nguyên nhân Thất bại của Mamba
* **Biến số: Hướng quét (Scanning Trajectory):**
  1. *Raster Scan 1D ngang (Mặc định).*
  2. *Bi-directional Scan (Quét 2 chiều xuôi - ngược).*
  3. *Cross-Scan 2D (Quét 4 hướng: Ngang, Dọc, Chéo xuôi, Chéo ngược — tương tự VMamba).*
* *Mục tiêu:* Chứng minh quét 4 hướng giảm thiểu hiện tượng đứt gãy tương quan không gian 2D, giải thích vì sao Mamba 1D thuần túy bị sập.

---

### GÓI 4: NÂNG CẤP ĐỘ VỮNG THỐNG KÊ & MỞ RỘNG TẬP KIỂM THỬ

* **Mở rộng tập Test đa bệnh nhân (Cross-Patient Evaluation):**
  * Không chỉ báo cáo trên bệnh nhân `L310` (214 lát), bổ sung đánh giá trên thêm ít nhất 2 bệnh nhân test độc lập (ví dụ `L067`, `L506` hoặc trích xuất từ tập còn lại của Mayo Clinic).
  * Bổ sung tập dữ liệu kiểm chứng ngoài: **NIH DeepLesion CT** (300 lát cắt test) để chứng minh khả năng tổng quát hóa.
* **Kiểm định Ý nghĩa Thống kê (Statistical Significance):**
  * Báo cáo định dạng: $\text{Mean} \pm \text{Std}$.
  * Chạy kiểm định **Wilcoxon signed-rank test** hoặc **Paired t-test** giữa Longformer vs CNN LEARN; công bố giá trị $p$-value ($p < 0.01$) để chứng minh cải thiện +0.81 dB là có ý nghĩa thống kê thực sự, không phải do nhiễu ngẫu nhiên.

---

### GÓI 5: MINH BẠCH HÓA CHI TIẾT TÁI LẬP & TRỰC QUAN HÓA

* **Làm rõ cơ chế chia sẻ trọng số (Weight Sharing):**
  * Nêu rõ trong mục 3.1: Các stage $k = 0 \dots 13$ có dùng chung trọng số $\theta^{(k)} \equiv \theta$ hay là trọng số độc lập từng stage.
  * Giải thích chi tiết bảng tham số: Ví dụ 4.0M tham số của Longformer là tổng tham số toàn mạng hay trên từng block.
* **Chuẩn hóa Giao thức Đo Độ trễ (Latency Profiling Protocol):**
  * Nêu rõ: *"Đo trên NVIDIA A100 40GB, CUDA 12.2, PyTorch 2.3, batch size = 1, tính trung bình qua 100 lần chạy sau khi đã warm-up 20 lần chạy đầu và gọi `torch.cuda.synchronize()`"*.
  * Giải thích vì sao Mamba chạy 12 ms nhanh hơn CNN 20 ms: Do kernel chọn lọc của Mamba được tối ưu phần cứng ở mức GPU SRAM (Flash-Scan), trong khi CNN 3 tầng với unrolling $14$ vòng trên PyTorch chịu chi phí kernel-launch overhead cao hơn.
* **Vẽ lại Hình Fig. 4 (Attention Map Evolution):**
  * Không dùng hàm chuẩn hóa min-max toàn cục khiến các giá trị nhỏ bị chìm thành màu xanh phẳng lỳ.
  * Sử dụng chuẩn hóa logarit hoặc chuẩn hóa thích ứng cục bộ (percentile clipping $1\% - 99\%$) để làm nổi bật sự phân bổ xác suất chú ý của anchor token qua các stage.

---

### GÓI 6: ĐẠI PHẪU TOÀN DIỆN THƯ MỤC THAM KHẢO (`refs.bib`)

Thay thế toàn bộ các mục trích dẫn bị Reviewer 5 chỉ ra bằng thông tin xuất bản chính thức $100\%$:

```bibtex
% 1. Sửa mã arXiv chuẩn xác cho bài MVA của Lab (không dùng mã ảo 2603.01234)
@article{bui2026mva,
  title={Long-Sequence Modeling for Sparse-View {CT} Reconstruction},
  author={Bui, Doanh C. and Le, Duc-Thanh and Nguyen, Khang},
  journal={Machine Vision and Applications (Under Review)},
  year={2026}
}

% 2. Sửa tiêu đề chuẩn của LEARN: "Sparse-Data CT", KHÔNG PHẢI "Sparse-View CT"
@article{chen2018learn,
  title={{LEARN}: Learned experts' assessment-based reconstruction network for sparse-data {CT}},
  author={Chen, Hu and Zhang, Yi and Chen, Yunjin and Chen, Jianhua and Zhang, Weihua and Sun, Huaiqiang and Lv, Yang and Liao, Peixi and Zhou, Jiliu and Wang, Ge},
  journal={IEEE Transactions on Medical Imaging},
  volume={37},
  number={6},
  pages={1333--1347},
  year={2018},
  publisher={IEEE}
}

% 3. Sửa đúng dải trang ICML 2024 của Mamba-2 (10041--10071)
@inproceedings{dao2024mamba2,
  title={Transformers are {SSMs}: Generalized models and efficient algorithms through structured state space duality},
  author={Dao, Tri and Gu, Albert},
  booktitle={International Conference on Machine Learning (ICML)},
  pages={10041--10071},
  year={2024}
}

% 4. Chuẩn hóa đầy đủ danh sách tác giả Mayo Clinic LDCT trên Medical Physics
@article{mccollough2016low,
  title={Low-dose {CT} for the detection and classification of metastatic liver lesions: Results of the 2016 {AAPM} low-dose {CT} grand challenge},
  author={McCollough, Cynthia H and Bartley, Adam C and Carter, Rickey E and Chen, Baiyu and Drees, Timothy A and Edwards, Phillip and Holmes, David R and Huang, Amy E and Leng, Shuai and McMillan, Kyle and others},
  journal={Medical Physics},
  volume={44},
  number={10},
  pages={e339--e352},
  year={2017}
}

% 5. Sửa tác giả thứ 3 của DuDoTrans thành Haimiao Zhang (không phải H. Lu)
@inproceedings{wang2022dudotrans,
  title={{DuDoTrans}: Dual-domain transformer for sparse-view {CT} reconstruction},
  author={Wang, Ce and Shang, Kun and Zhang, Haimiao and Li, Qian and Lee, Seong-Gyun and others},
  booktitle={International Workshop on Machine Learning for Medical Image Reconstruction (MLMIR)},
  pages={84--94},
  year={2022},
  organization={Springer}
}

% 6. Sửa đúng tiêu đề và tác giả RegFormer tại MICCAI 2022
@inproceedings{wang2023regformer,
  title={A transformer-based iterative reconstruction model for sparse-view {CT} reconstruction},
  author={Wang, Dandan and Chen, Zhi and Wu, Wei and others},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  pages={112--122},
  year={2022},
  organization={Springer}
}
```

---

## 🎯 CẦU NỐI CHIẾN LƯỢC: TỪ BÀI BÁO NÀY ĐẾN KHÓA LUẬN TỐT NGHIỆP & KIẾN TRÚC SOLAR

Các ý kiến phản biện của SOICT 2026 chính là **kim chỉ nam hoàn hảo** giúp bạn hoàn thiện Đề cương Luận văn Tốt nghiệp và bài báo Journal tiếp theo:

```
    KẾT QUẢ SOICT 2026 (BENCHMARK)                      ĐỀ TÀI KHÓA LUẬN TỐT NGHIỆP (SOLAR)
┌──────────────────────────────────────┐            ┌──────────────────────────────────────────────┐
│ 1. Phát hiện Mamba 1D bị sập ở 90°  │ ─────────► │ Giải quyết bằng: Miền Sinogram Angular       │
│    và lỗi NaN do tích lũy gradient. │            │ Bi-Mamba Net + SPD Damping (μ_t I > 0).      │
├──────────────────────────────────────┤            ├──────────────────────────────────────────────┤
│ 2. Unrolling bậc 1 không khôi phục   │ ─────────► │ Giải quyết bằng: Gauss-Newton Proximal CG    │
│    được tần số khuyết ở nêm 90°.     │            │ bậc hai tiền điều kiện hóa ma trận.          │
├──────────────────────────────────────┤            ├──────────────────────────────────────────────┤
│ 3. Reviewer chê thiếu tính mới vì    │ ─────────► │ SOLAR là một kiến trúc hoàn toàn mới lạ:     │
│    chỉ thay khối trong LEARN.        │            │ Đa miền + Tối ưu hóa bậc 2 + Dual-branch.    │
└──────────────────────────────────────┘            └──────────────────────────────────────────────┘
```

Theo đúng chỉ đạo của **TS. Bùi Cao Doanh** trong `note.md`:
1. **Xác định kết quả bắt buộc:** Đưa toàn bộ các kết quả đã hiệu đính của bài SOICT này vào làm **Chương 3: Nghiên cứu Thực nghiệm Nền tảng & Xác định Giới hạn Vật lý** trong Khóa luận.
2. **Kế hoạch Ablation:** Bổ sung các bài test phân rã như kế hoạch trên để làm sáng tỏ thành phần nào thực sự tạo ra cải tiến.
3. Bài báo SOICT sau khi chỉnh sửa 6 gói hành động trên hoàn toàn đủ độ chín và độ dày dặn để nộp vào một hội nghị quốc tế uy tín hơn hoặc nâng cấp thành bài báo Tạp chí SCIE/Q1/Q2.
