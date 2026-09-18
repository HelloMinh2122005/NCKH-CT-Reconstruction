# 🌟 KIẾN TRÚC ĐỘT PHÁ MIỀN KÉP: SOLAR_DualMamba
**Tái Tạo Ảnh Cắt Lớp CT Góc Giới Hạn (Limited-Angle CT Reconstruction)**  
*Second-Order Dual-Domain Newton-CG Unrolling with Angular Sinogram Bi-Mamba and Image-Domain Swin-Window Regularizer*

---

## 1. Tổng Quan & Mục Tiêu Nghiên Cứu

Trong bài toán **Tái tạo ảnh CT góc giới hạn (Limited-Angle CT — LA-CT)**, cung quét bị thu hẹp (ví dụ chỉ quét $[-60^\circ, +60^\circ]$ thay vì đầy đủ $180^\circ + \text{fan}$) nhằm giảm liều bức xạ tia X cho bệnh nhân tuân theo nguyên tắc **ALARA** ($50\% - 75\%$ liều chiếu).

Tuy nhiên, việc thiếu hụt góc chiếu tạo ra **vùng khuyết hình nêm (Missing Wedge)** nghiêm trọng trong miền tần số Fourier:
1. Ma trận Hessian $A^T A$ bị suy biến dị hướng nặng nề.
2. Các vệt sọc nhiễu (**Streak Artifacts**) lan truyền dữ dội trong miền ảnh.
3. Nếu chỉ xử lý đơn miền (chỉ trong Image Domain), mô hình bị ép phải tự "bịa" ra thông tin từ con số không, dễ dẫn đến hiện tượng ảo giác (hallucination) hoặc suy giảm cấu trúc giải phẫu.

**SOLAR_DualMamba** là kiến trúc đột phá SOTA đầu tiên kết hợp hài hòa hai trường phái mạnh nhất hiện nay:
- **Miền Chiếu (Sinogram Domain):** Mô hình hóa chuỗi góc chiếu bằng **Angular Bi-Mamba** (Bidirectional State Space Model) để ngoại suy và vá lành dải khuyết hình sin dựa trên bản chất liên tục của phép biến đổi Radon.
- **Cầu Nối Vật Lý:** Toán tử chiếu vi phân Fan-beam chuẩn xác $A$ và $A^T$ (ASTRA CUDA).
- **Miền Ảnh (Image Domain):** Động cơ tối ưu hóa bậc 2 **Newton-CG (SafeCGSolver)** kết hợp khối điều hòa **Swin Window Attention (W-MSA 8x8)** và **Local Res-CNN 3x3** để giam hãm nhiễu vệt và khôi phục độ sắc nét viền mô xương.

---

## 2. Sơ Đồ Luồng Dữ Liệu Miền Kép (Dual-Domain Pipeline)

```
[Sinogram Thô y (B x 1 x 64 x 512)]
                │
                ▼
┌────────────────────────────────────────────────────────┐
│ 1. SINOGRAM DOMAIN: Angular Bi-Mamba Extrapolator      │
│ - Detector 1D Conv (tương quan cảm biến)               │
│ - Forward SSM (-60° -> +60°) & Backward SSM (+60° -> -60°)│
│ - Gated Silu Fusion: y_restored = y + delta_y          │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼  [Chiếu Vi Phân A^T(.)]
┌────────────────────────────────────────────────────────┐
│ 2. IMAGE DOMAIN: SOLAR Second-Order Newton-CG Engine   │
│                                                        │
│   For t = 0 ... 7 (8 Stages Unrolling):                │
│     ┌──────────────────────────────────────────────┐   │
│     │ Khối Điều Hòa Kép Swin-RegFormer:            │   │
│     │ - Local Res-CNN 3x3 (viền mô xương)          │   │
│     │ - Non-local Swin W-MSA 8x8 (giam hãm vệt)    │   │
│     │ -> Gradient Điều Hòa: grad_R                 │   │
│     └──────────────────────┬───────────────────────┘   │
│                            ▼                           │
│     b_t = λ_t * A^T(y_restored) + μ_t * x_t - grad_R   │
│                            │                           │
│                            ▼                           │
│     [SafeCGSolver: Matrix-Free Newton-CG (4 steps)]    │
│     (λ_t * A^T A + μ_t * I) x_{t+1} = b_t              │
│     (Strictly SPD: μ_t > 0, 100% không NaN)            │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
       [Ảnh Tái Tạo Cuối Cùng x_T (256 x 256)]
```

---

## 3. Chi Tiết Toán Học & Vật Lý Các Thành Phần

### 3.1. Miền Chiếu: Angular Bi-Mamba (Sinogram Inpainting)
Theo định lý biến đổi Radon, hàm suy giảm tia X $f(x, y)$ khi quay qua các góc chiếu $\theta$ sẽ tạo thành các quỹ đạo hình sin trên Sinogram:
$$p(\theta, s) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(x, y) \delta(x \cos\theta + y \sin\theta - s) dx dy$$
Khi góc chiếu $\theta$ biến thiên liên tục, từng detector $s$ ghi nhận một chuỗi tín hiệu mượt mà. 

Khối **Angular Bi-Mamba** coi trục góc $\theta$ ($V = 64$ views) là chiều thời gian của hệ phương trình trạng thái liên tục:
$$h'(t) = \mathbf{A} h(t) + \mathbf{B} x(t)$$
$$y(t) = \mathbf{C} h(t) + \mathbf{D} x(t)$$
- **Quét xuôi (Forward):** Nắm bắt xu hướng biến thiên từ góc bắt đầu $-60^\circ \to +60^\circ$.
- **Quét ngược (Backward):** Nắm bắt chiều ngược lại $+60^\circ \to -60^\circ$.
- Cơ chế chọn lọc (Selective Scan) giúp mạng tự động học cách bù đắp phần tín hiệu suy hao ở hai biên của cung quét mà không gây bùng nổ tính toán (độ phức tạp tuyến tính $\mathcal{O}(V)$).

### 3.2. Miền Ảnh: Động Cơ Newton-CG Bậc 2 (`SafeCGSolver`)
Thay vì bước giảm gradient bậc 1 truyền thống dễ bị rung lắc Zigzag trên địa hình Hessian suy biến, `SOLAR_DualMamba` giải phương trình đạo hàm bậc 2 tại mỗi stage:
$$(\lambda_t A^T A + \mu_t I) x_{t+1} = \lambda_t A^T(\hat{y}) + \mu_t x_t - \nabla \mathcal{R}(x_t)$$
- **Matrix-Free:** Không lưu trữ ma trận Hessian, tính $H(p) = \lambda_t A^T(A(p)) + \mu_t p$ trực tiếp qua 1 lần chiếu thuận $A$ và 1 lần chiếu ngược $A^T$.
- **Strictly SPD:** Tham số hóa $\mu_t = \text{Softplus}(\text{raw\_mu}_t) + 10^{-4} > 0$, đảm bảo nghiệm CG luôn duy nhất và hội tụ tuyệt đối không phát sinh NaN.

### 3.3. Khối Điều Hòa Kép Miền Ảnh (`RegFormerDualBranchRegularizer`)
- **Nhánh Cục Bộ:** 3 tầng Residual Conv $3 \times 3$ với LeakyReLU giữ trọn chi tiết tần số cao.
- **Nhánh Phi Cục Bộ:** Swin Window Attention chia ảnh thành $1.024$ cửa sổ $8 \times 8$, tính toán tương quan chú ý bên trong từng ô kết hợp ma trận vị trí tương đối 2D $B_{\text{rel}}$. Cơ chế này đóng vai trò "tường lửa" giam hãm nhiễu vệt nêm khuyết, không cho rò rỉ ra toàn ảnh.
- **Chia Sẻ Trọng Số Tuần Hoàn (Recurrent Weight Sharing):** Cả 8 stages dùng chung 1 bộ điều hòa và 1 bộ Sinogram Bi-Mamba, đưa tổng số tham số mô hình về mức siêu nhẹ **$\sim 150\text{ K}$ tham số** (nhỏ hơn 8 lần so với RegFormer 1.23M tham số).

---

## 4. Bảng Tham Số Kỹ Thuật

| Tham số | Giá trị | Ý nghĩa |
| :--- | :--- | :--- |
| **Hình học chiếu** | Fan-beam geometry | Chuẩn máy CT y tế Siemens Somatom (`src_radius=600`, `det_radius=290`) |
| **Dải góc quét** | $[-60^\circ, +60^\circ]$ (120° span) | Cung quét chuẩn Benchmark quốc tế; giảm $66.7\%$ liều tia X |
| **Số góc chiếu (Views)** | `64` views | Bước góc $\Delta \theta \approx 1.875^\circ$ |
| **Số phần tử cảm biến** | `512` detectors | Độ phân giải chiếu detector |
| **Độ phân giải ảnh** | $256 \times 256$ | Kích thước lưới tái tạo |
| **Số stages unrolling** | `8` stages | Số vòng tối ưu hóa bậc 2 Newton-CG |
| **Số bước lặp CG** | `4` steps / stage | Số vòng lặp Conjugate Gradient Matrix-Free |
| **Kích thước cửa sổ Swin**| $8 \times 8$ | Phạm vi giam hãm nhiễu vệt (64 tokens/cửa sổ) |
| **Tổng số tham số** | $\sim 150\text{ K}$ tham số | $\sim 0.60\text{ MB}$, siêu nhẹ, chống Overfitting tối đa |

---

## 5. Hướng Dẫn Thực Thi

### Huấn luyện (50 Epochs):
```bash
sbatch scripts/train_solar_dualmamba_la.sh
```

### Kiểm thử Benchmark độc lập trên Patient L310 (214 lát cắt):
```bash
sbatch scripts/test_solar_dualmamba_la.sh
```
