# Quy trình Chuẩn bị Dữ liệu Limited-Angle CT (LA-CT)

Tài liệu này hướng dẫn chi tiết cách tạo và tiền xử lý dữ liệu cho bài toán **Limited-Angle CT Reconstruction** từ bộ dữ liệu CT gốc (AAPM Mayo Clinic LDCT).

---

## 1. Động Lực Nghiên Cứu & Bài Toán Limited-Angle CT

### Động lực chính: Giảm thiểu liều bức xạ tia X chiếu vào người bệnh nhân
Trong chẩn đoán hình ảnh y khoa, việc tiếp xúc nhiều với bức xạ tia X mang lại nguy cơ tích tụ phóng xạ và tiềm ẩn các rủi ro sức khỏe nghiêm trọng (như ung thư). Tuân theo nguyên tắc y tế **ALARA (As Low As Reasonably Achievable)**, bài toán **Limited-Angle CT** ra đời nhằm giải quyết mục tiêu cốt lõi:
1. **Cắt giảm trực tiếp liều bức xạ (Radiation Dose Reduction):** Thay vì để nguồn phát tia X quay trọn vòng $360^\circ$ quanh cơ thể, ta chỉ cho phát tia trong một cung góc hẹp (ví dụ: $90^\circ, 120^\circ$ hoặc $150^\circ$), giúp giảm ngay từ **50% đến 75% tổng lượng tia X** chiếu vào bệnh nhân.
2. **Bảo vệ các cơ quan nhạy cảm với phóng xạ (Organ Shielding):** Giới hạn góc quét cho phép bác sĩ điều chỉnh hướng chiếu tia chỉ đi qua vùng tổn thương, tránh chiếu trực diện vào các cơ quan dễ tổn thương do tia X (như tuyến giáp, mắt, tuyến vú, cơ quan sinh sản).
3. **Rút ngắn thời gian chụp:** Quét trong dải góc hẹp giúp giảm đáng kể thời gian quét, hạn chế nhiễu do cử động của bệnh nhân (motion artifacts) và rất hữu ích trong phẫu thuật can thiệp khẩn cấp (C-arm CT).

### Thách thức toán học (The Missing Wedge Problem):
Khi dải góc quét bị giới hạn $\Delta \theta < 180^\circ$:
- Theo **Định lý Lát cắt Fourier (Fourier Slice Theorem)**, miền tần số $k$-space bị khuyết một vùng hình nêm lớn (**Missing Wedge**).
- Phương pháp giải tích cổ điển (Filtered Backprojection - FBP) sẽ cho ra ảnh bị suy giảm chất lượng trầm trọng: xuất hiện vệt sọc dài (heavy streak artifacts), mất biên cạnh sắc nét và méo dạng cấu trúc giải phẫu dọc theo phương thiếu dữ liệu chiếu.
- Vì vậy, việc ứng dụng các mô hình học sâu (Deep Learning) nhằm tái tạo và phục hồi thông tin vùng bị khuyết là giải pháp đột phá để thu được ảnh CT chất lượng chẩn đoán cao với liều tia cực thấp.

---

## 2. Thông số Hình học & Vật lý (Fan-Beam Geometry)
Mô phỏng phép chiếu được thực hiện qua thư viện **ODL (Operator Discretization Library)** và **ASTRA Toolbox** (`astra_cuda`):

| Tham số | Giá trị mặc định | Giải thích |
| :--- | :--- | :--- |
| **Kích thước miền vật lý** | $200 \times 200\text{ mm}$ | Không gian tái tạo ảnh thực tế |
| **Độ phân giải ảnh gốc** | $512 \times 512$ | Kích thước lát cắt chuẩn DICOM |
| **Độ phân giải đầu ra** | $256 \times 256$ | Kích thước ảnh sau resize đưa vào Deep Learning |
| **Bán kính nguồn (src_radius)** | $600\text{ mm}$ | Khoảng cách từ nguồn phát đến tâm quay (Isocenter) |
| **Bán kính detector (det_radius)** | $290\text{ mm}$ | Khoảng cách từ tâm quay đến dải cảm biến (Detector) |
| **Số lượng Detector** | $512$ | Kích thước partition detector $[-480, 480]\text{ mm}$ |
| **Bộ lọc FBP** | Ram-Lak (`frequency_scaling=0.9`) | Bộ lọc cắt cao (High-pass ramp filter) chuẩn FBP |

---

## 3. Lý Do Lựa Chọn & Ý Nghĩa Chi Tiết Của Từng Tham Số

### 1. Dải góc quét: `start_ang = -np.pi / 3` ($-60^\circ$) và `end_ang = np.pi / 3` ($+60^\circ$)
- **Tại sao đối xứng qua $0^\circ$ ($[-60^\circ, +60^\circ]$)?**
  Cơ thể người có tính đối xứng giải phẫu hai bên (trục đối xứng dọc cơ thể). Việc đặt dải góc quét đối xứng qua hướng $0^\circ$ (hướng trước-sau AP hoặc sau-trước PA) giúp chùm tia X đi đều qua cả 2 bên cơ thể, tạo sự đồng nhất cho việc quan sát và bảo vệ các cơ quan nhạy cảm đối xứng (như 2 lá phổi, thận, tuyến vú).
- **Tại sao là $120^\circ$?**
  - Giảm chính xác **$66.7\%$ liều phóng xạ** so với quay đủ $360^\circ$.
  - Đây là **chuẩn Benchmark quốc tế** được dùng rộng rãi nhất trong các công trình NCKH SOTA (như *FBPConvNet, LEARN, DuDoNet, RegFormer*), giúp kết quả nghiên cứu của bạn có thể so sánh trực tiếp, công bằng với các công bố quốc tế.
  - Vùng thiếu góc ($240^\circ$) đủ lớn để chứng minh tính ưu việt của AI so với giải thuật truyền thống.

### 2. Số lượng góc chiếu: `num_view = 64`
- **Độ phân giải góc lý tưởng:** Với dải $120^\circ$, bước nhảy giữa các góc chiếu là $\Delta \theta = \frac{120^\circ}{64} \approx 1.875^\circ$/view. Mật độ này đủ dày để bài toán là **Góc giới hạn thuần túy (Pure Limited-Angle CT)**. Mô hình sẽ tập trung toàn bộ năng lực vào khôi phục **Missing Wedge**, tránh bị nhiễu bởi hiện tượng lấy mẫu quá thưa (Sparse-view under-sampling).
- **Thuận lợi cho kiến trúc Deep Learning ($64 = 2^6$):** 
  Các mô hình hiện đại (U-Net, Vision Transformer, LongNet, Mamba) thực hiện downsampling/upsampling qua nhiều tầng ($64 \rightarrow 32 \rightarrow 16 \rightarrow 8 \rightarrow 4$). Số 64 là lũy thừa của 2, tránh hoàn toàn lỗi kích thước lẻ hoặc padding sai lệch trên ma trận Sinogram `(Batch, 1, 64, 256)`.

### 3. Kích thước Detector: `num_detectors = 512` & dải $[-480, 480]\text{ mm}$
- Số lượng 512 detectors khớp hoàn toàn 1:1 với ma trận ảnh gốc $512 \times 512$.
- Dải độ rộng detector $960\text{ mm}$ đảm bảo toàn bộ vùng vật thể (FOV $200 \times 200\text{ mm}$) nằm trọn trong chùm tia quét Fan-beam, loại bỏ triệt để hiện tượng cắt cụt dữ liệu biên (**Truncation Artifacts**).

### 4. Bán kính hình học: `src_radius = 600 mm` & `det_radius = 290 mm`
- Cấu hình này mô phỏng chính xác hình học chùm tia rẽ quạt (**Fan-Beam CT**) của các dòng máy CT y khoa thương mại (như Siemens SOMATOM Definition AS+ trong bộ dataset Mayo Clinic).
- Tổng khoảng cách từ nguồn tới đầu thu ($SDD = 600 + 290 = 890\text{ mm}$) đảm bảo hệ số phóng đại hình học (Geometric Magnification $M = \frac{SDD}{SOD} \approx 1.48$) đạt chuẩn y tế.

### 5. Mô phỏng nhiễu: `poisson_level = 1e6` & `gaussian_level = 0.05`
- `poisson_level = 1e6` ($I_0 = 10^6$ photons): Mô phỏng hiện tượng thống kê hạt photon tới detector (Quantum Noise) khi giảm liều tia.
- `gaussian_level = 0.05`: Mô phỏng nhiễu nhiệt và nhiễu đọc điện tử (Electronic Readout Noise) của bảng mạch cảm biến.

### 6. Độ phân giải đầu ra: `input_size = 256`
- Resize từ $512 \times 512$ về $256 \times 256$ giúp giảm $4$ lần số lượng điểm ảnh, cho phép các mô hình nặng (như Transformer/Mamba có độ phức tạp $O(N)$ hoặc $O(N^2)$) huấn luyện nhanh, batch size lớn hơn trên GPU A100 mà vẫn bảo toàn đầy đủ các chi tiết giải phẫu học quan trọng.

---

## 4. Các cấu hình góc quét Limited-Angle phổ biến

| Tên cấu hình | Dải góc (Độ) | Dải góc (Radian) | Số góc chiếu (Views) | Giảm liều tia X ước tính |
| :--- | :--- | :--- | :--- | :--- |
| **LA-120° (Chuẩn Benchmark)** | $[-60^\circ, +60^\circ]$ | $[-\pi/3, +\pi/3]$ | $64$ hoặc $96$ views | $\approx 66.7\%$ liều quét $360^\circ$ |
| **LA-150° (Dễ hơn)** | $[-75^\circ, +75^\circ]$ | $[-5\pi/12, +5\pi/12]$ | $64$ hoặc $96$ views | $\approx 58.3\%$ liều quét $360^\circ$ |
| **LA-90° (Cực hạn / Thách thức)** | $[-45^\circ, +45^\circ]$ | $[-\pi/4, +\pi/4]$ | $48$ hoặc $64$ views | $\approx 75.0\%$ liều quét $360^\circ$ |

---

## 5. Hướng dẫn chạy tiền xử lý (Generate Dataset)

### Chạy trực tiếp qua Python:
```bash
# 1. Tạo tập dữ liệu dải góc 120 độ, 64 views, không nhiễu
python prepare_data_sinogram_LA.py \
    --data_dir /datastore/uittogether3/LuuTru/Thanhld/CT-Reconstruction/split/ \
    --output_dir /datastore/uittogether3/LuuTru/MinhPD/dataset/limited_angle/ \
    --angle_range_deg 120.0 \
    --num_view 64 \
    --input_size 256 \
    --poisson_level 0 \
    --gaussian_level 0

# 2. Tạo tập dữ liệu dải góc 90 độ có nhiễu mô phỏng Low-Dose
python prepare_data_sinogram_LA.py \
    --data_dir /datastore/uittogether3/LuuTru/Thanhld/CT-Reconstruction/split/ \
    --output_dir /datastore/uittogether3/LuuTru/MinhPD/dataset/limited_angle/ \
    --angle_range_deg 90.0 \
    --num_view 64 \
    --input_size 256 \
    --poisson_level 1000000 \
    --gaussian_level 0.05
```

### Chạy qua Slurm Job trên Cluster:
```bash
sbatch ../scripts/generate_la_dataset.sh
```

---

## 6. Cấu trúc thư mục dữ liệu được sinh ra

```text
/datastore/uittogether3/LuuTru/MinhPD/dataset/limited_angle/
├── train/
│   └── limited_ang_120deg_numview_64_size_256_noise_0/
│       ├── sino/       # Chứa các file .npy sinogram góc giới hạn
│       └── fbp_u/      # Chứa các file .npy ảnh FBP thô (bị missing-wedge artifact)
└── test/
    └── limited_ang_120deg_numview_64_size_256_noise_0/
        ├── sino/
        └── fbp_u/
```
