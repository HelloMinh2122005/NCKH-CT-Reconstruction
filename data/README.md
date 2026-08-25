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
| **Kích thước miền vật lý** | $200 \times 200\text{ mm}$ | Không gian tái tạo ảnh |
| **Độ phân giải ảnh gốc** | $512 \times 512$ | Kích thước lát cắt chuẩn DICOM |
| **Độ phân giải đầu ra** | $256 \times 256$ | Kích thước ảnh sau resize đưa vào Deep Learning |
| **Bán kính nguồn (src_radius)** | $600\text{ mm}$ | Khoảng cách từ nguồn phát đến tâm quay |
| **Bán kính detector (det_radius)** | $290\text{ mm}$ | Khoảng cách từ tâm quay đến detector |
| **Số lượng Detector** | $512$ | Kích thước partition detector $[-480, 480]$ |
| **Bộ lọc FBP** | Ram-Lak (`frequency_scaling=0.9`) | Bộ lọc chuẩn FBP tái tạo sơ bộ |

---

## 3. Các cấu hình góc quét Limited-Angle phổ biến

| Tên cấu hình | Dải góc (Độ) | Dải góc (Radian) | Số góc chiếu (Views) | Giảm liều tia X ước tính |
| :--- | :--- | :--- | :--- | :--- |
| **LA-120° (Chuẩn Benchmark)** | $[-60^\circ, +60^\circ]$ | $[-\pi/3, +\pi/3]$ | $64$ hoặc $96$ views | $\approx 66.7\%$ liều quét $360^\circ$ |
| **LA-150° (Dễ hơn)** | $[-75^\circ, +75^\circ]$ | $[-5\pi/12, +5\pi/12]$ | $64$ hoặc $96$ views | $\approx 58.3\%$ liều quét $360^\circ$ |
| **LA-90° (Cực hạn / Thách thức)** | $[-45^\circ, +45^\circ]$ | $[-\pi/4, +\pi/4]$ | $48$ hoặc $64$ views | $\approx 75.0\%$ liều quét $360^\circ$ |

---

## 4. Hướng dẫn chạy tiền xử lý (Generate Dataset)

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

## 5. Cấu trúc thư mục dữ liệu được sinh ra

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
