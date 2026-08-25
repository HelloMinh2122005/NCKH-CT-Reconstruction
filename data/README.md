# Quy trình Chuẩn bị Dữ liệu Limited-Angle CT (LA-CT)

Tài liệu này hướng dẫn chi tiết cách tạo và tiền xử lý dữ liệu cho bài toán **Limited-Angle CT Reconstruction** từ bộ dữ liệu CT gốc (AAPM Mayo Clinic LDCT).

---

## 1. Giới thiệu Bài toán Limited-Angle CT
Trong chụp cắt lớp vi tính thông thường, đầu phát tia X và máy thu (detector) quay trọn một góc $360^\circ$ (hoặc $180^\circ + 2\gamma$ đối với chùm tia Fan-Beam). 
Tuy nhiên trong thực tế lâm sàng (như chụp C-arm trong phẫu thuật, nha khoa, chụp X-quang tuyến vú cắt lớp số - DBT) hoặc kiểm tra công nghiệp, góc quay bị cản trở bởi cấu trúc cơ thể hoặc thiết bị:
- **Góc quét bị giới hạn (Limited Angular Range):** $\Delta \theta < 180^\circ$ (ví dụ: $90^\circ, 120^\circ, 140^\circ, 150^\circ$).
- **Vấn đề toán học (Missing Wedge Problem):** Theo định lý lát cắt Fourier (Fourier Slice Theorem), việc thiếu một phần góc quét dẫn đến vùng khuyết hình nêm trong miền tần số. Do đó, ảnh tái tạo bằng giải thuật FBP cổ điển bị vệt sọc nặng (streak artifacts), mờ biên và méo dạng cấu trúc nghiêm trọng hơn bài toán Sparse-view rất nhiều.

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

| Tên cấu hình | Dải góc (Độ) | Dải góc (Radian) | Số góc chiếu (Views) |
| :--- | :--- | :--- | :--- |
| **LA-120° (Chuẩn Benchmark)** | $[-60^\circ, +60^\circ]$ | $[-\pi/3, +\pi/3]$ | $64$ hoặc $96$ views |
| **LA-150° (Dễ hơn)** | $[-75^\circ, +75^\circ]$ | $[-5\pi/12, +5\pi/12]$ | $64$ hoặc $96$ views |
| **LA-90° (Cực hạn / Thách thức)** | $[-45^\circ, +45^\circ]$ | $[-\pi/4, +\pi/4]$ | $48$ hoặc $64$ views |

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
