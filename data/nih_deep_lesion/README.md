# Module Dữ Liệu NIH DeepLesion CT (Fan-Beam Limited-Angle CT)

Thư mục này chứa toàn bộ các script nạp và tiền xử lý dữ liệu cho tập dữ liệu tổn thương đa tạng **NIH DeepLesion CT Dataset** trong bài toán Tái tạo ảnh Cắt lớp CT Góc Giới hạn (Limited-Angle CT Reconstruction).

---

## 1. Danh Mục Tệp Tin
* `CTSlice_Provider_LA_DeepLesion.py`: Dataset loader đọc ảnh 16-bit PNG, giải mã Hounsfield Unit theo chuẩn NIH ($\text{HU} = \text{PixelValue} - 32768$), cắt ngưỡng cửa sổ y tế $[-1000, 1000]\text{ HU}$, mô phỏng phép chiếu quạt Fan-Beam ODL/ASTRA và nạp cache `.npy`.
* `datamodule_LA_DeepLesion.py`: PyTorch Lightning DataModule chuẩn hóa quản lý các DataLoader cho Train (1,999 lát cắt), Validation (199 lát cắt) và Test (299 lát cắt).
* `prepare_data_sinogram_LA_DeepLesion.py`: Script sinh hàng loạt mảng sinogram và FBP thô cho cả 2 cung quét $120^\circ$ và $90^\circ$.

---

## 2. Thông Số Hình Học Chiếu (Fan-Beam Geometry Đồng Nhất)
Để đảm bảo tính so sánh công bằng và khoa học với AAPM Mayo Clinic:
* **Hình học:** Fan-Beam Geometry (`src_radius=600mm`, `det_radius=290mm`).
* **Số kênh cảm biến detector:** 512 channels (chiều dài $[-480, 480]\text{ mm}$).
* **Kích thước ảnh:** $256 \times 256$ pixels.
* **Bộ lọc FBP:** Ram-Lak filter (frequency scaling $0.9$, nhân hệ số $\sqrt{2}$).
* **Dải góc quét:**
  * Cung quét chuẩn LA-120°: $[-60.0^\circ, +60.0^\circ]$, 64 views.
  * Cung quét cực hạn LA-90°: $[-45.0^\circ, +45.0^\circ]$, 64 views.

---

## 3. Thư Mục Lưu Trữ
* Dữ liệu cache `.npy` float32 được lưu tại:
  `/datastore/uittogether3/LuuTru/MinhPD/dataset/nih_deep_lesion/limited_angle/`
  * `train/`: 1,999 lát cắt từ `split_dl/train.csv`.
  * `val/`: 199 lát cắt từ `split_dl/val.csv`.
  * `test/`: 299 lát cắt từ `split_dl/test.csv`.
