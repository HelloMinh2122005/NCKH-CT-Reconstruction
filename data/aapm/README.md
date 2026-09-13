# Module Dữ Liệu AAPM Mayo Clinic Low Dose CT (Fan-Beam Limited-Angle CT)

Thư mục này chứa toàn bộ các script nạp và tiền xử lý dữ liệu cho bộ dữ liệu chuẩn y tế **AAPM Mayo Clinic Low Dose CT Grand Challenge (2016)** trong bài toán Tái tạo ảnh Cắt lớp CT Góc Giới hạn (Limited-Angle CT Reconstruction).

---

## 1. Danh Mục Tệp Tin
* `CTSlice_Provider_LA.py`: Dataset loader đọc ảnh y tế DICOM 16-bit (`.IMA`), chuyển đổi Hounsfield Unit (HU), cắt ngưỡng cửa sổ mô mềm $[-1000, 1000]\text{ HU}$, mô phỏng phép chiếu quạt Fan-Beam ODL/ASTRA và nạp cache `.npy`.
* `datamodule_LA.py`: PyTorch Lightning DataModule chuẩn hóa phục vụ huấn luyện các mô hình Baseline (LEARN) và Đề xuất (SOLAR).
* `prepare_data_sinogram_LA.py`: Script sinh hàng loạt mảng sinogram thô và ảnh tái tạo FBP thô (Filtered Backprojection) cho cả 2 cung quét $120^\circ$ và $90^\circ$.

---

## 2. Thông Số Vật Lý & Hình Học Chiếu (Fan-Beam Geometry)
* **Hình học:** Fan-Beam Geometry (chuẩn máy CT Siemens Somatom).
* **Khoảng cách Nguồn - Tâm (`src_radius`):** $600.0\text{ mm}$.
* **Khoảng cách Tâm - Detector (`det_radius`):** $290.0\text{ mm}$.
* **Số kênh cảm biến detector:** 512 channels (chiều dài thanh dò $[-480, 480]\text{ mm}$).
* **Kích thước ảnh:** $256 \times 256$ pixels (rescale từ $512 \times 512$ gốc).
* **Bộ lọc FBP:** Ram-Lak filter (frequency scaling $0.9$, nhân hệ số $\sqrt{2}$).
* **Dải góc quét:**
  * Cung quét chuẩn LA-120°: $[-60.0^\circ, +60.0^\circ]$, 64 views (bước $\Delta \theta \approx 1.875^\circ$).
  * Cung quét cực hạn LA-90°: $[-45.0^\circ, +45.0^\circ]$, 64 views (bước $\Delta \theta \approx 1.406^\circ$).

---

## 3. Cấu Trúc Dữ Liệu Lưu Trữ
* Dữ liệu cache `.npy` float32 được lưu tại:
  `/datastore/uittogether3/LuuTru/MinhPD/dataset/limited_angle/` (hoặc `dataset/aapm/`)
  * `train/120deg/`: `sino/` (kích thước $[1, 64, 512]$) và `fbp_u/` (kích thước $[1, 256, 256]$) từ 1,920 lát cắt (8 bệnh nhân).
  * `train/90deg/`: Tương tự cho cung quét $90^\circ$.
  * `test/120deg/`: Dữ liệu kiểm thử độc lập từ bệnh nhân `Patient L310` (214 lát cắt).
  * `test/90deg/`: Dữ liệu stress-test góc hẹp từ bệnh nhân `Patient L310` (214 lát cắt).
