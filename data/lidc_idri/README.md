# Module Dữ Liệu LIDC-IDRI Thoracic CT (Fan-Beam Limited-Angle CT)

Thư mục này chứa toàn bộ các script tải, nạp và tiền xử lý dữ liệu cho bộ dữ liệu lồng ngực quốc tế **LIDC-IDRI (The Lung Image Database Consortium and Image Database Resource Initiative)** trong bài toán Tái tạo ảnh Cắt lớp CT Góc Giới hạn (Limited-Angle CT Reconstruction).

---

## 1. Danh Mục Tệp Tin
* `download_lidc_subset.py`: Script tự động tải 12 ca CT lồng ngực tiêu biểu (~2,400 lát cắt trục) từ TCIA REST API trực tiếp về máy chủ.
* `CTSlice_Provider_LA_LIDC.py`: Dataset loader đọc ảnh DICOM 16-bit (`.dcm`), chuyển đổi Hounsfield Unit (`HU = pixel * RescaleSlope + RescaleIntercept`), cắt ngưỡng cửa sổ y tế $[-1000, 1000]\text{ HU}$, mô phỏng phép chiếu quạt Fan-Beam ODL/ASTRA và nạp cache `.npy`.
* `datamodule_LA_LIDC.py`: PyTorch Lightning DataModule chuẩn hóa quản lý các DataLoader cho Train (8 bệnh nhân), Validation (2 bệnh nhân) và Test (2 bệnh nhân).
* `prepare_data_sinogram_LA_LIDC.py`: Script sinh hàng loạt mảng sinogram và FBP thô cho cả 2 cung quét $120^\circ$ và $90^\circ$.

---

## 2. Thông Số Hình Học Chiếu (Fan-Beam Geometry)
* **Hình học:** Fan-Beam Geometry (`src_radius=600mm`, `det_radius=290mm`).
* **Số kênh cảm biến detector:** 512 channels (chiều dài $[-480, 480]\text{ mm}$).
* **Kích thước ảnh:** $256 \times 256$ pixels.
* **Bộ lọc FBP:** Ram-Lak filter (frequency scaling $0.9$, nhân hệ số $\sqrt{2}$).
* **Dải góc quét:**
  * Cung quét chuẩn LA-120°: $[-60.0^\circ, +60.0^\circ]$, 64 views.
  * Cung quét cực hạn LA-90°: $[-45.0^\circ, +45.0^\circ]$, 64 views.

---

## 3. Thư Mục Lưu Trữ
* Dữ liệu ảnh DICOM gốc:
  `/datastore/uittogether3/LuuTru/MinhPD/dataset/lidc_idri/dicom/`
* Dữ liệu cache `.npy` float32:
  `/datastore/uittogether3/LuuTru/MinhPD/dataset/lidc_idri/limited_angle/`
