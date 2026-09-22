# Hướng Dẫn Sử Dụng Bộ Khối 3D Custom Shapes Cho Draw.io
*(Áp dụng cho sơ đồ FISTA-Net / LEARN_MVA Limited-Angle CT)*

Thư mục lưu trữ: `uittogether3-slurm-server/MinhPD/docs/drawio_blocks/`

---

## 📦 Danh Sách Các Khối Custom Shape Đã Tạo

| STT | Tên Khối (Shape Name) | File XML | Chức Năng Cụ Thể |
|:---:|:---|:---|:---|
| 1 | **ImageHolderPlate** | `01_ImageHolderPlate.xml` | Tấm nghiêng 3D chuyên dụng để lồng ảnh Sinogram $y$, ảnh $x^{(0)}$, $x^{(14)}$, Ground Truth $x^*$. |
| 2 | **Conv3DBlock** | `02_Conv3DBlock.xml` | Khối hộp 3D Isometric Tensor Feature Map cho tầng tích chập Conv (Forward $A$, Adjoint $A^T$, Conv1, Conv2, Conv3). |
| 3 | **PatchTokenizerBlock** | `03_PatchTokenizerBlock.xml` | Khối 3D chia lưới Patch $4\times4$ ô vuông thể hiện toán tử cắt token vi mô $T_{2\times2}$ ($16\text{k}$ tokens). |
| 4 | **SequenceEngineBlock** | `04_SequenceEngineBlock.xml` | Khối 3D trung tâm thể hiện cỗ máy xử lý chuỗi dài $\mathcal{O}(N)$ Attention (Longformer / LongNet / Mamba). |
| 5 | **UntokenizerBlock** | `05_UntokenizerBlock.xml` | Khối 3D nếp gấp hội tụ thể hiện toán tử đảo $T_{2\times2}^{-1}$ gấp các token lại thành lưới ảnh 2D. |
| 6 | **SummationNode3D** | `06_SummationNode3D.xml` | Nút tròn cộng Gradient 3D $\oplus$ có dấu cộng dày ở tâm và 8 điểm neo kết nối đa hướng. |
| 7 | **StageContainerBox** | `07_StageContainerBox.xml` | Khung container unrolling stage bo góc viền đứt nét kèm huy hiệu tiêu đề phía trên. |

---

## 🚀 2 Cách Nạp Khối Vào Draw.io Cực Kỳ Nhanh Chóng

### Cách 1: Chèn Từng Khối Qua Menu "Insert Shape" (Rất Tiện Khi Cần 1 Khối Cụ Thể)
1. Trên thanh menu trên cùng của **Draw.io**, chọn:
   $$\text{Arrange} \longrightarrow \text{Insert} \longrightarrow \text{Shape...}$$
2. Mở file `.xml` của khối bạn muốn dùng (ví dụ: `01_ImageHolderPlate.xml` hoặc `02_Conv3DBlock.xml`), copy toàn bộ nội dung.
3. Dán vào khung soạn thảo XML của Draw.io rồi bấm nút **Apply**.
4. Khối 3D sẽ xuất hiện ngay trên bản vẽ! Bạn có thể tự do:
   - Thay đổi màu sắc (Fill Color): Vàng hổ phách cho nhánh vật lý, Tím/Cyan cho nhánh tiên nghiệm, Xám đen cho ảnh CT.
   - Đổi độ dày viền (Line width), đổi màu viền (Stroke color).
   - Kéo dãn kích thước to/nhỏ bằng chuột (chuẩn `aspect="variable"`).

---

### Cách 2: Nạp Nguyên Bộ Thư Viện 7 Khối Vào Cột Công Cụ Trái (Khuyên Dùng)
1. Mở Draw.io.
2. Kéo thả trực tiếp file tổng hợp:
   `uittogether3-slurm-server/MinhPD/docs/drawio_blocks/fista_net_drawio_library.xml`
   vào cửa sổ Draw.io.
   *(Hoặc vào menu **File** $\to$ **Open Library from** $\to$ **Device...** $\to$ Chọn file `fista_net_drawio_library.xml`)*.
3. Một bảng công cụ mới mang tên **fista_net_drawio_library** sẽ xuất hiện ở cột bên trái chứa đầy đủ cả 7 biểu tượng 3D. Mỗi lần cần vẽ, bạn chỉ việc kéo thả ra trang vẽ!

---

## 🎨 Hướng Dẫn Chèn & Lồng Ảnh Vào Khối `ImageHolderPlate`
1. Kéo khối `ImageHolderPlate` ra màn hình vẽ.
2. Thả ảnh cắt lớp CT hoặc Sinogram vào Draw.io.
3. Kéo ảnh đặt đè lên mặt trước của `ImageHolderPlate` (căn chỉnh theo khung viền chỉ dẫn bên trong - Inner Inset Frame).
4. Nhấp chuột phải vào ảnh $\to$ **To Front** (hoặc chọn khối tấm plate $\to$ **To Back**).
5. Bạn có thể chọn cả 2 đối tượng (Tấm plate + Ảnh) rồi bấm **Ctrl + G** để Group lại thành một khối thống nhất!
