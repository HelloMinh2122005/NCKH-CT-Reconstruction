cd ~/projects/PlotNeuralNet/pyexamples
../tikzmake.sh fista_net_arch


Bạn là chuyên gia về Deep Learning và đồ họa vector với thư viện PlotNeuralNet.
Hãy viết một file script Python hoàn chỉnh đặt trong thư mục `PlotNeuralNet/pyexamples/` để mô phỏng kiến trúc mạng [TÊN_MẠNG_CỦA_BẠN].

### 1. Ràng buộc kỹ thuật của thư viện PlotNeuralNet (BẮT BUỘC TUÂN THỦ):
- Chỉ import và sử dụng các hàm có sẵn trong `pycore.tikzeng`:
  + `to_Conv(name, s_filer, n_filer, offset="(x,y,z)", to="(prev-east)", width, height, depth, caption)`
  + `to_ConvConvRelu(name, s_filer, n_filer=(c1, c2), offset, to, width=(w1, w2), height, depth, caption)` (lưu ý: n_filer và width BẮT BUỘC là tuple)
  + `to_Pool(name, offset, to, width, height, depth, opacity, caption)`
  + `to_SoftMax(name, s_filer, offset, to, width, height, depth, opacity, caption)`
  + `to_Sum(name, offset, to, radius, opacity)`
  + `to_connection(of, to)`: mũi tên thẳng nối 2 layer
  + `to_skip(of, to, pos=1.25)`: đường nối tắt (skip connection) hình chữ U
- TUYỆT ĐỐI KHÔNG tự bịa ra các hàm như to_Linear, to_BatchNorm, to_Attention. Nếu có các thành phần này, hãy quy đổi biểu diễn thành `to_Conv` hoặc `to_SoftMax` với caption tương ứng.

### 2. Nguyên tắc bố cục không gian 3D:
- Chiều ngang `width`: Thể hiện số kênh (Channels / Feature Maps).
- Chiều cao `height` và sâu `depth`: Thể hiện kích thước không gian ảnh (Spatial H, W).
- Định vị tương đối: Luôn đặt layer sau tựa vào layer trước bằng `to="(tên_layer_trước-east)"` với `offset="(dx, 0, 0)"` (thường dx từ 1.5 đến 2.5 để không bị đè lên nhau).

### 3. Mô tả kiến trúc mạng cần vẽ:
[MÔ TẢ CHI TIẾT CÁC TẦNG CỦA BẠN HOẶC DÁN CODE PYTORCH MODEL VÀO ĐÂY]

### 4. Định dạng đầu ra yêu cầu:
- Trả về code Python hoàn chỉnh có hàm `main()` gọi `to_generate(arch, namefile + '.tex')`.


Hãy viết một script Python `fista_net_learn_mva.py` trong `PlotNeuralNet/pyexamples/` để vẽ chi tiết cho bài toán Limited-Angle CT Reconstruction.

Kiến trúc tuần follow theo /home/phandinhminh/Downloads/kltn/agents-research/uittogether3-slurm-server/MinhPD/docs/LEARN_MVA_diagram.html