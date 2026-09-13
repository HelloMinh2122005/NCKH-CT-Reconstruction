import os
import glob
import math
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import odl
from odl.contrib import torch as odl_torch
from skimage.io import imread

# Tránh xung đột nhiều phiên bản OpenMP runtime khi nạp thư viện C++ / CUDA
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class LimitedAngleCT_DeepLesion_Provider(Dataset):
    """
    Dataset loader cho bài toán Tái tạo ảnh CT góc giới hạn (Limited-Angle CT - LA-CT)
    trên bộ dữ liệu y tế NIH DeepLesion CT Dataset.

    Bộ dữ liệu chứa các lát cắt CT ngực/bụng trục (axial slices) định dạng ảnh 16-bit PNG,
    trong đó giá trị mức xám lưu trữ Hounsfield Unit theo công thức:
        HU = PixelValue - 32768
    """
    def __init__(
        self,
        base_path=None,
        data_dir="/datastore/uittogether3/LuuTru/Thanhld/CT-Reconstruction/minideeplesion/minideeplesion/",
        split_dir="/datastore/uittogether3/LuuTru/Thanhld/CT-Reconstruction/split_dl/",
        cache_dir="/datastore/uittogether3/LuuTru/MinhPD/dataset/nih_deep_lesion/limited_angle/",
        start_ang=-np.pi / 3,       # Góc bắt đầu (mặc định -60° cho dải 120° [-60°, +60°])
        end_ang=np.pi / 3,          # Góc kết thúc (mặc định +60°)
        num_view=64,                # Số lượng góc chiếu (views)
        num_detectors=512,          # Số lượng cảm biến detector
        poission_level=1e6,         # Mức photon mô phỏng nhiễu Poisson (0: không nhiễu)
        gaussian_level=0.05,        # Mức nhiễu Gaussian (0: không nhiễu)
        test=False,                 # Cờ bật tập Test
        valid=False,                # Cờ bật tập Validation
        input_size=256,             # Kích thước ảnh chuẩn hóa (256x256)
        transform=None,             # Phép biến đổi bổ sung
        use_precomputed=False,      # Đọc trực tiếp từ file cache .npy
        precomputed_setting=None,   # Thư mục cấu hình cache (ví dụ: limited_ang_120deg_numview_64_size_256_noise_0)
        return_path=False           # Trả về đường dẫn file gốc (cho script sinh cache)
    ):
        if base_path is not None:
            if os.path.exists(os.path.join(base_path, "train.csv")):
                split_dir = base_path
            elif os.path.exists(os.path.join(base_path, "train")):
                cache_dir = base_path

        self.data_dir = data_dir
        self.split_dir = split_dir
        self.cache_dir = cache_dir
        self.input_size = input_size
        self.transform = transform
        self.poission_level = poission_level
        self.gaussian_level = gaussian_level
        self.num_view = num_view
        self.num_detectors = num_detectors
        self.start_ang = start_ang
        self.end_ang = end_ang
        self.use_precomputed = use_precomputed
        self.precomputed_setting = precomputed_setting
        self.return_path = return_path

        # -------------------------------------------------------------
        # 1. Xác định phân vùng dữ liệu theo file CSV (Train, Val, Test)
        # -------------------------------------------------------------
        if valid:
            self.split_name = "val"
            csv_path = os.path.join(self.split_dir, "val.csv")
        elif test:
            self.split_name = "test"
            csv_path = os.path.join(self.split_dir, "test.csv")
        else:
            self.split_name = "train"
            csv_path = os.path.join(self.split_dir, "train.csv")

        # Đọc danh sách đường dẫn các file ảnh từ CSV
        self.slices_path = []
        if os.path.exists(csv_path):
            with open(csv_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and os.path.exists(line):
                        self.slices_path.append(line)
        else:
            # Fallback nếu không có CSV: quét trực tiếp từ thư mục
            self.slices_path = sorted(glob.glob(os.path.join(self.data_dir, "*", "*.png")))

        # -------------------------------------------------------------
        # 2. Thiết lập cấu hình Cache precomputed nếu có yêu cầu
        # -------------------------------------------------------------
        if self.use_precomputed:
            if self.precomputed_setting is None:
                deg = int(np.round((end_ang - start_ang) * 180.0 / np.pi))
                noise_flag = "0" if (poission_level == 0 and gaussian_level == 0) else "1e6"
                self.precomputed_setting = f"limited_ang_{deg}deg_numview_{num_view}_size_{input_size}_noise_{noise_flag}"

            self.cache_split_dir = os.path.join(self.cache_dir, self.split_name, self.precomputed_setting)
            self.sino_cache_dir = os.path.join(self.cache_split_dir, "sino")
            self.fbp_cache_dir = os.path.join(self.cache_split_dir, "fbp_u")

        # -------------------------------------------------------------
        # 3. Khởi tạo hình học Fan-beam ODL/ASTRA nếu chạy chế độ tính online
        # -------------------------------------------------------------
        if not self.use_precomputed:
            self._init_odl_geometry()

    def _init_odl_geometry(self):
        """
        Khởi tạo hình học Fan-Beam chuẩn máy CT y tế:
        - Domain vật lý: [-200, 200] mm
        - Độ phân giải: 512x512 gốc
        - Khoảng cách Nguồn - Tâm: 600 mm, Tâm - Detector: 290 mm
        - Số lượng kênh cảm biến detector: 512
        """
        xx = 200.0
        self.space = odl.uniform_discr([-xx, -xx], [xx, xx], [512, 512], dtype="float32")
        angle_partition = odl.uniform_partition(self.start_ang, self.end_ang, self.num_view)
        detector_partition = odl.uniform_partition(-480.0, 480.0, self.num_detectors)

        geometry = odl.tomo.FanBeamGeometry(
            angle_partition, detector_partition, src_radius=600.0, det_radius=290.0
        )

        try:
            operator = odl.tomo.RayTransform(self.space, geometry, impl="astra_cuda")
        except Exception:
            operator = odl.tomo.RayTransform(self.space, geometry)

        fbp = odl.tomo.fbp_op(operator, filter_type="Ram-Lak", frequency_scaling=0.9) * np.sqrt(2)

        self.op_layer_forward = odl_torch.operator.OperatorModule(operator)
        self.op_layer_fbp = odl_torch.operator.OperatorModule(fbp)

    def _read_and_process_png(self, slice_path):
        """
        Đọc ảnh 16-bit PNG của NIH DeepLesion, chuyển đổi HU:
            HU = raw_pixel - 32768
        Sau đó cắt ngưỡng cửa sổ y tế [-1000, 1000] HU và chuẩn hóa [0, 1].
        """
        raw_image = imread(slice_path).astype(np.float32)
        hu_image = raw_image - 32768.0

        # Cắt ngưỡng cửa sổ mô mềm y tế [-1000, 1000] HU và co giãn tuyến tính về [0, 1]
        hu_clipped = np.clip(hu_image, -1000.0, 1000.0)
        norm_image = (hu_clipped + 1000.0) / 2000.0

        # Chuyển đổi thành PIL Image để resize mượt mà về 256x256
        pil_img = Image.fromarray(norm_image.astype(np.float32))
        if pil_img.size != (self.input_size, self.input_size):
            pil_img = pil_img.resize((self.input_size, self.input_size), Image.Resampling.BILINEAR)

        ground_truth = np.array(pil_img, dtype=np.float32)
        return ground_truth

    def __len__(self):
        return len(self.slices_path)

    def __getitem__(self, index):
        slice_path = self.slices_path[index]
        ground_truth_256 = self._read_and_process_png(slice_path)
        phantom = torch.from_numpy(ground_truth_256).unsqueeze(0).float()

        # Đường dẫn file cache tương ứng
        # Ví dụ slice_path: .../minideeplesion/000075_02_01/049.png -> file_id: 000075_02_01_049.npy
        rel_parts = slice_path.split(os.sep)
        case_id = rel_parts[-2]
        slice_id = os.path.splitext(rel_parts[-1])[0]
        cache_filename = f"{case_id}_{slice_id}.npy"

        if self.use_precomputed:
            sino_file = os.path.join(self.sino_cache_dir, cache_filename)
            fbp_file = os.path.join(self.fbp_cache_dir, cache_filename)

            if os.path.exists(sino_file) and os.path.exists(fbp_file):
                sino_noisy = torch.from_numpy(np.load(sino_file)).float()
                fbp_u = torch.from_numpy(np.load(fbp_file)).float()
                if sino_noisy.ndim == 2:
                    sino_noisy = sino_noisy.unsqueeze(0)
                if fbp_u.ndim == 2:
                    fbp_u = fbp_u.unsqueeze(0)

                if self.return_path:
                    return slice_path, phantom, fbp_u, sino_noisy
                return phantom, fbp_u, sino_noisy

        # Chế độ tính online (hoặc phục vụ sinh cache)
        # Nâng kích thước lên 512x512 để thực hiện phép chiếu vật lý chuẩn xác
        pil_512 = Image.fromarray(ground_truth_256).resize((512, 512), Image.Resampling.BILINEAR)
        phantom_512 = torch.from_numpy(np.array(pil_512, dtype=np.float32)).unsqueeze(0).unsqueeze(0).float()

        # Chiếu thuận Radon Fan-beam
        sino = self.op_layer_forward(phantom_512).squeeze(0)  # Shape: (1, num_view, num_detectors)

        # Mô phỏng nhiễu thực tế (nếu được kích hoạt)
        if self.poission_level > 0:
            scale_val = float(self.poission_level)
            normalized_sino = torch.exp(-sino / (sino.max() + 1e-7))
            th_data = np.random.poisson(scale_val * normalized_sino.numpy())
            sino_noisy = -torch.log((torch.from_numpy(th_data).float() + 1e-7) / scale_val)
            sino_noisy = sino_noisy * sino.max()
        else:
            sino_noisy = sino.clone()

        if self.gaussian_level > 0:
            noise = float(self.gaussian_level) * torch.randn_like(sino_noisy)
            sino_noisy = sino_noisy + noise

        # Chiếu ngược có lọc FBP (Ram-Lak)
        fbp_512 = self.op_layer_fbp(sino_noisy.unsqueeze(0)).squeeze(0)  # Shape: (1, 512, 512)

        # Resize FBP về kích thước 256x256
        fbp_np = fbp_512.squeeze(0).numpy()
        fbp_pil = Image.fromarray(fbp_np).resize((self.input_size, self.input_size), Image.Resampling.BILINEAR)
        fbp_u = torch.from_numpy(np.array(fbp_pil, dtype=np.float32)).unsqueeze(0).float()

        if self.return_path:
            return slice_path, phantom, fbp_u, sino_noisy
        return phantom, fbp_u, sino_noisy
