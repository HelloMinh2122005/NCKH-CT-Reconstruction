import os
import glob
import math
import numpy as np
import torch
import pydicom
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import odl
from odl.contrib import torch as odl_torch

# Tránh xung đột nhiều phiên bản OpenMP runtime khi nạp thư viện C++ / CUDA
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class LimitedAngleCT_Provider(Dataset):
    """
    Dataset loader cho bài toán Tái tạo ảnh CT góc giới hạn (Limited-Angle CT - LA-CT).
    Hỗ trợ đọc dữ liệu DICOM từ tập dữ liệu AAPM Mayo Clinic Low-Dose CT (LDCT)
    kết hợp với bộ nhớ đệm cache .npy (Sinogram + FBP) để tối ưu hóa tốc độ huấn luyện.
    """
    def __init__(
        self,
        base_path=None,
        dicom_dir="/datastore/uittogether3/LuuTru/Thanhld/CT-Reconstruction/split/",
        cache_dir="/datastore/uittogether3/LuuTru/MinhPD/dataset/limited_angle/",
        start_ang=-np.pi / 3,       # Góc bắt đầu (mặc định: -60 độ = -pi/3)
        end_ang=np.pi / 3,          # Góc kết thúc (mặc định: +60 độ = +pi/3)
        num_view=64,                # Số lượng góc chiếu (views)
        num_detectors=512,          # Số lượng cảm biến trên detector
        poission_level=0.0,         # Mức photon mô phỏng nhiễu Poisson (0 nếu không nhiễu)
        gaussian_level=0.0,         # Độ lệch chuẩn của nhiễu Gaussian (0 nếu không nhiễu)
        test=False,                 # Cờ bật tập Test
        valid=False,                # Cờ bật tập Validation
        input_size=256,             # Kích thước không gian ảnh đầu ra (256x256)
        transform=None,             # Phép biến đổi augment/resize
        use_precomputed=True,       # Đọc từ file cache .npy đã tính sẵn
        precomputed_setting=None,   # Tên tag cấu hình cache
        return_path=False,          # Trả về kèm đường dẫn file (dành cho script sinh data)
        patient_list=None           # Danh sách bệnh nhân tùy biến (nếu None sẽ lấy mặc định)
    ):
        # Xử lý thông minh đường dẫn DICOM và Cache
        if base_path is not None:
            # Kiểm tra xem base_path là thư mục DICOM hay Cache
            if os.path.exists(os.path.join(base_path, "train", "L067")):
                dicom_dir = base_path
            elif os.path.exists(os.path.join(base_path, "train")):
                cache_dir = base_path

        self.dicom_dir = dicom_dir
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
        # 1. Phân chia bệnh nhân theo chuẩn tập AAPM Mayo Clinic LDCT
        # -------------------------------------------------------------
        if patient_list is not None:
            patients = patient_list
            split_name = "test" if test else "train"
        elif valid:
            patients = ["L333"]
            split_name = "train"
        elif test:
            patients = ["L310"]
            split_name = "test"
        else:
            patients = ["L067", "L096", "L109", "L143", "L192", "L286", "L291", "L506"]
            split_name = "train"

        paths = []
        for patient_id in patients:
            pattern = glob.glob(
                os.path.join(self.dicom_dir, split_name, patient_id, "full_3mm", f"{patient_id}_FD_3_1.CT.*.*.*.*.*.*.*.*.*.IMA")
            )
            paths.extend(pattern)

        # Sắp xếp danh sách đường dẫn theo thứ tự nhất quán
        self.slices_path = sorted(paths)
        split_label = "VALID" if valid else ("TEST" if test else "TRAIN")
        print(f"[{split_label}] Loaded {len(self.slices_path)} CT slices from {self.dicom_dir} (split: {split_name}).")

        # -------------------------------------------------------------
        # 2. Thiết lập cơ chế nạp dữ liệu: từ cache (.npy) hoặc qua ODL
        # -------------------------------------------------------------
        if self.use_precomputed and self.precomputed_setting:
            self.sino_dir = os.path.join(self.cache_dir, split_name, self.precomputed_setting, "sino")
            self.fbp_dir = os.path.join(self.cache_dir, split_name, self.precomputed_setting, "fbp_u")
        else:
            self.radon_op, self.fbp_op = self._build_limited_angle_operators(
                start_ang=self.start_ang,
                end_ang=self.end_ang,
                num_view=self.num_view,
                num_detectors=self.num_detectors
            )

    def _build_limited_angle_operators(self, start_ang, end_ang, num_view, num_detectors):
        """
        Khởi tạo toán tử chiếu Radon (Forward) và FBP (Backward) theo hình học chùm tia quạt (Fan-Beam).
        """
        xx = 200
        space = odl.uniform_discr([-xx, -xx], [xx, xx], [512, 512], dtype='float32')
        angles = np.array(num_view).astype(int)

        angle_partition = odl.uniform_partition(start_ang, end_ang, angles)
        detector_partition = odl.uniform_partition(-480, 480, num_detectors)

        geometry = odl.tomo.FanBeamGeometry(
            angle_partition,
            detector_partition,
            src_radius=600,
            det_radius=290
        )

        impl = 'astra_cuda' if torch.cuda.is_available() else 'astra_cpu'
        operator = odl.tomo.RayTransform(space, geometry, impl=impl)
        fbp = odl.tomo.fbp_op(operator, filter_type='Ram-Lak', frequency_scaling=0.9) * np.sqrt(2)

        op_layer = odl_torch.operator.OperatorModule(operator)
        op_layer_fbp = odl_torch.operator.OperatorModule(fbp)

        return op_layer, op_layer_fbp

    def __len__(self):
        """Trả về tổng số lượng lát cắt CT trong tập dữ liệu."""
        return len(self.slices_path)

    def __getitem__(self, index):
        """
        Lấy mẫu dữ liệu tại vị trí index.
        Trả về:
          - Nếu return_path == True: (slice_path, phantom, fbp_u, sino)
          - Nếu return_path == False (mặc định): (phantom, fbp_u, sino)
        """
        slice_path = self.slices_path[index]
        file_stem = os.path.basename(slice_path).split(".IMA")[0]

        # -------------------------------------------------------------
        # Đọc ảnh DICOM gốc làm Ground Truth (phantom)
        # -------------------------------------------------------------
        dcm = pydicom.read_file(slice_path)
        data_slice = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
        data_slice = data_slice.astype(float)
        data_slice = (data_slice - np.min(data_slice)) / (np.max(data_slice) - np.min(data_slice) + 1e-8)
        phantom = torch.from_numpy(data_slice).unsqueeze(0).float()

        # -------------------------------------------------------------
        # Nhánh 1: Nạp từ file npy đã tính sẵn (Tối ưu tốc độ huấn luyện)
        # -------------------------------------------------------------
        if self.use_precomputed and self.precomputed_setting:
            sino_file = os.path.join(self.sino_dir, f"{file_stem}.npy")
            fbp_file = os.path.join(self.fbp_dir, f"{file_stem}.npy")
            
            sino = torch.from_numpy(np.load(sino_file)).float()
            fbp_u = torch.from_numpy(np.load(fbp_file)).float()
        # -------------------------------------------------------------
        # Nhánh 2: Tính toán động on-the-fly (Dùng khi sinh dữ liệu mới)
        # -------------------------------------------------------------
        else:
            sino = self.radon_op(phantom)

            if self.poission_level > 0:
                scale_val = torch.tensor(float(self.poission_level))
                norm_sino = torch.exp(-sino / (sino.max() + 1e-8))
                th_data = np.random.poisson((scale_val * norm_sino).cpu().numpy())
                sino_noisy = -torch.log(torch.from_numpy(th_data).float() / scale_val + 1e-8) * sino.max()
            else:
                sino_noisy = sino

            if self.gaussian_level > 0:
                noise = float(self.gaussian_level) * torch.randn_like(sino_noisy)
                sino_noisy = sino_noisy + noise

            fbp_u = self.fbp_op(sino_noisy)
            sino = sino_noisy

        # -------------------------------------------------------------
        # 3. Resize ảnh về độ phân giải mô hình yêu cầu (vd: 256x256)
        # -------------------------------------------------------------
        if self.transform is not None:
            phantom = self.transform(phantom)
            fbp_u = self.transform(fbp_u)
        elif phantom.shape[-1] != self.input_size:
            resizer = transforms.Resize((self.input_size, self.input_size))
            phantom = resizer(phantom)
            fbp_u = resizer(fbp_u)

        if self.return_path:
            return slice_path, phantom, fbp_u, sino
        return phantom, fbp_u, sino
