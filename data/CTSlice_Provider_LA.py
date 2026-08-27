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
    Hỗ trợ đọc dữ liệu DICOM từ tập dữ liệu AAPM Mayo Clinic Low-Dose CT (LDCT).
    """
    def __init__(
        self,
        base_path=None,
        dicom_dir="/datastore/uittogether3/LuuTru/Thanhld/CT-Reconstruction/split/",
        cache_dir="/datastore/uittogether3/LuuTru/MinhPD/dataset/limited_angle/",
        start_ang=-np.pi / 3,       # Góc bắt đầu (ví dụ: -60 độ cho dải 120 độ [-60°, +60°])
        end_ang=np.pi / 3,          # Góc kết thúc (ví dụ: +60 độ)
        num_view=64,                # Số lượng góc chiếu (views) được lấy mẫu trong dải góc giới hạn
        num_detectors=512,          # Số lượng cảm biến trên detector
        poission_level=1e6,         # Mức photon mô phỏng nhiễu Poisson (càng nhỏ nhiễu càng lớn, 0 là không nhiễu)
        gaussian_level=0.05,        # Độ lệch chuẩn của nhiễu Gaussian (mô phỏng nhiễu điện tử)
        test=False,                 # Cờ bật tập Test
        valid=False,                # Cờ bật tập Validation
        input_size=256,             # Kích thước không gian ảnh đầu ra (mặc định 256x256)
        transform=None,             # Các phép biến đổi augment/resize (nếu có)
        use_precomputed=False,      # Đọc từ file cache .npy đã tính sẵn thay vì tính on-the-fly
        precomputed_setting=None,   # Tên thư mục cấu hình cache (ví dụ: limited_ang_120deg_numview_64_size_256_noise_0)
        return_path=False,          # Cờ trả về kèm đường dẫn file (dành cho script sinh dữ liệu)
        patient_list=None           # Danh sách mã bệnh nhân tùy chọn (nếu có)
    ):
        # Hỗ trợ nhận diện thông minh khi truyền base_path
        if base_path is not None:
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
            # Tập Validation: 1 bệnh nhân
            patients = ["L333"]
            split_name = "train"
        elif test:
            # Tập Test: 1 bệnh nhân
            patients = ["L310"]
            split_name = "test"
        else:
            # Tập Train: 8 bệnh nhân
            patients = ["L067", "L096", "L109", "L143", "L192", "L286", "L291", "L506"]
            split_name = "train"

        paths = []
        for patient_id in patients:
            pattern = glob.glob(
                os.path.join(self.dicom_dir, split_name, patient_id, "full_3mm", f"{patient_id}_FD_3_1.CT.*.*.*.*.*.*.*.*.*.IMA")
            )
            paths.extend(pattern)

        # Sắp xếp danh sách đường dẫn theo thứ tự để đảm bảo tính nhất quán (reproducibility)
        self.slices_path = sorted(paths)
        split_label = "VALID" if valid else ("TEST" if test else "TRAIN")
        print(f"[{split_label}] Loaded {len(self.slices_path)} CT slices from {self.dicom_dir} (split: {split_name}).")

        # -------------------------------------------------------------
        # 2. Thiết lập cơ chế nạp dữ liệu: từ cache (.npy) hoặc qua ODL
        # -------------------------------------------------------------
        if self.use_precomputed and self.precomputed_setting:
            # Trỏ trực tiếp tới thư mục sinogram và FBP đã tính sẵn để tăng tốc nạp dữ liệu
            self.sino_dir = os.path.join(self.cache_dir, split_name, self.precomputed_setting, "sino")
            self.fbp_dir = os.path.join(self.cache_dir, split_name, self.precomputed_setting, "fbp_u")
        else:
            # Xây dựng các toán tử chiếu tia Radon và FBP góc giới hạn bằng ODL và ASTRA GPU
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
        # Không gian tái tạo ảnh 2D: kích thước thực tế [-200mm, 200mm] với độ phân giải lưới 512x512
        xx = 200
        space = odl.uniform_discr([-xx, -xx], [xx, xx], [512, 512], dtype='float32')
        angles = np.array(num_view).astype(int)

        # Phân hoạch góc giới hạn: chỉ quét từ start_ang đến end_ang
        angle_partition = odl.uniform_partition(start_ang, end_ang, angles)
        # Phân hoạch dải cảm biến (detector) từ -480mm đến 480mm
        detector_partition = odl.uniform_partition(-480, 480, num_detectors)

        # Cấu hình hình học chùm tia quạt (Fan-Beam Geometry) mô phỏng máy CT thực tế
        # src_radius: Khoảng cách từ nguồn phát tia X tới tâm quay (600mm)
        # det_radius: Khoảng cách từ tâm quay tới detector (290mm)
        geometry = odl.tomo.FanBeamGeometry(
            angle_partition,
            detector_partition,
            src_radius=600,
            det_radius=290
        )

        # Toán tử chiếu thuận Radon (Ray Transform) tăng tốc bằng GPU CUDA ASTRA
        impl = 'astra_cuda' if torch.cuda.is_available() else 'astra_cpu'
        operator = odl.tomo.RayTransform(space, geometry, impl=impl)
        # Toán tử chiếu ngược có lọc (FBP) sử dụng bộ lọc Ram-Lak
        fbp = odl.tomo.fbp_op(operator, filter_type='Ram-Lak', frequency_scaling=0.9) * np.sqrt(2)

        # Đóng gói toán tử ODL thành PyTorch Module để có thể xử lý trực tiếp Tensor trên GPU
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
        # Hiệu chuẩn đơn vị Hounsfield Unit (HU): pixel_val * slope + intercept
        data_slice = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
        data_slice = data_slice.astype(float)
        # Chuẩn hóa giá trị pixel về dải [0, 1]
        data_slice = (data_slice - np.min(data_slice)) / (np.max(data_slice) - np.min(data_slice) + 1e-8)
        phantom = torch.from_numpy(data_slice).unsqueeze(0).float()

        # -------------------------------------------------------------
        # Nhánh 1: Nạp từ file npy đã tính sẵn (Tối ưu tốc độ huấn luyện)
        # -------------------------------------------------------------
        if self.use_precomputed and self.precomputed_setting:
            sino_file = os.path.join(self.sino_dir, f"{file_stem}.npy")
            fbp_file = os.path.join(self.fbp_dir, f"{file_stem}.npy")
            
            # Đọc sinogram và ảnh FBP từ file cache
            sino = torch.from_numpy(np.load(sino_file)).float()
            fbp_u = torch.from_numpy(np.load(fbp_file)).float()
        # -------------------------------------------------------------
        # Nhánh 2: Tính toán động on-the-fly (Dùng khi sinh dữ liệu mới)
        # -------------------------------------------------------------
        else:
            # Chiếu tia X thuận (Forward Projection) trong dải góc giới hạn để tạo Sinogram
            sino = self.radon_op(phantom)

            # Thêm nhiễu Poisson (mô phỏng số lượng photon tới detector theo định luật Beer-Lambert)
            if self.poission_level > 0:
                scale_val = torch.tensor(float(self.poission_level))
                norm_sino = torch.exp(-sino / (sino.max() + 1e-8))
                th_data = np.random.poisson((scale_val * norm_sino).cpu().numpy())
                sino_noisy = -torch.log(torch.from_numpy(th_data).float() / scale_val + 1e-8) * sino.max()
            else:
                sino_noisy = sino

            # Thêm nhiễu Gaussian (mô phỏng nhiễu mạch điện tử của cảm biến)
            if self.gaussian_level > 0:
                noise = float(self.gaussian_level) * torch.randn_like(sino_noisy)
                sino_noisy = sino_noisy + noise

            # Tái tạo ảnh ban đầu bằng FBP từ sinogram góc giới hạn (sẽ xuất hiện artifacts do thiếu góc quét)
            fbp_u = self.fbp_op(sino_noisy)
            sino = sino_noisy

        # -------------------------------------------------------------
        # 3. Resize kích thước ảnh về độ phân giải mô hình yêu cầu (vd: 256x256)
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
