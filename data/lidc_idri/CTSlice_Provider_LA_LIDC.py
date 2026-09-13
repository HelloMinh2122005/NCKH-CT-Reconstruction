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


class LimitedAngleCT_LIDC_Provider(Dataset):
    """
    Dataset loader cho bài toán Tái tạo ảnh CT góc giới hạn (Limited-Angle CT - LA-CT)
    trên bộ dữ liệu y tế quốc tế LIDC-IDRI (The Lung Image Database Consortium - Thoracic CT).

    Bộ dữ liệu chứa các ca chụp CT lồng ngực (phổi, phế quản, xương sườn) định dạng DICOM 16-bit.
    Giá trị mức xám lưu trữ Hounsfield Unit theo chuẩn DICOM:
        HU = PixelArray * RescaleSlope + RescaleIntercept
    """
    def __init__(
        self,
        base_path=None,
        dicom_dir="/datastore/uittogether3/LuuTru/MinhPD/dataset/lidc_idri/dicom/",
        cache_dir="/datastore/uittogether3/LuuTru/MinhPD/dataset/lidc_idri/limited_angle/",
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
        precomputed_setting=None,   # Thư mục cấu hình cache
        return_path=False,          # Trả về đường dẫn file gốc
        patient_list=None           # Danh sách bệnh nhân chỉ định (nếu có)
    ):
        if base_path is not None:
            if os.path.exists(os.path.join(base_path, "LIDC-IDRI-0001")):
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
        # 1. Phân chia tập bệnh nhân LIDC-IDRI (Train / Val / Test)
        # -------------------------------------------------------------
        if patient_list is not None:
            self.patients = patient_list
            self.split_name = "test" if test else "train"
        elif valid:
            self.split_name = "val"
            self.patients = ["LIDC-IDRI-0009", "LIDC-IDRI-0010"]
        elif test:
            self.split_name = "test"
            self.patients = ["LIDC-IDRI-0011", "LIDC-IDRI-0012"]
        else:
            self.split_name = "train"
            self.patients = [
                "LIDC-IDRI-0001", "LIDC-IDRI-0002", "LIDC-IDRI-0003", "LIDC-IDRI-0004",
                "LIDC-IDRI-0005", "LIDC-IDRI-0006", "LIDC-IDRI-0007", "LIDC-IDRI-0008"
            ]

        # Quét toàn bộ file .dcm của các bệnh nhân trong split
        self.slices_path = []
        for pat in self.patients:
            pat_dir = os.path.join(self.dicom_dir, pat)
            if os.path.exists(pat_dir):
                dcm_files = sorted(glob.glob(os.path.join(pat_dir, "**", "*.dcm"), recursive=True))
                self.slices_path.extend(dcm_files)

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

    def _read_and_process_dicom(self, dcm_path):
        """
        Đọc file DICOM lồng ngực LIDC-IDRI, giải mã Hounsfield Unit:
            HU = pixel_array * RescaleSlope + RescaleIntercept
        Cắt ngưỡng [-1000, 1000] HU và chuẩn hóa [0, 1].
        """
        dcm = pydicom.dcmread(dcm_path, force=True)
        slope = getattr(dcm, "RescaleSlope", 1.0)
        intercept = getattr(dcm, "RescaleIntercept", -1024.0)

        raw_array = dcm.pixel_array.astype(np.float32)
        hu_image = raw_array * float(slope) + float(intercept)

        # Cắt ngưỡng cửa sổ mô mềm/phổi y tế [-1000, 1000] HU
        hu_clipped = np.clip(hu_image, -1000.0, 1000.0)
        norm_image = (hu_clipped + 1000.0) / 2000.0

        # Resize về kích thước chuẩn input_size (256x256)
        pil_img = Image.fromarray(norm_image.astype(np.float32))
        if pil_img.size != (self.input_size, self.input_size):
            pil_img = pil_img.resize((self.input_size, self.input_size), Image.Resampling.BILINEAR)

        ground_truth = np.array(pil_img, dtype=np.float32)
        return ground_truth

    def __len__(self):
        return len(self.slices_path)

    def __getitem__(self, index):
        dcm_path = self.slices_path[index]
        ground_truth_256 = self._read_and_process_dicom(dcm_path)
        phantom = torch.from_numpy(ground_truth_256).unsqueeze(0).float()

        # Tạo tên file cache từ path: .../LIDC-IDRI-0001/.../000050.dcm -> LIDC-IDRI-0001_000050.npy
        pat_id = "UNKNOWN"
        for p in self.patients:
            if p in dcm_path:
                pat_id = p
                break
        slice_name = os.path.splitext(os.path.basename(dcm_path))[0]
        cache_filename = f"{pat_id}_{slice_name}.npy"

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
                    return dcm_path, phantom, fbp_u, sino_noisy
                return phantom, fbp_u, sino_noisy

        # Chế độ tính online (hoặc phục vụ sinh cache)
        pil_512 = Image.fromarray(ground_truth_256).resize((512, 512), Image.Resampling.BILINEAR)
        phantom_512 = torch.from_numpy(np.array(pil_512, dtype=np.float32)).unsqueeze(0).unsqueeze(0).float()

        # Chiếu thuận Radon Fan-beam
        sino = self.op_layer_forward(phantom_512).squeeze(0)  # Shape: (1, num_view, num_detectors)

        # Nhiễu Poisson & Gaussian
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
        fbp_512 = self.op_layer_fbp(sino_noisy.unsqueeze(0)).squeeze(0)

        # Resize FBP về 256x256
        fbp_np = fbp_512.squeeze(0).numpy()
        fbp_pil = Image.fromarray(fbp_np).resize((self.input_size, self.input_size), Image.Resampling.BILINEAR)
        fbp_u = torch.from_numpy(np.array(fbp_pil, dtype=np.float32)).unsqueeze(0).float()

        if self.return_path:
            return dcm_path, phantom, fbp_u, sino_noisy
        return phantom, fbp_u, sino_noisy
