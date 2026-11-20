import os
import glob
import math
import re
import numpy as np
import torch
try:
    import pydicom
except ImportError:
    pydicom = None
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
try:
    import odl
    from odl.contrib import torch as odl_torch
except ImportError:
    odl = None
    odl_torch = None

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
        patient_list=None,          # Danh sách mã bệnh nhân tùy chọn (nếu có)
        gt_only=False               # Cờ chỉ đọc ảnh Ground Truth từ DICOM (bỏ qua khởi tạo toán tử ODL/ASTRA)
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
        self.gt_only = gt_only

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

        split_label = "VALID" if valid else ("TEST" if test else "TRAIN")

        # -------------------------------------------------------------
        # 2. Thiết lập cơ chế nạp dữ liệu: từ cache (.npy) hoặc qua ODL
        # -------------------------------------------------------------
        self.sino_dir = None
        self.fbp_dir = None
        self.gt_dir = None
        self.cache_only = False
        if self.use_precomputed and self.precomputed_setting:
            # Trỏ trực tiếp tới thư mục sinogram và FBP đã tính sẵn để tăng tốc nạp dữ liệu
            self.sino_dir = os.path.join(self.cache_dir, split_name, self.precomputed_setting, "sino")
            self.fbp_dir = os.path.join(self.cache_dir, split_name, self.precomputed_setting, "fbp_u")
            # [2026-11-20 T0.4] Tìm thư mục Ground Truth (.npy) đã tiền xử lý để không cần đọc DICOM gốc từ raw/
            self.gt_dir = self._resolve_gt_dir(split_name)
        elif self.gt_only:
            # Chế độ thuần Ground Truth: bỏ qua khởi tạo ODL/ASTRA
            self.radon_op, self.fbp_op = None, None
        else:
            # Xây dựng các toán tử chiếu tia Radon và FBP góc giới hạn bằng ODL và ASTRA GPU
            self.radon_op, self.fbp_op = self._build_limited_angle_operators(
                start_ang=self.start_ang,
                end_ang=self.end_ang,
                num_view=self.num_view,
                num_detectors=self.num_detectors
            )

        paths = []
        if self.use_precomputed and self.precomputed_setting and self.gt_dir is not None:
            # Chế độ thuần cache: quét trực tiếp từ thư mục gt/ và đối chiếu sino/ + fbp_u/ (không chạm split/ hay raw/)
            paths = self._discover_cached_slices(patients)
            self.cache_only = True
            source_dir = self.gt_dir
        else:
            for patient_id in patients:
                # Hỗ trợ linh hoạt các định dạng tên file ảnh DICOM/IMA (legacy Mayo Clinic hoặc symlink đánh số chuẩn)
                patterns = [
                    os.path.join(self.dicom_dir, split_name, patient_id, "full_3mm", f"{patient_id}_FD_3_1.CT.*.IMA"),
                    os.path.join(self.dicom_dir, split_name, patient_id, "full_3mm", f"{patient_id}_FD_3_1.CT.*.*.*.*.*.*.*.*.*.IMA"),
                    os.path.join(self.dicom_dir, split_name, patient_id, "full_3mm", "*.IMA"),
                    os.path.join(self.dicom_dir, split_name, patient_id, "full_3mm", "*.dcm")
                ]
                for p in patterns:
                    matches = glob.glob(p)
                    if matches:
                        paths.extend(matches)
                        break

            dangling = [p for p in paths if not os.path.exists(p)]
            if self.use_precomputed and self.precomputed_setting:
                # Phát hiện sớm trường hợp thiếu gt cache VÀ symlink split/ bị gãy khi raw/ đã xóa
                if dangling or not paths:
                    raise FileNotFoundError(
                        f"CRITICAL ERROR: Không tìm thấy cache Ground Truth (.npy) tại '{self.cache_dir}/{split_name}/gt_size_{self.input_size}' "
                        f"và phát hiện {len(dangling)}/{len(paths)} tệp DICOM thiếu/symlink gãy tại '{self.dicom_dir}' (split: {split_name})."
                    )
            elif dangling or (self.gt_only and not paths):
                raise FileNotFoundError(
                    f"CRITICAL ERROR: Phát hiện {len(dangling)}/{len(paths)} tệp DICOM thiếu/symlink gãy tại '{self.dicom_dir}' (split: {split_name})."
                )
            source_dir = self.dicom_dir

        # Sắp xếp danh sách đường dẫn theo thứ tự để đảm bảo tính nhất quán (reproducibility)
        self.slices_path = sorted(paths)
        print(f"[{split_label}] Loaded {len(self.slices_path)} CT slices from {source_dir} (split: {split_name}).")

    def _resolve_gt_dir(self, split_name):
        """
        Tìm thư mục Ground Truth (.npy) theo thứ tự ưu tiên:
        1. <cache_dir>/<split>/<precomputed_setting>/gt
        2. <cache_dir>/<split>/gt_size_<input_size>
        3. <cache_dir>/<split>/gt
        Chỉ kích hoạt chế độ thuần cache khi sino/ và fbp_u/ cùng tồn tại hợp lệ.
        """
        if not (os.path.isdir(self.sino_dir) and os.path.isdir(self.fbp_dir)):
            return None

        candidates = [
            os.path.join(self.cache_dir, split_name, self.precomputed_setting, "gt"),
            os.path.join(self.cache_dir, split_name, f"gt_size_{self.input_size}"),
            os.path.join(self.cache_dir, split_name, "gt"),
        ]
        for cand in candidates:
            if os.path.isdir(cand):
                with os.scandir(cand) as entries:
                    if any(entry.name.endswith(".npy") and entry.is_file() for entry in entries):
                        return cand
        return None

    def _discover_cached_slices(self, patients):
        """
        Quét trực tiếp danh sách lát cắt từ thư mục gt_dir cho từng bệnh nhân và đối chiếu cặp sino/ + fbp_u/.
        Dùng regex tiền tố chặt để L1 không bao giờ khớp nhầm L10.
        """
        all_gt_files = sorted(f for f in os.listdir(self.gt_dir) if f.endswith(".npy"))
        sino_files = set(os.listdir(self.sino_dir))
        fbp_files = set(os.listdir(self.fbp_dir))

        paths = []
        for patient_id in patients:
            # Khớp chính xác tiền tố patient_id kết thúc bằng '_', '.' hoặc hết chuỗi
            pid_pattern = re.compile(rf"^{re.escape(patient_id)}(?:_|\.|$)")
            matched = [fname for fname in all_gt_files if pid_pattern.match(os.path.splitext(fname)[0])]
            if not matched:
                raise FileNotFoundError(
                    f"CRITICAL ERROR: Không tìm thấy lát cắt cache Ground Truth nào cho bệnh nhân '{patient_id}' tại '{self.gt_dir}'."
                )

            for fname in matched:
                if fname not in sino_files and not os.path.exists(os.path.join(self.sino_dir, fname)):
                    raise FileNotFoundError(
                        f"CRITICAL ERROR: Thiếu tệp cache sinogram tương ứng '{os.path.join(self.sino_dir, fname)}'."
                    )
                if fname not in fbp_files and not os.path.exists(os.path.join(self.fbp_dir, fname)):
                    raise FileNotFoundError(
                        f"CRITICAL ERROR: Thiếu tệp cache FBP tương ứng '{os.path.join(self.fbp_dir, fname)}'."
                    )
                paths.append(os.path.join(self.gt_dir, fname))

        if not paths:
            raise FileNotFoundError(
                f"CRITICAL ERROR: Không tìm thấy tệp cache hợp lệ nào trong '{self.gt_dir}' cho danh sách bệnh nhân {patients}."
            )
        return paths

    def _load_dicom_phantom(self, slice_path):
        """
        Đọc và hiệu chuẩn ảnh DICOM gốc thành Tensor Ground Truth (1, 512, 512) dải [0, 1].
        """
        if pydicom is None:
            raise ImportError("Thiếu thư viện 'pydicom' để đọc tệp DICOM gốc.")
        if not os.path.exists(slice_path):
            raise FileNotFoundError(
                f"CRITICAL ERROR: Không tìm thấy tệp DICOM tại '{slice_path}' (đích thực tế: '{os.path.realpath(slice_path)}')."
            )
        if os.path.getsize(slice_path) == 0:
            raise ValueError(
                f"CRITICAL ERROR: Phát hiện tệp DICOM rỗng 0 byte tại '{slice_path}' (đích thực tế: '{os.path.realpath(slice_path)}')."
            )

        # -------------------------------------------------------------
        # Đọc ảnh DICOM gốc làm Ground Truth (phantom)
        # -------------------------------------------------------------
        dcm = pydicom.read_file(slice_path)
        # Hiệu chuẩn đơn vị Hounsfield Unit (HU): pixel_val * slope + intercept
        data_slice = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
        data_slice = data_slice.astype(float)
        # Chuẩn hóa giá trị pixel về dải [0, 1]
        data_slice = (data_slice - np.min(data_slice)) / (np.max(data_slice) - np.min(data_slice) + 1e-8)
        # Tensor ảnh Ground Truth chuẩn hóa (1, 512, 512)
        phantom = torch.from_numpy(data_slice).unsqueeze(0).float()
        return phantom

    def _build_limited_angle_operators(self, start_ang, end_ang, num_view, num_detectors):
        """
        Khởi tạo toán tử chiếu Radon (Forward) và FBP (Backward) theo hình học chùm tia quạt (Fan-Beam).
        """
        if odl is None or odl_torch is None:
            raise ImportError("Thiếu thư viện 'odl' để khởi tạo toán tử chiếu tia Radon và FBP.")

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
        # [2026-11-20 T0.4] Tách phần mở rộng bằng os.path.splitext để hỗ trợ cả .IMA, .dcm và .npy thuần cache
        file_stem = os.path.splitext(os.path.basename(slice_path))[0]

        if self.gt_only:
            phantom = self._load_dicom_phantom(slice_path)
            if tuple(phantom.shape[-2:]) != (self.input_size, self.input_size):
                if self.transform is not None:
                    phantom = self.transform(phantom)
                else:
                    phantom = transforms.Resize((self.input_size, self.input_size))(phantom)
            if self.return_path:
                return slice_path, phantom, None, None
            return phantom, None, None

        # -------------------------------------------------------------
        # Nhánh 1: Nạp từ file npy đã tính sẵn (Tối ưu tốc độ huấn luyện)
        # -------------------------------------------------------------
        if self.use_precomputed and self.precomputed_setting:
            sino_file = os.path.join(self.sino_dir, f"{file_stem}.npy")
            fbp_file = os.path.join(self.fbp_dir, f"{file_stem}.npy")

            # Đọc sinogram và ảnh FBP từ file cache
            # Định dạng tensor: sino (1, num_view, num_detectors), fbp_u (1, input_size, input_size)
            sino = torch.from_numpy(np.load(sino_file)).float()
            fbp_u = torch.from_numpy(np.load(fbp_file)).float()
            if sino.ndim == 2:
                sino = sino.unsqueeze(0)
            if fbp_u.ndim == 2:
                fbp_u = fbp_u.unsqueeze(0)

            if self.gt_dir is not None:
                gt_file = os.path.join(self.gt_dir, f"{file_stem}.npy")
                if not os.path.exists(gt_file):
                    raise FileNotFoundError(f"Thiếu tệp Ground Truth cache: {gt_file}")
                # Đọc Ground Truth từ cache .npy dạng (1, input_size, input_size), tuyệt đối không gọi pydicom
                phantom = torch.from_numpy(np.load(gt_file)).float()
                if phantom.ndim == 2:
                    phantom = phantom.unsqueeze(0)
            else:
                phantom = self._load_dicom_phantom(slice_path)
        # -------------------------------------------------------------
        # Nhánh 2: Tính toán động on-the-fly (Dùng khi sinh dữ liệu mới)
        # -------------------------------------------------------------
        else:
            phantom = self._load_dicom_phantom(slice_path)
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
        need_resize_phantom = tuple(phantom.shape[-2:]) != (self.input_size, self.input_size)
        need_resize_fbp = tuple(fbp_u.shape[-2:]) != (self.input_size, self.input_size)
        if self.transform is not None:
            if not self.cache_only or need_resize_phantom:
                phantom = self.transform(phantom)
            if not self.cache_only or need_resize_fbp:
                fbp_u = self.transform(fbp_u)
        elif need_resize_phantom or need_resize_fbp:
            resizer = transforms.Resize((self.input_size, self.input_size))
            if need_resize_phantom:
                phantom = resizer(phantom)
            if need_resize_fbp:
                fbp_u = resizer(fbp_u)

        if self.return_path:
            return slice_path, phantom, fbp_u, sino
        return phantom, fbp_u, sino
