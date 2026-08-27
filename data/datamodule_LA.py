import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# Import Provider dữ liệu linh hoạt (hỗ trợ cả khi chạy từ thư mục gốc hoặc trong subfolder)
try:
    from data.CTSlice_Provider_LA import LimitedAngleCT_Provider
except ImportError:
    from CTSlice_Provider_LA import LimitedAngleCT_Provider


class LimitedAngleCTDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule quản lý toàn bộ luồng nạp dữ liệu (DataLoaders)
    cho bài toán Tái tạo ảnh CT góc giới hạn (Limited-Angle CT).
    
    DataModule này chịu trách nhiệm:
    1. Quản lý phân chia tập Train (8 bệnh nhân), Validation (L333), và Test (L310) theo chuẩn AAPM LDCT.
    2. Nạp song song dữ liệu ảnh gốc DICOM (Ground Truth) và dữ liệu sinogram/FBP đã tính sẵn (.npy cache).
    3. Cung cấp các DataLoader tối ưu bộ nhớ GPU với cờ pin_memory và đa luồng num_workers.
    """
    def __init__(
        self,
        dicom_dir: str = "/datastore/uittogether3/LuuTru/Thanhld/CT-Reconstruction/split/", # Thư mục chứa ảnh DICOM (.IMA) gốc
        cache_dir: str = "/datastore/uittogether3/LuuTru/MinhPD/dataset/limited_angle/",   # Thư mục chứa file cache .npy (sino và fbp_u)
        data_dir: str = None,                                               # Tham số tương thích ngược nếu truyền data_dir chung
        batch_size: int = 1,                                                # Kích thước batch (mặc định = 1 lát cắt mỗi step)
        num_workers: int = 4,                                               # Số luồng CPU nạp dữ liệu song song
        setting_tag: str = "limited_ang_120deg_numview_64_size_256_noise_0",# Tên định danh thư mục cache tương ứng với cấu hình vật lý
        start_ang: float = -3.1415926535 / 3,                               # Góc quét bắt đầu tính theo Radian (-60 độ = -pi/3)
        end_ang: float = 3.1415926535 / 3,                                  # Góc quét kết thúc tính theo Radian (+60 độ = +pi/3)
        num_view: int = 64,                                                 # Số góc chiếu (projection views) trong dải quét giới hạn
        num_detectors: int = 512,                                           # Số lượng phần tử cảm biến trên thanh detector
        input_size: int = 256,                                              # Kích thước không gian ảnh tái tạo (256x256 pixel)
        poisson_level: float = 0.0,                                         # Mức photon mô phỏng nhiễu Poisson (0 = không thêm nhiễu)
        gaussian_level: float = 0.0,                                        # Độ lệch chuẩn mô phỏng nhiễu Gaussian (0 = không thêm nhiễu)
        use_precomputed: bool = True,                                       # Đọc trực tiếp từ file cache .npy để đạt tốc độ tối đa
        train_patients: list = None,                                        # Danh sách mã bệnh nhân cho tập huấn luyện (Train)
        val_patients: list = None,                                          # Danh sách mã bệnh nhân cho tập kiểm định (Validation)
        test_patients: list = None                                          # Danh sách mã bệnh nhân cho tập kiểm thử độc lập (Test)
    ):
        super().__init__()
        
        # Xử lý tương thích ngược: tự động phát hiện nếu người dùng truyền một thư mục dữ liệu tổng quát
        if data_dir is not None:
            if os.path.exists(os.path.join(data_dir, "train", "L067")):
                dicom_dir = data_dir
            else:
                cache_dir = data_dir

        self.dicom_dir = dicom_dir
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.setting_tag = setting_tag
        self.start_ang = start_ang
        self.end_ang = end_ang
        self.num_view = num_view
        self.num_detectors = num_detectors
        self.input_size = input_size
        self.poisson_level = poisson_level
        self.gaussian_level = gaussian_level
        self.use_precomputed = use_precomputed
        self.train_patients = train_patients
        self.val_patients = val_patients
        self.test_patients = test_patients

        # Pipeline biến đổi ảnh: Đảm bảo độ phân giải không gian luôn khớp với input_size (vd: 256x256)
        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size))
        ])

    def setup(self, stage=None):
        """
        Khởi tạo các tập Dataset tương ứng với từng giai đoạn thực thi (fit: Train/Val, test: Test).
        Được PyTorch Lightning tự động gọi trước khi bắt đầu huấn luyện hoặc đánh giá.
        """
        # Giai đoạn Huấn luyện & Kiểm định (fit stage)
        if stage == "fit" or stage is None:
            # 1. Khởi tạo Dataset Huấn luyện (Train Dataset)
            self.train_dataset = LimitedAngleCT_Provider(
                dicom_dir=self.dicom_dir,
                cache_dir=self.cache_dir,
                start_ang=self.start_ang,
                end_ang=self.end_ang,
                num_view=self.num_view,
                num_detectors=self.num_detectors,
                input_size=self.input_size,
                transform=self.transform,
                poission_level=self.poisson_level,
                gaussian_level=self.gaussian_level,
                use_precomputed=self.use_precomputed,
                precomputed_setting=self.setting_tag,
                patient_list=self.train_patients,
                test=False,
                valid=False
            )
            # 2. Khởi tạo Dataset Kiểm định (Validation Dataset - mặc định bệnh nhân L333)
            self.val_dataset = LimitedAngleCT_Provider(
                dicom_dir=self.dicom_dir,
                cache_dir=self.cache_dir,
                start_ang=self.start_ang,
                end_ang=self.end_ang,
                num_view=self.num_view,
                num_detectors=self.num_detectors,
                input_size=self.input_size,
                transform=self.transform,
                poission_level=self.poisson_level,
                gaussian_level=self.gaussian_level,
                use_precomputed=self.use_precomputed,
                precomputed_setting=self.setting_tag,
                patient_list=self.val_patients,
                test=False,
                valid=True if self.val_patients is None else False
            )

        # Giai đoạn Kiểm thử độc lập (test stage)
        if stage == "test" or stage is None:
            # 3. Khởi tạo Dataset Kiểm thử (Test Dataset - mặc định bệnh nhân L310)
            self.test_dataset = LimitedAngleCT_Provider(
                dicom_dir=self.dicom_dir,
                cache_dir=self.cache_dir,
                start_ang=self.start_ang,
                end_ang=self.end_ang,
                num_view=self.num_view,
                num_detectors=self.num_detectors,
                input_size=self.input_size,
                transform=self.transform,
                poission_level=self.poisson_level,
                gaussian_level=self.gaussian_level,
                use_precomputed=self.use_precomputed,
                precomputed_setting=self.setting_tag,
                patient_list=self.test_patients,
                test=True
            )

    def train_dataloader(self):
        """
        DataLoader cho tập huấn luyện:
        - Bật xáo trộn dữ liệu (shuffle=True) để mô hình học tổng quát hóa.
        - Bật pin_memory khi có GPU để tăng tốc độ chuyển dữ liệu từ RAM CPU sang VRAM GPU.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )

    def val_dataloader(self):
        """
        DataLoader cho tập kiểm định:
        - Không xáo trộn dữ liệu (shuffle=False) để theo dõi trực quan cố định qua các Epoch.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )

    def test_dataloader(self):
        """
        DataLoader cho tập kiểm thử độc lập:
        - Không xáo trộn dữ liệu (shuffle=False) để đo lường benchmark PSNR, SSIM chuẩn xác.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )


# Alias tương thích ngược với các script cũ
CTDataModule_LA = LimitedAngleCTDataModule
