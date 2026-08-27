import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl

try:
    from data.CTSlice_Provider_LA import LimitedAngleCT_Provider
except ImportError:
    from CTSlice_Provider_LA import LimitedAngleCT_Provider


class LimitedAngleCTDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule quản lý toàn bộ luồng nạp dữ liệu (DataLoaders)
    cho bài toán Tái tạo ảnh CT góc giới hạn (Limited-Angle CT).
    """
    def __init__(
        self,
        dicom_dir: str = "/datastore/uittogether3/LuuTru/Thanhld/CT-Reconstruction/split/",
        cache_dir: str = "/datastore/uittogether3/LuuTru/MinhPD/dataset/limited_angle/",
        data_dir: str = None,                                               # Tương thích ngược nếu truyền data_dir
        batch_size: int = 1,                                                # Kích thước batch cho mỗi bước huấn luyện
        num_workers: int = 4,                                               # Số tiến trình nạp dữ liệu đa luồng
        setting_tag: str = "limited_ang_120deg_numview_64_size_256_noise_0",# Tên thư mục cấu hình cache dữ liệu
        start_ang: float = -3.1415926535 / 3,                               # Góc chiếu bắt đầu (-60 độ tính theo Radian)
        end_ang: float = 3.1415926535 / 3,                                  # Góc chiếu kết thúc (+60 độ tính theo Radian)
        num_view: int = 64,                                                 # Số góc chiếu trong dải góc giới hạn
        num_detectors: int = 512,                                           # Số lượng cảm biến detector
        input_size: int = 256,                                              # Kích thước không gian ảnh (256x256)
        poisson_level: float = 0.0,                                         # Mức nhiễu Poisson
        gaussian_level: float = 0.0,                                        # Mức nhiễu Gaussian
        use_precomputed: bool = True,                                       # Đọc từ cache .npy để tăng tốc
        train_patients: list = None,
        val_patients: list = None,
        test_patients: list = None
    ):
        super().__init__()
        # Xử lý tương thích ngược
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

        # Pipeline tiền xử lý: Resize ảnh về kích thước mong muốn
        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size))
        ])

    def setup(self, stage=None):
        """
        Khởi tạo các tập Dataset tương ứng với từng giai đoạn (fit: train/val, test: test).
        """
        # Giai đoạn huấn luyện và kiểm định (fit)
        if stage == "fit" or stage is None:
            # 1. Tập Train
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
            # 2. Tập Validation
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

        # Giai đoạn kiểm thử (test)
        if stage == "test" or stage is None:
            # 3. Tập Test
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
        """DataLoader cho tập huấn luyện (shuffle=True)."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )

    def val_dataloader(self):
        """DataLoader cho tập validation (shuffle=False)."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )

    def test_dataloader(self):
        """DataLoader cho tập kiểm thử (shuffle=False)."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )


# Alias tương thích ngược
CTDataModule_LA = LimitedAngleCTDataModule
