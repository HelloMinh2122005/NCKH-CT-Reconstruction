import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from CTSlice_Provider_LA import LimitedAngleCT_Provider


class LimitedAngleCTDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule quản lý toàn bộ luồng nạp dữ liệu (DataLoaders)
    cho bài toán Tái tạo ảnh CT góc giới hạn (Limited-Angle CT).
    """
    def __init__(
        self,
        data_dir: str,                                                      # Thư mục gốc chứa dữ liệu
        batch_size: int = 4,                                                # Kích thước batch cho mỗi bước huấn luyện
        num_workers: int = 4,                                               # Số tiến trình nạp dữ liệu đa luồng
        setting_tag: str = "limited_ang_120deg_numview_64_size_256_noise_0",# Tên thư mục cấu hình cache dữ liệu
        start_ang: float = -3.1415926535 / 3,                               # Góc chiếu bắt đầu (-60 độ tính theo Radian)
        end_ang: float = 3.1415926535 / 3,                                  # Góc chiếu kết thúc (+60 độ tính theo Radian)
        num_view: int = 64,                                                 # Số góc chiếu trong dải góc giới hạn
        input_size: int = 256,                                              # Kích thước không gian ảnh (256x256)
        poisson_level: float = 0.0,                                         # Mức nhiễu Poisson
        gaussian_level: float = 0.0,                                        # Mức nhiễu Gaussian
        use_precomputed: bool = True                                        # Mặc định đọc từ file .npy đã tiền xử lý để tăng tốc
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.setting_tag = setting_tag
        self.start_ang = start_ang
        self.end_ang = end_ang
        self.num_view = num_view
        self.input_size = input_size
        self.poisson_level = poisson_level
        self.gaussian_level = gaussian_level
        self.use_precomputed = use_precomputed

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
            # 1. Tập Train (8 bệnh nhân huấn luyện)
            self.train_dataset = LimitedAngleCT_Provider(
                base_path=self.data_dir,
                start_ang=self.start_ang,
                end_ang=self.end_ang,
                num_view=self.num_view,
                input_size=self.input_size,
                transform=self.transform,
                poission_level=self.poisson_level,
                gaussian_level=self.gaussian_level,
                use_precomputed=self.use_precomputed,
                precomputed_setting=self.setting_tag,
                test=False,
                valid=False
            )
            # 2. Tập Validation (bệnh nhân L333)
            self.val_dataset = LimitedAngleCT_Provider(
                base_path=self.data_dir,
                start_ang=self.start_ang,
                end_ang=self.end_ang,
                num_view=self.num_view,
                input_size=self.input_size,
                transform=self.transform,
                poission_level=self.poisson_level,
                gaussian_level=self.gaussian_level,
                use_precomputed=self.use_precomputed,
                precomputed_setting=self.setting_tag,
                test=False,
                valid=True
            )

        # Giai đoạn kiểm thử (test)
        if stage == "test" or stage is None:
            # 3. Tập Test (bệnh nhân L310)
            self.test_dataset = LimitedAngleCT_Provider(
                base_path=self.data_dir,
                start_ang=self.start_ang,
                end_ang=self.end_ang,
                num_view=self.num_view,
                input_size=self.input_size,
                transform=self.transform,
                poission_level=self.poisson_level,
                gaussian_level=self.gaussian_level,
                use_precomputed=self.use_precomputed,
                precomputed_setting=self.setting_tag,
                test=True
            )

    def train_dataloader(self):
        """DataLoader cho tập huấn luyện (bật xáo trộn dữ liệu shuffle=True)."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        """DataLoader cho tập validation (không xáo trộn dữ liệu shuffle=False)."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        """DataLoader cho tập kiểm thử (không xáo trộn dữ liệu shuffle=False)."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
