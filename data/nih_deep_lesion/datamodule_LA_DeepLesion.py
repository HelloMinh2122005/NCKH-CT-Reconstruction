import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# Import Provider dữ liệu DeepLesion linh hoạt
try:
    from data.nih_deep_lesion.CTSlice_Provider_LA_DeepLesion import LimitedAngleCT_DeepLesion_Provider
except ImportError:
    from CTSlice_Provider_LA_DeepLesion import LimitedAngleCT_DeepLesion_Provider


class LimitedAngleCT_DeepLesion_DataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule quản lý toàn bộ luồng nạp dữ liệu (DataLoaders)
    cho bài toán Tái tạo ảnh CT góc giới hạn (Limited-Angle CT) trên bộ dữ liệu NIH DeepLesion.
    
    DataModule này chịu trách nhiệm:
    1. Quản lý phân chia tập Train (1,999 lát cắt), Validation (199 lát cắt) và Test (299 lát cắt) từ file CSV.
    2. Nạp dữ liệu ảnh gốc 16-bit PNG (Ground Truth) và dữ liệu sinogram/FBP đã tính sẵn (.npy cache).
    3. Cung cấp các DataLoader tối ưu bộ nhớ GPU với cờ pin_memory và đa luồng num_workers.
    """
    def __init__(
        self,
        data_dir: str = "/datastore/uittogether3/LuuTru/Thanhld/CT-Reconstruction/minideeplesion/minideeplesion/",
        split_dir: str = "/datastore/uittogether3/LuuTru/Thanhld/CT-Reconstruction/split_dl/",
        cache_dir: str = "/datastore/uittogether3/LuuTru/MinhPD/dataset/nih_deep_lesion/limited_angle/",
        batch_size: int = 1,
        num_workers: int = 4,
        setting_tag: str = "limited_ang_120deg_numview_64_size_256_noise_0",
        start_ang: float = -3.1415926535 / 3,
        end_ang: float = 3.1415926535 / 3,
        num_view: int = 64,
        num_detectors: int = 512,
        input_size: int = 256,
        poisson_level: float = 0.0,
        gaussian_level: float = 0.0,
        use_precomputed: bool = True
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split_dir = split_dir
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

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str = None):
        """
        Khởi tạo các đối tượng Dataset cho từng giai đoạn fit/test.
        """
        if stage == "fit" or stage is None:
            # 1. Khởi tạo tập Huấn luyện (Train)
            self.train_dataset = LimitedAngleCT_DeepLesion_Provider(
                data_dir=self.data_dir,
                split_dir=self.split_dir,
                cache_dir=self.cache_dir,
                start_ang=self.start_ang,
                end_ang=self.end_ang,
                num_view=self.num_view,
                num_detectors=self.num_detectors,
                poission_level=self.poisson_level,
                gaussian_level=self.gaussian_level,
                test=False,
                valid=False,
                input_size=self.input_size,
                use_precomputed=self.use_precomputed,
                precomputed_setting=self.setting_tag
            )

            # 2. Khởi tạo tập Kiểm định (Validation)
            self.val_dataset = LimitedAngleCT_DeepLesion_Provider(
                data_dir=self.data_dir,
                split_dir=self.split_dir,
                cache_dir=self.cache_dir,
                start_ang=self.start_ang,
                end_ang=self.end_ang,
                num_view=self.num_view,
                num_detectors=self.num_detectors,
                poission_level=self.poisson_level,
                gaussian_level=self.gaussian_level,
                test=False,
                valid=True,
                input_size=self.input_size,
                use_precomputed=self.use_precomputed,
                precomputed_setting=self.setting_tag
            )

        if stage == "test" or stage is None:
            # 3. Khởi tạo tập Kiểm thử Độc lập (Test)
            self.test_dataset = LimitedAngleCT_DeepLesion_Provider(
                data_dir=self.data_dir,
                split_dir=self.split_dir,
                cache_dir=self.cache_dir,
                start_ang=self.start_ang,
                end_ang=self.end_ang,
                num_view=self.num_view,
                num_detectors=self.num_detectors,
                poission_level=self.poisson_level,
                gaussian_level=self.gaussian_level,
                test=True,
                valid=False,
                input_size=self.input_size,
                use_precomputed=self.use_precomputed,
                precomputed_setting=self.setting_tag
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
