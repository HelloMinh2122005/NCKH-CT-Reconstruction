import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from CTSlice_Provider_LA import LimitedAngleCT_Provider


class LimitedAngleCTDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for Limited-Angle CT experiments.
    """
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 4,
        num_workers: int = 4,
        setting_tag: str = "limited_ang_120deg_numview_64_size_256_noise_0",
        start_ang: float = -3.1415926535 / 3,
        end_ang: float = 3.1415926535 / 3,
        num_view: int = 64,
        input_size: int = 256,
        poisson_level: float = 0.0,
        gaussian_level: float = 0.0,
        use_precomputed: bool = True
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

        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size))
        ])

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
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

        if stage == "test" or stage is None:
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
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
