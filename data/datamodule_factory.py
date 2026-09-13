import os
import sys
from pathlib import Path

# Đảm bảo thư mục gốc dự án luôn nằm trong sys.path để import an toàn từ mọi vị trí
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.datamodule_LA import LimitedAngleCTDataModule
from data.nih_deep_lesion.datamodule_LA_DeepLesion import LimitedAngleCT_DeepLesion_DataModule
from data.lidc_idri.datamodule_LA_LIDC import LimitedAngleCT_LIDC_DataModule


def get_datamodule(
    dataset_type: str = "aapm",
    dicom_dir: str = None,
    cache_dir: str = None,
    data_dir: str = None,
    split_dir: str = None,
    setting_tag: str = "limited_ang_120deg_numview_64_size_256_noise_0",
    start_ang: float = -3.1415926535 / 3,
    end_ang: float = 3.1415926535 / 3,
    num_view: int = 64,
    num_detectors: int = 512,
    input_size: int = 256,
    poisson_level: float = 0.0,
    gaussian_level: float = 0.0,
    use_precomputed: bool = True,
    batch_size: int = 1,
    num_workers: int = 4,
    train_patients: list = None,
    val_patients: list = None,
    test_patients: list = None,
):
    """
    Nhà máy khởi tạo DataModule (DataModule Factory) thống nhất cho bài toán Limited-Angle CT.
    
    Hỗ trợ chuyển đổi linh hoạt giữa 3 tập dữ liệu chuẩn:
    1. 'aapm': AAPM Mayo Clinic Low-Dose CT (2016) - Chuẩn tái tạo ảnh lát cắt bụng/ngực.
    2. 'deeplesion': NIH DeepLesion CT Dataset - Chuyên sâu về phát hiện và bù đắp thương tổn mô mềm.
    3. 'lidc': LIDC-IDRI Lung CT Dataset - Ảnh CT phổi/lồng ngực với độ chi tiết phế quản cao.
    
    Tham số:
        dataset_type: Tên loại tập dữ liệu ('aapm', 'deeplesion', 'lidc').
        dicom_dir: Thư mục chứa ảnh gốc DICOM hoặc PNG.
        cache_dir: Thư mục chứa sinogram và FBP .npy tiền xử lý.
        setting_tag: Chuỗi định danh cấu hình góc và phân giải (ví dụ: limited_ang_120deg_numview_64_size_256_noise_0).
        start_ang, end_ang: Góc quét bắt đầu và kết thúc tính bằng Radian.
        num_view: Số lượng góc chiếu (views).
        num_detectors: Số kênh cảm biến detector (mặc định: 512).
        input_size: Kích thước ảnh vuông (mặc định: 256).
        poisson_level, gaussian_level: Mức nhiễu mô phỏng (mặc định: 0.0).
        use_precomputed: Bật chế độ đọc trực tiếp từ cache .npy để tăng tốc tối đa.
        batch_size: Kích thước mini-batch (mặc định: 1).
        num_workers: Số luồng CPU tải dữ liệu song song.
    
    Trả về:
        pl.LightningDataModule tương ứng sẵn sàng truyền vào Trainer.fit().
    """
    dataset_type = (dataset_type or "aapm").lower()

    if dataset_type in ["deeplesion", "deep_lesion", "nih_deep_lesion"]:
        # Tự động gán đường dẫn cache chuẩn của DeepLesion nếu người dùng dùng path mặc định cũ của AAPM
        if cache_dir is None or "dataset/limited_angle" in cache_dir or "dataset/aapm" in cache_dir:
            cache_dir = "/datastore/uittogether3/LuuTru/MinhPD/dataset/nih_deep_lesion/limited_angle/"
        return LimitedAngleCT_DeepLesion_DataModule(
            data_dir=data_dir or "/datastore/uittogether3/LuuTru/Thanhld/CT-Reconstruction/minideeplesion/minideeplesion/",
            split_dir=split_dir or "/datastore/uittogether3/LuuTru/Thanhld/CT-Reconstruction/split_dl/",
            cache_dir=cache_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            setting_tag=setting_tag,
            start_ang=start_ang,
            end_ang=end_ang,
            num_view=num_view,
            num_detectors=num_detectors,
            input_size=input_size,
            poisson_level=poisson_level,
            gaussian_level=gaussian_level,
            use_precomputed=use_precomputed,
        )

    elif dataset_type in ["lidc", "lidc_idri"]:
        # Tự động gán đường dẫn dicom và cache chuẩn của LIDC-IDRI nếu người dùng dùng path mặc định cũ của AAPM
        if dicom_dir is None or "CT-Reconstruction/split" in dicom_dir or "dataset/aapm" in dicom_dir:
            dicom_dir = "/datastore/uittogether3/LuuTru/MinhPD/dataset/lidc_idri/dicom/"
        if cache_dir is None or "dataset/limited_angle" in cache_dir or "dataset/aapm" in cache_dir:
            cache_dir = "/datastore/uittogether3/LuuTru/MinhPD/dataset/lidc_idri/limited_angle/"
        return LimitedAngleCT_LIDC_DataModule(
            dicom_dir=dicom_dir,
            cache_dir=cache_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            setting_tag=setting_tag,
            start_ang=start_ang,
            end_ang=end_ang,
            num_view=num_view,
            num_detectors=num_detectors,
            input_size=input_size,
            poisson_level=poisson_level,
            gaussian_level=gaussian_level,
            use_precomputed=use_precomputed,
        )

    else:
        # Mặc định: Bộ dữ liệu chuẩn AAPM Mayo Clinic 2016
        if cache_dir is None:
            cache_dir = "/datastore/uittogether3/LuuTru/MinhPD/dataset/aapm/limited_angle/"
        return LimitedAngleCTDataModule(
            dicom_dir=dicom_dir or "/datastore/uittogether3/LuuTru/Thanhld/CT-Reconstruction/split/",
            cache_dir=cache_dir,
            setting_tag=setting_tag,
            start_ang=start_ang,
            end_ang=end_ang,
            num_view=num_view,
            num_detectors=num_detectors,
            input_size=input_size,
            poisson_level=poisson_level,
            gaussian_level=gaussian_level,
            use_precomputed=use_precomputed,
            batch_size=batch_size,
            num_workers=num_workers,
            train_patients=train_patients,
            val_patients=val_patients,
            test_patients=test_patients,
        )
