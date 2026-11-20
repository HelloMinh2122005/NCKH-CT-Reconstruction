"""
Các hàm kiểm tra bảo vệ tính toàn vẹn phân chia tập dữ liệu AAPM (T0.6).

Số lượng lát cắt chuẩn đóng băng theo `.lab/20261026-restart-intake/experiments/split.md`:
- Tập huấn luyện (train, 55 bệnh nhân): 8,218 lát cắt
- Tập kiểm định (val, bệnh nhân L110): 133 lát cắt
- Tập kiểm thử (test, 4 bệnh nhân L058/L064/L094/L107): 757 lát cắt
"""

from types import SimpleNamespace
from typing import Dict, Optional, Tuple


# Số lượng lát cắt chuẩn bắt buộc cho bộ dữ liệu AAPM Mayo Clinic LDCT
EXPECTED_AAPM_SPLIT_COUNTS: Dict[str, int] = {
    "train": 8218,
    "val": 133,
    "test": 757,
}

_STAGE_SPLITS: Dict[Optional[str], Tuple[str, ...]] = {
    "fit": ("train", "val"),
    "test": ("test",),
    None: ("train", "val", "test"),
    "all": ("train", "val", "test"),
}


def assert_aapm_split_counts(
    datamodule,
    stage: Optional[str],
    allow_custom_split: bool = False,
    dataset_type: str = "aapm",
) -> Dict[str, Optional[int]]:
    """
    Khẳng định số lượng lát cắt trên DataModule khớp chính xác chuẩn AAPM 8218 / 133 / 757.

    Tham số:
        datamodule: Đối tượng DataModule đã gọi setup(stage) có các thuộc tính *_dataset.
        stage: Giai đoạn thực thi ('fit', 'test', None hoặc 'all').
        allow_custom_split: Cho phép bỏ qua kiểm tra nghiêm ngặt khi chạy smoke test nhỏ có chủ đích.
        dataset_type: Loại tập dữ liệu (chỉ áp dụng kiểm tra cho 'aapm').

    Trả về:
        Từ điển ánh xạ tên phân chia -> số lượng lát cắt thực tế (hoặc {} nếu không phải 'aapm').

    Ném ngoại lệ:
        ValueError: Nếu stage không hợp lệ.
        AssertionError: Nếu bất kỳ phân chia nào lệch khỏi 8218 / 133 / 757 khi allow_custom_split=False.
    """
    # Bỏ qua kiểm tra AAPM nếu đang chạy bộ dữ liệu khác (LIDC-IDRI hoặc DeepLesion)
    ds_type = (dataset_type or "aapm").strip().lower()
    if ds_type != "aapm":
        return {}

    stage_key = stage.lower() if isinstance(stage, str) else stage
    if stage_key not in _STAGE_SPLITS:
        raise ValueError(
            f"Tham số stage='{stage}' không hợp lệ cho assert_aapm_split_counts "
            f"(chỉ chấp nhận 'fit', 'test', 'all' hoặc None)."
        )

    target_splits = _STAGE_SPLITS[stage_key]
    actual_counts: Dict[str, Optional[int]] = {}
    mismatches = []

    for split_name in target_splits:
        dataset = getattr(datamodule, f"{split_name}_dataset", None)
        actual = len(dataset) if dataset is not None else None
        actual_counts[split_name] = actual
        expected = EXPECTED_AAPM_SPLIT_COUNTS[split_name]
        if actual != expected:
            mismatches.append(f"{split_name}_dataset={actual} (kỳ vọng {expected})")

    if allow_custom_split:
        print(
            f"[WARN] --allow_custom_split được bật: bỏ qua kiểm tra nghiêm ngặt split AAPM "
            f"8218/133/757 ở stage='{stage}' (thực tế: {actual_counts})."
        )
        return actual_counts

    if mismatches:
        raise AssertionError(
            f"CRITICAL SPLIT GUARD ERROR (stage='{stage}'): Số lượng lát cắt AAPM không khớp "
            f"chuẩn đóng băng trong experiments/split.md (8218/133/757)! Sai lệch: {', '.join(mismatches)}. "
            f"Thực tế: {actual_counts}. Kiểm tra lại --train_patients / --val_patients / --test_patients "
            f"(hoặc truyền --allow_custom_split nếu đang chạy smoke test nhỏ có chủ đích)."
        )

    print(f"[SPLIT GUARD OK] AAPM stage='{stage}': {actual_counts}")
    return actual_counts


def assert_aapm_test_dataset_count(
    test_dataset,
    allow_custom_split: bool = False,
    dataset_type: str = "aapm",
) -> Optional[int]:
    """
    Kiểm tra trực tiếp số lượng lát cắt tập kiểm thử AAPM (kỳ vọng 757) cho các script không dùng DataModule.

    Tham số:
        test_dataset: Đối tượng Dataset kiểm thử (ví dụ LimitedAngleCT_Provider trong evaluate_fbp.py).
        allow_custom_split: Cho phép bỏ qua khi chạy thử nghiệm tập nhỏ có chủ đích.
        dataset_type: Tên tập dữ liệu (mặc định 'aapm').

    Trả về:
        Số lát cắt kiểm thử thực tế (hoặc None nếu không phải 'aapm').
    """
    counts = assert_aapm_split_counts(
        SimpleNamespace(test_dataset=test_dataset),
        stage="test",
        allow_custom_split=allow_custom_split,
        dataset_type=dataset_type,
    )
    return counts.get("test")
