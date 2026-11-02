import os
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import numpy as np
import torch
import pytorch_lightning as pl

# Tương thích PyTorch 2.6+
if hasattr(torch, "load"):
    _orig_torch_load = torch.load
    def _safe_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)
    torch.load = _safe_torch_load

from data.datamodule_factory import get_datamodule
from baselines.DuDoTrans.models import DuDoTrans_LA
from baselines.eval_utils import find_best_val_checkpoint, run_slice_by_slice_evaluation


def parse_args():
    parser = argparse.ArgumentParser(description="Kiểm thử và đánh giá mô hình Baseline DuDoTrans trên Limited-Angle CT")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Đường dẫn đến tệp trọng số checkpoint (.ckpt) hoặc thư mục chứa checkpoint để tự động chọn best val_psnr")
    parser.add_argument("--dataset_type", type=str, default="aapm", choices=["aapm", "deeplesion", "lidc"])
    parser.add_argument("--dicom_dir", type=str, default="/datastore/uittogether3/LuuTru/Thanhld/CT-Reconstruction/split/")
    parser.add_argument("--cache_dir", type=str, default="/datastore/uittogether3/LuuTru/MinhPD/dataset/aapm/limited_angle/")
    parser.add_argument("--angle_range_deg", type=float, default=90.0)
    parser.add_argument("--num_view", type=int, default=64)
    parser.add_argument("--num_detectors", type=int, default=512)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--poisson_level", type=float, default=0.0)
    parser.add_argument("--gaussian_level", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--test_patients",
        nargs="+",
        default=["L058", "L064", "L094", "L107"],
        help="Mã bệnh nhân kiểm thử độc lập (mặc định: L058 L064 L094 L107 theo split.md)"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Đường dẫn tệp CSV lưu kết quả đánh giá chi tiết theo từng lát cắt"
    )
    parser.add_argument(
        "--job_id",
        type=str,
        default=None,
        help="Mã Slurm Job ID phục vụ truy xuất nguồn gốc (mặc định: lấy từ biến môi trường SLURM_JOB_ID)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="DuDoTrans",
        help="Tên phương pháp đánh giá (mặc định: DuDoTrans)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # In thông tin Git commit phục vụ kiểm chứng nguồn gốc
    try:
        import subprocess
        git_hash = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        git_hash = "<unknown>"
    print(f"[INFO] Git rev-parse HEAD: {git_hash}")

    job_id = args.job_id or os.environ.get("SLURM_JOB_ID", "local")
    print(f"[INFO] Slurm Job ID: {job_id}")

    angle_range_deg = args.angle_range_deg
    start_ang = np.deg2rad(-angle_range_deg / 2.0)
    end_ang = np.deg2rad(angle_range_deg / 2.0)

    # Lựa chọn checkpoint: nếu truyền vào thư mục hoặc không truyền, tự động tìm best val_psnr
    if args.ckpt_path is None or os.path.isdir(args.ckpt_path):
        search_dir = args.ckpt_path if args.ckpt_path else os.path.join(PROJECT_ROOT, f"saved_models/DuDoTrans_{int(angle_range_deg)}")
        ckpt_path = find_best_val_checkpoint(search_dir)
    else:
        ckpt_path = args.ckpt_path

    setting_tag = f"limited_ang_{int(angle_range_deg)}deg_numview_{args.num_view}_size_{args.input_size}"
    if args.poisson_level > 0:
        setting_tag += f"_poisson_{int(args.poisson_level)}"
    else:
        setting_tag += "_noise_0"

    print("=" * 80)
    print("🔍 ĐÁNH GIÁ MÔ HÌNH BASELINE: DuDoTrans (Dual-Domain Transformer)")
    print(f"- Tệp checkpoint: {ckpt_path}")
    print(f"- Tập dữ liệu: {args.dataset_type.upper()} | Dải góc: {angle_range_deg}° (Views: {args.num_view})")
    print(f"- Bệnh nhân kiểm thử: {args.test_patients}")
    if args.output_csv:
        print(f"- File CSV xuất kết quả per-slice: {args.output_csv}")
    print("=" * 80)

    dm = get_datamodule(
        dataset_type=args.dataset_type,
        dicom_dir=args.dicom_dir,
        cache_dir=args.cache_dir,
        setting_tag=setting_tag,
        start_ang=start_ang,
        end_ang=end_ang,
        num_view=args.num_view,
        num_detectors=args.num_detectors,
        input_size=args.input_size,
        poisson_level=args.poisson_level,
        gaussian_level=args.gaussian_level,
        use_precomputed=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_patients=args.test_patients,
    )

    model = DuDoTrans_LA.load_from_checkpoint(
        ckpt_path,
        num_view=args.num_view,
        num_detectors=args.num_detectors,
        start_ang=start_ang,
        end_ang=end_ang,
        input_size=args.input_size,
    )
    model.eval()

    # Thực hiện đánh giá chi tiết theo từng lát cắt và xuất CSV nếu có yêu cầu
    if args.output_csv:
        run_slice_by_slice_evaluation(
            model=model,
            datamodule=dm,
            method_name=args.method,
            angle_deg=angle_range_deg,
            output_csv=args.output_csv,
            job_id=job_id,
        )
    else:
        trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
        )
        results = trainer.test(model, datamodule=dm)
        print("\n📊 KẾT QUẢ ĐÁNH GIÁ ĐỊNH LƯỢNG (DuDoTrans):")
        print(results)


if __name__ == "__main__":
    main()
