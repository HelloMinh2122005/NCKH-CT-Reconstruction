#!/usr/bin/env python3
# ==============================================================================
# KIỂM THỬ ĐƠN VỊ CHO CÁC CỜ BIẾN THỂ LOẠI BỎ THÀNH PHẦN (ABLATION SWITCHES)
# CỦA MÔ HÌNH SOLAR_DualMamba (--ablate {none, a, b, c, d})
# Tuân thủ: spec-1.md, spec-1-decisions.md (GT-02 / Wave B / P5.1)
# ==============================================================================

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Kiểm tra sự sẵn có của PyTorch trong môi trường thực thi
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import pytorch_lightning as pl
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # Giả lập các module phụ thuộc cụm để CLI test có thể chạy cục bộ
    for mod in [
        "torch",
        "torch.nn",
        "torch.nn.functional",
        "torchvision",
        "pytorch_lightning",
        "pytorch_lightning.callbacks",
        "pytorch_lightning.loggers",
        "torchmetrics",
        "torchmetrics.functional",
        "torchmetrics.functional.image",
        "odl",
        "odl.contrib",
        "odl.contrib.torch",
        "data",
        "data.datamodule_factory",
    ]:
        if mod not in sys.modules:
            sys.modules[mod] = MagicMock()


class TestSolarAblateCLI(unittest.TestCase):
    """Kiểm thử đơn vị tham số dòng lệnh --ablate trong train và test script."""

    def test_train_cli_default_ablate_is_none(self):
        """Xác minh cờ --ablate trong train_solar_dualmamba_la mặc định là 'none'."""
        from baselines.SOLAR_DualMamba.train_solar_dualmamba_la import parse_args

        with patch.object(sys, "argv", ["train_solar_dualmamba_la.py"]):
            args = parse_args()
            self.assertTrue(hasattr(args, "ablate"))
            self.assertEqual(args.ablate, "none")

    def test_train_cli_valid_ablate_choices(self):
        """Xác minh train_solar_dualmamba_la chấp nhận toàn bộ các lựa chọn ['none', 'a', 'b', 'c', 'd']."""
        from baselines.SOLAR_DualMamba.train_solar_dualmamba_la import parse_args

        for choice in ["none", "a", "b", "c", "d"]:
            with patch.object(sys, "argv", ["train_solar_dualmamba_la.py", "--ablate", choice]):
                args = parse_args()
                self.assertEqual(args.ablate, choice)

    def test_train_cli_invalid_ablate_rejected(self):
        """Xác minh train_solar_dualmamba_la từ chối lựa chọn ablate không hợp lệ."""
        from baselines.SOLAR_DualMamba.train_solar_dualmamba_la import parse_args

        with patch.object(sys, "argv", ["train_solar_dualmamba_la.py", "--ablate", "invalid_choice"]):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_test_cli_default_ablate_is_none(self):
        """Xác minh cờ --ablate trong test_solar_dualmamba_la mặc định là 'none'."""
        from baselines.SOLAR_DualMamba.test_solar_dualmamba_la import parse_args

        with patch.object(sys, "argv", ["test_solar_dualmamba_la.py"]):
            args = parse_args()
            self.assertTrue(hasattr(args, "ablate"))
            self.assertEqual(args.ablate, "none")

    def test_test_cli_valid_ablate_choices(self):
        """Xác minh test_solar_dualmamba_la chấp nhận các lựa chọn ['none', 'a', 'b', 'c', 'd']."""
        from baselines.SOLAR_DualMamba.test_solar_dualmamba_la import parse_args

        for choice in ["none", "a", "b", "c", "d"]:
            with patch.object(sys, "argv", ["test_solar_dualmamba_la.py", "--ablate", choice]):
                args = parse_args()
                self.assertEqual(args.ablate, choice)

    def test_test_cli_invalid_ablate_rejected(self):
        """Xác minh test_solar_dualmamba_la từ chối lựa chọn ablate không hợp lệ."""
        from baselines.SOLAR_DualMamba.test_solar_dualmamba_la import parse_args

        with patch.object(sys, "argv", ["test_solar_dualmamba_la.py", "--ablate", "unknown"]):
            with self.assertRaises(SystemExit):
                parse_args()


# ==============================================================================
# BỘ KIỂM THỬ U1 - U8 DÀNH CHO MÔ HÌNH (spec-1 §6)
# Chạy khi môi trường có PyTorch (trên HPC-UIT hoặc env có PyTorch CPU)
# ==============================================================================

class LinearStubOperator(nn.Module if HAS_TORCH else object):
    """Toán tử tuyến tính thay thế Radon A và FBP F khi không có thư viện ODL / ASTRA."""
    def __init__(self, in_shape, out_shape):
        if HAS_TORCH:
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(out_shape)

    def forward(self, x):
        # x: (B, 1, H_in, W_in) -> (B, 1, H_out, W_out)
        return self.pool(x)


def create_tiny_solar_model(ablate="none", seed=42):
    """Tạo mô hình SOLAR_DualMamba_LA với cấu hình siêu nhỏ (Tiny config) từ spec-1 §6."""
    from baselines.SOLAR_DualMamba.models import SOLAR_DualMamba_LA

    if HAS_TORCH:
        torch.manual_seed(seed)

    # Monkeypatch hàm radon_transform để chạy thuần CPU mà không cần ODL / ASTRA
    orig_radon_transform = SOLAR_DualMamba_LA.radon_transform

    def stub_radon_transform(self_model, num_view=8, start_ang=0, end_ang=0, num_detectors=64, input_size=32):
        # A: (B, 1, 32, 32) -> (B, 1, 8, 64)
        stub_a = LinearStubOperator((32, 32), (num_view, num_detectors))
        # F: (B, 1, 8, 64) -> (B, 1, 32, 32)
        stub_f = LinearStubOperator((num_view, num_detectors), (input_size, input_size))
        return stub_a, stub_f

    SOLAR_DualMamba_LA.radon_transform = stub_radon_transform
    try:
        model = SOLAR_DualMamba_LA(
            n_iterations=2,
            cg_iters=2,
            num_view=8,
            num_detectors=64,
            input_size=32,
            feature_dim=8,
            window_size=8,
            sino_loss_weight=0.1,
            initial_lr=1e-4,
            final_lr=1e-5,
            ablate=ablate,
        )
    finally:
        SOLAR_DualMamba_LA.radon_transform = orig_radon_transform

    return model


@unittest.skipUnless(HAS_TORCH, "PyTorch không có trong môi trường cục bộ; kiểm thử U1-U8 sẽ chạy trên HPC-UIT")
class TestSolarAblateModelU1toU8(unittest.TestCase):
    """
    Bộ kiểm thử đơn vị U1 - U8 từ spec-1.md §6.
    Cấu hình tiny: B=1, input_size=32, num_view=8, num_detectors=64, feature_dim=8, window_size=8, K=2, cg_iters=2.
    """

    def test_u1_shape_every_switch(self):
        """U1: Shape contract cho mọi switch {none, a, b, c, d} -> x: (1,1,32,32), y_hat: (1,1,8,64), float32."""
        for switch in ["none", "a", "b", "c", "d"]:
            model = create_tiny_solar_model(ablate=switch, seed=42)
            model.eval()
            x_fbp = torch.randn(1, 1, 32, 32, dtype=torch.float32)
            sino = torch.randn(1, 1, 8, 64, dtype=torch.float32)
            with torch.no_grad():
                x_recon, y_hat = model(x_fbp, sino)

            self.assertEqual(x_recon.shape, (1, 1, 32, 32), f"Lỗi shape ảnh tái tạo tại switch {switch}")
            self.assertEqual(y_hat.shape, (1, 1, 8, 64), f"Lỗi shape sinogram tại switch {switch}")
            self.assertEqual(x_recon.dtype, torch.float32)
            self.assertEqual(y_hat.dtype, torch.float32)

    def test_u2_bit_identical_none(self):
        """U2: Bit-identity giữa ablate='none' và khởi tạo ngẫu nhiên cùng hạt giống."""
        # Khởi tạo mô hình 1
        model_1 = create_tiny_solar_model(ablate="none", seed=100)
        # Khởi tạo mô hình 2 với cùng hạt giống
        model_2 = create_tiny_solar_model(ablate="none", seed=100)

        # Kiểm tra khớp chính xác từng khóa tham số trong state_dict
        self.assertEqual(set(model_1.state_dict().keys()), set(model_2.state_dict().keys()))
        for k in model_1.state_dict():
            self.assertTrue(
                torch.equal(model_1.state_dict()[k], model_2.state_dict()[k]),
                f"Lệch trọng số tại khóa tham số {k}"
            )

        # Kiểm tra đầu ra khi forward cùng một đầu vào
        x_fbp = torch.randn(1, 1, 32, 32, dtype=torch.float32)
        sino = torch.randn(1, 1, 8, 64, dtype=torch.float32)
        with torch.no_grad():
            out1_x, out1_y = model_1(x_fbp, sino)
            out2_x, out2_y = model_2(x_fbp, sino)

        self.assertTrue(torch.equal(out1_x, out2_x), "Lệch đầu ra ảnh tái tạo tại ablate='none'")
        self.assertTrue(torch.equal(out1_y, out2_y), "Lệch đầu ra sinogram tại ablate='none'")

    def test_u3_parameter_accounting(self):
        """
        U3: Phân nhóm named_parameters() theo tiền tố:
        - (a): không có sino_restoration.*
        - (b): không có raw_lambda.*, raw_mu.*; có raw_alpha.* (đúng K phần tử)
        - (c): chỉ khác sino_restoration.mamba_blocks.*
        - (d): chỉ khác image_regularizer.nonlocal_branch.*
        """
        model_none = create_tiny_solar_model(ablate="none", seed=42)
        model_a = create_tiny_solar_model(ablate="a", seed=42)
        model_b = create_tiny_solar_model(ablate="b", seed=42)
        model_c = create_tiny_solar_model(ablate="c", seed=42)
        model_d = create_tiny_solar_model(ablate="d", seed=42)

        # (a): hoàn toàn vắng mặt sino_restoration
        params_a = dict(model_a.named_parameters())
        self.assertFalse(any(k.startswith("sino_restoration") for k in params_a))

        # (b): vắng mặt raw_lambda và raw_mu, có raw_alpha với đúng 2 phần tử (K=2)
        params_b = dict(model_b.named_parameters())
        self.assertFalse(any(k.startswith("raw_lambda") for k in params_b))
        self.assertFalse(any(k.startswith("raw_mu") for k in params_b))
        alpha_keys = [k for k in params_b if k.startswith("raw_alpha")]
        self.assertEqual(len(alpha_keys), 2)

        # (c): các tham số khác sino_restoration.mamba_blocks phải trùng tên và số phần tử
        params_c = dict(model_c.named_parameters())
        non_mamba_none = {k: v.shape for k, v in model_none.named_parameters() if not k.startswith("sino_restoration.mamba_blocks")}
        non_mamba_c = {k: v.shape for k, v in params_c.items() if not k.startswith("sino_restoration.mamba_blocks")}
        self.assertEqual(non_mamba_none, non_mamba_c)

        # (d): các tham số khác image_regularizer.nonlocal_branch phải trùng tên và shape
        params_d = dict(model_d.named_parameters())
        non_swin_none = {k: v.shape for k, v in model_none.named_parameters() if not k.startswith("image_regularizer.nonlocal_branch")}
        non_swin_d = {k: v.shape for k, v in params_d.items() if not k.startswith("image_regularizer.nonlocal_branch")}
        self.assertEqual(non_swin_none, non_swin_d)

    def test_u4_finite(self):
        """U4: Tính hữu hạn: forward + backward không phát sinh NaN hoặc Inf trong loss và grad."""
        for switch in ["none", "a", "b", "c", "d"]:
            model = create_tiny_solar_model(ablate=switch, seed=42)
            model.train()

            phantom = torch.randn(1, 1, 32, 32, dtype=torch.float32)
            fbp_u = torch.randn(1, 1, 32, 32, dtype=torch.float32)
            sino = torch.randn(1, 1, 8, 64, dtype=torch.float32)

            loss = model.training_step((phantom, fbp_u, sino), batch_idx=0)
            self.assertFalse(torch.isnan(loss).item(), f"NaN loss tại switch {switch}")
            self.assertFalse(torch.isinf(loss).item(), f"Inf loss tại switch {switch}")

            loss.backward()
            for name, p in model.named_parameters():
                if p.grad is not None:
                    self.assertFalse(torch.isnan(p.grad).any().item(), f"NaN grad tại {name} (switch {switch})")
                    self.assertFalse(torch.isinf(p.grad).any().item(), f"Inf grad tại {name} (switch {switch})")

    def test_u5_switch_a_identity_and_zero_loss(self):
        """U5: Đối với biến thể (a): ŷ bằng y (torch.equal) và loss_sino == 0."""
        model_a = create_tiny_solar_model(ablate="a", seed=42)
        model_a.eval()

        phantom = torch.randn(1, 1, 32, 32, dtype=torch.float32)
        fbp_u = torch.randn(1, 1, 32, 32, dtype=torch.float32)
        sino = torch.randn(1, 1, 8, 64, dtype=torch.float32)

        x_recon, y_hat = model_a(fbp_u, sino)
        self.assertTrue(torch.equal(y_hat, sino), "ŷ phải bằng đúng y trong biến thể (a)")

        # Kiểm tra loss_sino
        loss_sino = F.mse_loss(y_hat, sino)
        self.assertEqual(loss_sino.item(), 0.0, "loss_sino phải bằng 0 trong biến thể (a)")

    def test_u6_operator_call_count(self):
        """U6: Số lượt gọi toán tử Radon A: per stage b = 1 call; none = 1 + cg_iters."""
        class CountingOperator(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner
                self.calls = 0
            def forward(self, x):
                self.calls += 1
                return self.inner(x)

        # Kiểm tra biến thể (b): 1 call mỗi stage -> K=2 stages thì gọi 2 lần A
        model_b = create_tiny_solar_model(ablate="b", seed=42)
        counter_b = CountingOperator(model_b.forward_module)
        model_b.forward_module = counter_b

        x_fbp = torch.randn(1, 1, 32, 32, dtype=torch.float32)
        sino = torch.randn(1, 1, 8, 64, dtype=torch.float32)
        model_b.eval()
        with torch.no_grad():
            _ = model_b(x_fbp, sino)
        self.assertEqual(counter_b.calls, 2, f"Biến thể b tại K=2 phải gọi A đúng 2 lần, thực tế gọi {counter_b.calls}")

        # Kiểm tra biến thể none: 1 + cg_iters (2 + 1 = 3 lần mỗi stage) -> K=2 stages thì gọi 6 lần A
        model_none = create_tiny_solar_model(ablate="none", seed=42)
        counter_none = CountingOperator(model_none.forward_module)
        model_none.forward_module = counter_none
        model_none.eval()
        with torch.no_grad():
            _ = model_none(x_fbp, sino)
        # Mỗi stage gọi 1 lần r0 và tối đa cg_iters lần trong vòng lặp CG
        self.assertLessEqual(counter_none.calls, 2 * (1 + 2))
        self.assertGreater(counter_none.calls, 2)

    def test_u7_diagnostic_q1_hessian(self):
        """U7: Báo cáo chẩn đoán tính đối xứng và chặn dưới dương của F*A."""
        # Chạy kiểm tra chẩn đoán trên tensor ngẫu nhiên
        # |<u, FAv> - <FAu, v>| / (||u|| * ||FAv||)
        stub_a = LinearStubOperator((32, 32), (8, 64))
        stub_f = LinearStubOperator((8, 64), (32, 32))

        u = torch.randn(1, 1, 32, 32)
        v = torch.randn(1, 1, 32, 32)
        fav = stub_f(stub_a(v))
        fau = stub_f(stub_a(u))

        dot1 = torch.sum(u * fav)
        dot2 = torch.sum(fau * v)
        asym = torch.abs(dot1 - dot2) / (torch.norm(u) * torch.norm(fav) + 1e-7)
        # Chẩn đoán báo cáo chỉ số (không assert thất bại theo đặc tả spec-1 §6 U7)
        print(f"\n[U7 Diagnostic] Relative asymmetry: {asym.item():.6f}")

    def test_u8_hparams_roundtrip(self):
        """U8: Checkpoint lưu với ablate='d' lưu đúng ablate='d' trong hyperparameters."""
        model_d = create_tiny_solar_model(ablate="d", seed=42)
        self.assertEqual(model_d.hparams.ablate, "d")
        self.assertEqual(model_d.ablate, "d")


if __name__ == "__main__":
    unittest.main()
