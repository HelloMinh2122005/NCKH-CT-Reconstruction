#!/usr/bin/env python3
"""
Unit tests cho T0.6: Ràng buộc nghiêm ngặt kích thước phân chia tập AAPM (8218 / 133 / 757)
tại helper data/split_guards.py và tất cả các điểm vào train/test Tier 1 + evaluate_fbp + eval_utils.
"""

import ast
import importlib
import os
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

TIER1_TRAIN_SCRIPTS = [
    "baselines/SOLAR_DualMamba/train_solar_dualmamba_la.py",
    "baselines/LEARN/train_learn_la.py",
    "baselines/RegFormer/train_regformer_la.py",
    "baselines/DuDoTrans/train_dudotrans_la.py",
]

TIER1_TEST_SCRIPTS = [
    "baselines/SOLAR_DualMamba/test_solar_dualmamba_la.py",
    "baselines/LEARN/test_learn_la.py",
    "baselines/RegFormer/test_regformer_la.py",
    "baselines/DuDoTrans/test_dudotrans_la.py",
]


class _SizedList:
    """Đối tượng nhẹ hỗ trợ len() kích thước tùy ý mà không tốn RAM."""

    def __init__(self, n: int):
        self._n = n

    def __len__(self) -> int:
        return self._n


def _build_lightweight_ml_stubs():
    """Tạo stub cho các thư viện cluster-only (torch, pytorch_lightning, models) khi chạy tại local."""
    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        max_memory_allocated=lambda: 0,
    )
    torch_mod.load = lambda *a, **kw: {}
    torch_mod.no_grad = MagicMock

    pl_mod = types.ModuleType("pytorch_lightning")
    pl_mod.seed_everything = lambda *a, **kw: None
    pl_mod.Trainer = MagicMock()
    pl_callbacks = types.ModuleType("pytorch_lightning.callbacks")
    ckpt_cb = MagicMock()
    ckpt_cb.best_model_score = 30.0
    ckpt_cb.best_model_path = "/tmp/fake.ckpt"
    pl_callbacks.ModelCheckpoint = MagicMock(return_value=ckpt_cb)
    pl_callbacks.LearningRateMonitor = MagicMock()
    pl_loggers = types.ModuleType("pytorch_lightning.loggers")
    pl_loggers.TensorBoardLogger = MagicMock()

    np_mod = sys.modules.get("numpy")
    if np_mod is None:
        import numpy as np_mod  # noqa: F401

    stubs = {
        "torch": torch_mod,
        "pytorch_lightning": pl_mod,
        "pytorch_lightning.callbacks": pl_callbacks,
        "pytorch_lightning.loggers": pl_loggers,
        "data.datamodule_factory": types.ModuleType("data.datamodule_factory"),
    }
    stubs["data.datamodule_factory"].get_datamodule = MagicMock()

    # Stub các lớp mô hình baseline để tránh kéo torch.nn khi kiểm thử CLI & split guard ở local
    model_specs = {
        "baselines.SOLAR_DualMamba": ("baselines.SOLAR_DualMamba.models", "SOLAR_DualMamba_LA"),
        "baselines.LEARN": ("baselines.LEARN.models", "LEARN_LA"),
        "baselines.RegFormer": ("baselines.RegFormer.models", "RegFormer_LA"),
        "baselines.DuDoTrans": ("baselines.DuDoTrans.models", "DuDoTrans_LA"),
    }
    for pkg_name, (mod_name, cls_name) in model_specs.items():
        pkg_mod = types.ModuleType(pkg_name)
        pkg_mod.__path__ = [os.path.join(REPO_ROOT, pkg_name.replace(".", os.sep))]
        mdl_mod = types.ModuleType(mod_name)
        cls_mock = MagicMock()
        cls_mock.load_from_checkpoint = MagicMock(return_value=MagicMock())
        setattr(mdl_mod, cls_name, cls_mock)
        stubs[pkg_name] = pkg_mod
        stubs[mod_name] = mdl_mod

    return stubs


def _is_real_module(mod) -> bool:
    return mod is not None and hasattr(mod, "__file__") and mod.__file__ is not None


_SAVED_MODULES = None


def setUpModule():
    global _SAVED_MODULES
    _SAVED_MODULES = sys.modules.copy()
    if not _is_real_module(sys.modules.get("torch")):
        try:
            import torch  # noqa: F401
        except ImportError:
            pass
    if not _is_real_module(sys.modules.get("torch")):
        for mod_name in list(sys.modules.keys()):
            if any(
                mod_name.startswith(prefix)
                for prefix in (
                    "baselines.SOLAR_DualMamba",
                    "baselines.LEARN",
                    "baselines.RegFormer",
                    "baselines.DuDoTrans",
                    "baselines.eval_utils",
                    "scripts.evaluate_fbp",
                )
            ):
                del sys.modules[mod_name]
        for name, mod in _build_lightweight_ml_stubs().items():
            sys.modules[name] = mod


def tearDownModule():
    global _SAVED_MODULES
    if _SAVED_MODULES is not None:
        for name in list(sys.modules.keys()):
            if name not in _SAVED_MODULES:
                del sys.modules[name]
        for name, mod in _SAVED_MODULES.items():
            sys.modules[name] = mod
        _SAVED_MODULES = None


class TestSplitGuardsHelper(unittest.TestCase):
    def test_expected_aapm_split_counts_constant(self):
        """Hằng số EXPECTED_AAPM_SPLIT_COUNTS khớp chính xác experiments/split.md (8218/133/757)."""
        from data.split_guards import EXPECTED_AAPM_SPLIT_COUNTS

        self.assertEqual(EXPECTED_AAPM_SPLIT_COUNTS, {"train": 8218, "val": 133, "test": 757})

    def test_assert_aapm_split_counts_passes_on_canonical_counts(self):
        """assert_aapm_split_counts cho phép đi qua khi số lát cắt khớp chính xác 8218 / 133 / 757."""
        from data.split_guards import assert_aapm_split_counts

        dm_fit = SimpleNamespace(train_dataset=_SizedList(8218), val_dataset=_SizedList(133))
        res_fit = assert_aapm_split_counts(dm_fit, "fit")
        self.assertEqual(res_fit, {"train": 8218, "val": 133})

        dm_test = SimpleNamespace(test_dataset=_SizedList(757))
        res_test = assert_aapm_split_counts(dm_test, "test")
        self.assertEqual(res_test, {"test": 757})

        dm_all = SimpleNamespace(
            train_dataset=_SizedList(8218),
            val_dataset=_SizedList(133),
            test_dataset=_SizedList(757),
        )
        res_all = assert_aapm_split_counts(dm_all, None)
        self.assertEqual(res_all, {"train": 8218, "val": 133, "test": 757})

    def test_assert_aapm_split_counts_fails_on_wrong_counts(self):
        """assert_aapm_split_counts ném AssertionError tức thì khi bất kỳ tập train/val/test nào lệch số lát."""
        from data.split_guards import assert_aapm_split_counts, assert_aapm_test_dataset_count

        # Sai train (99 thay vì 8218)
        dm_bad_train = SimpleNamespace(train_dataset=_SizedList(99), val_dataset=_SizedList(133))
        with self.assertRaises(AssertionError) as ctx_train:
            assert_aapm_split_counts(dm_bad_train, "fit")
        self.assertIn("8218", str(ctx_train.exception))
        self.assertIn("99", str(ctx_train.exception))

        # Sai val (0 hoặc 132 thay vì 133)
        dm_bad_val = SimpleNamespace(train_dataset=_SizedList(8218), val_dataset=_SizedList(132))
        with self.assertRaises(AssertionError) as ctx_val:
            assert_aapm_split_counts(dm_bad_val, "fit")
        self.assertIn("133", str(ctx_val.exception))
        self.assertIn("132", str(ctx_val.exception))

        # Sai test (210 thay vì 757)
        dm_bad_test = SimpleNamespace(test_dataset=_SizedList(210))
        with self.assertRaises(AssertionError) as ctx_test:
            assert_aapm_split_counts(dm_bad_test, "test")
        self.assertIn("757", str(ctx_test.exception))
        self.assertIn("210", str(ctx_test.exception))

        # Dataset bị None
        dm_none = SimpleNamespace(train_dataset=None, val_dataset=_SizedList(133))
        with self.assertRaises(AssertionError):
            assert_aapm_split_counts(dm_none, "fit")

        # Kiểm tra helper cho evaluate_fbp.py
        self.assertEqual(assert_aapm_test_dataset_count(_SizedList(757)), 757)
        with self.assertRaises(AssertionError):
            assert_aapm_test_dataset_count(_SizedList(210))

    def test_allow_custom_split_opt_out_and_non_aapm_bypass(self):
        """--allow_custom_split=True cho phép tập nhỏ có chủ đích; dataset khác aapm được bỏ qua."""
        from data.split_guards import assert_aapm_split_counts

        dm_smoke = SimpleNamespace(train_dataset=_SizedList(99), val_dataset=_SizedList(10))
        res_opt_out = assert_aapm_split_counts(dm_smoke, "fit", allow_custom_split=True)
        self.assertEqual(res_opt_out, {"train": 99, "val": 10})

        res_lidc = assert_aapm_split_counts(dm_smoke, "fit", allow_custom_split=False, dataset_type="lidc")
        self.assertEqual(res_lidc, {})


class TestEntryPointsSplitGuardWiring(unittest.TestCase):
    def _parse_ast(self, relpath: str):
        full_path = Path(REPO_ROOT) / relpath
        self.assertTrue(full_path.exists(), f"Không tìm thấy tệp {relpath}")
        source = full_path.read_text(encoding="utf-8")
        return source, ast.parse(source, filename=str(full_path))

    def _collect_called_names(self, tree: ast.AST) -> set:
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    names.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    names.add(node.func.attr)
        return names

    def test_all_tier1_train_and_test_scripts_wire_split_guard_ast(self):
        """AST check: Tất cả 8 script Tier 1 + evaluate_fbp + eval_utils đều gọi assert_aapm_split_counts & có --allow_custom_split."""
        for relpath in TIER1_TRAIN_SCRIPTS + TIER1_TEST_SCRIPTS:
            source, tree = self._parse_ast(relpath)
            called = self._collect_called_names(tree)
            self.assertIn("assert_aapm_split_counts", called, f"{relpath} thiếu lời gọi assert_aapm_split_counts")
            self.assertIn("--allow_custom_split", source, f"{relpath} thiếu cờ CLI --allow_custom_split")

        eval_fbp_src, eval_fbp_tree = self._parse_ast("scripts/evaluate_fbp.py")
        eval_fbp_called = self._collect_called_names(eval_fbp_tree)
        self.assertTrue(
            "assert_aapm_test_dataset_count" in eval_fbp_called or "assert_aapm_split_counts" in eval_fbp_called,
            "scripts/evaluate_fbp.py thiếu split-count assert cho AAPM test",
        )
        self.assertIn("--allow_custom_split", eval_fbp_src)

        _, eval_utils_tree = self._parse_ast("baselines/eval_utils.py")
        self.assertIn("assert_aapm_split_counts", self._collect_called_names(eval_utils_tree))

    def test_tier1_train_entrypoints_runtime_fail_fast_and_opt_out(self):
        """Thực thi main() trên 4 train entry points: mặc định lỗi ngay khi 99/133; --allow_custom_split cho qua."""
        train_modules = [
            "baselines.SOLAR_DualMamba.train_solar_dualmamba_la",
            "baselines.LEARN.train_learn_la",
            "baselines.RegFormer.train_regformer_la",
            "baselines.DuDoTrans.train_dudotrans_la",
        ]
        for mod_name in train_modules:
            mod = importlib.import_module(mod_name)
            with patch.object(sys, "argv", [mod_name]):
                args_default = mod.parse_args()
                self.assertFalse(args_default.allow_custom_split, f"{mod_name}: allow_custom_split mặc định phải là False")

            fake_dm = MagicMock()
            fake_dm.train_dataset = _SizedList(99)
            fake_dm.val_dataset = _SizedList(133)
            fake_trainer = MagicMock()

            with patch.object(mod, "get_datamodule", return_value=fake_dm), \
                 patch.object(mod.pl, "Trainer", return_value=fake_trainer), \
                 patch("os.makedirs"):
                with patch.object(sys, "argv", [mod_name, "--dataset_type", "aapm"]):
                    with self.assertRaises(AssertionError):
                        mod.main()
                    self.assertEqual(fake_trainer.fit.call_count, 0)

                with patch.object(sys, "argv", [mod_name, "--dataset_type", "aapm", "--allow_custom_split"]):
                    mod.main()
                    self.assertEqual(fake_trainer.fit.call_count, 1)

    def test_tier1_test_entrypoints_runtime_fail_fast_and_opt_out(self):
        """Thực thi main() trên 4 test entry points: mặc định lỗi ngay khi test != 757; --allow_custom_split cho qua."""
        test_modules = [
            "baselines.SOLAR_DualMamba.test_solar_dualmamba_la",
            "baselines.LEARN.test_learn_la",
            "baselines.RegFormer.test_regformer_la",
            "baselines.DuDoTrans.test_dudotrans_la",
        ]
        for mod_name in test_modules:
            mod = importlib.import_module(mod_name)
            with patch.object(sys, "argv", [mod_name]):
                args_default = mod.parse_args()
                self.assertFalse(args_default.allow_custom_split, f"{mod_name}: allow_custom_split mặc định phải là False")

            fake_dm = MagicMock()
            fake_dm.test_dataset = _SizedList(210)
            fake_trainer = MagicMock()

            with patch.object(mod, "get_datamodule", return_value=fake_dm), \
                 patch.object(mod.pl, "Trainer", return_value=fake_trainer):
                with patch.object(sys, "argv", [mod_name, "--dataset_type", "aapm", "--ckpt_path", "/tmp/dummy.ckpt"]):
                    with self.assertRaises(AssertionError):
                        mod.main()
                    self.assertEqual(fake_trainer.test.call_count, 0)

                with patch.object(
                    sys,
                    "argv",
                    [mod_name, "--dataset_type", "aapm", "--ckpt_path", "/tmp/dummy.ckpt", "--allow_custom_split"],
                ):
                    mod.main()
                    self.assertEqual(fake_trainer.test.call_count, 1)

    def test_eval_utils_and_evaluate_fbp_runtime_split_guard(self):
        """Thực thi eval_utils.run_slice_by_slice_evaluation & evaluate_fbp_for_angle: lỗi khi != 757, cho qua khi allow_custom_split=True."""
        from baselines import eval_utils
        from scripts import evaluate_fbp

        # 1) eval_utils.run_slice_by_slice_evaluation
        fake_dm = MagicMock()
        fake_dm.test_dataset = _SizedList(10)
        fake_dm.test_dataloader.return_value = []
        fake_model = MagicMock()
        fake_model.to.return_value = fake_model

        with self.assertRaises(AssertionError):
            eval_utils.run_slice_by_slice_evaluation(
                model=fake_model,
                datamodule=fake_dm,
                method_name="TEST",
                angle_deg=120.0,
                device="cpu",
            )

        # Với allow_custom_split=True phải đi qua guard (nên dùng allow_custom_split=True khi test)
        with patch("numpy.mean", return_value=0.0), patch("numpy.std", return_value=0.0):
            res = eval_utils.run_slice_by_slice_evaluation(
                model=fake_model,
                datamodule=fake_dm,
                method_name="TEST",
                angle_deg=120.0,
                device="cpu",
                allow_custom_split=True,
            )
            self.assertEqual(res["method"], "TEST")

        # 2) scripts/evaluate_fbp.py
        args_strict = evaluate_fbp.parse_args(["--allow_custom_split"] if False else [])
        self.assertFalse(args_strict.allow_custom_split)
        args_opt_out = evaluate_fbp.parse_args(["--allow_custom_split"])
        self.assertTrue(args_opt_out.allow_custom_split)

        fake_provider = MagicMock()
        fake_provider.__len__.return_value = 5
        fake_tv = types.ModuleType("torchvision")
        fake_tv_tr = types.ModuleType("torchvision.transforms")
        fake_tv_tr.Compose = MagicMock()
        fake_tv_tr.Resize = MagicMock()
        fake_tv.transforms = fake_tv_tr

        with patch.object(evaluate_fbp, "HAS_TORCH", True), \
             patch.object(evaluate_fbp, "LimitedAngleCT_Provider", return_value=fake_provider), \
             patch.object(evaluate_fbp, "torch", create=True) as mock_torch, \
             patch.dict(sys.modules, {"torchvision": fake_tv, "torchvision.transforms": fake_tv_tr}):
            mock_torch.cuda.is_available.return_value = False
            mock_torch.device.return_value = "cpu"
            args_strict.output_dir = "/tmp/fbp_test_guard_strict"
            with patch("os.makedirs"):
                with self.assertRaises(AssertionError):
                    evaluate_fbp.evaluate_fbp_for_angle(120.0, args_strict)


if __name__ == "__main__":
    unittest.main()
