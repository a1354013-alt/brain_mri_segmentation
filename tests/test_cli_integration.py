import importlib
import sys
import types
import unittest
from pathlib import Path


def _install_stub(name: str, module: types.ModuleType) -> None:
    sys.modules[name] = module


def _make_torch_stub():
    torch = types.ModuleType("torch")

    class _Device:
        def __init__(self, dev_type: str):
            self.type = dev_type

        def __str__(self) -> str:
            return self.type

    def device(dev: str):
        return _Device(dev)

    torch.device = device
    torch.manual_seed = lambda *_args, **_kwargs: None
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.load = lambda *_args, **_kwargs: {}

    # torch.utils.data.DataLoader stub
    torch_utils = types.ModuleType("torch.utils")
    torch_utils_data = types.ModuleType("torch.utils.data")

    class DataLoader:
        def __init__(self, dataset, **kwargs):
            self.dataset = dataset
            self.kwargs = kwargs

    torch_utils_data.DataLoader = DataLoader
    torch_utils.data = torch_utils_data

    _install_stub("torch", torch)
    _install_stub("torch.utils", torch_utils)
    _install_stub("torch.utils.data", torch_utils_data)
    return torch


def _make_numpy_stub():
    np = types.ModuleType("numpy")
    np.random = types.SimpleNamespace(seed=lambda *_args, **_kwargs: None)
    np.uint8 = int
    np.float32 = float
    return np


def _make_config_stub():
    cfg = types.ModuleType("config")
    cfg.PROJECT_ROOT = Path(".").resolve()
    cfg.DATA_DIR = Path("data/Brats")
    cfg.OUTPUT_DIR = Path("outputs")

    cfg.CHECKPOINT_PATH = cfg.OUTPUT_DIR / "best_checkpoint.pth"
    cfg.MODEL_STATE_PATH = cfg.OUTPUT_DIR / "best_model_state.pth"

    cfg.LAST_CHECKPOINT_PATH = cfg.OUTPUT_DIR / "last_checkpoint.pth"
    cfg.LAST_MODEL_STATE_PATH = cfg.OUTPUT_DIR / "last_model_state.pth"
    cfg.LOG_FILE = cfg.OUTPUT_DIR / "training_log.csv"
    cfg.TENSORBOARD_DIR = cfg.OUTPUT_DIR / "tensorboard"

    cfg.DEMO_OUTPUT_DIR = cfg.OUTPUT_DIR / "demo"
    cfg.DEMO_CHECKPOINT_PATH = cfg.DEMO_OUTPUT_DIR / "best_checkpoint_demo.pth"
    cfg.DEMO_MODEL_STATE_PATH = cfg.DEMO_OUTPUT_DIR / "best_model_state_demo.pth"
    cfg.DEMO_LAST_CHECKPOINT_PATH = cfg.DEMO_OUTPUT_DIR / "last_checkpoint_demo.pth"
    cfg.DEMO_LAST_MODEL_STATE_PATH = cfg.DEMO_OUTPUT_DIR / "last_model_state_demo.pth"
    cfg.DEMO_TENSORBOARD_DIR = cfg.DEMO_OUTPUT_DIR / "tensorboard"
    cfg.DEMO_LOG_FILE = cfg.DEMO_OUTPUT_DIR / "training_log_demo.csv"

    cfg.N_CHANNELS = 4
    cfg.N_CLASSES = 1
    cfg.DROPOUT_P = 0.2
    cfg.THRESHOLD = 0.5

    cfg.IMAGE_SIZE = 128
    cfg.BATCH_SIZE = 4
    cfg.EPOCHS = 1
    cfg.LEARNING_RATE = 1e-4
    cfg.WEIGHT_DECAY = 1e-5

    cfg.TRAIN_VAL_SPLIT = 0.8
    cfg.NUM_WORKERS = 0
    cfg.PERSISTENT_WORKERS = False
    cfg.PREFETCH_FACTOR = 2
    cfg.PIN_MEMORY = False
    cfg.USE_PROXY_CACHE = False

    cfg.NEG_SLICE_PROB = 0.3
    cfg.MC_ITERATIONS = 3
    cfg.DEMO_MC_ITERATIONS = 2

    cfg.RANDOM_SEED = 42
    cfg.DICE_WEIGHT = 1.0
    cfg.BCE_WEIGHT = 0.5

    cfg.DEVICE = types.SimpleNamespace(type="cpu")

    def set_seed(_seed: int = 42):
        cfg.RANDOM_SEED = int(_seed)

    def set_device(dev: str | None):
        if dev is None:
            return
        cfg.DEVICE = types.SimpleNamespace(type=dev)

    def apply_overrides(**kwargs):
        if kwargs.get("device") is not None:
            set_device(kwargs["device"])
        if kwargs.get("RANDOM_SEED") is not None:
            cfg.RANDOM_SEED = int(kwargs["RANDOM_SEED"])
        if kwargs.get("MC_ITERATIONS") is not None:
            cfg.MC_ITERATIONS = int(kwargs["MC_ITERATIONS"])
        if kwargs.get("NEG_SLICE_PROB") is not None:
            cfg.NEG_SLICE_PROB = float(kwargs["NEG_SLICE_PROB"])

    cfg.set_seed = set_seed
    cfg.apply_overrides = apply_overrides
    return cfg


def _make_models_stub():
    models = types.ModuleType("models")

    class AttentionUNet:
        def __init__(self, *_args, **_kwargs):
            pass

        def to(self, _device):
            return self

        def eval(self):
            return self

        def load_state_dict(self, _sd):
            return None

    models.AttentionUNet = AttentionUNet
    return models


def _make_utils_stub(calls: dict):
    utils = types.ModuleType("utils")

    class BraTSDataset:
        def __init__(self, _data_dir, patient_ids, _image_size, mode="train", prepared_cache=None, output_dir=None):
            calls.setdefault("dataset_inits", []).append(
                {
                    "mode": mode,
                    "patient_ids": list(patient_ids),
                    "prepared_cache": prepared_cache,
                    "output_dir": output_dir,
                }
            )
            if calls.get("force_empty_dataset"):
                self.valid_patient_ids = []
            else:
                self.valid_patient_ids = list(patient_ids)
            self.patient_cache = {pid: {"norm_stats": {}} for pid in self.valid_patient_ids}

        def __len__(self):
            return len(self.valid_patient_ids)

        def __getitem__(self, _idx):
            # image, mask dummy objects with the minimal surface used by infer_command
            img = types.SimpleNamespace(unsqueeze=lambda *_args, **_kwargs: "img_batch", numpy=lambda: [[0]])
            msk = types.SimpleNamespace(numpy=lambda: [[0]])
            return img, msk

        @staticmethod
        def quick_validate_patient(_data_dir, _pid, **kwargs):
            calls.setdefault("quick_validate_calls", []).append(kwargs)
            return True

    def mc_dropout_inference(_model, _img_batch, n_iterations, device, method="var"):
        calls["mc_dropout"] = {"n_iterations": n_iterations, "device": device, "method": method}
        # prediction, uncertainty
        return [[[0]]], [[[0]]]

    def plot_results_with_uncertainty(*_args, **_kwargs):
        calls["plot_called"] = True

    class _Arr:
        def __init__(self, name: str):
            self.name = name

        def astype(self, _dtype):
            return self

    def predict_patient_volume(*_args, **kwargs):
        calls["predict_patient_volume"] = {"kwargs": kwargs}
        ref_img = object()
        return _Arr("pred"), _Arr("unc"), _Arr("prob"), ref_img

    def save_nifti_like(_ref_img, data, save_path):
        calls.setdefault("save_nifti_like", []).append(
            {"data": getattr(data, "name", None), "save_path": str(save_path)}
        )

    utils.BraTSDataset = BraTSDataset
    utils.mc_dropout_inference = mc_dropout_inference
    utils.plot_results_with_uncertainty = plot_results_with_uncertainty
    utils.predict_patient_volume = predict_patient_volume
    utils.save_nifti_like = save_nifti_like
    return utils


def _make_train_stub(calls: dict):
    train = types.ModuleType("train")

    class Trainer:
        def __init__(self, **kwargs):
            calls["trainer_init"] = kwargs

        def train(self):
            calls["trainer_train_called"] = True

    train.Trainer = Trainer
    return train


def import_main_with_stubs(include_train_module: bool, calls: dict):
    for name in [
        "main",
        "config",
        "models",
        "utils",
        "train",
        "torch",
        "torch.utils",
        "torch.utils.data",
        "numpy",
    ]:
        sys.modules.pop(name, None)

    _make_torch_stub()
    _install_stub("numpy", _make_numpy_stub())
    _install_stub("config", _make_config_stub())
    _install_stub("models", _make_models_stub())
    _install_stub("utils", _make_utils_stub(calls))
    if include_train_module:
        _install_stub("train", _make_train_stub(calls))

    return importlib.import_module("main")


class TestCliIntegration(unittest.TestCase):
    def test_train_val_cache_not_reused_and_split_safe(self):
        calls = {}
        m = import_main_with_stubs(include_train_module=True, calls=calls)

        # Patch helpers to avoid filesystem access and run-config writes.
        m.get_patient_ids = lambda _data_dir: ["p1", "p2", "p3", "p4"]
        m.save_run_config = lambda *_args, **_kwargs: None

        args = types.SimpleNamespace(
            command="train",
            data_dir=None,
            output_dir=None,
            device=None,
            image_size=None,
            batch_size=None,
            epochs=None,
            lr=None,
            weight_decay=None,
            num_workers=None,
            seed=123,
            mc_iterations=None,
            neg_slice_prob=None,
            no_proxy_cache=False,
        )
        m.train_command(args)

        inits = calls.get("dataset_inits", [])
        self.assertEqual(len(inits), 2, "Expected train_dataset and val_dataset to be created")
        self.assertIsNone(inits[0]["prepared_cache"])
        self.assertIsNone(inits[1]["prepared_cache"])
        self.assertEqual(inits[0]["mode"], "train")
        self.assertEqual(inits[1]["mode"], "val")
        self.assertTrue(set(inits[0]["patient_ids"]).isdisjoint(set(inits[1]["patient_ids"])))
        self.assertTrue(calls.get("trainer_train_called", False))

    def test_small_dataset_rejected(self):
        calls = {}
        m = import_main_with_stubs(include_train_module=True, calls=calls)
        m.get_patient_ids = lambda _data_dir: ["only_one"]
        m.save_run_config = lambda *_args, **_kwargs: None

        args = types.SimpleNamespace(command="train", seed=None)
        m.train_command(args)
        self.assertEqual(len(calls.get("dataset_inits", [])), 0)

    def test_infer_does_not_import_train(self):
        calls = {}
        m = import_main_with_stubs(include_train_module=False, calls=calls)
        self.assertNotIn("train", sys.modules)

        # Patch helpers to avoid filesystem access and run-config writes.
        m.get_patient_ids = lambda _data_dir: ["p1"]
        m.save_run_config = lambda *_args, **_kwargs: None

        args = types.SimpleNamespace(
            command="infer",
            patient_id="p1",
            uncertainty="var",
            save_nifti=False,
            save_prob=False,
            device="cpu",
            seed=None,
            data_dir=None,
            output_dir=None,
            image_size=None,
            batch_size=None,
            epochs=None,
            lr=None,
            weight_decay=None,
            num_workers=None,
            mc_iterations=None,
            neg_slice_prob=None,
            no_proxy_cache=False,
        )

        # Should not raise (and should not import train).
        m.infer_command(args)
        self.assertNotIn("train", sys.modules)
        self.assertIn("mc_dropout", calls)
        self.assertEqual(calls["mc_dropout"]["method"], "var")
        self.assertTrue(calls.get("quick_validate_calls"))
        # Two-phase validation: we should have at least one strict scan on the selected pid.
        self.assertTrue(any(c.get("strict_tumor_check") for c in calls["quick_validate_calls"]))

    def test_device_override_passed_to_mc_dropout(self):
        calls = {}
        m = import_main_with_stubs(include_train_module=False, calls=calls)
        m.get_patient_ids = lambda _data_dir: ["p1"]
        m.save_run_config = lambda *_args, **_kwargs: None

        args = types.SimpleNamespace(
            command="infer",
            patient_id="p1",
            uncertainty="entropy",
            save_nifti=False,
            save_prob=False,
            device="cuda",
            seed=None,
            data_dir=None,
            output_dir=None,
            image_size=None,
            batch_size=None,
            epochs=None,
            lr=None,
            weight_decay=None,
            num_workers=None,
            mc_iterations=7,
            neg_slice_prob=None,
            no_proxy_cache=False,
        )

        m.infer_command(args)
        self.assertEqual(calls["mc_dropout"]["method"], "entropy")
        # We only check that device is explicitly forwarded after overrides.
        self.assertEqual(getattr(calls["mc_dropout"]["device"], "type", None), "cuda")
        self.assertEqual(calls["mc_dropout"]["n_iterations"], 7)

    def test_infer_save_nifti_path_calls(self):
        calls = {}
        m = import_main_with_stubs(include_train_module=False, calls=calls)
        m.get_patient_ids = lambda _data_dir: ["p1"]
        m.save_run_config = lambda *_args, **_kwargs: None

        args = types.SimpleNamespace(
            command="infer",
            patient_id="p1",
            uncertainty="var",
            save_nifti=True,
            save_prob=True,
            device="cpu",
            seed=None,
            data_dir=None,
            output_dir=None,
            image_size=None,
            batch_size=None,
            epochs=None,
            lr=None,
            weight_decay=None,
            num_workers=None,
            mc_iterations=None,
            neg_slice_prob=None,
            no_proxy_cache=False,
        )

        m.infer_command(args)
        self.assertIn("predict_patient_volume", calls)
        saved = calls.get("save_nifti_like", [])
        self.assertEqual(len(saved), 3)
        save_paths = [s["save_path"] for s in saved]
        self.assertTrue(any(p.endswith("_pred.nii.gz") for p in save_paths))
        self.assertTrue(any(p.endswith("_uncertainty.nii.gz") for p in save_paths))
        self.assertTrue(any(p.endswith("_prob.nii.gz") for p in save_paths))

    def test_demo_empty_dataset_exits_safely(self):
        calls = {"force_empty_dataset": True}
        m = import_main_with_stubs(include_train_module=True, calls=calls)
        m.get_patient_ids = lambda _data_dir: ["p1", "p2"]
        m.save_run_config = lambda *_args, **_kwargs: None

        args = types.SimpleNamespace(
            command="demo",
            patient_id=None,
            uncertainty="var",
            save_nifti=False,
            save_prob=False,
            device="cpu",
            seed=None,
            data_dir=None,
            output_dir=None,
            image_size=None,
            batch_size=None,
            epochs=None,
            lr=None,
            weight_decay=None,
            num_workers=None,
            mc_iterations=None,
            neg_slice_prob=None,
            no_proxy_cache=False,
        )
        m.demo_command(args)
        self.assertFalse(calls.get("trainer_train_called", False))


if __name__ == "__main__":
    unittest.main()
