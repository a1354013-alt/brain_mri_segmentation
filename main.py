"""
Main CLI for Brain MRI Segmentation Project (v3.1 stable iteration).
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Avoid writing bytecode caches into the repo (helps keep the workspace clean on Windows/AV-restricted hosts).
sys.dont_write_bytecode = True

import config

# Avoid noisy matplotlib cache permission issues on some Windows environments.
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.getenv("TEMP", ".")) / "bms_mpl_cache"))


def worker_init_fn(worker_id: int) -> None:
    """Seed RNGs per DataLoader worker for better reproducibility."""
    seed = config.RANDOM_SEED + worker_id
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
    except Exception:
        pass


def get_patient_ids(data_dir: Path) -> list[str]:
    if not data_dir.exists():
        print(f"Error: Data directory not found at {data_dir}")
        return []
    return sorted([d.name for d in data_dir.iterdir() if d.is_dir()])


def quick_validate_two_phase(pid: str, *, require_tumor: bool) -> bool:
    """
    Two-phase patient validation for CLI auto-selection.

    Phase 1 (fast): file existence + NIfTI readability + shape sanity +
    modality/seg shape consistency.
    Phase 2 (strict): full seg scan for tumor presence
    (only if phase 1 passes and require_tumor=True).
    """
    from utils import BraTSDataset

    # Phase 1: structural validation only. Do not require tumor here
    # to avoid false negatives from sparse tumor slices.
    if not BraTSDataset.quick_validate_patient(
        config.DATA_DIR,
        pid,
        require_tumor=False,
        strict_tumor_check=False,
    ):
        return False

    if not require_tumor:
        return True

    return BraTSDataset.quick_validate_patient(
        config.DATA_DIR,
        pid,
        require_tumor=True,
        strict_tumor_check=True,
    )


def apply_overrides_from_args(args: argparse.Namespace) -> dict:
    overrides = {
        "data_dir": getattr(args, "data_dir", None),
        "output_dir": getattr(args, "output_dir", None),
        "device": getattr(args, "device", None),
        "IMAGE_SIZE": getattr(args, "image_size", None),
        "BATCH_SIZE": getattr(args, "batch_size", None),
        "EPOCHS": getattr(args, "epochs", None),
        "LEARNING_RATE": getattr(args, "lr", None),
        "WEIGHT_DECAY": getattr(args, "weight_decay", None),
        "NUM_WORKERS": getattr(args, "num_workers", None),
        "NEG_SLICE_PROB": getattr(args, "neg_slice_prob", None),
        "MC_ITERATIONS": getattr(args, "mc_iterations", None),
        "RANDOM_SEED": getattr(args, "seed", None),
        "USE_PROXY_CACHE": (False if getattr(args, "no_proxy_cache", False) else None),
    }
    overrides_applied = {k: v for k, v in overrides.items() if v is not None}
    config.apply_overrides(**overrides_applied)
    if getattr(args, "seed", None) is not None:
        config.set_seed(int(args.seed))
    else:
        config.set_seed()
    return overrides_applied


def _get_git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=config.PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        commit = result.stdout.strip()
        return commit or None
    except Exception:
        return None


def fail(message: str, exit_code: int = 1) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(exit_code)


def save_run_config(
    command: str,
    args: argparse.Namespace,
    overrides_applied: dict,
    *,
    model_info: dict | None = None,
    selected_patient_id: str | None = None,
    valid_patient_count: int | None = None,
) -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    args_whitelist = [
        "command",
        "patient_id",
        "uncertainty",
        "save_nifti",
        "save_prob",
        "device",
        "seed",
        "data_dir",
        "output_dir",
        "image_size",
        "batch_size",
        "epochs",
        "lr",
        "weight_decay",
        "num_workers",
        "mc_iterations",
        "neg_slice_prob",
        "no_proxy_cache",
    ]
    args_payload = {k: getattr(args, k, None) for k in args_whitelist}

    model_payload = {
        "model_loaded": None,
        "weights_source": None,
        "checkpoint_path": None,
    }
    if isinstance(model_info, dict):
        model_payload.update({k: model_info.get(k) for k in model_payload.keys()})

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "project_version": config.PROJECT_VERSION,
        "python_version": platform.python_version(),
        "git_commit_hash": _get_git_commit_hash(),
        "command": command,
        "selected_patient_id": selected_patient_id,
        "valid_patient_count": valid_patient_count,
        "args": args_payload,
        "overrides_applied": overrides_applied,
        "config": {
            "DATA_DIR": str(config.DATA_DIR),
            "OUTPUT_DIR": str(config.OUTPUT_DIR),
            "DEVICE": str(config.DEVICE),
            "IMAGE_SIZE": config.IMAGE_SIZE,
            "BATCH_SIZE": config.BATCH_SIZE,
            "EPOCHS": config.EPOCHS,
            "LEARNING_RATE": config.LEARNING_RATE,
            "WEIGHT_DECAY": config.WEIGHT_DECAY,
            "NUM_WORKERS": config.NUM_WORKERS,
            "USE_PROXY_CACHE": config.USE_PROXY_CACHE,
            "NEG_SLICE_PROB": config.NEG_SLICE_PROB,
            "MC_ITERATIONS": config.MC_ITERATIONS,
            "RANDOM_SEED": config.RANDOM_SEED,
            "DICE_WEIGHT": config.DICE_WEIGHT,
            "BCE_WEIGHT": config.BCE_WEIGHT,
        },
        "model": model_payload,
    }
    out_path = config.OUTPUT_DIR / f"run_config_{command}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def _torch_safe_load(path: Path, device):
    """
    Best-effort safe load across PyTorch versions.

    If supported, try `weights_only=True` first (safer: avoids unpickling arbitrary
    objects), then fall back to legacy loading for older checkpoints or file formats.
    """
    import torch

    kwargs = {"map_location": device}
    try:
        if "weights_only" in inspect.signature(torch.load).parameters:
            try:
                return torch.load(path, **kwargs, weights_only=True)
            except Exception:
                return torch.load(path, **kwargs)
    except (TypeError, ValueError):
        pass
    return torch.load(path, **kwargs)


def _is_state_dict(obj) -> bool:
    if not isinstance(obj, dict) or not obj:
        return False
    if not all(isinstance(k, str) for k in obj.keys()):
        return False
    try:
        import torch

        return any(torch.is_tensor(v) for v in obj.values())
    except Exception:
        return False


def load_model_weights(model, *, device) -> dict:
    """
    Load model weights deterministically and return audit metadata.

    Priority:
    1. `best_model_state.pth` as a raw state_dict
    2. `best_checkpoint.pth` as a checkpoint dict with `model_state_dict`
    3. random initialization fallback
    """
    if config.MODEL_STATE_PATH.exists():
        obj = _torch_safe_load(config.MODEL_STATE_PATH, device)
        if not _is_state_dict(obj):
            raise ValueError(f"Expected state_dict at {config.MODEL_STATE_PATH}, got {type(obj)}")
        model.load_state_dict(obj)
        print(f"Loaded model state from {config.MODEL_STATE_PATH}")
        return {
            "model_loaded": True,
            "weights_source": "best_model_state",
            "checkpoint_path": str(config.MODEL_STATE_PATH),
        }

    if config.CHECKPOINT_PATH.exists():
        obj = _torch_safe_load(config.CHECKPOINT_PATH, device)
        if not isinstance(obj, dict) or "model_state_dict" not in obj:
            raise ValueError(f"Expected checkpoint dict with 'model_state_dict' at {config.CHECKPOINT_PATH}")
        state_dict = obj["model_state_dict"]
        if not _is_state_dict(state_dict):
            raise ValueError(f"Checkpoint 'model_state_dict' is not state_dict-like at {config.CHECKPOINT_PATH}")
        model.load_state_dict(state_dict)
        print(f"Loaded model state from checkpoint {config.CHECKPOINT_PATH}")
        return {
            "model_loaded": True,
            "weights_source": "checkpoint",
            "checkpoint_path": str(config.CHECKPOINT_PATH),
        }

    print("Warning: No model found. Using random weights.")
    return {
        "model_loaded": False,
        "weights_source": "random_init",
        "checkpoint_path": None,
    }


def enforce_python_version_policy(command: str | None) -> None:
    """
    Prefer Python 3.10/3.11 for full CLI runs.

    Some lightweight tests may still pass on newer Python versions, but the full CLI
    depends on PyTorch and the scientific stack, which can lag behind new interpreter releases.
    """
    if sys.version_info < (3, 13):
        return
    if command not in {"train", "infer", "demo"}:
        return

    print("Error: Python 3.13+ is not a supported environment for the full CLI in this project.")
    print("Recommended: Python 3.10 or 3.11 in a clean virtual environment.")
    print("Note: some lightweight tests may still pass on newer Python versions,")
    print("but train / infer / demo depend on PyTorch and scientific packages that may not.")
    print(f"Detected Python: {sys.version.split()[0]}")
    raise SystemExit(2)


def train_command(args: argparse.Namespace) -> None:
    print(f"\nTraining Mode ({config.PROJECT_VERSION})")
    overrides_applied = apply_overrides_from_args(args)

    from torch.utils.data import DataLoader

    from models import AttentionUNet

    try:
        from utils import BraTSDataset
    except ModuleNotFoundError as e:
        fail(f"Error: Missing dependency: {e.name}. Install: pip install -r requirements.txt", 2)
    except MemoryError:
        fail(
            "Error: Failed to import dataset utilities due to MemoryError. "
            "Please use a less constrained environment.",
            2,
        )

    patient_ids = get_patient_ids(config.DATA_DIR)
    if len(patient_ids) == 0:
        fail("Error: No data found in DATA_DIR. Please run download script first.", 1)
    if len(patient_ids) < 2:
        fail("Error: Need at least 2 patients to create a non-empty train/val split.", 1)

    rnd = random.Random(int(config.RANDOM_SEED))
    rnd.shuffle(patient_ids)

    split_idx = int(len(patient_ids) * float(config.TRAIN_VAL_SPLIT))
    split_idx = max(1, min(len(patient_ids) - 1, split_idx))
    train_ids = patient_ids[:split_idx]
    val_ids = patient_ids[split_idx:]

    print(f"Data Split: {len(train_ids)} train, {len(val_ids)} val")
    print(f"Train PIDs (first 3): {train_ids[:3]}")
    print(f"Val PIDs (first 3): {val_ids[:3]}")

    # Build train/val caches independently to avoid train-cache-only ids leaking into val.
    train_dataset = BraTSDataset(
        config.DATA_DIR,
        train_ids,
        config.IMAGE_SIZE,
        mode="train",
        output_dir=config.OUTPUT_DIR,
    )
    val_dataset = BraTSDataset(
        config.DATA_DIR,
        val_ids,
        config.IMAGE_SIZE,
        mode="val",
        output_dir=config.OUTPUT_DIR,
    )

    if len(train_dataset) == 0:
        fail("Error: Train dataset contains 0 valid patients after scanning. Check data integrity.", 1)
    if len(val_dataset) == 0:
        fail("Error: Val dataset contains 0 valid patients after scanning. Check data integrity or split seed.", 1)

    save_run_config(
        "train",
        args,
        overrides_applied=overrides_applied,
        valid_patient_count=len(train_dataset) + len(val_dataset),
    )

    dl_kwargs = {
        "num_workers": int(config.NUM_WORKERS),
        "worker_init_fn": worker_init_fn,
    }
    if int(config.NUM_WORKERS) > 0:
        dl_kwargs["persistent_workers"] = bool(config.PERSISTENT_WORKERS)
        dl_kwargs["prefetch_factor"] = int(config.PREFETCH_FACTOR)
    if config.DEVICE.type == "cuda":
        dl_kwargs["pin_memory"] = bool(config.PIN_MEMORY)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, **dl_kwargs)

    model = AttentionUNet(config.N_CHANNELS, config.N_CLASSES, config.DROPOUT_P).to(config.DEVICE)
    use_amp = config.DEVICE.type == "cuda"

    try:
        from train import Trainer
    except ModuleNotFoundError as e:
        fail(f"Error: Missing training dependency: {e.name}. Install: pip install -r requirements.txt", 2)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config.DEVICE,
        output_dir=config.OUTPUT_DIR,
        checkpoint_path=config.CHECKPOINT_PATH,
        model_state_path=config.MODEL_STATE_PATH,
        last_checkpoint_path=config.LAST_CHECKPOINT_PATH,
        last_model_state_path=config.LAST_MODEL_STATE_PATH,
        log_file=config.LOG_FILE,
        tensorboard_dir=config.TENSORBOARD_DIR,
        use_amp=use_amp,
        total_epochs=config.EPOCHS,
    )
    trainer.train()


def infer_command(args: argparse.Namespace) -> None:
    print(f"\nInference Mode ({config.PROJECT_VERSION})")
    overrides_applied = apply_overrides_from_args(args)

    try:
        import numpy as np
    except MemoryError:
        fail("Error: NumPy failed to import due to MemoryError. Please use a less constrained environment.", 2)

    from models import AttentionUNet

    try:
        from utils import (
            BraTSDataset,
            mc_dropout_inference,
            plot_results_with_uncertainty,
            predict_patient_volume,
            save_nifti_like,
        )
    except ModuleNotFoundError as e:
        fail(f"Error: Missing dependency: {e.name}. Install: pip install -r requirements.txt", 2)
    except MemoryError:
        fail(
            "Error: Failed to import inference utilities due to MemoryError. "
            "Please use a less constrained environment.",
            2,
        )

    model = AttentionUNet(config.N_CHANNELS, config.N_CLASSES, config.DROPOUT_P).to(config.DEVICE)
    model_info = load_model_weights(model, device=config.DEVICE)

    patient_ids = get_patient_ids(config.DATA_DIR)
    if len(patient_ids) == 0:
        fail("Error: No data found in DATA_DIR.", 1)

    target_patient = args.patient_id
    dataset = None

    if target_patient:
        if quick_validate_two_phase(target_patient, require_tumor=True):
            dataset = BraTSDataset(
                config.DATA_DIR,
                [target_patient],
                config.IMAGE_SIZE,
                mode="val",
                output_dir=config.OUTPUT_DIR,
            )
        else:
            fail(f"Error: Patient {target_patient} is invalid, missing, or incompatible. Please provide a valid --patient_id.", 1)

    if target_patient is None:
        for pid in patient_ids:
            if quick_validate_two_phase(pid, require_tumor=True):
                target_patient = pid
                dataset = BraTSDataset(
                    config.DATA_DIR,
                    [pid],
                    config.IMAGE_SIZE,
                    mode="val",
                    output_dir=config.OUTPUT_DIR,
                )
                print(f"Automatically selected valid patient: {target_patient}")
                break

    if dataset is None or len(dataset) == 0:
        fail("Error: No valid patients found in DATA_DIR.", 1)

    save_run_config(
        "infer",
        args,
        overrides_applied=overrides_applied,
        model_info=model_info,
        selected_patient_id=target_patient,
        valid_patient_count=len(dataset),
    )

    image, mask = dataset[0]
    prediction, uncertainty = mc_dropout_inference(
        model,
        image.unsqueeze(0),
        n_iterations=config.MC_ITERATIONS,
        device=config.DEVICE,
        method=args.uncertainty,
    )

    save_path = config.OUTPUT_DIR / "inference" / f"{target_patient}_seg.png"
    plot_results_with_uncertainty(image.numpy(), mask.numpy(), prediction[0], uncertainty[0], save_path=save_path)
    print(f"Saved to {save_path}")

    if args.save_nifti:
        norm_stats = None
        try:
            norm_stats = dataset.patient_cache.get(target_patient, {}).get("norm_stats")
        except Exception:
            norm_stats = None

        pred_vol, unc_vol, prob_vol, ref_img = predict_patient_volume(
            model=model,
            data_dir=config.DATA_DIR,
            pid=target_patient,
            image_size=config.IMAGE_SIZE,
            device=config.DEVICE,
            n_iterations=config.MC_ITERATIONS,
            method=args.uncertainty,
            norm_stats=norm_stats,
        )
        out_dir = config.OUTPUT_DIR / "inference_nifti"
        save_nifti_like(ref_img, pred_vol.astype(np.uint8), out_dir / f"{target_patient}_pred.nii.gz")
        save_nifti_like(ref_img, unc_vol.astype(np.float32), out_dir / f"{target_patient}_uncertainty.nii.gz")
        if args.save_prob:
            save_nifti_like(ref_img, prob_vol.astype(np.float32), out_dir / f"{target_patient}_prob.nii.gz")
        print(f"Saved NIfTI outputs to {out_dir}")


def demo_command(args: argparse.Namespace) -> None:
    print(f"\nDemo Mode ({config.PROJECT_VERSION})")
    overrides_applied = apply_overrides_from_args(args)

    from torch.utils.data import DataLoader

    from models import AttentionUNet

    try:
        from utils import BraTSDataset, mc_dropout_inference, plot_results_with_uncertainty
    except ModuleNotFoundError as e:
        fail(f"Error: Missing dependency: {e.name}. Install: pip install -r requirements.txt", 2)
    except MemoryError:
        fail("Error: Failed to import demo utilities due to MemoryError. Please use a less constrained environment.", 2)

    patient_ids = get_patient_ids(config.DATA_DIR)
    if len(patient_ids) == 0:
        fail("Error: No data found in DATA_DIR.", 1)

    demo_ids = []
    for pid in patient_ids:
        if quick_validate_two_phase(pid, require_tumor=True):
            demo_ids.append(pid)
        if len(demo_ids) >= 2:
            break

    if not demo_ids:
        fail("Error: No valid patients found for Demo.", 1)

    train_dataset = BraTSDataset(
        config.DATA_DIR,
        demo_ids,
        config.IMAGE_SIZE,
        mode="train",
        output_dir=config.DEMO_OUTPUT_DIR,
    )
    if len(train_dataset) == 0:
        fail("Error: Demo dataset contains 0 valid patients after scanning. Check data integrity.", 1)

    save_run_config(
        "demo",
        args,
        overrides_applied=overrides_applied,
        valid_patient_count=len(train_dataset),
    )
    demo_loader = DataLoader(train_dataset, batch_size=1, num_workers=0, worker_init_fn=worker_init_fn)

    model = AttentionUNet(config.N_CHANNELS, config.N_CLASSES, config.DROPOUT_P).to(config.DEVICE)

    try:
        from train import Trainer
    except ModuleNotFoundError as e:
        fail(f"Error: Missing training dependency: {e.name}. Install: pip install -r requirements.txt", 2)

    trainer = Trainer(
        model=model,
        train_loader=demo_loader,
        val_loader=None,
        device=config.DEVICE,
        output_dir=config.DEMO_OUTPUT_DIR,
        checkpoint_path=config.DEMO_CHECKPOINT_PATH,
        model_state_path=config.DEMO_MODEL_STATE_PATH,
        last_checkpoint_path=config.DEMO_LAST_CHECKPOINT_PATH,
        last_model_state_path=config.DEMO_LAST_MODEL_STATE_PATH,
        log_file=config.DEMO_LOG_FILE,
        tensorboard_dir=None,
        use_amp=False,
        total_epochs=1,
    )
    trainer.train()

    # Run one demo inference example after the short training pass.
    image, mask = train_dataset[0]
    prediction, uncertainty = mc_dropout_inference(
        model,
        image.unsqueeze(0),
        n_iterations=config.DEMO_MC_ITERATIONS,
        device=config.DEVICE,
    )
    save_path = config.DEMO_OUTPUT_DIR / "demo_inference.png"
    plot_results_with_uncertainty(image.numpy(), mask.numpy(), prediction[0], uncertainty[0], save_path=save_path)
    print(f"Demo completed. Results in {config.DEMO_OUTPUT_DIR}")


def main() -> int:
    parser = argparse.ArgumentParser(description=f"Brain MRI Segmentation ({config.PROJECT_VERSION})")
    subparsers = parser.add_subparsers(dest="command")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data_dir", type=str, default=None)
    common.add_argument("--output_dir", type=str, default=None)
    common.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None)
    common.add_argument("--image_size", type=int, default=None)
    common.add_argument("--batch_size", type=int, default=None)
    common.add_argument("--epochs", type=int, default=None)
    common.add_argument("--lr", type=float, default=None)
    common.add_argument("--weight_decay", type=float, default=None)
    common.add_argument("--num_workers", type=int, default=None)
    common.add_argument("--seed", type=int, default=None)
    common.add_argument("--mc_iterations", type=int, default=None)
    common.add_argument("--neg_slice_prob", type=float, default=None)
    common.add_argument("--no_proxy_cache", action="store_true")

    subparsers.add_parser("train", parents=[common])

    infer_p = subparsers.add_parser("infer", parents=[common])
    infer_p.add_argument("--patient_id", type=str)
    infer_p.add_argument("--uncertainty", choices=["var", "entropy"], default="var")
    infer_p.add_argument("--save_nifti", action="store_true", help="Save 3D NIfTI outputs (pred/uncertainty).")
    infer_p.add_argument("--save_prob", action="store_true", help="Also save mean probability NIfTI.")

    subparsers.add_parser("demo", parents=[common])

    args = parser.parse_args()
    enforce_python_version_policy(args.command)

    if args.command == "train":
        train_command(args)
    elif args.command == "infer":
        infer_command(args)
    elif args.command == "demo":
        demo_command(args)
    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    sys.exit(main())
