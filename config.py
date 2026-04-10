"""
Project configuration for Brain MRI Segmentation (v3.1 stable iteration).
"""

from __future__ import annotations

import os
import random
from pathlib import Path

# Keep CPU thread usage conservative by default. This reduces memory pressure on constrained hosts.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "Brats"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Release / runtime metadata
PROJECT_VERSION = "v3.1 stable iteration"
CHECKPOINT_PATH = OUTPUT_DIR / "best_checkpoint.pth"
LAST_CHECKPOINT_PATH = OUTPUT_DIR / "last_checkpoint.pth"
MODEL_STATE_PATH = OUTPUT_DIR / "best_model_state.pth"
LAST_MODEL_STATE_PATH = OUTPUT_DIR / "last_model_state.pth"
LOG_FILE = OUTPUT_DIR / "training_log.csv"
TENSORBOARD_DIR = OUTPUT_DIR / "tensorboard"
SKIPPED_LOG = OUTPUT_DIR / "skipped_patients.txt"

# Demo output paths
DEMO_OUTPUT_DIR = OUTPUT_DIR / "demo"
DEMO_CHECKPOINT_PATH = DEMO_OUTPUT_DIR / "best_checkpoint_demo.pth"
DEMO_MODEL_STATE_PATH = DEMO_OUTPUT_DIR / "best_model_state_demo.pth"
DEMO_LAST_CHECKPOINT_PATH = DEMO_OUTPUT_DIR / "last_checkpoint_demo.pth"
DEMO_LAST_MODEL_STATE_PATH = DEMO_OUTPUT_DIR / "last_model_state_demo.pth"
DEMO_TENSORBOARD_DIR = DEMO_OUTPUT_DIR / "tensorboard"
DEMO_LOG_FILE = DEMO_OUTPUT_DIR / "training_log_demo.csv"

# Model
N_CHANNELS = 4  # FLAIR, T1, T1ce, T2
N_CLASSES = 1  # Binary segmentation (whole tumor vs background)
DROPOUT_P = 0.2
THRESHOLD = 0.5
OVERLAY_ALPHA = 0.35

# Training
IMAGE_SIZE = 128
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP_VALUE = 1.0

# Loss mixing (logits-based BCE + Dice).
DICE_WEIGHT = 1.0
BCE_WEIGHT = 0.5

# Data / loading
TRAIN_VAL_SPLIT = 0.8
NUM_WORKERS = 4
USE_PROXY_CACHE = True  # Reuse nibabel proxies to reduce repeated file-open overhead.
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2

# Slice sampling: mix tumor / non-tumor slices during training to reduce false positives.
NEG_SLICE_PROB = 0.3

# Dataset statistics: slices sampled per patient per modality to estimate normalization stats.
STATS_N_SLICES = 16

# MC Dropout
MC_ITERATIONS = 20
DEMO_MC_ITERATIONS = 5

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Randomness
RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible runs."""
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_paths(output_dir: Path) -> None:
    """Update OUTPUT_DIR and all derived output paths."""
    global OUTPUT_DIR
    global CHECKPOINT_PATH, LAST_CHECKPOINT_PATH, MODEL_STATE_PATH, LAST_MODEL_STATE_PATH
    global LOG_FILE, TENSORBOARD_DIR, SKIPPED_LOG
    global DEMO_OUTPUT_DIR, DEMO_CHECKPOINT_PATH, DEMO_MODEL_STATE_PATH
    global DEMO_LAST_CHECKPOINT_PATH, DEMO_LAST_MODEL_STATE_PATH, DEMO_TENSORBOARD_DIR, DEMO_LOG_FILE

    OUTPUT_DIR = Path(output_dir).resolve()
    CHECKPOINT_PATH = OUTPUT_DIR / "best_checkpoint.pth"
    LAST_CHECKPOINT_PATH = OUTPUT_DIR / "last_checkpoint.pth"
    MODEL_STATE_PATH = OUTPUT_DIR / "best_model_state.pth"
    LAST_MODEL_STATE_PATH = OUTPUT_DIR / "last_model_state.pth"
    LOG_FILE = OUTPUT_DIR / "training_log.csv"
    TENSORBOARD_DIR = OUTPUT_DIR / "tensorboard"
    SKIPPED_LOG = OUTPUT_DIR / "skipped_patients.txt"

    DEMO_OUTPUT_DIR = OUTPUT_DIR / "demo"
    DEMO_CHECKPOINT_PATH = DEMO_OUTPUT_DIR / "best_checkpoint_demo.pth"
    DEMO_MODEL_STATE_PATH = DEMO_OUTPUT_DIR / "best_model_state_demo.pth"
    DEMO_LAST_CHECKPOINT_PATH = DEMO_OUTPUT_DIR / "last_checkpoint_demo.pth"
    DEMO_LAST_MODEL_STATE_PATH = DEMO_OUTPUT_DIR / "last_model_state_demo.pth"
    DEMO_TENSORBOARD_DIR = DEMO_OUTPUT_DIR / "tensorboard"
    DEMO_LOG_FILE = DEMO_OUTPUT_DIR / "training_log_demo.csv"


def set_data_dir(data_dir: Path) -> None:
    """Update DATA_DIR for CLI overrides."""
    global DATA_DIR
    DATA_DIR = Path(data_dir).resolve()


def set_device(device: str | None) -> None:
    """Override DEVICE selection (`cpu`, `cuda`, or None)."""
    global DEVICE
    if device is None:
        return
    device = device.lower().strip()
    if device == "cpu":
        DEVICE = torch.device("cpu")
    elif device == "cuda":
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        raise ValueError(f"Unknown device override: {device}")


def apply_overrides(**kwargs) -> None:
    """
    Best-effort config overrides from CLI. Unknown keys are ignored.

    This keeps the project simple without adding extra configuration dependencies.
    """
    if kwargs.get("output_dir") is not None:
        set_paths(Path(kwargs["output_dir"]))
    if kwargs.get("data_dir") is not None:
        set_data_dir(Path(kwargs["data_dir"]))
    if kwargs.get("device") is not None:
        set_device(str(kwargs["device"]))

    for key in [
        "IMAGE_SIZE",
        "BATCH_SIZE",
        "EPOCHS",
        "LEARNING_RATE",
        "WEIGHT_DECAY",
        "NUM_WORKERS",
        "USE_PROXY_CACHE",
        "PIN_MEMORY",
        "PERSISTENT_WORKERS",
        "PREFETCH_FACTOR",
        "NEG_SLICE_PROB",
        "RANDOM_SEED",
        "MC_ITERATIONS",
        "DICE_WEIGHT",
        "BCE_WEIGHT",
        "STATS_N_SLICES",
    ]:
        if kwargs.get(key) is not None:
            globals()[key] = kwargs[key]
