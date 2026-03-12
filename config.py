"""
Configuration file for Brain MRI Segmentation Project (v2.3)
"""
import torch
import random
import numpy as np
from pathlib import Path

# ==================== 路徑設定 ====================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "Brats"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_PATH = OUTPUT_DIR / "best_checkpoint.pth"
MODEL_STATE_PATH = OUTPUT_DIR / "best_model_state.pth"
LOG_FILE = OUTPUT_DIR / "training_log.csv"
TENSORBOARD_DIR = OUTPUT_DIR / "tensorboard"
SKIPPED_LOG = OUTPUT_DIR / "skipped_patients.txt"

# Demo 模式路徑
DEMO_OUTPUT_DIR = OUTPUT_DIR / "demo"
DEMO_CHECKPOINT_PATH = DEMO_OUTPUT_DIR / "best_checkpoint_demo.pth"
DEMO_MODEL_STATE_PATH = DEMO_OUTPUT_DIR / "best_model_state_demo.pth"
DEMO_TENSORBOARD_DIR = DEMO_OUTPUT_DIR / "tensorboard"
DEMO_LOG_FILE = DEMO_OUTPUT_DIR / "training_log_demo.csv"

# ==================== 模型參數 ====================
N_CHANNELS = 4  # FLAIR, T1, T1ce, T2
N_CLASSES = 1   # Binary segmentation (Whole Tumor)
DROPOUT_P = 0.2
THRESHOLD = 0.5

# ==================== 訓練參數 ====================
IMAGE_SIZE = 128
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP_VALUE = 1.0

# ==================== 資料增強參數 ====================
TRAIN_VAL_SPLIT = 0.8
NUM_WORKERS = 4

# ==================== MC Dropout 參數 ====================
MC_ITERATIONS = 20
DEMO_MC_ITERATIONS = 5

# ==================== 裝置設定 ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== Random Seed ====================
RANDOM_SEED = 42

def set_seed(seed: int = RANDOM_SEED) -> None:
    """
    固定所有隨機種子以確保可重現性
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
