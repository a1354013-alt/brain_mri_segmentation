"""
Configuration file for Brain MRI Segmentation Project
"""
import torch
import random
import numpy as np

# ==================== 路徑設定 ====================
DATA_DIR = "data/Brats"
OUTPUT_DIR = "outputs"
MODEL_SAVE_PATH = "outputs/best_model.pth"
LOG_FILE = "outputs/training_log.csv"
TENSORBOARD_DIR = "outputs/tensorboard"

# ==================== 模型參數 ====================
N_CHANNELS = 4  # FLAIR, T1, T1ce, T2
N_CLASSES = 1   # Binary segmentation
DROPOUT_P = 0.2

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

# ==================== 裝置設定 ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== Random Seed ====================
RANDOM_SEED = 42

def set_seed(seed: int = RANDOM_SEED) -> None:
    """
    固定所有隨機種子以確保可重現性
    
    Args:
        seed: 隨機種子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
