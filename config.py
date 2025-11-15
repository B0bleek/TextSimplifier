# config.py
import os
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"

# модели
MODEL_NAME = "cointegrated/rut5-base"
MAX_SOURCE_LEN = 512
MAX_TARGET_LEN = 128

BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 8
EPOCHS = 3
LR = 3e-5
SEED = 42

# Включаем GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Создаём папки
MODELS_DIR.mkdir(exist_ok=True, parents=True)