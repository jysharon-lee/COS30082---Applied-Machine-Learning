import os, random
from pathlib import Path
import numpy as np
import torch


class _Config:
    # ---- Paths: in Colab you only need to override DATA_ROOT, CKPT_DIR, RESULTS_DIR ----
    DATA_ROOT = Path("data")
    CKPT_DIR = Path("checkpoints")
    RESULTS_DIR = Path("results")

    @property
    def TRAIN_TXT(self):
        return self.DATA_ROOT / "train.txt"

    @property
    def TEST_TXT(self):
        return self.DATA_ROOT / "test.txt"

    # ---- Data / training ----
    NUM_CLASSES = 200
    SEED = 42
    VAL_FRAC = 0.15
    BATCH_SIZE = 32                      # T4 (16GB) + AMP: 32 is safe, try 64 for ResNet/EffNet
    NUM_WORKERS = min(8, os.cpu_count() or 2)
    UNIFORM_RES = True                   # True: all models 224px | False: EffNet-B2 at native 260px


CFG = _Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = CFG.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True