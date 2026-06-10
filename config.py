"""Hyperparameters and path configuration for bone age model."""

import argparse
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_CSV = DATA_DIR / "boneage-training-dataset.csv"
TRAIN_IMG_DIR = DATA_DIR / "boneage-training-dataset" / "boneage-training-dataset"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_NAME = "efficientnet_b4"
PRETRAINED = True
NUM_FEATURES = 1792  # EfficientNet-B4 feature dim after global avg pool
HIDDEN_DIM = 512
DROPOUT = 0.3

# ── Data ───────────────────────────────────────────────────────────────────────
IMG_SIZE = 500
VAL_SPLIT = 0.15  # fraction of training data used for validation
NUM_WORKERS = min(4, os.cpu_count() // 4 or 1) if os.cpu_count() else 2
PERSISTENT_WORKERS = True  # keep worker processes alive between batches (Windows-safe)

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 4  # effective batch = BATCH_SIZE × GRAD_ACCUM_STEPS = 64
EPOCHS = 35
WARMUP_EPOCHS = 5  # backbone frozen
LR = 1e-3          # warmup phase LR
FINETUNE_LR = 1e-4 # fine-tune phase LR
WEIGHT_DECAY = 1e-4
SEED = 42
USE_AMP = True     # mixed precision

# ── ImageNet normalization ─────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    """Override config values from the command line."""
    parser = argparse.ArgumentParser(description="Bone Age Training Config")
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad_accum_steps", type=int, default=GRAD_ACCUM_STEPS,
                        help="Gradient accumulation steps (effective batch = batch_size × this)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--warmup_epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--finetune_lr", type=float, default=FINETUNE_LR)
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    parser.add_argument("--val_split", type=float, default=VAL_SPLIT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--compile", action="store_true", help="Compile model with torch.compile")
    parser.add_argument("--no_prefetch", action="store_true", help="Disable asynchronous CUDA data prefetching")
    return parser.parse_args()
