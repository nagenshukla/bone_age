"""Utility functions: checkpointing, seeding, logging."""

import random
import torch
import numpy as np
from pathlib import Path


def set_seed(seed: int):
    """Reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # auto-tune convolutions for fixed input size


def save_checkpoint(model, optimizer, epoch, val_mad, path: Path):
    """Save model + optimizer state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_mad": val_mad,
        },
        path,
    )
    print(f"  Checkpoint saved -> {path}  (MAD={val_mad:.2f} months)")


def load_checkpoint(path: Path, model, optimizer=None, device="cuda"):
    """Load model (and optionally optimizer) from checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"  Loaded checkpoint from epoch {ckpt['epoch']} (MAD={ckpt['val_mad']:.2f})")
    return ckpt["epoch"], ckpt["val_mad"]


class EarlyStopping:
    """Stop training when validation MAD stops improving."""

    def __init__(self, patience: int = 7, min_delta: float = 0.05):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None

    def __call__(self, val_mad: float) -> bool:
        if self.best_score is None or val_mad < self.best_score - self.min_delta:
            self.best_score = val_mad
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


class AverageMeter:
    """Track running average of a metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


class CudaPrefetcher:
    """Prefetch batches to GPU using a separate CUDA stream.

    Overlaps CPU→GPU data transfer with GPU computation so the GPU
    never waits for host-side copies.  Works with num_workers=0.
    """

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream()

    def __iter__(self):
        self._iter = iter(self.loader)
        self._preload()
        return self

    def _preload(self):
        try:
            self._next = next(self._iter)
        except StopIteration:
            self._next = None
            return
        with torch.cuda.stream(self.stream):
            self._next = tuple(
                t.to(self.device, non_blocking=True) if isinstance(t, torch.Tensor) else t
                for t in self._next
            )

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self._next
        if batch is None:
            raise StopIteration
        self._preload()
        return batch

    def __len__(self):
        return len(self.loader)
