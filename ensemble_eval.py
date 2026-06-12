"""Evaluate an ensemble of checkpoints.

Averages per-image predictions across all member models, then reports overall
MAD plus bias slices (gender / age bin / pseudo-machine cluster) — the same
bias-aware view as evaluate_bias.py, so numbers are directly comparable to the
single-model results.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config
from dataset import BoneAgeDataset, load_dataframes
from transforms import get_val_transforms
from model import BoneAgeModel
from utils import load_checkpoint
from evaluate import collect_predictions
from evaluate_bias import slice_report, derive_clusters


def main():
    p = argparse.ArgumentParser(description="Ensemble bone age evaluation")
    p.add_argument("--checkpoints", type=str, required=True,
                   help="Comma-separated checkpoint paths")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--tta", action="store_true", help="Test-time augmentation per member")
    p.add_argument("--n_clusters", type=int, default=config.N_MACHINE_CLUSTERS)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = [s.strip() for s in args.checkpoints.split(",") if s.strip()]
    print(f"Ensembling {len(paths)} models: {paths}")

    _, val_df = load_dataframes(config.TRAIN_CSV, val_split=config.VAL_SPLIT, seed=config.SEED)
    val_df = val_df.reset_index(drop=True)
    val_ds = BoneAgeDataset(val_df, config.TRAIN_IMG_DIR, transform=get_val_transforms(),
                            cache_in_ram=False, preprocess=False)
    nw = 0 if sys.platform == "win32" else config.NUM_WORKERS
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=nw, pin_memory=(device.type == "cuda"))

    model = BoneAgeModel(pretrained=False).to(device)
    member_preds, targets = [], None
    for path in paths:
        ckpt = load_checkpoint(Path(path), model, device=device)
        mean = ckpt.get("target_mean", 0.0)
        std = ckpt.get("target_std", 1.0)
        preds, tg = collect_predictions(model, val_loader, device, tta=args.tta)
        preds = preds * std + mean
        member_preds.append(preds)
        targets = tg
        print(f"  member {Path(path).name}: MAD {np.abs(preds - tg).mean():.2f} months")

    preds = np.mean(member_preds, axis=0)
    abs_err = np.abs(preds - targets)
    signed = preds - targets
    print(f"\nENSEMBLE ({len(paths)} models) Overall MAD: {abs_err.mean():.2f} months  (n={len(preds)})")

    gender = np.where(val_df["male"].to_numpy().astype(bool), "male", "female")
    slice_report("By gender:", gender, abs_err, signed)

    age_bin = pd.cut(targets / 12.0, bins=[0, 4, 8, 12, 16, 100],
                     labels=["0-4y", "4-8y", "8-12y", "12-16y", "16y+"]).astype(str)
    slice_report("By age bin:", age_bin, abs_err, signed)

    clusters = derive_clusters(val_df, config.TRAIN_IMG_DIR, args.n_clusters, config.SEED)
    slice_report("By pseudo-machine cluster:",
                 np.array([f"c{c}" for c in clusters]), abs_err, signed)

    print("\nBias check: small cluster spreads (< ~1 month) => machine-invariant.")


if __name__ == "__main__":
    main()
