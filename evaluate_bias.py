"""Bias-aware evaluation: slice error by pseudo-machine cluster, gender, age bin.

Answers the question "does prediction error depend on the X-ray machine?".
There is no machine label in the CSV, so we derive pseudo-machine clusters from
low-level intensity/resolution statistics of the ORIGINAL images (the machine
fingerprint) via KMeans, then report MAD and mean signed error per slice.

A machine-invariant model shows small spread (< ~1 month) across clusters.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import BoneAgeDataset, load_dataframes
from transforms import get_val_transforms
from model import BoneAgeModel
from utils import load_checkpoint
from evaluate import collect_predictions


def image_stats(path: Path) -> list:
    """Machine-fingerprint features computed on the raw (un-preprocessed) image."""
    g = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    p5, p50, p95 = np.percentile(g, [5, 50, 95])
    h, w = g.shape
    return [g.mean(), g.std(), p5, p50, p95, float((g < 10).mean()), h, w, h / w]


def derive_clusters(df, img_dir, n_clusters, seed) -> np.ndarray:
    feats = [image_stats(Path(img_dir) / f"{int(i)}.png")
             for i in tqdm(df["id"], desc="Image stats", file=sys.stdout)]
    X = StandardScaler().fit_transform(np.asarray(feats))
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    return km.fit_predict(X)


def slice_report(name, labels, abs_err, signed_err):
    print(f"\n{name}")
    print(f"  {'group':<14}{'n':>6}{'MAD':>9}{'mean err':>11}")
    mads, errs = [], []
    for g in sorted(pd.unique(labels), key=str):
        m = labels == g
        mad, err = abs_err[m].mean(), signed_err[m].mean()
        mads.append(mad); errs.append(err)
        print(f"  {str(g):<14}{int(m.sum()):>6}{mad:>9.2f}{err:>11.2f}")
    print(f"  -> MAD spread (max-min):      {max(mads) - min(mads):.2f} months")
    print(f"  -> mean-err spread (max-min): {max(errs) - min(errs):.2f} months")


def main():
    p = argparse.ArgumentParser(description="Bias-aware bone age evaluation")
    p.add_argument("--checkpoint", type=str, default=str(config.CHECKPOINT_DIR / "best_model.pth"))
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--no_preprocess", action="store_true", help="Disable bias-normalization pipeline")
    p.add_argument("--n_clusters", type=int, default=config.N_MACHINE_CLUSTERS)
    p.add_argument("--tta", action="store_true", help="Test-time augmentation (flip + rotations)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess = config.PREPROCESS and not args.no_preprocess

    _, val_df = load_dataframes(config.TRAIN_CSV, val_split=config.VAL_SPLIT, seed=config.SEED)
    val_df = val_df.reset_index(drop=True)

    val_ds = BoneAgeDataset(val_df, config.TRAIN_IMG_DIR, transform=get_val_transforms(),
                            cache_in_ram=False, preprocess=preprocess)
    num_workers = 0 if sys.platform == "win32" else config.NUM_WORKERS
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=(device.type == "cuda"))

    model = BoneAgeModel(pretrained=False).to(device)
    ckpt = load_checkpoint(Path(args.checkpoint), model, device=device)
    target_mean = ckpt.get("target_mean", 0.0)
    target_std = ckpt.get("target_std", 1.0)
    preds, targets = collect_predictions(model, val_loader, device, tta=args.tta)
    preds = preds * target_std + target_mean

    abs_err = np.abs(preds - targets)
    signed_err = preds - targets
    print(f"\nOverall MAD: {abs_err.mean():.2f} months  (n={len(preds)})")

    gender = np.where(val_df["male"].to_numpy().astype(bool), "male", "female")
    slice_report("By gender:", gender, abs_err, signed_err)

    age_bin = pd.cut(targets / 12.0, bins=[0, 4, 8, 12, 16, 100],
                     labels=["0-4y", "4-8y", "8-12y", "12-16y", "16y+"]).astype(str)
    slice_report("By age bin:", age_bin, abs_err, signed_err)

    clusters = derive_clusters(val_df, config.TRAIN_IMG_DIR, args.n_clusters, config.SEED)
    slice_report("By pseudo-machine cluster:",
                 np.array([f"c{c}" for c in clusters]), abs_err, signed_err)

    print("\nBias check: small cluster spreads (< ~1 month) => machine-invariant.")


if __name__ == "__main__":
    main()
