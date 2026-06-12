"""Evaluate a trained model: metrics, scatter plots, sample predictions."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import BoneAgeDataset, load_dataframes
from transforms import get_val_transforms
from model import BoneAgeModel
from utils import load_checkpoint


@torch.no_grad()
def tta_forward(model, images, genders):
    """Average predictions over horizontal flip + small rotations (image-only TTA)."""
    import torchvision.transforms.functional as TF
    views = [images, torch.flip(images, dims=[3]),
             TF.rotate(images, 10), TF.rotate(images, -10)]
    return torch.stack([model(v, genders) for v in views], dim=0).mean(dim=0)


@torch.no_grad()
def collect_predictions(model, loader, device, tta=False):
    """Run inference on a dataloader, return (predictions, targets) in months."""
    model.eval()
    all_preds, all_targets = [], []

    for images, genders, targets in tqdm(loader, desc="Evaluating", file=sys.stdout):
        images = images.to(device, non_blocking=True)
        genders = genders.to(device, non_blocking=True)

        out = tta_forward(model, images, genders) if tta else model(images, genders)
        all_preds.append(out.squeeze(1).cpu().numpy())
        all_targets.append(targets.numpy())

    return np.concatenate(all_preds), np.concatenate(all_targets)


def compute_metrics(preds, targets):
    """Compute regression metrics."""
    errors = preds - targets
    abs_errors = np.abs(errors)

    metrics = {
        "MAD (months)": abs_errors.mean(),
        "Median AE (months)": np.median(abs_errors),
        "RMSE (months)": np.sqrt((errors ** 2).mean()),
        "Within 12 months (%)": (abs_errors < 12).mean() * 100,
        "Within 6 months (%)": (abs_errors < 6).mean() * 100,
        "N samples": len(preds),
    }
    return metrics


def plot_scatter(preds, targets, save_path: Path):
    """Predicted vs actual bone age scatter plot."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(targets, preds, alpha=0.3, s=10, c="steelblue")

    # Perfect prediction line
    lims = [0, max(targets.max(), preds.max()) + 10]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")

    ax.set_xlabel("Actual Bone Age (months)")
    ax.set_ylabel("Predicted Bone Age (months)")
    ax.set_title("Bone Age: Predicted vs Actual")
    ax.legend()
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Scatter plot saved -> {save_path}")
    plt.close(fig)


def plot_error_distribution(preds, targets, save_path: Path):
    """Histogram of prediction errors."""
    errors = preds - targets

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(errors, bins=60, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Prediction Error (months)")
    ax.set_ylabel("Count")
    ax.set_title(f"Error Distribution (MAD={np.abs(errors).mean():.2f} months)")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Error histogram saved -> {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate bone age model")
    parser.add_argument(
        "--checkpoint", type=str,
        default=str(config.CHECKPOINT_DIR / "best_model.pth"),
        help="Path to model checkpoint",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--visualize", action="store_true", help="Generate plots")
    parser.add_argument("--no_preprocess", action="store_true", help="Disable bias-normalization preprocessing")
    parser.add_argument("--tta", action="store_true", help="Test-time augmentation (flip + rotations)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data (validation split)
    _, val_df = load_dataframes(config.TRAIN_CSV, val_split=config.VAL_SPLIT, seed=config.SEED)
    preprocess = config.PREPROCESS and not args.no_preprocess
    val_ds = BoneAgeDataset(val_df, config.TRAIN_IMG_DIR, transform=get_val_transforms(),
                            preprocess=preprocess)
    num_workers = config.NUM_WORKERS
    if sys.platform == "win32":
        num_workers = 0
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # Load model
    model = BoneAgeModel(pretrained=False).to(device)
    ckpt = load_checkpoint(Path(args.checkpoint), model, device=device)
    target_mean = ckpt.get("target_mean", 0.0)
    target_std = ckpt.get("target_std", 1.0)

    # Predict (model outputs z-normalized space if trained that way; un-normalize.
    # Eval dataset keeps targets in months, so only preds need un-normalizing.)
    preds, targets = collect_predictions(model, val_loader, device, tta=args.tta)
    preds = preds * target_std + target_mean

    # Metrics
    metrics = compute_metrics(preds, targets)
    print(f"\n{'='*40}")
    print("  Evaluation Results")
    print(f"{'='*40}")
    for k, v in metrics.items():
        print(f"  {k:.<28} {v:.2f}" if isinstance(v, float) else f"  {k:.<28} {v}")
    print(f"{'='*40}\n")

    # Plots
    if args.visualize:
        output_dir = config.PROJECT_ROOT / "results"
        output_dir.mkdir(exist_ok=True)
        plot_scatter(preds, targets, output_dir / "scatter_pred_vs_actual.png")
        plot_error_distribution(preds, targets, output_dir / "error_distribution.png")


if __name__ == "__main__":
    main()
