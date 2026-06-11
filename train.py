"""Training loop with two-phase strategy and mixed precision."""

import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import GradScaler
from tqdm import tqdm

import config
from config import parse_args
from dataset import BoneAgeDataset, load_dataframes
from transforms import get_train_transforms, get_val_transforms
from model import BoneAgeModel
from utils import set_seed, save_checkpoint, load_checkpoint, EarlyStopping, AverageMeter, CudaPrefetcher


def train_one_epoch(model, loader, optimizer, scaler, device, use_amp, accum_steps=1, target_std=1.0):
    """Run one training epoch with gradient accumulation. Returns average loss (months)."""
    model.train()
    loss_meter = AverageMeter()
    criterion = nn.L1Loss()

    optimizer.zero_grad(set_to_none=True)

    # Disable the live bar when stdout is redirected to a file (keeps logs tiny);
    # interactive terminals still get the progress bar.
    pbar = tqdm(loader, desc="  Train", leave=False, file=sys.stdout,
                disable=not sys.stdout.isatty())
    for i, (images, genders, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        genders = genders.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).unsqueeze(1)

        with torch.amp.autocast("cuda", enabled=use_amp):
            preds = model(images, genders)
            loss = criterion(preds, targets) / accum_steps  # scale for accumulation

        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        loss_meter.update(loss.item() * accum_steps * target_std, images.size(0))  # log in months
        pbar.set_postfix(loss=f"{loss_meter.avg:.2f}")

    # Handle remaining gradients if dataset isn't divisible by accum_steps
    if (i + 1) % accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return loss_meter.avg


@torch.no_grad()
def validate(model, loader, device, use_amp, target_std=1.0):
    """Run validation. Returns average MAE (MAD) in months.

    Targets/preds are in z-normalized space when target_std != 1; since the
    transform is linear, MAD in months = MAD_normalized * target_std.
    """
    model.eval()
    mad_meter = AverageMeter()

    for images, genders, targets in loader:
        images = images.to(device, non_blocking=True)
        genders = genders.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).unsqueeze(1)

        with torch.amp.autocast("cuda", enabled=use_amp):
            preds = model(images, genders)

        mae = torch.abs(preds - targets).mean().item()
        mad_meter.update(mae, images.size(0))

    return mad_meter.avg * target_std


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config.USE_AMP and not args.no_amp and device.type == "cuda"
    accum_steps = args.grad_accum_steps
    preprocess = config.PREPROCESS and not args.no_preprocess

    # Cap CUDA to dedicated VRAM only — prevent silent spill into shared memory
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.95)

    effective_batch = args.batch_size * accum_steps
    print(f"Device: {device} | AMP: {use_amp}")
    print(f"Batch: {args.batch_size} x {accum_steps} accum = {effective_batch} effective")
    if device.type == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {vram_gb:.1f} GB (capped at 95% = {vram_gb * 0.95:.1f} GB)")

    # ── Data ───────────────────────────────────────────────────────────────
    train_df, val_df = load_dataframes(
        config.TRAIN_CSV, val_split=args.val_split, seed=args.seed
    )

    # Target z-normalization stats computed on the TRAIN split only.
    target_mean = float(train_df["boneage"].mean())
    target_std = float(train_df["boneage"].std())
    ckpt_extra = {"target_mean": target_mean, "target_std": target_std}
    best_path = config.CHECKPOINT_DIR / f"best_model{args.tag}.pth"
    final_path = config.CHECKPOINT_DIR / f"final_model{args.tag}.pth"
    print(f"Target normalization: mean={target_mean:.1f}, std={target_std:.1f} months")

    print(f"Bias-normalization preprocessing: {preprocess}")
    train_ds = BoneAgeDataset(train_df, config.TRAIN_IMG_DIR,
                               transform=get_train_transforms(skip_resize=True),
                               base_size=args.img_size, cache_in_ram=True,
                               preprocess=preprocess,
                               target_mean=target_mean, target_std=target_std)
    val_ds = BoneAgeDataset(val_df, config.TRAIN_IMG_DIR,
                             transform=get_val_transforms(skip_resize=True),
                             base_size=args.img_size, cache_in_ram=True,
                             preprocess=preprocess,
                             target_mean=target_mean, target_std=target_std)

    # Age-balanced sampling to counter regression-to-the-mean at age extremes.
    ages = train_df["boneage"].to_numpy()
    age_edges = np.array([0, 48, 96, 144, 192])  # months: 0-4, 4-8, 8-12, 12-16, 16y+
    bin_idx = np.clip(np.digitize(ages, age_edges) - 1, 0, len(age_edges) - 1)
    counts = np.bincount(bin_idx, minlength=len(age_edges))
    sample_weights = torch.as_tensor((1.0 / np.maximum(counts, 1))[bin_idx], dtype=torch.double)
    train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    print(f"Age-balanced sampler: bin counts {counts.tolist()}")

    num_workers = args.num_workers
    if sys.platform == "win32" and num_workers > 0:
        print(f"  [Windows Safeguard] RAM caching is active. Overriding num_workers from {num_workers} to 0 to prevent IPC serialization hang.")
        num_workers = 0

    persistent = num_workers > 0
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=persistent,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    # ── Prefetcher ─────────────────────────────────────────────────────────
    use_prefetch = not args.no_prefetch and device.type == "cuda"
    if use_prefetch:
        print("Using asynchronous CUDA data prefetcher...")
        train_loader = CudaPrefetcher(train_loader, device)
        val_loader = CudaPrefetcher(val_loader, device)

    # ── Model ──────────────────────────────────────────────────────────────
    model = BoneAgeModel().to(device)
    scaler = GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    best_mad = float("inf")

    if args.checkpoint:
        ckpt = load_checkpoint(args.checkpoint, model, device=device)
        start_epoch, best_mad = ckpt["epoch"], ckpt["val_mad"]

    if args.compile:
        print("Compiling model components with torch.compile()...")
        model.backbone = torch.compile(model.backbone)
        model.head = torch.compile(model.head)

    # ── Phase 1: Warmup (backbone frozen) ──────────────────────────────────
    model.freeze_backbone()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=config.WEIGHT_DECAY,
    )

    print(f"\n{'='*60}")
    print(f"Phase 1: Warmup ({args.warmup_epochs} epochs, backbone frozen)")
    print(f"{'='*60}")

    for epoch in range(start_epoch, args.warmup_epochs):
        print(f"\nEpoch {epoch+1}/{args.warmup_epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, use_amp, accum_steps, target_std)
        val_mad = validate(model, val_loader, device, use_amp, target_std)
        print(f"  Train Loss: {train_loss:.2f} | Val MAD: {val_mad:.2f} months")

        if val_mad < best_mad:
            best_mad = val_mad
            save_checkpoint(
                model, optimizer, epoch,
                val_mad, best_path,
                extra=ckpt_extra,
            )

    # ── Phase 2: Fine-tune (full model) ────────────────────────────────────
    model.unfreeze_backbone()
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.finetune_lr},
            {"params": model.head.parameters(), "lr": args.finetune_lr * 5},
        ],
        weight_decay=config.WEIGHT_DECAY,
    )

    total_finetune = args.epochs - args.warmup_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_finetune)
    early_stop = EarlyStopping(patience=7)

    print(f"\n{'='*60}")
    print(f"Phase 2: Fine-tune ({total_finetune} epochs, full model)")
    print(f"{'='*60}")

    for epoch in range(args.warmup_epochs, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, use_amp, accum_steps, target_std)
        val_mad = validate(model, val_loader, device, use_amp, target_std)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Train Loss: {train_loss:.2f} | Val MAD: {val_mad:.2f} months | LR: {lr_now:.2e}")

        if val_mad < best_mad:
            best_mad = val_mad
            save_checkpoint(
                model, optimizer, epoch,
                val_mad, best_path,
                extra=ckpt_extra,
            )

        if early_stop(val_mad):
            print(f"\n  Early stopping at epoch {epoch+1}. Best MAD: {best_mad:.2f}")
            break

    # Save final model regardless
    save_checkpoint(
        model, optimizer, epoch,
        val_mad, final_path,
        extra=ckpt_extra,
    )

    print(f"\n{'='*60}")
    print(f"Training complete. Best validation MAD: {best_mad:.2f} months")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
