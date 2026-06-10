"""PyTorch Dataset for the RSNA Bone Age dataset."""

import os
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from tqdm import tqdm


class BoneAgeDataset(Dataset):
    """
    Loads hand X-ray images with bone age labels and gender.

    CSV columns expected: id, boneage, male
    Images: <img_dir>/<id>.png

    When cache_in_ram=True (default), all images are loaded and resized
    at init time, eliminating disk I/O during training.
    """

    def __init__(self, df: pd.DataFrame, img_dir: str | Path, transform=None,
                 base_size: int | None = None, cache_in_ram: bool = True):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.cache_in_ram = cache_in_ram

        # Pre-compute stats for optional normalization of targets
        self.age_mean = self.df["boneage"].mean()
        self.age_std = self.df["boneage"].std()

        # Cache resized images in RAM to avoid per-batch disk reads
        self._cache = []
        if cache_in_ram:
            # RAM guard: auto-disable if cache would exceed 40% of available RAM
            est_bytes = len(self.df) * ((base_size or 500) ** 2) * 3
            est_gb = est_bytes / 1e9
            try:
                import psutil
                avail_gb = psutil.virtual_memory().available / 1e9
                if est_gb > avail_gb * 0.4:
                    print(f"  [RAM Guard] Cache ~{est_gb:.1f}GB exceeds 40% of {avail_gb:.1f}GB available. Disabling cache.")
                    cache_in_ram = False
                    self.cache_in_ram = False
                else:
                    print(f"  [RAM Guard] Cache ~{est_gb:.1f}GB OK ({avail_gb:.1f}GB available)")
            except ImportError:
                pass  # psutil not installed, skip guard
            resize_tf = None
            if base_size:
                from torchvision import transforms as T
                resize_tf = T.Resize((base_size, base_size))
            print(f"  Caching {len(self.df)} images in RAM using parallel workers...")
            
            def load_and_resize(idx):
                row = self.df.iloc[idx]
                img_path = self.img_dir / f"{int(row['id'])}.png"
                img = Image.open(img_path).convert("RGB")
                if resize_tf:
                    img = resize_tf(img)
                return idx, img

            # Pre-allocate cache list
            self._cache = [None] * len(self.df)
            
            # Using thread pool to read and decode in parallel
            max_workers = min(8, (os.cpu_count() or 4) // 2)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(tqdm(
                    executor.map(load_and_resize, range(len(self.df))),
                    total=len(self.df),
                    desc="  Cache",
                    leave=False
                ))
            
            # Reconstruct list maintaining original dataframe order
            for idx, img in results:
                self._cache[idx] = img

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image from cache or disk
        if self.cache_in_ram:
            image = self._cache[idx]
        else:
            img_path = self.img_dir / f"{int(row['id'])}.png"
            image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Gender: 1 = male, 0 = female
        gender = torch.tensor([float(row["male"])], dtype=torch.float32)

        # Target: bone age in months
        bone_age = torch.tensor(row["boneage"], dtype=torch.float32)

        return image, gender, bone_age


def load_dataframes(csv_path: str | Path, val_split: float = 0.15, seed: int = 42):
    """
    Read the CSV and split into train / validation DataFrames.

    Returns:
        train_df, val_df
    """
    df = pd.read_csv(csv_path)

    # Ensure expected columns exist
    required = {"id", "boneage", "male"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {required}, got {set(df.columns)}")

    # Stratified-ish split by shuffling
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split_idx = int(len(df) * (1 - val_split))
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    print(f"Train: {len(train_df)} | Val: {len(val_df)}")
    return train_df, val_df
