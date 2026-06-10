"""Data augmentation and preprocessing transforms."""

from torchvision import transforms
import config


def get_train_transforms(skip_resize=False):
    """Training augmentations: geometric + color jitter."""
    ops = []
    if not skip_resize:
        ops.append(transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)))
    ops += [
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ]
    return transforms.Compose(ops)


def get_val_transforms(skip_resize=False):
    """Validation: deterministic resize + normalize only."""
    ops = []
    if not skip_resize:
        ops.append(transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ]
    return transforms.Compose(ops)
