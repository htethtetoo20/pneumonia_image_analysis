"""Dataset helpers and DataLoaders for chest X-ray images."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

from cxr_pneumonia.config import Config

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms(image_size: int, train: bool = False) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def resolve_split_dir(data_dir: Path, split: str) -> Path:
    """Resolve train/val/test folder (handles nested chest_xray/chest_xray)."""
    candidates = [
        data_dir / split,
        data_dir / "chest_xray" / split,
    ]
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(
        f"Could not find '{split}' under {data_dir}. "
        "Run: python scripts/download_data.py"
    )


def build_dataset(data_dir: Path, split: str, image_size: int, train: bool) -> datasets.ImageFolder:
    split_dir = resolve_split_dir(data_dir, split)
    return datasets.ImageFolder(str(split_dir), transform=get_transforms(image_size, train=train))


def class_weights_from_dataset(dataset: datasets.ImageFolder) -> torch.Tensor:
    counts = Counter(dataset.targets)
    n_classes = len(dataset.classes)
    weights = torch.zeros(n_classes, dtype=torch.float32)
    total = len(dataset.targets)
    for cls_idx, count in counts.items():
        weights[cls_idx] = total / (n_classes * count)
    return weights


def make_weighted_sampler(dataset: datasets.ImageFolder) -> WeightedRandomSampler:
    counts = Counter(dataset.targets)
    sample_weights = [1.0 / counts[t] for t in dataset.targets]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def create_dataloaders(cfg: Config) -> dict[str, DataLoader]:
    """Create train/val/test loaders. Train uses weighted sampling for imbalance."""
    train_ds = build_dataset(cfg.data_path, "train", cfg.image_size, train=True)
    val_ds = build_dataset(cfg.data_path, "val", cfg.image_size, train=False)
    test_ds = build_dataset(cfg.data_path, "test", cfg.image_size, train=False)

    # Align class order with config when possible
    if list(train_ds.classes) != cfg.class_names:
        # ImageFolder sorts alphabetically; NORMAL < PNEUMONIA — matches default
        pass

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=make_weighted_sampler(train_ds),
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return {"train": train_loader, "val": val_loader, "test": test_loader}


def count_split(data_dir: Path, split: str) -> dict[str, int]:
    split_dir = resolve_split_dir(data_dir, split)
    counts: dict[str, int] = {}
    for class_dir in sorted(split_dir.iterdir()):
        if class_dir.is_dir():
            counts[class_dir.name] = sum(
                1 for p in class_dir.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"}
            )
    return counts
