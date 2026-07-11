#!/usr/bin/env python3
"""Download the Kaggle Chest X-Ray Pneumonia dataset into data/chest_xray."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import kagglehub


DATASET = "paultimothymooney/chest-xray-pneumonia"


def find_chest_xray_root(download_root: Path) -> Path:
    """Locate the folder that contains train/val/test splits."""
    direct = download_root / "chest_xray"
    if (direct / "train").is_dir():
        return direct
    if (download_root / "train").is_dir():
        return download_root
    for candidate in download_root.rglob("train"):
        parent = candidate.parent
        if (parent / "test").is_dir():
            return parent
    raise FileNotFoundError(f"Could not find train/test splits under {download_root}")


def count_images(directory: Path) -> int:
    return sum(1 for p in directory.rglob("*") if p.suffix.lower() in {".jpeg", ".jpg", ".png"})


def rebuild_validation(dest: Path, val_fraction: float = 0.1, seed: int = 42) -> None:
    """
    The Kaggle val split is tiny (16 images). Move a stratified fraction of
    train into val so early stopping / model selection is meaningful.
    Test split is left untouched.
    """
    train_dir = dest / "train"
    val_dir = dest / "val"
    if count_images(val_dir) >= 100:
        return

    rng = random.Random(seed)
    # Move existing tiny val images back into train first
    for cls_dir in val_dir.iterdir():
        if not cls_dir.is_dir():
            continue
        target = train_dir / cls_dir.name
        target.mkdir(parents=True, exist_ok=True)
        for img in cls_dir.iterdir():
            if img.is_file():
                shutil.move(str(img), str(target / img.name))

    for cls_dir in sorted(p for p in train_dir.iterdir() if p.is_dir()):
        images = [
            p for p in cls_dir.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"}
        ]
        rng.shuffle(images)
        n_val = max(1, int(len(images) * val_fraction))
        val_cls = val_dir / cls_dir.name
        val_cls.mkdir(parents=True, exist_ok=True)
        for img in images[:n_val]:
            shutil.move(str(img), str(val_cls / img.name))
    print(f"Rebuilt validation set (~{val_fraction:.0%} stratified from train).")


def print_counts(dest: Path) -> None:
    for split in ("train", "val", "test"):
        split_dir = dest / split
        if not split_dir.is_dir():
            print(f"Warning: missing split {split}")
            continue
        for cls in sorted(p.name for p in split_dir.iterdir() if p.is_dir()):
            n = sum(
                1
                for p in (split_dir / cls).iterdir()
                if p.suffix.lower() in {".jpeg", ".jpg", ".png"}
            )
            print(f"  {split}/{cls}: {n} images")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data/chest_xray"),
        help="Destination directory for the dataset",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-copy even if destination already has train/",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of train moved into val when official val is tiny",
    )
    args = parser.parse_args()

    dest: Path = args.dest
    if (dest / "train").is_dir() and not args.force:
        print(f"Dataset already present at {dest.resolve()}")
        if count_images(dest / "val") < 100:
            rebuild_validation(dest, val_fraction=args.val_fraction)
            print_counts(dest)
        print("Use --force to re-download/copy.")
        return

    print(f"Downloading {DATASET} via kagglehub...")
    cache_path = Path(kagglehub.dataset_download(DATASET))
    print(f"Cached at {cache_path}")

    source = find_chest_xray_root(cache_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest)

    rebuild_validation(dest, val_fraction=args.val_fraction)
    print_counts(dest)
    print(f"Done. Data ready at {dest.resolve()}")


if __name__ == "__main__":
    main()
