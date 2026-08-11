#!/usr/bin/env python3
"""Download the Kaggle Chest X-Ray Pneumonia dataset into data/chest_xray."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import kagglehub

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cxr_pneumonia.subtype import hold_out_patients, patient_id

DATASET = "paultimothymooney/chest-xray-pneumonia"
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}


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
    return sum(1 for p in directory.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def rebuild_validation(dest: Path, val_fraction: float = 0.1, seed: int = 42) -> None:
    """
    Carve a real validation set out of train (the official one is 16 images).

    Grouped by patient: splitting by image put 272 children on both sides of
    the boundary, so early stopping was selecting on data it had already seen.
    Folds any existing val back into train first, so re-running with a
    different fraction/seed is safe. Test split is never touched.
    """
    train_dir = dest / "train"
    val_dir = dest / "val"

    # Recompute from the full pool each time, rather than layering on a previous split.
    if val_dir.is_dir():
        for cls_dir in sorted(p for p in val_dir.iterdir() if p.is_dir()):
            target = train_dir / cls_dir.name
            target.mkdir(parents=True, exist_ok=True)
            for img in cls_dir.iterdir():
                if img.is_file():
                    shutil.move(str(img), str(target / img.name))

    class_of = {}
    pool = []
    for cls_dir in sorted(p for p in train_dir.iterdir() if p.is_dir()):
        for img in sorted(cls_dir.iterdir()):
            if img.suffix.lower() in IMAGE_SUFFIXES:
                pool.append(img)
                class_of[img] = cls_dir.name

    val_images = hold_out_patients(pool, val_fraction, seed, stratum_of=class_of.__getitem__)
    for img in sorted(val_images):
        val_cls = val_dir / class_of[img]
        val_cls.mkdir(parents=True, exist_ok=True)
        shutil.move(str(img), str(val_cls / img.name))

    print(f"Rebuilt validation set (~{val_fraction:.0%} of train, split by patient).")


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
                if p.suffix.lower() in IMAGE_SUFFIXES
            )
            print(f"  {split}/{cls}: {n} images")


def check_no_leakage(dest: Path) -> None:
    """Fail loudly if any patient has images in both train and val."""
    def patients(split: str) -> set[str]:
        split_dir = dest / split
        return {
            patient_id(p)
            for p in split_dir.rglob("*")
            if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file()
        }

    overlap = patients("train") & patients("val")
    print(f"  train/val patient overlap: {len(overlap)}")
    if overlap:
        raise SystemExit("Patient leakage between train and val -- split is broken.")


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
        help="Fraction of train held out as val",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the patient-grouped validation split",
    )
    parser.add_argument(
        "--resplit",
        action="store_true",
        help="Redo the train/val split on data already on disk (no re-download)",
    )
    args = parser.parse_args()

    dest: Path = args.dest
    if (dest / "train").is_dir() and not args.force:
        print(f"Dataset already present at {dest.resolve()}")
        if args.resplit or count_images(dest / "val") < 100:
            rebuild_validation(dest, val_fraction=args.val_fraction, seed=args.seed)
            print_counts(dest)
            check_no_leakage(dest)
        else:
            print("Pass --resplit to redo the train/val split, or --force to re-download.")
        return

    print(f"Downloading {DATASET} via kagglehub...")
    cache_path = Path(kagglehub.dataset_download(DATASET))
    print(f"Cached at {cache_path}")

    source = find_chest_xray_root(cache_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest)

    rebuild_validation(dest, val_fraction=args.val_fraction, seed=args.seed)
    print_counts(dest)
    check_no_leakage(dest)
    print(f"Done. Data ready at {dest.resolve()}")


if __name__ == "__main__":
    main()
