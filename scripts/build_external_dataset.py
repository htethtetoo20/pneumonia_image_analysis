#!/usr/bin/env python3
"""
Build an external NORMAL/PNEUMONIA test set from the COVID-19 Radiography Database.

Every number this project has reported so far comes from one hospital
(Guangzhou, pediatric, one imaging setup), and CXR models are notorious for
latching onto scanner/demographic characteristics rather than pathology. This
builds a held-out test set from a genuinely different source (Qatar/Dhaka,
adult, different equipment, PNG) so transfer can be measured, not assumed.
Deliberately never used for training.

Class mapping:  Normal -> NORMAL,  Viral Pneumonia -> PNEUMONIA

"Lung_Opacity" is excluded by default -- it's documented as non-COVID lung
infection (broader than pneumonia), which would make disagreements ambiguous.
Pass --include-lung-opacity to add it anyway. COVID images are excluded too;
see confounding_study.py for that extension.

    python scripts/build_external_dataset.py
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATASET = "tawsifurrahman/covid19-radiography-database"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

# Source folder -> our class name.
CLASS_MAP = {
    "Normal": "NORMAL",
    "Viral Pneumonia": "PNEUMONIA",
}
LUNG_OPACITY = ("Lung_Opacity", "PNEUMONIA")


def find_dataset_root(cache: Path) -> Path:
    """Locate COVID-19_Radiography_Dataset inside the kagglehub cache."""
    direct = cache / "COVID-19_Radiography_Dataset"
    if direct.is_dir():
        return direct
    for candidate in cache.rglob("COVID-19_Radiography_Dataset"):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find COVID-19_Radiography_Dataset under {cache}")


def collect(class_dir: Path) -> list[Path]:
    """Images for one class (masks/ holds lung segmentations, not X-rays, so skip it)."""
    images_dir = class_dir / "images"
    search_root = images_dir if images_dir.is_dir() else class_dir
    return sorted(p for p in search_root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=ROOT / "data" / "external")
    parser.add_argument(
        "--balanced",
        action="store_true",
        default=True,
        help="Subsample the larger class so the test set is balanced (default)",
    )
    parser.add_argument("--no-balanced", dest="balanced", action="store_false")
    parser.add_argument("--include-lung-opacity", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="Seed for subsampling")
    parser.add_argument("--copy", action="store_true", help="Copy instead of hardlinking")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    import kagglehub

    print(f"Resolving {DATASET} via kagglehub...")
    source = find_dataset_root(Path(kagglehub.dataset_download(DATASET)))
    print(f"Source: {source}")

    mapping = dict(CLASS_MAP)
    if args.include_lung_opacity:
        mapping[LUNG_OPACITY[0]] = LUNG_OPACITY[1]

    by_class: dict[str, list[Path]] = {}
    for src_name, cls in mapping.items():
        found = collect(source / src_name)
        if not found:
            raise SystemExit(f"No images found in {source / src_name}")
        by_class.setdefault(cls, []).extend(found)
        print(f"  {src_name:<16} -> {cls:<10} {len(found)} images")

    if args.balanced:
        smallest = min(len(v) for v in by_class.values())
        rng = random.Random(args.seed)
        for cls, paths in by_class.items():
            if len(paths) > smallest:
                by_class[cls] = sorted(rng.sample(paths, smallest))
        print(f"\nBalanced to {smallest} images per class (seed {args.seed})")

    test_dir = args.dest / "test"
    if args.dest.exists():
        if not args.force:
            raise SystemExit(f"{args.dest} already exists. Pass --force to rebuild it.")
        shutil.rmtree(args.dest)

    for cls, paths in sorted(by_class.items()):
        out = test_dir / cls
        out.mkdir(parents=True, exist_ok=True)
        for src in paths:
            # Source names collide across folders (e.g. "Normal-1.png"); prefix
            # with the originating folder to keep them distinct.
            dest = out / f"{src.parent.parent.name}__{src.name}".replace(" ", "_")
            if args.copy:
                shutil.copy2(src, dest)
                continue
            try:
                os.link(src, dest)
            except OSError:
                # kagglehub cache may sit on another volume; symlink is fine
                # here because nothing moves these files afterwards.
                dest.symlink_to(src.resolve())

    print(f"\nBuilt external test set at {test_dir}")
    for cls in sorted(by_class):
        n = sum(1 for p in (test_dir / cls).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        print(f"  {cls:<10} {n} images")
    print("\nEvaluate with: python scripts/external_validate.py")


if __name__ == "__main__":
    main()
