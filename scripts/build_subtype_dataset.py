#!/usr/bin/env python3
"""
Build a 3-class Normal / Bacterial / Viral dataset from the binary chest_xray data.

No download needed -- sub-type labels are already in the Kermany filenames
(see cxr_pneumonia.subtype). Differs from the binary layout in two ways:

* Train and val are rebuilt from scratch, **grouped by patient**. The binary
  pipeline's rebuild_validation() splits by image, which puts 272 children on
  both sides of the boundary -- early stopping then selects on memorised data.
* The official test folder is carried over untouched (patient IDs never pooled
  with train, since numbering restarts between folders), so results stay
  comparable with published numbers.

Hardlinked by default (no extra disk, survives download_data.py --resplit
shuffling files between train/val). Pass --copy for a portable directory.

    python scripts/build_subtype_dataset.py
    python scripts/build_subtype_dataset.py --copy --val-fraction 0.15
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cxr_pneumonia.subtype import (
    SUBTYPE_CLASSES,
    UnparseableFilename,
    hold_out_patients,
    patient_id,
    subtype_of,
)

IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}


def iter_images(directory: Path) -> list[Path]:
    """All images under a split folder, skipping __MACOSX and other junk."""
    if not directory.is_dir():
        return []
    return sorted(
        p
        for p in directory.rglob("*")
        if p.suffix.lower() in IMAGE_SUFFIXES
        and p.is_file()
        and not any(part.startswith((".", "__MACOSX")) for part in p.parts)
    )


def label_images(paths: list[Path]) -> tuple[list[tuple[Path, str, str]], list[Path]]:
    """Annotate each path with (subtype, patient_id); return failures separately."""
    labelled: list[tuple[Path, str, str]] = []
    skipped: list[Path] = []
    for path in paths:
        try:
            labelled.append((path, subtype_of(path), patient_id(path)))
        except UnparseableFilename:
            skipped.append(path)
    return labelled, skipped


def split_by_patient(
    labelled: list[tuple[Path, str, str]],
    val_fraction: float,
    seed: int,
) -> tuple[list[tuple[Path, str, str]], list[tuple[Path, str, str]]]:
    """Partition labelled images into (train, val), holding out whole patients."""
    subtype_of_path = {path: subtype for path, subtype, _ in labelled}
    val_paths = hold_out_patients(
        [path for path, _, _ in labelled],
        val_fraction,
        seed,
        stratum_of=subtype_of_path.__getitem__,
    )
    train_items = [i for i in labelled if i[0] not in val_paths]
    val_items = [i for i in labelled if i[0] in val_paths]
    return train_items, val_items


def emit(items: list[tuple[Path, str, str]], split_dir: Path, copy: bool) -> None:
    """Write one split's images into per-class folders."""
    for cls in SUBTYPE_CLASSES:
        (split_dir / cls).mkdir(parents=True, exist_ok=True)

    for source, subtype, pid in items:
        dest = split_dir / subtype / source.name
        if dest.exists() or dest.is_symlink():
            # Defensive: source folders have unique names, but never silently drop.
            dest = split_dir / subtype / f"{pid}__{source.name}"
        if copy:
            shutil.copy2(source, dest)
            continue
        try:
            os.link(source, dest)
        except OSError:
            # Different filesystem: symlink instead (breaks if source moves later).
            dest.symlink_to(source.resolve())


def report(name: str, items: list[tuple[Path, str, str]]) -> None:
    counts = Counter(subtype for _, subtype, _ in items)
    patients = len({pid for _, _, pid in items})
    total = len(items)
    breakdown = "  ".join(f"{cls}: {counts.get(cls, 0):>5}" for cls in SUBTYPE_CLASSES)
    print(f"  {name:<6} {breakdown}   total: {total:>5}  patients: {patients:>4}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "chest_xray")
    parser.add_argument("--dest", type=Path, default=ROOT / "data" / "chest_xray_subtype")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of hardlinking")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing destination")
    args = parser.parse_args()

    if not (args.source / "train").is_dir():
        raise SystemExit(f"No train/ under {args.source}. Run scripts/download_data.py first.")

    if args.dest.exists():
        if not args.force:
            raise SystemExit(f"{args.dest} already exists. Pass --force to rebuild it.")
        shutil.rmtree(args.dest)

    # Fold the binary pipeline's image-level val back into train, then resplit by patient.
    pool_paths = iter_images(args.source / "train") + iter_images(args.source / "val")
    test_paths = iter_images(args.source / "test")

    pool, pool_skipped = label_images(pool_paths)
    test_items, test_skipped = label_images(test_paths)
    for path in pool_skipped + test_skipped:
        print(f"  ! skipped unparseable filename: {path.name}")

    train_items, val_items = split_by_patient(pool, args.val_fraction, args.seed)

    emit(train_items, args.dest / "train", args.copy)
    emit(val_items, args.dest / "val", args.copy)
    emit(test_items, args.dest / "test", args.copy)

    print(f"\nBuilt {'copies' if args.copy else 'hardlinks'} at {args.dest}\n")
    report("train", train_items)
    report("val", val_items)
    report("test", test_items)

    train_patients = {pid for _, _, pid in train_items}
    val_patients = {pid for _, _, pid in val_items}
    overlap = train_patients & val_patients
    print(f"\n  train/val patient overlap: {len(overlap)}")
    if overlap:
        raise SystemExit("Patient leakage between train and val -- split is broken.")
    if pool_skipped or test_skipped:
        raise SystemExit(f"{len(pool_skipped) + len(test_skipped)} files were unlabelled; fix before training.")
    print("  no leakage. Train with: python scripts/train.py --config configs/subtype.yaml")


if __name__ == "__main__":
    main()
