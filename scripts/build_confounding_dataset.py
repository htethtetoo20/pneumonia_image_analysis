#!/usr/bin/env python3
"""
Build the 5-class dataset AND a source-labelled copy of the same images.

The 5-class task (Normal/Bacterial/Viral/COVID-19/Tuberculosis) pools three
hospitals, so disease label and data source are almost perfectly correlated --
a CNN can score ~97% by telling child CXRs (Kermany) from adult ones (Qatar)
and never read a lung (see DeGrave et al., Nature Machine Intelligence 2021).

This emits two parallel datasets over the *identical* images -- one labelled
by disease (data/confounding/), one by source (data/confounding_source/).
Training on both is the control: if the source classifier scores as high as
the disease one, the disease number isn't evidence of pathology detection.
compute_source_oracle() prints the floor any honest 5-class result must clear.

Kermany keeps its patient-grouped split; the Qatar sets ship no patient IDs so
they're split at image level (80/10/10) -- a limitation that can only inflate
the COVID/TB numbers.

    python scripts/build_confounding_dataset.py
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

COVID_DATASET = "tawsifurrahman/covid19-radiography-database"
TB_DATASET = "tawsifurrahman/tuberculosis-tb-chest-xray-dataset"

SOURCE_KERMANY = "KERMANY_PEDIATRIC"
SOURCE_COVID = "QATAR_COVID"
SOURCE_TB = "QATAR_TB"

DISEASE_CLASSES = ["BACTERIAL", "COVID", "NORMAL", "TUBERCULOSIS", "VIRAL"]
SOURCE_CLASSES = [SOURCE_KERMANY, SOURCE_COVID, SOURCE_TB]

SPLITS = ("train", "val", "test")


def images_in(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    search = directory / "images" if (directory / "images").is_dir() else directory
    return sorted(
        p
        for p in search.rglob("*")
        if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file() and not p.name.startswith(".")
    )


def find_under(cache: Path, name: str) -> Path:
    direct = cache / name
    if direct.is_dir():
        return direct
    for candidate in cache.rglob(name):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find {name} under {cache}")


def split_image_level(paths: list[Path], seed: int) -> dict[str, list[Path]]:
    """80/10/10 split, matching the proportions Kermany's splits already have."""
    rng = random.Random(seed)
    shuffled = list(paths)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train, n_val = int(n * 0.8), int(n * 0.1)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def link(src: Path, dest: Path, copy: bool) -> None:
    if copy:
        shutil.copy2(src, dest)
        return
    try:
        os.link(src, dest)
    except OSError:
        dest.symlink_to(src.resolve())


def compute_source_oracle(counts: dict[str, dict[str, int]]) -> None:
    """Accuracy from knowing only the data source (perfect for COVID/TB, 1/3 credit within Kermany's 3 classes)."""
    test = counts["test"]
    kermany = test["NORMAL"] + test["BACTERIAL"] + test["VIRAL"]
    single_source = test["COVID"] + test["TUBERCULOSIS"]
    total = kermany + single_source
    oracle = (single_source + kermany / 3.0) / total
    majority = max(test.values()) / total
    print("\nBaselines on the 5-class test set:")
    print(f"  majority class          {majority:.4f}")
    print(f"  source-only oracle      {oracle:.4f}  <- knows the dataset, guesses within Kermany")
    print("  A 5-class model must beat the source oracle to show it reads pathology.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subtype-dir", type=Path, default=ROOT / "data" / "chest_xray_subtype")
    parser.add_argument("--disease-dest", type=Path, default=ROOT / "data" / "confounding")
    parser.add_argument("--source-dest", type=Path, default=ROOT / "data" / "confounding_source")
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=1500,
        help="Cap each class so training stays tractable and no class dominates",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not (args.subtype_dir / "train").is_dir():
        raise SystemExit(f"No train/ under {args.subtype_dir}. Run scripts/build_subtype_dataset.py first.")

    import kagglehub

    covid_root = find_under(Path(kagglehub.dataset_download(COVID_DATASET)), "COVID-19_Radiography_Dataset")
    tb_root = find_under(Path(kagglehub.dataset_download(TB_DATASET)), "TB_Chest_Radiography_Database")

    rng = random.Random(args.seed)

    # Kermany keeps its patient-grouped split; just carry the assignment over.
    plan: dict[str, list[tuple[Path, str, str]]] = {s: [] for s in SPLITS}
    for split in SPLITS:
        for cls in ("BACTERIAL", "NORMAL", "VIRAL"):
            found = images_in(args.subtype_dir / split / cls)
            if len(found) > args.max_per_class:
                found = sorted(rng.sample(found, args.max_per_class))
            plan[split].extend((p, cls, SOURCE_KERMANY) for p in found)

    # Qatar sets have no patient IDs, so image-level split is the only option.
    for label, source, paths in (
        ("COVID", SOURCE_COVID, images_in(covid_root / "COVID")),
        ("TUBERCULOSIS", SOURCE_TB, images_in(tb_root / "Tuberculosis")),
    ):
        if not paths:
            raise SystemExit(f"No images found for {label}")
        if len(paths) > args.max_per_class:
            paths = sorted(rng.sample(paths, args.max_per_class))
        for split, subset in split_image_level(paths, args.seed).items():
            plan[split].extend((p, label, source) for p in subset)

    for dest in (args.disease_dest, args.source_dest):
        if dest.exists():
            if not args.force:
                raise SystemExit(f"{dest} already exists. Pass --force to rebuild it.")
            shutil.rmtree(dest)

    counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        for classes, dest, index in (
            (DISEASE_CLASSES, args.disease_dest, 1),
            (SOURCE_CLASSES, args.source_dest, 2),
        ):
            for cls in classes:
                (dest / split / cls).mkdir(parents=True, exist_ok=True)
            for item in plan[split]:
                src = item[0]
                # Names collide across the three sources; prefix to keep unique.
                name = f"{item[2]}__{src.name}".replace(" ", "_")
                link(src, dest / split / item[index] / name, args.copy)
        counts[split] = Counter(cls for _, cls, _ in plan[split])

    print(f"\nDisease labels -> {args.disease_dest}")
    print(f"Source labels  -> {args.source_dest}   (same images)")
    header = "".join(f"{c[:9]:>13}" for c in DISEASE_CLASSES)
    print(f"\n  {'split':<7}{header}{'total':>9}")
    for split in SPLITS:
        row = "".join(f"{counts[split].get(c, 0):>13}" for c in DISEASE_CLASSES)
        print(f"  {split:<7}{row}{sum(counts[split].values()):>9}")

    print(f"\n  {'split':<7}" + "".join(f"{s:>20}" for s in SOURCE_CLASSES))
    for split in SPLITS:
        src_counts = Counter(s for _, _, s in plan[split])
        print(f"  {split:<7}" + "".join(f"{src_counts.get(s, 0):>20}" for s in SOURCE_CLASSES))

    compute_source_oracle(counts)
    print("\nNext:")
    print("  python scripts/train.py --config configs/confounding.yaml")
    print("  python scripts/train.py --config configs/confounding_source.yaml")


if __name__ == "__main__":
    main()
