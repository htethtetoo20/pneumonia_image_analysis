#!/usr/bin/env python3
"""
Measure how much of a model's Grad-CAM attention falls outside the lungs.

Images come from the config's own split (default test), so attention is never
measured on data the model trained on. Masks come from the COVID-19
Radiography Database, which is the only source that ships lung segmentations;
build_confounding_dataset.py prefixes filenames as SOURCE__original.png, so
the original stem still resolves to its mask.

  outside_fraction  share of CAM mass outside the lung mask. The "no
                    preference" baseline is 1 - lung_area_fraction, NOT zero:
                    lungs cover roughly a quarter of the frame, so scattering
                    attention at random already lands ~75% of it outside.
                    Read outside_fraction only against that baseline.

  concentration     (mass inside/area inside) / (mass outside/area outside).
                    Above 1 = denser attention in the lungs; at or below 1 =
                    no preference for lung tissue. This is the number to quote.

Attention landing on lungs is not proof the model reads pathology -- a source
confound lives inside the lung field too (scanner, body habitus, age all
differ there). Pair this with the source-classifier control, never alone.

    python scripts/lung_mask_attention.py --checkpoint artifacts/confounding/best.pt \
        --config configs/confounding.yaml --class-dir COVID
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from cxr_pneumonia.config import load_config
from cxr_pneumonia.data import get_transforms
from cxr_pneumonia.model import get_target_layer, load_checkpoint
from cxr_pneumonia.train import get_device

COVID_DATASET = "tawsifurrahman/covid19-radiography-database"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def find_dataset_root(cache: Path) -> Path:
    direct = cache / "COVID-19_Radiography_Dataset"
    if direct.is_dir():
        return direct
    for candidate in cache.rglob("COVID-19_Radiography_Dataset"):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find COVID-19_Radiography_Dataset under {cache}")


def build_mask_index(source: Path) -> dict[str, Path]:
    """Map original image stem -> lung mask, across every class folder that ships masks."""
    index: dict[str, Path] = {}
    for masks_dir in sorted(source.glob("*/masks")):
        for path in masks_dir.iterdir():
            if path.suffix.lower() in IMAGE_SUFFIXES:
                index[path.stem] = path
    if not index:
        raise SystemExit(f"No masks/ folders under {source}")
    return index


def original_stem(name: str) -> str:
    """Undo the SOURCE__ prefix build_confounding_dataset.py adds, to recover the mask key."""
    return Path(name.split("__", 1)[-1]).stem


def paired_from_split(split_dir: Path, masks: dict[str, Path]) -> tuple[list[tuple[Path, Path]], int]:
    """Pair our split's images with their masks; count the ones that have none."""
    if not split_dir.is_dir():
        raise SystemExit(f"No such split folder: {split_dir}")
    pairs: list[tuple[Path, Path]] = []
    unmatched = 0
    for image in sorted(split_dir.iterdir()):
        if image.suffix.lower() not in IMAGE_SUFFIXES or not image.is_file():
            continue
        mask = masks.get(original_stem(image.name))
        if mask is None:
            unmatched += 1
            continue
        pairs.append((image, mask))
    if not pairs:
        raise SystemExit(
            f"No image/mask pairs in {split_dir}. Only classes sourced from the "
            "COVID-19 Radiography Database ship lung masks."
        )
    return pairs, unmatched


def analyse(cam_map: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Split one heatmap's mass by whether it falls inside the lung mask."""
    total = cam_map.sum()
    if total <= 0:
        return {}
    inside_area = float(mask.mean())
    if inside_area <= 0 or inside_area >= 1:
        return {}

    inside_mass = float(cam_map[mask].sum() / total)
    outside_mass = 1.0 - inside_mass
    concentration = (inside_mass / inside_area) / (outside_mass / (1 - inside_area)) if outside_mass > 0 else np.inf
    return {
        "outside_fraction": outside_mass,
        "lung_area_fraction": inside_area,
        "concentration": float(concentration),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "confounding.yaml")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts" / "confounding" / "best.pt")
    parser.add_argument("--class-dir", default="COVID", help="Class folder inside the split")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, default=150, help="Images to analyse")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    import kagglehub

    # Masks only exist upstream; the images themselves come from our own split
    # so attention is never measured on data the model was trained on.
    source = find_dataset_root(Path(kagglehub.dataset_download(COVID_DATASET)))
    masks_by_stem = build_mask_index(source)

    data_root = Path(cfg.data_dir)
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    split_dir = data_root / args.split / args.class_dir
    pairs, unmatched = paired_from_split(split_dir, masks_by_stem)
    if unmatched:
        print(f"  ! {unmatched} images in {split_dir.name} have no lung mask; skipped")

    rng = np.random.default_rng(args.seed)
    if len(pairs) > args.n:
        pairs = [pairs[i] for i in sorted(rng.choice(len(pairs), args.n, replace=False))]

    device = get_device()
    model = load_checkpoint(str(args.checkpoint), device, num_classes=cfg.num_classes)
    if device.type == "mps":
        # Grad-CAM's backward pass is unreliable on MPS; CPU is slower but correct.
        model = model.to("cpu")
        device = torch.device("cpu")

    transform = get_transforms(cfg.image_size, train=False)
    size = (cfg.image_size, cfg.image_size)
    records: list[dict] = []
    predictions: list[int] = []

    with GradCAM(model=model, target_layers=[get_target_layer(model)]) as cam:
        for image_path, mask_path in pairs:
            tensor = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                predicted = int(model(tensor).argmax(dim=1).item())
            predictions.append(predicted)

            heatmap = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(predicted)])[0]
            mask = np.asarray(Image.open(mask_path).convert("L").resize(size), dtype=np.float32) > 127
            stats = analyse(heatmap, mask)
            if stats:
                records.append(stats)

    if not records:
        raise SystemExit("No usable heatmaps produced.")

    summary = {
        key: {
            "mean": float(np.mean([r[key] for r in records])),
            "median": float(np.median([r[key] for r in records])),
        }
        for key in ("outside_fraction", "lung_area_fraction", "concentration")
    }
    summary["n_images"] = len(records)
    summary["class_dir"] = args.class_dir
    summary["split"] = args.split
    summary["images_from"] = str(split_dir)
    summary["checkpoint"] = str(args.checkpoint)
    # Stored alongside the result so outside_fraction is never read as if 0 were the baseline.
    summary["random_baseline_outside_fraction"] = 1.0 - summary["lung_area_fraction"]["mean"]
    summary["predicted_class_counts"] = {
        cfg.class_names[i]: int(predictions.count(i)) for i in sorted(set(predictions))
    }

    out_dir = args.out or (cfg.artifacts_path / "attention")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"lung_attention_{args.split}_{args.class_dir.replace(' ', '_')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lung_area = summary["lung_area_fraction"]["mean"]
    outside = summary["outside_fraction"]["mean"]
    baseline = 1 - lung_area
    concentration = summary["concentration"]["mean"]
    verdict = "MORE lung-focused than chance" if outside < baseline else "no better than chance"

    print(f"\nLung-mask attention -- {args.class_dir}, {args.split} split, {len(records)} images")
    print(f"  lungs occupy            {lung_area:.3f} of the image")
    print(f"  CAM mass outside lungs  {outside:.3f}")
    print(f"  random baseline         {baseline:.3f}   <- compare against this, not 0")
    print(f"  -> {verdict} by {abs(baseline - outside):.3f}")
    print(f"  concentration ratio     {concentration:.3f}   (1.0 = no lung preference)")
    print(f"  predicted: {summary['predicted_class_counts']}")
    print("\n  Note: lung-focused attention does not rule out a source confound -- the")
    print("  confound lives inside the lung field too. Read with the source control.")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
