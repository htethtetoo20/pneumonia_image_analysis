#!/usr/bin/env python3
"""
Measure how much of a model's Grad-CAM attention falls outside the lungs.

Turns "reads pathology vs reads provenance" into a number, using the lung
segmentation masks shipped with the COVID-19 Radiography Database: what
fraction of each heatmap's mass lands outside the lung fields?

  outside_fraction  share of CAM mass outside the lung mask. Compare against
                    lung_area_fraction -- random attention scores
                    outside_fraction ~= 1 - lung_area_fraction, so that's the
                    "no preference" baseline, not zero.

  concentration     (mass inside/area inside) / (mass outside/area outside).
                    Above 1 = denser attention in the lungs; at or below 1 =
                    no preference for lung tissue.

Only classes with masks can be scored (COVID-19 Radiography images). Pass
--class-dir to point at Normal, Viral Pneumonia, etc.

    python scripts/lung_mask_attention.py --checkpoint artifacts/confounding/best.pt \
        --config configs/confounding.yaml --n 150
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


def paired_images_and_masks(class_dir: Path) -> list[tuple[Path, Path]]:
    """Match each image to its mask by filename; skip any without a partner."""
    images_dir, masks_dir = class_dir / "images", class_dir / "masks"
    if not (images_dir.is_dir() and masks_dir.is_dir()):
        raise SystemExit(f"{class_dir} has no images/ + masks/ pair")
    masks = {p.stem: p for p in masks_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}
    pairs = [
        (img, masks[img.stem])
        for img in sorted(images_dir.iterdir())
        if img.suffix.lower() in IMAGE_SUFFIXES and img.stem in masks
    ]
    if not pairs:
        raise SystemExit(f"No image/mask pairs found in {class_dir}")
    return pairs


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
    parser.add_argument("--class-dir", default="COVID", help="Folder inside the COVID-19 Radiography Database")
    parser.add_argument("--n", type=int, default=150, help="Images to analyse")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    import kagglehub

    source = find_dataset_root(Path(kagglehub.dataset_download(COVID_DATASET)))
    pairs = paired_images_and_masks(source / args.class_dir)
    rng = np.random.default_rng(args.seed)
    if len(pairs) > args.n:
        pairs = [pairs[i] for i in sorted(rng.choice(len(pairs), args.n, replace=False))]

    cfg = load_config(args.config)
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
    summary["checkpoint"] = str(args.checkpoint)
    summary["predicted_class_counts"] = {
        cfg.class_names[i]: int(predictions.count(i)) for i in sorted(set(predictions))
    }

    out_dir = args.out or (cfg.artifacts_path / "attention")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"lung_attention_{args.class_dir.replace(' ', '_')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lung_area = summary["lung_area_fraction"]["mean"]
    outside = summary["outside_fraction"]["mean"]
    print(f"\nLung-mask attention -- {args.class_dir}, {len(records)} images")
    print(f"  lungs occupy            {lung_area:.3f} of the image")
    print(f"  CAM mass outside lungs  {outside:.3f}   (random attention would give {1 - lung_area:.3f})")
    print(f"  concentration ratio     {summary['concentration']['mean']:.3f}   (1.0 = no lung preference)")
    print(f"  predicted: {summary['predicted_class_counts']}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
