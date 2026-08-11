#!/usr/bin/env python3
"""
Evaluate a trained checkpoint on the external (different-hospital) test set.

Answers what in-distribution accuracy can't: does the model transfer? A large
drop here means the internal score was measuring site characteristics
(scanner, age range, hospital) as much as pathology. A 3-class checkpoint is
collapsed to binary (PNEUMONIA = BACTERIAL u VIRAL) by summing probabilities,
so both models are scored on the same task with a real AUC, not just labels.

    python scripts/external_validate.py --checkpoint artifacts/best.pt
    python scripts/external_validate.py --config configs/subtype.yaml \
        --checkpoint artifacts/subtype/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cxr_pneumonia.config import load_config
from cxr_pneumonia.data import build_dataset
from cxr_pneumonia.evaluate import bootstrap_ci, collect_predictions, compute_metrics, plot_confusion_matrix
from cxr_pneumonia.model import load_checkpoint
from cxr_pneumonia.train import get_device

BINARY_CLASSES = ["NORMAL", "PNEUMONIA"]


def operating_point_analysis(y_true: np.ndarray, p_pneumonia: np.ndarray) -> dict:
    """
    Separate a discrimination failure from a calibration failure.

    If AUC holds up but accuracy collapses, the model still ranks cases
    correctly and only its 0.5 cut-point is wrong -- fixable by recalibration,
    not retraining. Threshold is tuned on the external set itself, so this is
    an oracle upper bound, not honest generalisation.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    accuracies = [(t, accuracy_score(y_true, (p_pneumonia >= t).astype(int))) for t in thresholds]
    best_t, best_acc = max(accuracies, key=lambda pair: pair[1])
    predicted = (p_pneumonia >= best_t).astype(int)
    return {
        "threshold": float(best_t),
        "accuracy": float(best_acc),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "note": "threshold tuned on the external set; oracle upper bound, not honest generalisation",
    }


def collapse_to_binary(probs: np.ndarray, class_names: list[str]) -> np.ndarray:
    """Sum multi-class probabilities into NORMAL vs PNEUMONIA so a 3-class model scores against binary ground truth."""
    normal_idx = [i for i, n in enumerate(class_names) if n == "NORMAL"]
    if not normal_idx:
        raise ValueError(f"No NORMAL class in {class_names}; cannot collapse to binary.")
    p_normal = probs[:, normal_idx].sum(axis=1)
    return np.column_stack([p_normal, 1.0 - p_normal])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts" / "best.pt")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "external")
    parser.add_argument("--out", type=Path, default=None, help="Where to write metrics (default: <artifacts>/external)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not (args.data_dir / "test").is_dir():
        raise SystemExit(f"No test/ under {args.data_dir}. Run scripts/build_external_dataset.py first.")

    device = get_device()
    model = load_checkpoint(str(args.checkpoint), device, num_classes=cfg.num_classes)

    dataset = build_dataset(args.data_dir, "test", cfg.image_size, train=False)
    if list(dataset.classes) != BINARY_CLASSES:
        raise SystemExit(f"External set has classes {dataset.classes}, expected {BINARY_CLASSES}")

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    y_true, _, y_prob = collect_predictions(model, loader, device)

    if cfg.num_classes != 2:
        print(f"Collapsing {cfg.class_names} -> {BINARY_CLASSES}")
        y_prob = collapse_to_binary(y_prob, cfg.class_names)
    y_pred = y_prob.argmax(axis=1)

    metrics = compute_metrics(y_true, y_pred, y_prob, BINARY_CLASSES)
    metrics["ci_95"] = bootstrap_ci(y_true, y_pred, y_prob, BINARY_CLASSES, seed=cfg.seed)
    metrics["operating_point"] = operating_point_analysis(y_true, y_prob[:, 1])
    metrics["source_config"] = str(args.config)
    metrics["checkpoint"] = str(args.checkpoint)
    metrics["external_data"] = str(args.data_dir)

    out_dir = args.out or (cfg.artifacts_path / "external")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics_external.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    plot_confusion_matrix(
        np.asarray(metrics["confusion_matrix"]), BINARY_CLASSES, out_dir / "confusion_matrix_external.png"
    )

    ci = metrics["ci_95"]
    print(f"\nExternal validation -- {args.checkpoint}")
    for name in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        if metrics[name] is None:
            continue
        bounds = f"  95% CI [{ci[name][0]:.3f}, {ci[name][1]:.3f}]" if name in ci else ""
        print(f"  {name:<10} {metrics[name]:.4f}{bounds}")
    cm = np.asarray(metrics["confusion_matrix"])
    print(f"\n  {'':<12}{'pred NORMAL':>13}{'pred PNEUM':>13}")
    for i, n in enumerate(BINARY_CLASSES):
        print(f"  {n:<12}{cm[i][0]:>13}{cm[i][1]:>13}")

    op = metrics["operating_point"]
    print(
        f"\n  at recalibrated threshold {op['threshold']:.2f}: "
        f"accuracy {op['accuracy']:.4f}  f1 {op['f1']:.4f}"
    )
    print("  (threshold tuned on this set -- an upper bound, not honest generalisation)")
    print(f"\nWrote {metrics_path}")


if __name__ == "__main__":
    main()
