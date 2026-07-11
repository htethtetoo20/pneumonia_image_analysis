"""Evaluation metrics and plots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from cxr_pneumonia.config import Config
from cxr_pneumonia.data import create_dataloaders
from cxr_pneumonia.model import load_checkpoint
from cxr_pneumonia.train import get_device


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels_list: list[int] = []
    preds_list: list[int] = []
    probs_list: list[float] = []

    for images, labels in tqdm(loader, desc="evaluate", leave=False):
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)
        labels_list.extend(labels.tolist())
        preds_list.extend(preds.cpu().tolist())
        probs_list.extend(probs.cpu().tolist())

    return (
        np.asarray(labels_list),
        np.asarray(preds_list),
        np.asarray(probs_list),
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, class_names: list[str]) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None,
        "classification_report": classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    return metrics


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: Path) -> None:
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Pneumonia)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def evaluate(cfg: Config, checkpoint: str | Path, split: str = "test") -> dict:
    device = get_device()
    cfg.artifacts_path.mkdir(parents=True, exist_ok=True)

    model = load_checkpoint(str(checkpoint), device, num_classes=cfg.num_classes)
    loaders = create_dataloaders(cfg)
    if split not in loaders:
        raise ValueError(f"Unknown split '{split}'. Expected one of {list(loaders)}")

    y_true, y_pred, y_prob = collect_predictions(model, loaders[split], device)
    metrics = compute_metrics(y_true, y_pred, y_prob, cfg.class_names)

    cm_path = cfg.artifacts_path / f"confusion_matrix_{split}.png"
    roc_path = cfg.artifacts_path / f"roc_{split}.png"
    metrics_path = cfg.artifacts_path / f"metrics_{split}.json"

    plot_confusion_matrix(np.asarray(metrics["confusion_matrix"]), cfg.class_names, cm_path)
    if metrics["roc_auc"] is not None:
        plot_roc(y_true, y_prob, roc_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps({k: metrics[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")}, indent=2))
    print(f"Wrote {metrics_path}")
    print(f"Wrote {cm_path}")
    return metrics
