"""Training loop with early stopping and checkpointing."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from cxr_pneumonia.config import Config
from cxr_pneumonia.data import class_weights_from_dataset, create_dataloaders
from cxr_pneumonia.model import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def run_epoch_eval(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[float] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    n = len(all_labels)
    acc = sum(int(y == p) for y, p in zip(all_labels, all_preds)) / max(n, 1)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float("nan")
    except ValueError:
        auc = float("nan")

    return {
        "loss": total_loss / max(n, 1),
        "accuracy": acc,
        "f1": float(f1),
        "auc": float(auc) if auc == auc else 0.0,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
    return total_loss / max(n, 1)


def train(cfg: Config) -> Path:
    set_seed(cfg.seed)
    device = get_device()
    cfg.artifacts_path.mkdir(parents=True, exist_ok=True)

    loaders = create_dataloaders(cfg)
    train_ds = loaders["train"].dataset
    weights = class_weights_from_dataset(train_ds).to(device)

    model = build_model(num_classes=cfg.num_classes, freeze_backbone=cfg.freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    best_f1 = -1.0
    patience_left = cfg.patience
    best_path = cfg.artifacts_path / "best.pt"
    history_path = cfg.artifacts_path / "train_history.csv"

    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_loss", "val_accuracy", "val_f1", "val_auc"],
        )
        writer.writeheader()

        for epoch in range(1, cfg.epochs + 1):
            train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
            val_metrics = run_epoch_eval(model, loaders["val"], criterion, device)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "val_auc": val_metrics["auc"],
            }
            writer.writerow(row)
            f.flush()
            print(
                f"Epoch {epoch}/{cfg.epochs}  "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_metrics['loss']:.4f}  "
                f"val_acc={val_metrics['accuracy']:.4f}  "
                f"val_f1={val_metrics['f1']:.4f}  "
                f"val_auc={val_metrics['auc']:.4f}"
            )

            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                patience_left = cfg.patience
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "epoch": epoch,
                        "val_f1": best_f1,
                        "class_names": cfg.class_names,
                        "config": {
                            "image_size": cfg.image_size,
                            "num_classes": cfg.num_classes,
                        },
                    },
                    best_path,
                )
                print(f"  saved best checkpoint → {best_path}")
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print("Early stopping.")
                    break

    return best_path
