"""Grad-CAM explanations for chest X-ray predictions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.utils.data import DataLoader, Subset

from cxr_pneumonia.config import Config
from cxr_pneumonia.data import IMAGENET_MEAN, IMAGENET_STD, build_dataset
from cxr_pneumonia.model import get_target_layer, load_checkpoint
from cxr_pneumonia.train import get_device


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """CHW tensor → HWC float image in [0, 1]."""
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    img = std * img + mean
    return np.clip(img, 0, 1).astype(np.float32)


def explain(
    cfg: Config,
    checkpoint: str | Path,
    split: str = "test",
    n: int = 16,
    out_dir: str | Path | None = None,
) -> Path:
    device = get_device()
    out_path = Path(out_dir) if out_dir else cfg.artifacts_path / "gradcam"
    out_path.mkdir(parents=True, exist_ok=True)

    model = load_checkpoint(str(checkpoint), device, num_classes=cfg.num_classes)
    dataset = build_dataset(cfg.data_path, split, cfg.image_size, train=False)

    # Sample up to n images, interleaved across classes (NORMAL, PNEUMONIA, ...)
    by_class: dict[int, list[int]] = {i: [] for i in range(len(dataset.classes))}
    for idx, (_, label) in enumerate(dataset.samples):
        by_class[label].append(idx)
    per_class = max(1, (n + len(by_class) - 1) // max(len(by_class), 1))
    class_iters = [idxs[:per_class] for idxs in by_class.values()]
    indices: list[int] = []
    for i in range(per_class):
        for class_idxs in class_iters:
            if i < len(class_idxs) and len(indices) < n:
                indices.append(class_idxs[i])

    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False)

    target_layers = [get_target_layer(model)]
    # MPS can be flaky with Grad-CAM; fall back to CPU for CAM if needed
    cam_device = device
    if device.type == "mps":
        model = model.to("cpu")
        cam_device = torch.device("cpu")

    saved = 0
    with GradCAM(model=model, target_layers=target_layers) as cam:
        for i, (images, labels) in enumerate(loader):
            images = images.to(cam_device)
            labels = labels.to(cam_device)
            with torch.no_grad():
                logits = model(images)
                pred = int(logits.argmax(dim=1).item())
                prob = float(torch.softmax(logits, dim=1)[0, pred].item())

            targets = [ClassifierOutputTarget(pred)]
            grayscale_cam = cam(input_tensor=images, targets=targets)[0]
            rgb = denormalize(images[0])
            overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)

            true_name = cfg.class_names[int(labels.item())]
            pred_name = cfg.class_names[pred]
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(rgb)
            axes[0].set_title(f"Original\ntrue={true_name}")
            axes[1].imshow(grayscale_cam, cmap="jet")
            axes[1].set_title("Grad-CAM")
            axes[2].imshow(overlay)
            axes[2].set_title(f"Overlay\npred={pred_name} ({prob:.2f})")
            for ax in axes:
                ax.axis("off")
            fig.suptitle(f"{split} sample {indices[i]}")
            fig.tight_layout()
            fig_path = out_path / f"gradcam_{split}_{i:02d}_true-{true_name}_pred-{pred_name}.png"
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
            saved += 1

    print(f"Saved {saved} Grad-CAM figures → {out_path}")
    return out_path


def predict_with_gradcam(
    cfg: Config,
    checkpoint: str | Path,
    image: Image.Image | str | Path,
) -> dict:
    """
    Predict NORMAL/PNEUMONIA and return Grad-CAM arrays for UI use.

    Returns dict with keys: pred_idx, pred_name, prob, probs, rgb, cam, overlay.
    """
    device = get_device()
    model = load_checkpoint(str(checkpoint), device, num_classes=cfg.num_classes)
    if device.type == "mps":
        model = model.to("cpu")
        device = torch.device("cpu")

    from cxr_pneumonia.data import get_transforms

    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    transform = get_transforms(cfg.image_size, train=False)
    tensor = transform(image).unsqueeze(0).to(device)

    target_layers = [get_target_layer(model)]
    with GradCAM(model=model, target_layers=target_layers) as cam:
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            pred = int(logits.argmax(dim=1).item())
            prob = float(probs[pred].item())
        grayscale_cam = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(pred)])[0]
        rgb = denormalize(tensor[0])
        overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)

    return {
        "pred_idx": pred,
        "pred_name": cfg.class_names[pred],
        "prob": prob,
        "probs": {name: float(probs[i].item()) for i, name in enumerate(cfg.class_names)},
        "rgb": rgb,
        "cam": grayscale_cam,
        "overlay": overlay,
    }


def explain_image_path(
    cfg: Config,
    checkpoint: str | Path,
    image_path: str | Path,
    out_path: str | Path | None = None,
) -> Path:
    """Run Grad-CAM on a single image file and save a 3-panel figure."""
    result = predict_with_gradcam(cfg, checkpoint, image_path)
    dest = Path(out_path) if out_path else cfg.artifacts_path / "gradcam" / f"single_{Path(image_path).stem}.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(result["rgb"])
    axes[0].set_title("Original")
    axes[1].imshow(result["cam"], cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(result["overlay"])
    axes[2].set_title(f"pred={result['pred_name']} ({result['prob']:.2f})")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    print(f"Saved {dest}")
    return dest
