"""ResNet50 classifier for binary pneumonia detection."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int = 2, freeze_backbone: bool = False, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False
    return model


def get_target_layer(model: nn.Module) -> nn.Module:
    """Last convolutional block for Grad-CAM."""
    return model.layer4[-1]


def load_checkpoint(
    path: str,
    device: torch.device,
    num_classes: int = 2,
    freeze_backbone: bool = False,
) -> nn.Module:
    model = build_model(num_classes=num_classes, freeze_backbone=freeze_backbone, pretrained=False)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
