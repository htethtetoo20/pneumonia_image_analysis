"""Configuration loading and defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    seed: int = 42
    data_dir: str = "data/chest_xray"
    artifacts_dir: str = "artifacts"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 3
    freeze_backbone: bool = False
    num_classes: int = 2
    class_names: list[str] = field(default_factory=lambda: ["NORMAL", "PNEUMONIA"])

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def artifacts_path(self) -> Path:
        return Path(self.artifacts_dir)


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Load config from YAML and apply optional keyword overrides."""
    values: dict[str, Any] = {}
    if path is not None:
        with open(path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        values.update(loaded)
    values.update({k: v for k, v in overrides.items() if v is not None})
    known = {f.name for f in Config.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return Config(**{k: v for k, v in values.items() if k in known})
