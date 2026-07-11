#!/usr/bin/env python3
"""Train the pneumonia classifier."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cxr_pneumonia.config import load_config
from cxr_pneumonia.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(
        args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
    )
    best = train(cfg)
    print(f"Best checkpoint: {best}")


if __name__ == "__main__":
    main()
