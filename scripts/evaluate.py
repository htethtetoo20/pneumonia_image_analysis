#!/usr/bin/env python3
"""Evaluate a trained checkpoint on a data split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cxr_pneumonia.config import load_config
from cxr_pneumonia.evaluate import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts" / "best.pt")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, data_dir=args.data_dir)
    evaluate(cfg, args.checkpoint, split=args.split)


if __name__ == "__main__":
    main()
