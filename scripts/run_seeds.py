#!/usr/bin/env python3
"""
Train and evaluate the same config under several random seeds.

A single training run can't distinguish a real improvement from a lucky
initialisation. Bootstrap CIs in evaluate.py cover test-set uncertainty; this
covers the other half -- uncertainty from training itself (weight init,
sampler draws, augmentation) with data held fixed. Each seed trains into its
own artifacts subdirectory; summary lands in <artifacts_dir>/seed_summary.json.

    python scripts/run_seeds.py --config configs/subtype.yaml --seeds 42 1337 7
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cxr_pneumonia.config import load_config
from cxr_pneumonia.evaluate import evaluate
from cxr_pneumonia.train import train

METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")


def summarise(per_seed: dict[int, dict]) -> dict:
    """Mean, sample sd and range for each metric across seeds."""
    summary: dict[str, dict] = {}
    for metric in METRICS:
        values = [m[metric] for m in per_seed.values() if m.get(metric) is not None]
        if not values:
            continue
        summary[metric] = {
            "mean": float(statistics.mean(values)),
            # stdev needs 2+ points.
            "sd": float(statistics.stdev(values)) if len(values) > 1 else None,
            "min": float(min(values)),
            "max": float(max(values)),
            "values": [float(v) for v in values],
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "subtype.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 7])
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where seed subdirectories go (default: the config's artifacts_dir)",
    )
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    out_root = args.out or (ROOT / base_cfg.artifacts_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    per_seed: dict[int, dict] = {}
    for seed in args.seeds:
        seed_dir = out_root / f"seed_{seed}"
        print(f"\n{'=' * 60}\nseed {seed} -> {seed_dir}\n{'=' * 60}")
        cfg = load_config(args.config, seed=seed, artifacts_dir=str(seed_dir))
        checkpoint = train(cfg)
        metrics = evaluate(cfg, checkpoint, split=args.split)
        per_seed[seed] = {k: metrics.get(k) for k in METRICS}

    summary = {
        "config": str(args.config),
        "split": args.split,
        "seeds": args.seeds,
        "per_seed": per_seed,
        "summary": summarise(per_seed),
    }
    out_path = out_root / "seed_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}\nAcross {len(args.seeds)} seeds ({args.split} split)\n{'=' * 60}")
    for metric, stats in summary["summary"].items():
        sd = f" +/- {stats['sd']:.4f}" if stats["sd"] is not None else ""
        spread = f"   [{stats['min']:.4f}, {stats['max']:.4f}]"
        print(f"  {metric:<10} {stats['mean']:.4f}{sd}{spread}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
