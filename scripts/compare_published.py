#!/usr/bin/env python3
"""
Print our results next to published ones on the same dataset.

Most published work on this dataset skips the official 624-image test folder
in favor of its own random split over all ~5,856 images, usually without
grouping by patient -- both choices inflate scores, so comparisons need that
caveat rather than a plain ranking. Published figures and their caveats live
in references/published_results.yaml; this script renders them against
whatever is currently in the artifacts directories.

    python scripts/compare_published.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TASK_TITLES = {
    "binary": "Normal vs Pneumonia",
    "subtype": "Normal vs Bacterial vs Viral (3-class)",
    "bacterial_vs_viral": "Bacterial vs Viral (2-class)",
}


def load_ours(spec: dict) -> dict | None:
    path = ROOT / spec["metrics_path"]
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def format_ci(metrics: dict, key: str) -> str:
    ci = (metrics.get("ci_95") or {}).get(key)
    return f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, default=ROOT / "references" / "published_results.yaml")
    args = parser.parse_args()

    with open(args.reference, encoding="utf-8") as f:
        ref = yaml.safe_load(f)

    by_task: dict[str, list[tuple[dict, dict]]] = {}
    for entry in ref["results"]:
        for task, payload in entry["tasks"].items():
            by_task.setdefault(task, []).append((entry, payload))

    for task in ("binary", "subtype", "bacterial_vs_viral"):
        if task not in by_task and task not in ref["ours"]:
            continue
        print(f"\n{TASK_TITLES.get(task, task)}")
        print("=" * 78)
        print(f"  {'source':<42}{'accuracy':>10}  {'split':<30}")
        print("  " + "-" * 82)

        for entry, payload in by_task.get(task, []):
            marker = "" if payload.get("comparable") else "  (*)"
            source = f"{entry['authors']} {entry['year']}{marker}"
            print(f"  {source:<42}{payload['accuracy']:>10.4f}  {payload['split']:<30}")

        spec = ref["ours"].get(task)
        if spec:
            metrics = load_ours(spec)
            if metrics is None:
                print(f"  {'ours (not yet evaluated)':<42}{'--':>10}  {spec['metrics_path']}")
            else:
                acc = metrics["accuracy"]
                print(f"  {'ours -- ' + spec['label']:<42}{acc:>10.4f}  {'official test set (624 images)':<30}")
                print(f"  {'':<42}{'':>10}  95% CI{format_ci(metrics, 'accuracy')}")
                if metrics.get("f1") is not None:
                    print(f"  {'':<42}{'':>10}  macro F1 {metrics['f1']:.4f}{format_ci(metrics, 'f1')}")

        notes = [(e, p) for e, p in by_task.get(task, []) if not p.get("comparable")]
        if notes:
            print("\n  (*) not directly comparable:")
            for entry, payload in notes:
                why = " ".join((payload.get("why") or "").split())
                print(f"      {entry['authors']} {entry['year']}: {why}")

    print("\n" + "=" * 78)
    print("Only entries without (*) share our test set. The rest are context, not a")
    print("leaderboard: a random split of all 5,856 images scores higher than the")
    print("official 624-image test folder regardless of model quality.")


if __name__ == "__main__":
    main()
