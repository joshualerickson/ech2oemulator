#!/usr/bin/env python3
"""Select a small, state-balanced training subset while preserving held-out rows."""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def select_sites(site_ids: list[str], count: int, seed: int) -> list[str]:
    by_state: dict[str, list[str]] = defaultdict(list)
    for site_id in sorted(site_ids):
        by_state[site_id[:2]].append(site_id)
    rng = random.Random(seed)
    for values in by_state.values():
        rng.shuffle(values)
    selected: list[str] = []
    while len(selected) < count:
        added = False
        for state in sorted(by_state):
            if by_state[state] and len(selected) < count:
                selected.append(by_state[state].pop())
                added = True
        if not added:
            break
    return sorted(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-site-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()
    with args.source_manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    train_sites = sorted({row["site_id"] for row in rows if row["split"] == "dev_train"})
    selected = select_sites(train_sites, args.train_site_count, args.seed)
    if len(selected) != args.train_site_count:
        raise ValueError(f"Requested {args.train_site_count} train sites but only selected {len(selected)}")
    subset = [row for row in rows if row["split"] != "dev_train" or row["site_id"] in selected]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(subset)
    summary = {
        "source_manifest": str(args.source_manifest), "seed": args.seed,
        "selected_train_sites": selected,
        "rows_by_split": {split: sum(row["split"] == split for row in subset) for split in sorted({row["split"] for row in subset})},
        "sites_by_split": {split: sorted({row["site_id"] for row in subset if row["split"] == split}) for split in sorted({row["split"] for row in subset})},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
