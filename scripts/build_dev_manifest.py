#!/usr/bin/env python3
"""Build a small, leakage-aware development manifest from completed daily QA files."""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.temporal_contract import TEMPORAL_CONTRACT


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_sites(site_ids: list[str], count: int, seed: int) -> list[str]:
    """Take a deterministic round-robin sample across available state prefixes."""
    by_state: dict[str, list[str]] = defaultdict(list)
    for site_id in site_ids:
        by_state[site_id[:2]].append(site_id)
    rng = random.Random(seed)
    for values in by_state.values():
        rng.shuffle(values)
    selected: list[str] = []
    while len(selected) < min(count, len(site_ids)):
        added = False
        for state in sorted(by_state):
            if by_state[state] and len(selected) < count:
                selected.append(by_state[state].pop())
                added = True
        if not added:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--site-count", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    daily_paths = sorted(args.daily_dir.glob("*_daily_screen.csv"))
    daily_by_site = {rows[0]["site_id"]: rows for path in daily_paths if (rows := read_csv(path))}
    if len(daily_by_site) < 12:
        raise ValueError("Need at least 12 completed site-day files for the development split")
    sites = select_sites(sorted(daily_by_site), args.site_count, args.seed)
    n_test = max(2, round(len(sites) * 0.15))
    n_val = max(2, round(len(sites) * 0.15))
    test_sites, val_sites, train_sites = sites[:n_test], sites[n_test:n_test + n_val], sites[n_test + n_val:]

    output_rows: list[dict[str, object]] = []
    embargo_days = args.sequence_length - 1
    for site_id in sites:
        daily = daily_by_site[site_id]
        by_date = {date.fromisoformat(row["date"]): row for row in daily}
        candidate_dates = sorted(by_date)
        temporal_start = candidate_dates[int(len(candidate_dates) * 0.85)]
        for target_date in candidate_dates:
            history = [by_date.get(target_date - timedelta(days=offset)) for offset in range(embargo_days, -1, -1)]
            if any(row is None for row in history) or not all(row["forcing_valid"] == "True" for row in history):
                continue
            target = by_date[target_date]
            if target["target_qa_valid"] != "True":
                continue
            if site_id in test_sites:
                split = "dev_spatial_test"
            elif site_id in val_sites:
                split = "dev_spatial_val"
            elif target_date >= temporal_start:
                split = "dev_temporal_val"
            # A train sequence ending at t and the first temporal-validation
            # sequence beginning at temporal_start overlap unless t is at
            # least one full sequence length earlier.
            elif target_date <= temporal_start - timedelta(days=args.sequence_length):
                split = "dev_train"
            else:
                continue
            water_year = target["water_year_end"]
            site_dir = args.data_root / site_id
            output_rows.append({
                "split": split, "site_id": site_id, "bbox_id": site_id,
                "water_year_end": water_year, "temporal_contract": TEMPORAL_CONTRACT, "start_date": history[0]["date"],
                "end_date": target["date"], "target_date": target["date"],
                "sequence_length": args.sequence_length, "target_time_index": target["target_time_index"], "forcing_start_index": history[0]["forcing_time_index"], "forcing_end_index": target["forcing_time_index"],
                "dynamic_sequence_source": str(site_dir), "static_source": str(site_dir / "Spatial"),
                "target_source": str(site_dir / f"{site_id}-{water_year}_subdaily.nc"),
            })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "dev_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0])); writer.writeheader(); writer.writerows(output_rows)
    summary = {"seed": args.seed, "sequence_length": args.sequence_length, "embargo_days": embargo_days,
               "selected_sites": sites, "train_sites": train_sites, "spatial_val_sites": val_sites,
               "spatial_test_sites": test_sites,
               "rows_by_split": {split: sum(row["split"] == split for row in output_rows) for split in sorted({row["split"] for row in output_rows})}}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
