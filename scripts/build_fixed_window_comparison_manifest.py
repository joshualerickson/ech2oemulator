#!/usr/bin/env python3
"""Rebuild a longer-window manifest while preserving an existing site split."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.temporal_contract import TEMPORAL_CONTRACT


FIELDS = (
    "split", "site_id", "bbox_id", "water_year_end", "temporal_contract", "start_date", "end_date", "target_date",
    "sequence_length", "target_time_index", "forcing_start_index", "forcing_end_index", "dynamic_sequence_source", "static_source", "target_source",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-manifest", type=Path, required=True,
                        help="Existing manifest whose site-to-split assignments are immutable.")
    parser.add_argument("--daily-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--months", type=int, nargs="+", default=[6, 7, 8, 9])
    args = parser.parse_args()
    if args.sequence_length < 1:
        parser.error("--sequence-length must be positive")

    with args.reference_manifest.open(newline="") as handle:
        reference = list(csv.DictReader(handle))
    sites: dict[str, dict[str, str]] = {}
    for row in reference:
        prior = sites.setdefault(row["site_id"], row)
        if prior["split"] != row["split"]:
            raise ValueError(f"{row['site_id']} has conflicting reference splits")

    months = set(args.months)
    rows: list[dict[str, object]] = []
    excluded: list[dict[str, str]] = []
    for site_id, source in sorted(sites.items()):
        matches = list(args.daily_dir.glob(f"{site_id}_*_daily_screen.csv"))
        if len(matches) != 1:
            excluded.append({"site_id": site_id, "split": source["split"], "reason": "missing_or_ambiguous_daily_screen"})
            continue
        with matches[0].open(newline="") as handle:
            daily = list(csv.DictReader(handle))
        by_date = {date.fromisoformat(row["date"]): row for row in daily}
        eligible = 0
        for target_date, target in sorted(by_date.items()):
            if target_date.month not in months or target["target_qa_valid"] != "True":
                continue
            history = [by_date.get(target_date - timedelta(days=offset)) for offset in range(args.sequence_length - 1, -1, -1)]
            if any(day is None or day["forcing_valid"] != "True" for day in history):
                continue
            rows.append({
                "split": source["split"], "site_id": site_id, "bbox_id": source["bbox_id"],
                "water_year_end": source["water_year_end"], "temporal_contract": TEMPORAL_CONTRACT, "start_date": history[0]["date"],
                "end_date": target["date"], "target_date": target["date"],
                "sequence_length": args.sequence_length, "target_time_index": target["target_time_index"], "forcing_start_index": history[0]["forcing_time_index"], "forcing_end_index": target["forcing_time_index"],
                "dynamic_sequence_source": source["dynamic_sequence_source"],
                "static_source": source["static_source"], "target_source": source["target_source"],
            })
            eligible += 1
        if not eligible:
            excluded.append({"site_id": site_id, "split": source["split"], "reason": "no_contiguous_eligible_windows"})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "purpose": "fixed site split rebuilt at a new temporal lookback",
        "reference_manifest": str(args.reference_manifest),
        "sequence_length": args.sequence_length,
        "months": sorted(months),
        "rows_by_split": dict(sorted(Counter(row["split"] for row in rows).items())),
        "sites_by_split": dict(sorted(Counter(row["split"] for row in rows).items())),
        "excluded_sites": excluded,
    }
    # Count distinct sites rather than sequence rows in the two site fields.
    summary["sites_by_split"] = {
        split: len({row["site_id"] for row in rows if row["split"] == split})
        for split in sorted({row["split"] for row in rows})
    }
    summary["eligible_sites_by_split"] = summary["sites_by_split"]
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
