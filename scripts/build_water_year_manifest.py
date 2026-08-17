#!/usr/bin/env python3
"""Collapse a fixed-window manifest to a stateful Jan--Sep replay manifest.

Targets are water-year arrays (Oct--Sep), while forcings are calendar-year
arrays (Jan--Dec).  The replay is deliberately limited to their date-aligned
Jan--Sep overlap; it must never silently use shared target/forcing band ids.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date, timedelta
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.temporal_contract import TEMPORAL_CONTRACT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--daily-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with args.source_manifest.open(newline="") as handle:
        source = list(csv.DictReader(handle))
    if {row.get("temporal_contract") for row in source} != {TEMPORAL_CONTRACT}:
        raise ValueError("Source manifest is not v2 calendar-forcing/target-water-year data; rebuild it from corrected daily screens.")
    unique: dict[tuple[str, str], dict[str, str]] = {}
    for row in source:
        unique.setdefault((row["split"], row["site_id"]), row)

    rows: list[dict[str, str]] = []
    excluded: list[dict[str, str]] = []
    for (split, site), source_row in sorted(unique.items()):
        paths = list(args.daily_dir.glob(f"{site}_*_daily_screen.csv"))
        if len(paths) != 1:
            excluded.append({"site_id": site, "split": split, "reason": "missing_or_ambiguous_daily_screen"})
            continue
        with paths[0].open(newline="") as handle:
            daily = list(csv.DictReader(handle))
        water_year = int(source_row["water_year_end"])
        overlap = [row for row in daily if row.get("forcing_available") == "True"]
        dates = [date.fromisoformat(row["date"]) for row in overlap]
        expected_start, expected_end = date(water_year, 1, 1), date(water_year, 9, 30)
        if (
            not overlap
            or dates[0] != expected_start
            or dates[-1] != expected_end
            or any(next_day - current != timedelta(days=1) for current, next_day in zip(dates, dates[1:]))
            or not all(row["forcing_valid"] == "True" for row in overlap)
        ):
            excluded.append({"site_id": site, "split": split, "reason": "forcing_gap_or_invalid_day_in_calendar_jan_sep_overlap"})
            continue
        rows.append({
            **{key: source_row[key] for key in ("split", "site_id", "bbox_id", "water_year_end", "temporal_contract", "dynamic_sequence_source", "static_source", "target_source")},
            "start_date": expected_start.isoformat(),
            "end_date": expected_end.isoformat(),
            "daily_screen_source": str(paths[0]),
        })

    if not rows:
        raise ValueError("No source-valid Jan--Sep calendar-forcing replays")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "source_manifest": str(args.source_manifest),
        "temporal_contract": TEMPORAL_CONTRACT,
        "protocol": "stateful_calendar_jan01_to_sep30",
        "sites_by_split": {split: sum(row["split"] == split for row in rows) for split in sorted({row["split"] for row in rows})},
        "excluded_sites": excluded,
        "replay_rows": len(rows),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(summary)


if __name__ == "__main__":
    main()
