#!/usr/bin/env python3
"""Summarize pure Jan--Sep forcing bbox completeness from daily QA screens.

No target artifacts or values are considered here.  A clean day requires all
six forcing rasters to be available and valid over the full target-support bbox.
The edge columns separately identify invalid perimeter pixels.  Longest
contiguous clean run supports fixed-window and stateful replay eligibility.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path


def longest_run(values: list[bool]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.daily_dir.glob("*_daily_screen.csv")):
        with path.open(newline="") as handle:
            daily = [row for row in csv.DictReader(handle) if date.fromisoformat(row["date"]).month in range(1, 10)]
        if not daily:
            continue
        clean = [row.get("forcing_available") == "True" and row.get("forcing_valid") == "True" for row in daily]
        edge = [int(row.get("forcing_edge_invalid_pixel_count") or 0) > 0 for row in daily]
        rows.append({
            "site_id": daily[0]["site_id"], "water_year_end": daily[0]["water_year_end"],
            "jan_sep_days_expected": len(daily), "jan_sep_forcing_clean_days": sum(clean),
            "jan_sep_forcing_unavailable_days": sum(row.get("forcing_available") != "True" for row in daily),
            "jan_sep_forcing_invalid_bbox_days": sum(row.get("forcing_available") == "True" and row.get("forcing_valid") != "True" for row in daily),
            "jan_sep_forcing_edge_na_days": sum(edge),
            "jan_sep_forcing_edge_invalid_pixels_total": sum(int(row.get("forcing_edge_invalid_pixel_count") or 0) for row in daily),
            "longest_contiguous_clean_forcing_days": longest_run(clean),
            "forcing_bbox_status": "clean_full_jan_sep" if all(clean) else "has_bbox_breaks",
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["site_id"])
        writer.writeheader()
        writer.writerows(rows)
    print({"output": str(args.output), "sites": len(rows), "clean_full_jan_sep": sum(row["forcing_bbox_status"] == "clean_full_jan_sep" for row in rows)})


if __name__ == "__main__":
    main()
