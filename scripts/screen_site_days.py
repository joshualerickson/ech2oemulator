#!/usr/bin/env python3
"""Persist a bounded site/day QA screen; intended for large bbox resumability."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.phase2_qc import iter_daily_screen_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", type=Path, required=True)
    parser.add_argument("--water-year-end", type=int, required=True)
    parser.add_argument("--time-start", type=int, required=True)
    parser.add_argument("--time-stop", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = list(
        iter_daily_screen_rows(
            args.site_dir,
            args.water_year_end,
            time_start=args.time_start,
            time_stop=args.time_stop,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} daily QA rows to {args.output}")


if __name__ == "__main__":
    main()
