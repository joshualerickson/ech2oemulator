#!/usr/bin/env python3
"""Build daily QA records and a contiguous sequence-target manifest for ECH2O."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.phase2_qc import iter_daily_screen_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema-report", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--site-offset", type=int, default=0)
    parser.add_argument("--site-limit", type=int)
    args = parser.parse_args()
    if args.sequence_length < 1:
        raise ValueError("sequence length must be positive")

    schema = json.loads(args.schema_report.read_text(encoding="utf-8"))
    source_valid = [site for site in schema["sites"] if not site["issues"]]
    selected = source_valid[args.site_offset :]
    if args.site_limit is not None:
        selected = selected[: args.site_limit]

    daily_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    for site in selected:
        site_id = str(site["site_id"])
        water_year_end = int(site["water_year_end"])
        site_dir = args.data_root / site_id
        rows = list(iter_daily_screen_rows(site_dir, water_year_end))
        daily_rows.extend(rows)
        by_date = {date.fromisoformat(str(row["date"])): row for row in rows}
        for target_date in sorted(by_date):
            start_date = target_date - timedelta(days=args.sequence_length - 1)
            sequence_dates = [start_date + timedelta(days=index) for index in range(args.sequence_length)]
            history = [by_date.get(current) for current in sequence_dates]
            if any(row is None for row in history):
                continue
            assert all(row is not None for row in history)
            target_row = by_date[target_date]
            assert target_row is not None
            if not all(bool(row["forcing_valid"]) for row in history):
                continue
            if not bool(target_row["target_qa_valid"]):
                continue
            manifest_rows.append(
                {
                    "site_id": site_id,
                    "bbox_id": site_id,
                    "water_year_end": water_year_end,
                    "start_date": start_date.isoformat(),
                    "end_date": target_date.isoformat(),
                    "target_date": target_date.isoformat(),
                    "sequence_length": args.sequence_length,
                    "dynamic_sequence_source": str(site_dir),
                    "static_source": str(site_dir / "Spatial"),
                    "target_source": str(site_dir / f"{site_id}-{water_year_end}_subdaily.nc"),
                    "crs": site["target_grid"]["crs"],
                    "target_width": site["target_grid"]["width"],
                    "target_height": site["target_grid"]["height"],
                }
            )
    stem = f"sequence_{args.sequence_length}_sites_{args.site_offset}_{args.site_offset + len(selected)}"
    write_csv(args.output_dir / f"{stem}_daily_screen.csv", daily_rows)
    write_csv(args.output_dir / f"{stem}_manifest.csv", manifest_rows)
    summary = {
        "source_valid_sites_selected": len(selected),
        "daily_row_count": len(daily_rows),
        "manifest_row_count": len(manifest_rows),
        "daily_forcing_invalid_count": sum(not bool(row["forcing_valid"]) for row in daily_rows),
        "daily_target_qa_invalid_count": sum(not bool(row["target_qa_valid"]) for row in daily_rows),
        "target_violation_day_counts": dict(
            Counter(
                target
                for row in daily_rows
                for target in ("soilmoisture", "tskin_am", "tskin_pm", "plc_am", "plc_pm")
                if int(row[f"{target}_plausibility_violation_count"]) > 0
            )
        ),
    }
    (args.output_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
