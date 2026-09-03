#!/usr/bin/env python3
"""Build a validated Seq90 recovery cohort from the original training archive.

This intentionally does not read ECH2O targets.  It identifies sites that were
missing from an MTBS input root but whose original forcing/static inputs remain
in a training archive, and checks that the VIIRS window plus five-day buffer can
be served by a complete 90-day forcing history.
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import date, timedelta
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rasterio

from src.data.phase2_qc import DYNAMIC_CHANNELS


def dated_bands(path: Path) -> list[date]:
    """Read and validate the daily band-description contract without pixels."""
    with rasterio.open(path) as dataset:
        try:
            days = [date.fromisoformat(f"{value[:4]}-{value[4:6]}-{value[6:8]}") for value in dataset.descriptions if value]
        except (IndexError, ValueError) as error:
            raise ValueError("invalid_band_descriptions") from error
        if len(days) != dataset.count:
            raise ValueError("missing_band_descriptions")
        if any(next_day - current != timedelta(days=1) for current, next_day in zip(days, days[1:])):
            raise ValueError("noncontiguous_band_dates")
        return days


def viirs_windows(path: Path) -> dict[str, tuple[date, date]]:
    windows: dict[str, tuple[date, date]] = {}
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            site = row.get("event_id", "").strip()
            start, end = row.get("viirs_start_date", "").strip(), row.get("viirs_end_date", "").strip()
            if not site or not start or not end:
                continue
            interval = (date.fromisoformat(start), date.fromisoformat(end))
            if interval[1] < interval[0]:
                raise ValueError(f"{site}: VIIRS end predates start")
            if site in windows and windows[site] != interval:
                raise ValueError(f"{site}: conflicting VIIRS intervals")
            windows[site] = interval
    return windows


def evaluate_site(site: str, root: Path, windows: dict[str, tuple[date, date]], buffer_days: int, sequence_length: int) -> dict[str, str | int | bool]:
    row: dict[str, str | int | bool] = {"site_id": site, "source_folder_exists": False, "runnable": False, "status": ""}
    folder = root / site
    if not folder.is_dir():
        row["status"] = "absent_from_training_archive"
        return row
    row["source_folder_exists"] = True
    if site not in windows:
        row["status"] = "absent_from_viirs_csv"
        return row
    all_days: list[date] | None = None
    for channel in DYNAMIC_CHANNELS:
        paths = sorted(folder.glob(f"{channel}_*.tif"))
        if len(paths) != 1:
            row["status"] = f"invalid_{channel}_file_count_{len(paths)}"
            return row
        try:
            days = dated_bands(paths[0])
        except ValueError as error:
            row["status"] = f"invalid_{channel}_{error}"
            return row
        if all_days is None:
            all_days = days
        elif days != all_days:
            row["status"] = f"forcing_dates_disagree_{channel}"
            return row
    assert all_days is not None
    viirs_start, viirs_end = windows[site]
    requested_start = viirs_start - timedelta(days=buffer_days)
    requested_end = viirs_end + timedelta(days=buffer_days)
    required_start = requested_start - timedelta(days=sequence_length - 1)
    row.update({
        "forcing_start": all_days[0].isoformat(), "forcing_end": all_days[-1].isoformat(),
        "viirs_start": viirs_start.isoformat(), "viirs_end": viirs_end.isoformat(),
        "requested_prediction_start": requested_start.isoformat(), "requested_prediction_end": requested_end.isoformat(),
        "required_history_start": required_start.isoformat(),
    })
    available = set(all_days)
    if any(day not in available for day in (required_start, requested_start, requested_end)):
        row["status"] = "insufficient_complete_90_day_forcing_window"
        return row
    row["runnable"] = True
    row["status"] = "runnable"
    row["requested_prediction_days"] = (requested_end - requested_start).days + 1
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnosis-csv", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--viirs-date-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--runnable-site-list", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=90)
    parser.add_argument("--date-buffer-days", type=int, default=5)
    args = parser.parse_args()
    if args.sequence_length < 1 or args.date_buffer_days < 0:
        parser.error("sequence length must be positive and buffer days non-negative")
    with args.diagnosis_csv.open(newline="", encoding="utf-8") as source:
        requested = [row["site_id"] for row in csv.DictReader(source) if row.get("diagnosis") == "missing_input_folder"]
    windows = viirs_windows(args.viirs_date_csv)
    rows = [evaluate_site(site, args.training_root, windows, args.date_buffer_days, args.sequence_length) for site in requested]
    fields = sorted({field for row in rows for field in row})
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    runnable = [str(row["site_id"]) for row in rows if row["runnable"]]
    args.runnable_site_list.write_text("\n".join(runnable) + ("\n" if runnable else ""), encoding="utf-8")
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1
    print({"requested_missing_sites": len(requested), "runnable_sites": len(runnable), "status_counts": counts, "output_csv": str(args.output_csv)})


if __name__ == "__main__":
    main()
