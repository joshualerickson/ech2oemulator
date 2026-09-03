#!/usr/bin/env python3
"""Atomically align completed 30 m NetCDF time axes with the 240 m contract."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

import h5py
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.predict_fire_seq90 import read_viirs_windows


SUFFIX = "_direct_30m.nc"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--viirs-date-csv", type=Path, required=True)
    parser.add_argument("--date-buffer-days", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.date_buffer_days < 0:
        parser.error("date-buffer-days cannot be negative")

    windows = read_viirs_windows(args.viirs_date_csv)
    repaired = already_correct = 0
    for path in sorted(args.output_dir.glob(f"*{SUFFIX}")):
        site_id = path.name[: -len(SUFFIX)]
        if site_id not in windows:
            raise ValueError(f"{site_id} is absent from {args.viirs_date_csv}")
        viirs_start, viirs_end = windows[site_id]
        first_day = viirs_start - timedelta(days=args.date_buffer_days)
        expected_count = (viirs_end - viirs_start).days + 1 + 2 * args.date_buffer_days
        expected_units = f"days since {first_day.isoformat()} 00:00:00"
        expected_values = np.arange(expected_count, dtype="int64")
        with h5py.File(path, "r") as dataset:
            time = dataset["time"]
            if time.shape != (expected_count,):
                raise ValueError(
                    f"{path}: time length {time.shape} does not match expected {(expected_count,)}"
                )
            correct = (
                time.attrs.get("units") == expected_units
                and time.attrs.get("calendar") == "proleptic_gregorian"
                and np.array_equal(time[:], expected_values)
            )
        if correct:
            already_correct += 1
            continue
        if args.dry_run:
            print({"site_id": site_id, "status": "would_repair", "units": expected_units})
            repaired += 1
            continue

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.stem}.timefix.", suffix=".tmp.nc", dir=path.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            shutil.copy2(path, temporary)
            with h5py.File(temporary, "r+") as dataset:
                dataset["time"][:] = expected_values
                dataset["time"].attrs["units"] = expected_units
                dataset["time"].attrs["calendar"] = "proleptic_gregorian"
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        Path(f"{path}.aux.xml").unlink(missing_ok=True)
        repaired += 1
        print({"site_id": site_id, "status": "repaired", "units": expected_units})
    print({"repaired": repaired, "already_correct": already_correct})


if __name__ == "__main__":
    main()
