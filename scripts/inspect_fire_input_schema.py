#!/usr/bin/env python3
"""Audit prediction-only fire folders without assuming forcing band dates."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import rasterio

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.phase2_qc import DYNAMIC_CHANNELS
from src.data.static_contract import STATIC_CHANNELS, resolve_ascii_static


def parse_dates(dataset: rasterio.DatasetReader, path: Path) -> list[date]:
    raw = dataset.descriptions
    if any(value is None for value in raw):
        raise ValueError(f"{path} has an undescribed forcing band")
    try:
        dates = [date.fromisoformat(f"{value[:4]}-{value[4:6]}-{value[6:8]}") for value in raw if value]
    except (IndexError, ValueError) as error:
        raise ValueError(f"{path} has non-YYYYMMDD band descriptions") from error
    if any(next_day - current != timedelta(days=1) for current, next_day in zip(dates, dates[1:])):
        raise ValueError(f"{path} forcing-band dates are not daily-contiguous")
    return dates


def inspect_site(site: Path) -> dict[str, object]:
    result: dict[str, object] = {"site_id": site.name, "issues": []}
    force_paths = []
    for channel in DYNAMIC_CHANNELS:
        matches = sorted(site.glob(f"{channel}_*.tif"))
        if len(matches) != 1:
            result["issues"].append(f"{channel}: expected_one_tif_found_{len(matches)}")  # type: ignore[index]
        else:
            force_paths.append(matches[0])
    static_missing = []
    for channel in STATIC_CHANNELS:
        if channel in {"twi", "fac", "tpi", "psst", "wbdef"}:
            continue
        try:
            resolve_ascii_static(site / "Spatial", channel)
        except FileNotFoundError:
            static_missing.append(channel)
    result["missing_selected_static_channels"] = static_missing
    if len(force_paths) != len(DYNAMIC_CHANNELS):
        return result
    reference = None
    for path in force_paths:
        try:
            with rasterio.open(path) as dataset:
                item = {
                    "path": str(path), "count": dataset.count, "width": dataset.width,
                    "height": dataset.height, "crs": str(dataset.crs),
                    "transform": tuple(dataset.transform), "date_range": [value.isoformat() for value in (parse_dates(dataset, path)[0], parse_dates(dataset, path)[-1])],
                }
        except Exception as error:
            result["issues"].append(str(error))  # type: ignore[index]
            continue
        if reference is None:
            reference = item
        elif {key: item[key] for key in ("count", "width", "height", "crs", "transform", "date_range")} != {key: reference[key] for key in ("count", "width", "height", "crs", "transform", "date_range")}:
            result["issues"].append(f"forcing metadata differs: {path.name}")  # type: ignore[index]
    if reference:
        result["forcing"] = reference
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--site-limit", type=int, default=0)
    args = parser.parse_args()
    sites = sorted(path for path in args.input_root.iterdir() if path.is_dir())
    if args.site_limit:
        sites = sites[:args.site_limit]
    records = []
    for number, site in enumerate(sites, start=1):
        records.append(inspect_site(site))
        if number % 10 == 0 or number == len(sites):
            print({"inspected": f"{number}/{len(sites)}"}, flush=True)
    result = {
        "input_root": str(args.input_root), "site_count": len(records),
        "sites": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print({"output": str(args.output), "site_count": len(records)})


if __name__ == "__main__":
    main()
