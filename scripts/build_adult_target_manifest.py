#!/usr/bin/env python3
"""Repoint a persisted ECH2O split at adult-response targets and screen dates.

Inputs and site assignments come from the source manifest. Adult target files
are flat under --target-root and must match the original target grid exactly.
Only selected dates with usable, plausible values in all five targets are kept.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import rasterio
import xarray as xr

from src.data.phase2_qc import TARGET_CHANNELS
from src.data.target_qc import TARGET_PLAUSIBILITY_RANGES
from src.data.target_source import read_target, target_time_count
from src.data.temporal_contract import TEMPORAL_CONTRACT, target_index_for_date


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def matching_grid(original: Path, adult: Path) -> bool:
    with rasterio.open(f"NETCDF:{original}:soilmoisture") as left, rasterio.open(
        f"NETCDF:{adult}:soilmoisture"
    ) as right:
        return (
            left.crs == right.crs
            and left.transform.almost_equals(right.transform)
            and left.width == right.width
            and left.height == right.height
            and left.count == right.count
        )


def screen_site(rows: list[dict[str, str]], adult: Path) -> tuple[set[int], dict[int, str]]:
    water_year = int(rows[0]["water_year_end"])
    indices = sorted({int(row["target_time_index"]) for row in rows})
    for row in rows:
        target_date = date.fromisoformat(row["target_date"])
        derived = target_index_for_date(target_date, water_year, 366)
        if int(row["target_time_index"]) != derived:
            raise ValueError(f"{row['site_id']} {target_date}: target index does not match Oct 1 water-year origin")
    rejected: dict[int, str] = {}
    with xr.open_dataset(adult, engine="h5netcdf", decode_cf=False, mask_and_scale=False) as dataset:
        counts = {name: target_time_count(dataset, name) for name in TARGET_CHANNELS}
        if len(set(counts.values())) != 1:
            raise ValueError(f"Target band counts disagree: {counts}")
        count = next(iter(counts.values()))
        expected = (date(water_year, 10, 1) - date(water_year - 1, 10, 1)).days
        if count != expected or indices[-1] >= count:
            raise ValueError(f"Expected {expected} water-year target bands, got {count}")
        for name in TARGET_CHANNELS:
            lower, upper = TARGET_PLAUSIBILITY_RANGES[name]
            # Read one variable at a time; peak memory stays bounded by one site.
            values = read_target(dataset, name, indices)
            if values.shape[0] != len(indices):
                raise ValueError(f"Unexpected selected-band shape for {name}: {values.shape}")
            finite = np.isfinite(values)
            bad = np.count_nonzero(finite & ((values < lower) | (values > upper)), axis=(1, 2))
            valid = np.count_nonzero(finite & (values >= lower) & (values <= upper), axis=(1, 2))
            for index, invalid_count, valid_count in zip(indices, bad, valid):
                if invalid_count:
                    rejected[index] = f"{name}_out_of_range"
                elif not valid_count and index not in rejected:
                    rejected[index] = f"{name}_no_valid_pixels"
    return set(indices) - set(rejected), rejected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--site-limit", type=int, help="Pilot only: inspect the first N sorted sites")
    args = parser.parse_args()
    if not args.target_root.is_dir():
        parser.error(f"Adult target directory does not exist: {args.target_root}")
    with args.source_manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        source_rows = list(reader)
    required = {"site_id", "split", "water_year_end", "temporal_contract", "target_date", "target_time_index", "target_source"}
    if not source_rows or not required <= set(fields):
        parser.error(f"Source manifest empty or missing fields: {sorted(required - set(fields))}")
    if {row["temporal_contract"] for row in source_rows} != {TEMPORAL_CONTRACT}:
        parser.error("Source manifest must use the calendar-forcing / water-year-target v2 contract")
    by_site: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        by_site[row["site_id"]].append(row)
    sites = sorted(by_site)
    if args.site_limit is not None:
        if args.site_limit < 1:
            parser.error("--site-limit must be positive")
        sites = sites[: args.site_limit]
    accepted: list[dict[str, str]] = []
    site_audit: list[dict[str, object]] = []
    rejected_days: list[dict[str, object]] = []
    for number, site in enumerate(sites, 1):
        rows = by_site[site]
        years = {row["water_year_end"] for row in rows}
        originals = {row["target_source"] for row in rows}
        splits = {row["split"] for row in rows}
        if len(years) != 1 or len(originals) != 1 or len(splits) != 1:
            raise ValueError(f"{site}: source manifest has conflicting split, water year, or target file")
        adult = args.target_root / f"{site}-{next(iter(years))}_subdaily_adult.nc"
        reason = None
        good: set[int] = set()
        bad: dict[int, str] = {}
        if not adult.is_file():
            reason = "missing_adult_target"
        else:
            try:
                if not matching_grid(Path(next(iter(originals))), adult):
                    reason = "adult_grid_differs_from_original"
                else:
                    good, bad = screen_site(rows, adult)
            except (OSError, ValueError, KeyError, rasterio.errors.RasterioIOError) as error:
                reason = f"invalid_adult_target: {error}"
        if reason is None:
            accepted.extend({**row, "target_source": str(adult)} for row in rows if int(row["target_time_index"]) in good)
            rejected_days.extend(
                {"site_id": site, "target_date": row["target_date"], "reason": bad[int(row["target_time_index"])]}
                for row in rows if int(row["target_time_index"]) in bad
            )
        site_audit.append({
            "site_id": site, "split": rows[0]["split"], "adult_target_source": str(adult),
            "source_rows": len(rows), "retained_rows": len(good) if reason is None else 0,
            "rejected_days": len(bad) if reason is None else len(rows), "site_issue": reason or "",
        })
        if number % 25 == 0 or number == len(sites):
            print({"screened_sites": f"{number}/{len(sites)}", "retained_rows": len(accepted)}, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "manifest.csv", accepted, fields)
    write_csv(args.output_dir / "site_audit.csv", site_audit, list(site_audit[0]))
    write_csv(args.output_dir / "rejected_days.csv", rejected_days, ["site_id", "target_date", "reason"])
    summary = {
        "source_manifest": str(args.source_manifest), "target_root": str(args.target_root),
        "scope": "pilot" if args.site_limit is not None else "full",
        "source_sites": len(sites), "retained_sites": len({row["site_id"] for row in accepted}),
        "source_rows": sum(len(by_site[site]) for site in sites), "retained_rows": len(accepted),
        "failed_target_day_rows": len(rejected_days),
        "missing_adult_file_rows": sum(int(row["source_rows"]) for row in site_audit if row["site_issue"] == "missing_adult_target"),
        "retained_sites_by_split": dict(sorted(Counter(row["split"] for row in site_audit if row["retained_rows"]).items())),
        "site_issues": dict(sorted(Counter(row["site_issue"] for row in site_audit if row["site_issue"]).items())),
        "rejected_day_reasons": dict(sorted(Counter(row["reason"] for row in rejected_days).items())),
        "date_contract": "adult target band 0 is October 1 of the preceding calendar year, confirmed by the ECH2O producer; forcing band 0 is January 1",
        "target_channels": list(TARGET_CHANNELS),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
