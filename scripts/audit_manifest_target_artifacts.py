#!/usr/bin/env python3
"""Score raw target spatial artifacts for sites in a persisted model manifest.

This audit is deliberately independent of forcing data and model outputs. It
uses a manifest only to preserve the original site/split cohort, then evaluates
the raw NetCDF target bands for the requested target months.  It does not make
an exclusion decision: downstream policy must be calibrated using curated
artifact and clean-control labels.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import xarray as xr

from scripts.benchmark_target_artifacts import finite_array, spatial_scores
from src.data.phase2_qc import TARGET_CHANNELS
from src.data.temporal_contract import target_dates


SCORE_NAMES = ("legacy_blockiness", "row_seam_ratio", "column_seam_ratio", "edge_gradient_ratio", "corner_jump_ratio", "high_frequency_ratio")


def parse_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    sites: dict[str, dict[str, str]] = {}
    for row in rows:
        if row.get("split") not in {"dev_train", "dev_spatial_val"}:
            continue
        prior = sites.setdefault(row["site_id"], row)
        if prior["split"] != row["split"] or prior["water_year_end"] != row["water_year_end"]:
            raise ValueError(f"{row['site_id']}: conflicting split or water-year values in manifest")
    if not sites:
        raise ValueError("Manifest contains no dev_train or dev_spatial_val sites")
    return sites


def parse_schema_report(path: Path) -> dict[str, dict[str, str]]:
    """Use every Phase 1 source-valid bundle, without inheriting an old split."""
    report = json.loads(path.read_text())
    sites: dict[str, dict[str, str]] = {}
    for site in report["sites"]:
        if site.get("issues"):
            continue
        grid = site["target_grid"]
        site_id = str(site["site_id"])
        sites[site_id] = {
            "site_id": site_id,
            "split": "source_eligible_pre_split",
            "water_year_end": str(site["water_year_end"]),
            "target_height": str(grid["height"]),
            "target_width": str(grid["width"]),
        }
    if not sites:
        raise ValueError("Schema report contains no source-valid bundles")
    return sites


def audit_site(args: tuple[dict[str, str], str, tuple[int, ...]]) -> list[dict[str, object]]:
    row, data_root_text, months = args
    site_id, water_year = row["site_id"], int(row["water_year_end"])
    path = Path(data_root_text) / site_id / f"{site_id}-{water_year}_subdaily.nc"
    if not path.is_file():
        raise FileNotFoundError(path)
    results: list[dict[str, object]] = []
    with xr.open_dataset(path, engine="h5netcdf", decode_cf=False, mask_and_scale=False) as dataset:
        count = int(dataset.sizes["time"])
        dates = target_dates(water_year, count)
        indices = [index for index, day in enumerate(dates) if day.month in months]
        for target in TARGET_CHANNELS:
            stack = finite_array(dataset[target].isel(time=indices).values)
            for local_index, source_index in enumerate(indices):
                results.append({
                    "site_id": site_id,
                    "split": row["split"],
                    "water_year_end": water_year,
                    "target": target,
                    "target_height": row.get("target_height", ""),
                    "target_width": row.get("target_width", ""),
                    "target_time_index": source_index,
                    "target_date": dates[source_index].isoformat(),
                    **spatial_scores(stack[local_index]),
                })
    return results


def metric_summary(values: list[float | None]) -> dict[str, float | None]:
    finite = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    if not finite.size:
        return {"median": None, "p95": None, "max": None}
    return {"median": float(np.median(finite)), "p95": float(np.percentile(finite, 95)), "max": float(np.max(finite))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="Persisted split manifest; audits dev_train and dev_spatial_val only.")
    source.add_argument("--schema-report", type=Path, help="Phase 1 report; audits every source-valid bundle before a new split.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--months", type=int, nargs="+", default=[6, 7, 8, 9])
    parser.add_argument("--workers", type=int, default=8, help="Independent NetCDF site reads; use a modest value on shared storage.")
    parser.add_argument("--site-limit", type=int, default=0, help="Optional deterministic smoke-test cap; 0 audits every manifest site.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if any(month < 1 or month > 12 for month in args.months):
        parser.error("--months must be calendar months 1..12")
    sites = parse_manifest(args.manifest) if args.manifest else parse_schema_report(args.schema_report)
    if args.site_limit < 0:
        parser.error("--site-limit must be non-negative")
    if args.site_limit:
        sites = dict(sorted(sites.items())[:args.site_limit])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(row, str(args.data_root), tuple(sorted(set(args.months)))) for _, row in sorted(sites.items())]
    records: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(audit_site, job): job[0] for job in jobs}
        for completed, future in enumerate(as_completed(futures), start=1):
            source = futures[future]
            try:
                records.extend(future.result())
            except Exception as error:  # retain a transparent failure record for any source issue
                failures.append({"site_id": source["site_id"], "split": source["split"], "error": repr(error)})
            if completed % 10 == 0 or completed == len(futures):
                print({"sites_completed": completed, "sites_total": len(futures), "failures": len(failures)}, flush=True)
    if not records:
        raise RuntimeError("No artifact records were generated")
    records.sort(key=lambda row: (str(row["split"]), str(row["site_id"]), str(row["target"]), int(row["target_time_index"])))
    with (args.output_dir / "manifest_target_artifact_daily.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["split"]), str(record["site_id"]), str(record["target"]))].append(record)
    site_target_summary = []
    for (split, site_id, target), values in sorted(grouped.items()):
        summary = {name: metric_summary([row[name] for row in values]) for name in SCORE_NAMES}
        # A transparent review marker calibrated only to the known severe Tskin
        # reference; it is not an automatic exclusion policy.
        seam_days_ge_8 = sum((row["row_seam_ratio"] or 0.0) >= 8.0 for row in values)
        site_target_summary.append({
            "split": split, "site_id": site_id, "target": target, "days_scored": len(values),
            "row_seam_days_ge_8_review": seam_days_ge_8,
            "row_seam_ratio": summary["row_seam_ratio"],
            "edge_gradient_ratio": summary["edge_gradient_ratio"],
            "corner_jump_ratio": summary["corner_jump_ratio"],
            "legacy_blockiness": summary["legacy_blockiness"],
        })
    (args.output_dir / "manifest_target_artifact_site_target_summary.json").write_text(json.dumps(site_target_summary, indent=2) + "\n")
    report = {
        "purpose": "raw target artifact candidate audit; no exclusions applied",
        "source_manifest": str(args.manifest) if args.manifest else None,
        "source_schema_report": str(args.schema_report) if args.schema_report else None,
        "months": sorted(set(args.months)),
        "site_counts_by_split": dict(Counter(row["split"] for row in sites.values())),
        "daily_records": len(records),
        "failures": failures,
        "review_marker": "row_seam_ratio >= 8; use only as a Tskin-oriented review flag, not a universal exclusion cutoff",
        "site_target_summary": "manifest_target_artifact_site_target_summary.json",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
