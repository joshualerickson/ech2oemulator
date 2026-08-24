#!/usr/bin/env python3
"""Build 75/25 site-level train/validation splits with terrain/climate coverage."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import convolve
from rasterio.warp import Resampling, reproject

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.static_contract import EXTERNAL_STATIC_SOURCES, Grid, read_ascii_on_target_grid, resolve_ascii_static
from src.data.paths import configured_data_root
from src.data.temporal_contract import TEMPORAL_CONTRACT


MANIFEST_FIELDS = (
    "split", "site_id", "bbox_id", "water_year_end", "temporal_contract", "start_date", "end_date", "target_date",
    "sequence_length", "target_time_index", "forcing_start_index", "forcing_end_index", "dynamic_sequence_source", "static_source", "target_source",
)


def daily_screen(daily_dir: Path, site_id: str) -> Path:
    matches = list(daily_dir.glob(f"{site_id}_*_daily_screen.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one daily-screen CSV for {site_id}; got {len(matches)}")
    return matches[0]


def qa_eligible_sites(path: Path, require_forcing_clean: bool) -> tuple[set[str], list[dict[str, str]]]:
    """Read the resolved pre-split QA decision table without inferring eligibility."""
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    eligible = {
        row["site_id"] for row in rows
        if row["training_site_status"] == "include_candidate"
        and (not require_forcing_clean or row.get("forcing_daily_status") == "forcing_clean")
    }
    if not eligible:
        raise ValueError("QA decision table yielded no eligible sites")
    return eligible, rows


def sequence_rows(site_id: str, daily_dir: Path, data_root: Path, months: set[int], length: int) -> list[dict[str, object]]:
    with daily_screen(daily_dir, site_id).open(newline="") as handle:
        daily = list(csv.DictReader(handle))
    by_date = {date.fromisoformat(row["date"]): row for row in daily}
    water_year = int(daily[0]["water_year_end"])
    root = data_root / site_id
    rows = []
    for target_date, target in sorted(by_date.items()):
        if target_date.month not in months or target["target_qa_valid"] != "True":
            continue
        history = [by_date.get(target_date - timedelta(days=offset)) for offset in range(length - 1, -1, -1)]
        if any(day is None or day["forcing_valid"] != "True" for day in history):
            continue
        rows.append({
            "split": "", "site_id": site_id, "bbox_id": site_id, "water_year_end": water_year, "temporal_contract": TEMPORAL_CONTRACT,
            "start_date": history[0]["date"], "end_date": target["date"], "target_date": target["date"],
            "sequence_length": length, "target_time_index": target["target_time_index"], "forcing_start_index": history[0]["forcing_time_index"], "forcing_end_index": target["forcing_time_index"], "dynamic_sequence_source": str(root), "static_source": str(root / "Spatial"),
            "target_source": str(root / f"{site_id}-{water_year}_subdaily.nc"),
        })
    return rows


def finite_median(values: np.ndarray, name: str, site_id: str) -> float:
    values = values[np.isfinite(values)]
    if not values.size:
        raise ValueError(f"No finite {name} values for {site_id}")
    return float(np.median(values))


def tpi_radius_five(dem: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of the static-contract 11x11 finite-neighbor TPI."""
    finite = np.isfinite(dem)
    kernel = np.ones((11, 11), dtype=np.float32)
    counts = convolve(finite.astype(np.float32), kernel, mode="constant", cval=0.0)
    sums = convolve(np.where(finite, dem, 0.0), kernel, mode="constant", cval=0.0)
    mean = np.divide(sums, counts, out=np.full_like(dem, np.nan), where=counts > 0)
    return np.where(finite, dem - mean, np.nan).astype(np.float32)


def reproject_source(source: rasterio.DatasetReader, grid: Grid) -> np.ndarray:
    destination = np.full((grid.height, grid.width), np.nan, dtype=np.float32)
    reproject(source=rasterio.band(source, 1), destination=destination, src_transform=source.transform, src_crs=source.crs, src_nodata=source.nodata, dst_transform=grid.transform, dst_crs=grid.crs, dst_nodata=np.nan, resampling=Resampling.bilinear)
    return destination


def site_covariates(site_id: str, rows: list[dict[str, object]], twi_source: rasterio.DatasetReader, wbdef_source: rasterio.DatasetReader) -> dict[str, float | int | str]:
    row = rows[0]
    with rasterio.open(f"NETCDF:{row['target_source']}:soilmoisture") as reference:
        grid = Grid(reference.width, reference.height, reference.transform, reference.crs)
    spatial = Path(row["static_source"])
    dem = read_ascii_on_target_grid(resolve_ascii_static(spatial, "dem"), grid)
    tpi = tpi_radius_five(dem)
    twi = reproject_source(twi_source, grid)
    wbdef = reproject_source(wbdef_source, grid)
    return {
        "site_id": site_id, "state": site_id[:2], "height": grid.height, "width": grid.width,
        "cell_count": grid.height * grid.width, "wbdef_median": finite_median(wbdef, "wbdef", site_id),
        "twi_median": finite_median(twi, "twi", site_id), "tpi_median": finite_median(tpi, "tpi", site_id),
        "tpi_iqr": float(np.nanpercentile(tpi, 75) - np.nanpercentile(tpi, 25)),
        "sequence_count": len(rows),
    }


def quantile_bins(values: np.ndarray, bins: int) -> np.ndarray:
    """Rank bins are stable even if a covariate has repeated raw values."""
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=int)
    result[order] = np.minimum(bins - 1, np.arange(len(values)) * bins // len(values))
    return result


def quotas_by_state(records: list[dict], fraction: float) -> dict[str, int]:
    groups = Counter(record["state"] for record in records)
    total = round(len(records) * fraction)
    exact = {state: count * fraction for state, count in groups.items()}
    quotas = {state: max(1, int(value)) for state, value in exact.items()}
    # Largest remainder adjustment reaches the global requested count exactly.
    while sum(quotas.values()) < total:
        choices = [state for state,count in groups.items() if quotas[state] < count - 1]
        state = max(choices, key=lambda value: (exact[value] - quotas[value], groups[value], value))
        quotas[state] += 1
    while sum(quotas.values()) > total:
        choices = [state for state in groups if quotas[state] > 1]
        state = min(choices, key=lambda value: (exact[value] - quotas[value], -groups[value], value))
        quotas[state] -= 1
    return quotas


def select_validation(records: list[dict], fraction: float, bins: int, seed: int) -> set[str]:
    rng = random.Random(seed)
    features = ("wbdef_median", "cell_count", "tpi_median", "twi_median")
    for feature in features:
        values = np.asarray([float(record[feature]) for record in records])
        for record, bin_id in zip(records, quantile_bins(values, bins)):
            record[f"{feature}_bin"] = int(bin_id)
    quotas = quotas_by_state(records, fraction)
    selected: set[str] = set()
    feature_counts: Counter[tuple[str, int]] = Counter()
    by_state: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_state[record["state"]].append(record)
    # Greedily spread validation sites over each marginal feature distribution.
    for state in sorted(by_state):
        pool = by_state[state].copy()
        for _ in range(quotas[state]):
            scores = []
            for record in pool:
                diversity = sum(1.0 / (1 + feature_counts[(feature, record[f"{feature}_bin"])]) for feature in features)
                scores.append((diversity + rng.random() * 1e-6, record))
            _, chosen = max(scores, key=lambda item: item[0])
            selected.add(chosen["site_id"])
            for feature in features:
                feature_counts[(feature, chosen[f"{feature}_bin"])] += 1
            pool.remove(chosen)
    return selected


def partition_summary(records: list[dict], split: str) -> dict:
    subset = [record for record in records if record["split"] == split]
    numeric = ("wbdef_median", "cell_count", "tpi_median", "twi_median", "tpi_iqr", "sequence_count")
    return {
        "site_count": len(subset), "sequence_count": sum(int(record["sequence_count"]) for record in subset),
        "sites_by_state": dict(sorted(Counter(record["state"] for record in subset).items())),
        "median": {name: float(np.median([float(record[name]) for record in subset])) for name in numeric},
        "quantile_bin_counts": {name: dict(sorted(Counter(record[f"{name}_bin"] for record in subset).items())) for name in ("wbdef_median", "cell_count", "tpi_median", "twi_median")},
    }


def write_covariate_cache(path: Path, records: list[dict]) -> None:
    if not records:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("site_id", "state", "height", "width", "cell_count", "wbdef_median", "twi_median", "tpi_median", "tpi_iqr", "sequence_count"))
        writer.writeheader(); writer.writerows(sorted(records, key=lambda record: record["site_id"]))


def read_covariate_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    numeric = {"height", "width", "cell_count", "wbdef_median", "twi_median", "tpi_median", "tpi_iqr", "sequence_count"}
    with path.open(newline="") as handle:
        return {row["site_id"]: {key: (float(value) if key in numeric and "." in value else int(value) if key in numeric else value) for key, value in row.items()} for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-dir", type=Path, required=True)
    parser.add_argument("--training-qa-site-decisions", type=Path, required=True,
                        help="Resolved QA site decisions; only include_candidate rows can enter the split.")
    parser.add_argument("--data-root", type=Path, default=configured_data_root(), help="Raw input root; defaults to ECH2O_DATA_ROOT.")
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--months", type=int, nargs="+", default=[6, 7, 8, 9])
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--strata-bins", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--allow-forcing-day-masks", action="store_true",
                        help="Permit include-candidate sites with forcing-day/bbox breaks. Off by default for contiguous recurrent baselines.")
    args = parser.parse_args()
    if args.data_root is None:
        parser.error("set ECH2O_DATA_ROOT or pass --data-root")
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between zero and one")

    with args.external_manifest.open(newline="") as handle:
        external_sites = {row["site_id"] for row in csv.DictReader(handle)}
    qa_sites, qa_rows = qa_eligible_sites(args.training_qa_site_decisions, not args.allow_forcing_day_masks)
    all_sites = sorted(path.name.split("_")[0] for path in args.daily_dir.glob("*_daily_screen.csv"))
    selected_sites = set(all_sites) & qa_sites - external_sites
    site_rows = {site: sequence_rows(site, args.daily_dir, args.data_root, set(args.months), args.sequence_length) for site in sorted(selected_sites)}
    site_rows = {site: rows for site, rows in site_rows.items() if rows}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "site_covariates_cache.csv"
    cached = read_covariate_cache(cache_path)
    records = []
    with rasterio.open(EXTERNAL_STATIC_SOURCES["twi"]) as twi_source, rasterio.open(EXTERNAL_STATIC_SOURCES["wbdef"]) as wbdef_source:
        for index, (site, rows) in enumerate(sorted(site_rows.items()), start=1):
            records.append(cached.get(site) or site_covariates(site, rows, twi_source, wbdef_source))
            if index % 10 == 0 or index == len(site_rows):
                write_covariate_cache(cache_path, records)
                print({"site_covariates": f"{index}/{len(site_rows)}"}, flush=True)
    validation_sites = select_validation(records, args.validation_fraction, args.strata_bins, args.seed)
    for record in records:
        record["split"] = "dev_spatial_val" if record["site_id"] in validation_sites else "dev_train"

    manifest_rows = []
    for record in records:
        for row in site_rows[record["site_id"]]:
            row["split"] = record["split"]
            manifest_rows.append(row)
    with (args.output_dir / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader(); writer.writerows(manifest_rows)
    audit_fields = tuple(records[0])
    with (args.output_dir / "site_split_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_fields)
        writer.writeheader(); writer.writerows(sorted(records, key=lambda record: record["site_id"]))
    summary = {
        "purpose": "site-level 75/25 state-and-covariate-balanced spatial validation split",
        "months": args.months, "sequence_length": args.sequence_length, "seed": args.seed,
        "training_qa_site_decisions": str(args.training_qa_site_decisions),
        "require_forcing_clean": not args.allow_forcing_day_masks,
        "qa_site_status_counts": dict(sorted(Counter(row["training_site_status"] for row in qa_rows).items())),
        "qa_eligible_sites_before_external_holdout": len(qa_sites),
        "qa_eligible_sites_removed_as_external": sorted(qa_sites & external_sites),
        "qa_eligible_sites_with_daily_screens": len(selected_sites),
        "external_test_sites_excluded": sorted(external_sites),
        "eligible_non_external_sites": len(records), "requested_validation_fraction": args.validation_fraction,
        "train": partition_summary(records, "dev_train"), "spatial_validation": partition_summary(records, "dev_spatial_val"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
