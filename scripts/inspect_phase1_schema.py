#!/usr/bin/env python3
"""Inspect the ECH2O bbox source schema without creating training samples.

The ECH2O ``*_YYYY_subdaily.nc`` filename year is the *water-year end year*.
When NetCDF time metadata cannot supply usable dates, day/band zero is therefore
``YYYY-1-10-01``.  This tool intentionally only reports source facts and data
contract violations; it does not resample, clip, or otherwise alter rasters.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import rasterio


DYNAMIC_CHANNELS = ("prcp", "srad", "tmin", "tmax", "rmin", "rmax")
TARGET_CHANNELS = ("soilmoisture", "tskin_am", "tskin_pm", "plc_am", "plc_pm")
NC_YEAR_RE = re.compile(r"^(?P<site>.+)-(?P<year>\d{4})_subdaily\.nc$")


def serialise_transform(transform: rasterio.Affine) -> list[float]:
    return [float(value) for value in transform[:6]]


def grid(dataset: rasterio.DatasetReader) -> dict[str, Any]:
    return {
        "width": dataset.width,
        "height": dataset.height,
        "crs": dataset.crs.to_string() if dataset.crs else None,
        "transform": serialise_transform(dataset.transform),
        "resolution": [float(abs(dataset.transform.a)), float(abs(dataset.transform.e))],
        "nodata": dataset.nodata,
    }


def relationship(source: dict[str, Any], target: dict[str, Any]) -> str:
    if source == target:
        return "exact"
    if source["crs"] != target["crs"] or source["resolution"] != target["resolution"]:
        return "different_crs_or_resolution"
    # Affine is serialised as (a, b, c, d, e, f): c/f are the origin.
    _, _, sx, _, _, sy = source["transform"]
    _, _, tx, _, _, ty = target["transform"]
    xres, yres = source["resolution"]
    if (
        source["width"] == target["width"] + 2
        and source["height"] == target["height"] + 2
        and abs(tx - (sx + xres)) < 1e-6
        and abs(ty - (sy - yres)) < 1e-6
    ):
        return "target_is_one_cell_inset"
    return "different_extent_or_shape"


def inspect_site(site_dir: Path, full_nodata_scan: bool) -> dict[str, Any]:
    nc_files = sorted(site_dir.glob("*_subdaily.nc"))
    result: dict[str, Any] = {"site_id": site_dir.name, "issues": []}
    if len(nc_files) != 1:
        result["issues"].append(f"expected_one_subdaily_netcdf_found_{len(nc_files)}")
        return result

    nc_path = nc_files[0]
    match = NC_YEAR_RE.match(nc_path.name)
    if not match or match.group("site") != site_dir.name:
        result["issues"].append("netcdf_filename_does_not_match_site_water_year_convention")
        return result
    water_year = int(match.group("year"))
    result["water_year_end"] = water_year

    forcing_grids: dict[str, dict[str, Any]] = {}
    forcing_counts: dict[str, int] = {}
    for name in DYNAMIC_CHANNELS:
        paths = sorted(site_dir.glob(f"{name}_*{water_year}*.tif"))
        if len(paths) != 1:
            result["issues"].append(f"forcing_{name}_matches_{len(paths)}")
            continue
        with rasterio.open(paths[0]) as dataset:
            forcing_grids[name] = grid(dataset)
            forcing_counts[name] = dataset.count
    result["forcing_band_counts"] = forcing_counts
    if len(set(forcing_counts.values())) > 1:
        result["issues"].append("forcing_band_counts_disagree")

    target_grids: dict[str, dict[str, Any]] = {}
    target_counts: dict[str, int] = {}
    valid_pixels: dict[str, int] = {}
    total_pixels: dict[str, int] = {}
    for name in TARGET_CHANNELS:
        try:
            with rasterio.open(f"NETCDF:{nc_path}:{name}") as dataset:
                target_grids[name] = grid(dataset)
                target_counts[name] = dataset.count
                if full_nodata_scan:
                    masks = dataset.read_masks()
                    valid_pixels[name] = int((masks > 0).sum())
                    total_pixels[name] = int(masks.size)
        except rasterio.errors.RasterioIOError:
            result["issues"].append(f"missing_target_{name}")
    result["target_band_counts"] = target_counts
    if len(set(target_counts.values())) > 1:
        result["issues"].append("target_band_counts_disagree")
    if valid_pixels:
        result["target_valid_pixel_counts"] = valid_pixels
        result["target_total_pixel_counts"] = total_pixels

    if "soilmoisture" not in target_grids:
        return result
    reference_target = target_grids["soilmoisture"]
    result["target_grid"] = reference_target
    if any(value != reference_target for value in target_grids.values()):
        result["issues"].append("target_grids_disagree")
    result["forcing_to_target_grid_relationship"] = {
        name: relationship(value, reference_target) for name, value in forcing_grids.items()
    }
    if set(result["forcing_to_target_grid_relationship"].values()) != {"target_is_one_cell_inset"}:
        result["issues"].append("forcing_target_grid_relationship_not_uniform_one_cell_inset")

    static_dir = site_dir / "Spatial"
    static_paths = sorted(static_dir.glob("*.asc"))
    result["static_channels"] = [path.stem for path in static_paths]
    static_relationships: Counter[str] = Counter()
    for path in static_paths:
        with rasterio.open(path) as dataset:
            static_relationships[relationship(grid(dataset), reference_target)] += 1
    result["static_to_target_grid_relationship_counts"] = dict(static_relationships)
    if not static_paths:
        result["issues"].append("no_static_ascii_rasters")

    target_steps = target_counts.get("soilmoisture")
    forcing_steps = forcing_counts.get("prcp")
    if target_steps and forcing_steps and target_steps != forcing_steps:
        result["issues"].append("target_and_forcing_band_counts_disagree")
    if target_steps:
        start = date(water_year - 1, 10, 1)
        end = start + timedelta(days=target_steps - 1)
        result["fallback_date_contract"] = {
            "source": "water_year_oct1_from_filename_due_to_unusable_or_untrusted_time_coordinate",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "step_count": target_steps,
        }
    return result


def build_report(
    data_root: Path,
    full_nodata_scan: bool,
    site_limit: int | None = None,
    site_offset: int = 0,
) -> dict[str, Any]:
    site_paths = [path for path in sorted(data_root.iterdir()) if path.is_dir()]
    site_paths = site_paths[site_offset:]
    if site_limit is not None:
        site_paths = site_paths[:site_limit]
    sites = []
    for index, path in enumerate(site_paths, start=1):
        sites.append(inspect_site(path, full_nodata_scan))
        gc.collect()
        if index % 25 == 0 or index == len(site_paths):
            print(f"inspected {index}/{len(site_paths)} sites", file=sys.stderr, flush=True)
    issue_counts = Counter(issue for site in sites for issue in site["issues"])
    target_grids = Counter(json.dumps(site["target_grid"], sort_keys=True) for site in sites if "target_grid" in site)
    static_names = Counter(name for site in sites for name in site.get("static_channels", []))
    forcing_relationships = Counter(
        relation
        for site in sites
        for relation in site.get("forcing_to_target_grid_relationship", {}).values()
    )
    target_valid = defaultdict(int)
    target_total = defaultdict(int)
    for site in sites:
        for name, value in site.get("target_valid_pixel_counts", {}).items():
            target_valid[name] += value
        for name, value in site.get("target_total_pixel_counts", {}).items():
            target_total[name] += value
    return {
        "data_root": str(data_root),
        "inspection_scope": "metadata plus full target mask scan" if full_nodata_scan else "metadata only",
        "calendar_contract": {
            "water_year_filename_semantics": "YYYY is water-year end year",
            "fallback_band_zero_date": "YYYY-1-10-01",
            "date_join_key": "ISO calendar date derived from water-year convention; never ordinal band number alone",
        },
        "dynamic_channel_order": list(DYNAMIC_CHANNELS),
        "target_channel_order": list(TARGET_CHANNELS),
        "site_count": len(sites),
        "eligible_bundle_count": sum(not site["issues"] for site in sites),
        "issue_counts": dict(issue_counts),
        "target_grid_variants": [{"grid": json.loads(key), "site_count": value} for key, value in target_grids.items()],
        "forcing_to_target_relationship_counts": dict(forcing_relationships),
        "static_channel_presence_counts": dict(sorted(static_names.items())),
        "target_valid_pixel_counts": dict(target_valid),
        "target_total_pixel_counts": dict(target_total),
        "sites": sites,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-target-nodata-scan", action="store_true")
    parser.add_argument("--site-limit", type=int)
    parser.add_argument("--site-offset", type=int, default=0)
    args = parser.parse_args()
    report = build_report(
        args.data_root,
        args.full_target_nodata_scan,
        args.site_limit,
        args.site_offset,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("site_count", "eligible_bundle_count", "issue_counts")}, indent=2))


if __name__ == "__main__":
    main()
