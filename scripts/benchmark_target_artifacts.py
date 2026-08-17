#!/usr/bin/env python3
"""Benchmark raw-target spatial-artifact scores on curated known sites.

This is deliberately pre-model QA: it reads target NetCDF fields only.  It
preserves NetCDF band indices (zero based) so labels can later be translated to
water-year ISO dates without coupling this check to forcing availability.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from src.data.phase2_qc import TARGET_CHANNELS
from src.data.temporal_contract import target_dates


CURATED_SITES = {
    "CA3982012144020181108": {
        "water_year_end": 2018,
        "label": "severe: tskin AM/PM all bands; soil moisture/PLC edge artifacts",
    },
    "CA4156412340420210801": {
        "water_year_end": 2021,
        "label": "moderate-to-severe: tskin AM/PM, severe near band 171 and persistent after band 250",
    },
}


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0 else float(numerator / denominator)


def finite_array(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).copy()
    result[result == np.float32(-1.175494e38)] = np.nan
    return result


def legacy_blockiness_score(arr: np.ndarray, block_sizes: tuple[int, ...] = (2, 4, 8, 16)) -> float | None:
    """Prior-project boundary heuristic, retained as a comparable benchmark."""
    finite = np.isfinite(arr)
    best: float | None = None
    for block_size in block_sizes:
        dx, dy = np.abs(arr[:, 1:] - arr[:, :-1]), np.abs(arr[1:, :] - arr[:-1, :])
        dx_valid, dy_valid = finite[:, 1:] & finite[:, :-1], finite[1:, :] & finite[:-1, :]
        values: list[float] = []
        for differences, valid, boundary in (
            (dx, dx_valid, (np.arange(arr.shape[1] - 1) + 1) % block_size == 0),
            (dy, dy_valid, (np.arange(arr.shape[0] - 1) + 1) % block_size == 0),
        ):
            boundary_mean = np.mean(differences[valid & boundary[None, :] if differences is dx else valid & boundary[:, None]])
            interior = ~boundary
            interior_mean = np.mean(differences[valid & interior[None, :] if differences is dx else valid & interior[:, None]])
            ratio = safe_ratio(float(boundary_mean), float(interior_mean))
            if ratio is not None:
                values.append(ratio)
        if values:
            best = max(best or 0.0, max(values))
    return best


def spatial_scores(arr: np.ndarray) -> dict[str, float | None]:
    """Scores for tiled boundaries, arbitrary long seams, and exterior artifacts."""
    if arr.ndim != 2 or min(arr.shape) < 4 or np.count_nonzero(np.isfinite(arr)) < 16:
        return {name: None for name in ("legacy_blockiness", "row_seam_ratio", "column_seam_ratio", "edge_gradient_ratio", "corner_jump_ratio", "high_frequency_ratio")}
    finite = np.isfinite(arr)
    horizontal = np.abs(arr[1:, :] - arr[:-1, :])
    vertical = np.abs(arr[:, 1:] - arr[:, :-1])
    horizontal_valid = finite[1:, :] & finite[:-1, :]
    vertical_valid = finite[:, 1:] & finite[:, :-1]

    def line_medians(difference: np.ndarray, valid: np.ndarray, axis: int) -> np.ndarray:
        return np.asarray([np.nanmedian(np.where(valid.take(i, axis=axis), difference.take(i, axis=axis), np.nan)) for i in range(difference.shape[axis])])

    row_lines = line_medians(horizontal, horizontal_valid, axis=0)
    column_lines = line_medians(vertical, vertical_valid, axis=1)
    overall = np.nanmedian(np.concatenate([horizontal[horizontal_valid], vertical[vertical_valid]]))
    row_ratio = safe_ratio(float(np.nanmax(row_lines)), float(overall))
    column_ratio = safe_ratio(float(np.nanmax(column_lines)), float(overall))
    edge_values = np.asarray([row_lines[0], row_lines[-1], column_lines[0], column_lines[-1]])
    edge_ratio = safe_ratio(float(np.nanmax(edge_values)), float(overall))
    corners = ((0, 0), (0, -1), (-1, 0), (-1, -1))
    corner_jumps = []
    for row, column in corners:
        inner_row = 1 if row == 0 else -2
        inner_column = 1 if column == 0 else -2
        if np.isfinite(arr[row, column]) and np.isfinite(arr[inner_row, inner_column]):
            corner_jumps.append(abs(float(arr[row, column] - arr[inner_row, inner_column])))
    corner_ratio = safe_ratio(max(corner_jumps) if corner_jumps else np.nan, float(overall))

    filled = np.where(finite, arr, np.nanmedian(arr[finite]))
    laplacian = np.abs(4 * filled[1:-1, 1:-1] - filled[:-2, 1:-1] - filled[2:, 1:-1] - filled[1:-1, :-2] - filled[1:-1, 2:])
    high_frequency_ratio = safe_ratio(float(np.nanmedian(laplacian)), float(overall))
    return {
        "legacy_blockiness": legacy_blockiness_score(arr),
        "row_seam_ratio": row_ratio,
        "column_seam_ratio": column_ratio,
        "edge_gradient_ratio": edge_ratio,
        "corner_jump_ratio": corner_ratio,
        "high_frequency_ratio": high_frequency_ratio,
    }


def robust_z(values: np.ndarray) -> np.ndarray:
    if not np.any(np.isfinite(values)):
        return np.full_like(values, np.nan)
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    return np.zeros_like(values) if not np.isfinite(mad) or mad == 0 else 0.67448975 * (values - median) / mad


def metric_summary(values: list[float | None]) -> dict[str, float | None]:
    finite = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    if not finite.size:
        return {"median": None, "p95": None, "max": None}
    return {"median": float(np.median(finite)), "p95": float(np.percentile(finite, 95)), "max": float(np.max(finite))}


def plot_scores(rows: list[dict[str, object]], site_id: str, output: Path) -> None:
    targets = ("tskin_am", "tskin_pm", "soilmoisture", "plc_am", "plc_pm")
    fig, axes = plt.subplots(len(targets), 1, figsize=(11, 12), sharex=True, constrained_layout=True)
    for axis, target in zip(axes, targets):
        subset = [row for row in rows if row["target"] == target]
        x = [int(row["target_time_index"]) for row in subset]
        for score, color in (("legacy_blockiness", "#7b61ff"), ("row_seam_ratio", "#e6550d"), ("edge_gradient_ratio", "#238b45")):
            axis.plot(x, [row[score] for row in subset], label=score.replace("_", " "), linewidth=1.1, color=color)
        axis.set_ylabel(target)
        axis.grid(alpha=0.25)
        if target.startswith("tskin"):
            axis.axvline(171, color="black", linestyle="--", alpha=0.5, linewidth=0.8)
            axis.axvline(250, color="black", linestyle=":", alpha=0.5, linewidth=0.8)
    axes[0].legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("Raw target NetCDF band index (zero based)")
    fig.suptitle(f"Raw-target artifact benchmark — {site_id}")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_peak_maps(fields: list[tuple[str, int, str, np.ndarray]], site_id: str, output: Path) -> None:
    """Visual companion: each panel is the day with maximum row-seam score."""
    fig, axes = plt.subplots(1, len(fields), figsize=(4 * len(fields), 4), constrained_layout=True)
    for axis, (target, index, day, field) in zip(axes, fields):
        finite = field[np.isfinite(field)]
        lo, hi = np.nanpercentile(finite, (2, 98)) if finite.size else (0.0, 1.0)
        image = axis.imshow(field, cmap="turbo", vmin=lo, vmax=hi, interpolation="nearest")
        axis.set_title(f"{target}\nband {index} — {day}")
        axis.set_axis_off()
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
    fig.suptitle(f"Peak row-seam raw targets — {site_id}")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--additional-site",
        action="append",
        default=[],
        metavar="SITE_ID:WATER_YEAR_END",
        help="Add a comparison site, for example CA4198012316420170811:2017.",
    )
    parser.add_argument("--only-additional", action="store_true", help="Run only the sites supplied by --additional-site.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {"purpose": "pre-rerun curated raw-target artifact benchmark", "sites": {}}
    benchmark_sites = {} if args.only_additional else dict(CURATED_SITES)
    for item in args.additional_site:
        try:
            site_id, year_text = item.rsplit(":", 1)
            benchmark_sites[site_id] = {"water_year_end": int(year_text), "label": "comparison control"}
        except ValueError as error:
            parser.error(f"--additional-site must be SITE_ID:WATER_YEAR_END, got {item!r}: {error}")
    for site_id, metadata in benchmark_sites.items():
        water_year = int(metadata["water_year_end"])
        path = args.data_root / site_id / f"{site_id}-{water_year}_subdaily.nc"
        if not path.is_file():
            raise FileNotFoundError(path)
        site_rows: list[dict[str, object]] = []
        peak_fields: list[tuple[str, int, str, np.ndarray]] = []
        with xr.open_dataset(path, engine="h5netcdf", decode_cf=False, mask_and_scale=False) as dataset:
            count = int(dataset.sizes["time"])
            dates = target_dates(water_year, count)
            for target in TARGET_CHANNELS:
                stack = finite_array(dataset[target].values)
                target_rows = []
                for index, day in enumerate(dates):
                    scores = spatial_scores(stack[index])
                    target_rows.append({"site_id": site_id, "water_year_end": water_year, "target": target, "target_time_index": index, "target_date": day.isoformat(), **scores})
                for name in ("legacy_blockiness", "row_seam_ratio", "column_seam_ratio", "edge_gradient_ratio", "corner_jump_ratio", "high_frequency_ratio"):
                    values = np.asarray([np.nan if row[name] is None else row[name] for row in target_rows], dtype=float)
                    for row, value in zip(target_rows, robust_z(values)):
                        row[f"{name}_temporal_robust_z"] = None if not np.isfinite(value) else float(value)
                rows.extend(target_rows)
                site_rows.extend(target_rows)
                peak = max(target_rows, key=lambda row: -np.inf if row["row_seam_ratio"] is None else float(row["row_seam_ratio"]))
                peak_fields.append((target, int(peak["target_time_index"]), str(peak["target_date"]), stack[int(peak["target_time_index"])].copy()))
        plot_scores(site_rows, site_id, args.output_dir / f"{site_id}_artifact_scores.png")
        plot_peak_maps(peak_fields, site_id, args.output_dir / f"{site_id}_peak_seam_maps.png")
        summary["sites"][site_id] = {
            **metadata,
            "per_target": {
                target: {
                    name: metric_summary([row[name] for row in site_rows if row["target"] == target])
                    for name in ("legacy_blockiness", "row_seam_ratio", "column_seam_ratio", "edge_gradient_ratio", "corner_jump_ratio", "high_frequency_ratio")
                }
                for target in TARGET_CHANNELS
            },
        }
    with (args.output_dir / "artifact_benchmark_daily.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "artifact_benchmark_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output_dir": str(args.output_dir), "rows": len(rows), "sites": list(summary["sites"])}, indent=2))


if __name__ == "__main__":
    main()
