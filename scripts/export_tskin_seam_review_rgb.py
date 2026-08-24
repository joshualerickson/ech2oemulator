#!/usr/bin/env python3
"""Export temporal RGB previews for manual Tskin seam review.

Each preview maps the target field on the day before, the selected worst-seam
day, and the day after to R, G, and B.  It is a visual review aid, not a target
artifact classifier: the accompanying CSV preserves the exact dates and raw
artifact scores used to select each image.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scripts.benchmark_target_artifacts import finite_array


TSKIN_TARGETS = {"tskin_am", "tskin_pm"}


def numeric(row: dict[str, str], name: str) -> float:
    value = row.get(name, "")
    return float(value) if value not in {"", None} else float("nan")


def rgb_preview(fields: np.ndarray) -> np.ndarray:
    """Jointly stretch a [3, H, W] temporal stack to a displayable RGB image."""
    finite = fields[np.isfinite(fields)]
    if not finite.size:
        return np.full((*fields.shape[1:], 3), 0.25, dtype=np.float32)
    low, high = np.percentile(finite, (2, 98))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        high = low + 1.0
    rgb = np.moveaxis(np.clip((fields - low) / (high - low), 0.0, 1.0), 0, -1)
    # Keep missing support visually obvious without treating it as data.
    rgb[~np.all(np.isfinite(rgb), axis=-1)] = (0.23, 0.23, 0.31)
    return rgb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-target-stats", type=Path, required=True)
    parser.add_argument("--daily-artifact-audit", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--row-seam-median", type=float, default=4.0)
    parser.add_argument("--column-seam-median", type=float, default=4.0)
    args = parser.parse_args()
    if args.row_seam_median <= 0 or args.column_seam_median <= 0:
        parser.error("seam thresholds must be positive")

    with args.site_target_stats.open(newline="") as handle:
        selected = [
            row for row in csv.DictReader(handle)
            if row["target"] in TSKIN_TARGETS
            and (numeric(row, "row_seam_ratio_median") > args.row_seam_median
                 or numeric(row, "column_seam_ratio_median") > args.column_seam_median)
        ]
    with args.daily_artifact_audit.open(newline="") as handle:
        daily = list(csv.DictReader(handle))
    daily_by_pair: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in daily:
        if row["target"] in TSKIN_TARGETS:
            daily_by_pair.setdefault((row["site_id"], row["target"]), []).append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, object]] = []
    for row in sorted(selected, key=lambda item: (item["site_id"], item["target"])):
        site_id, target = row["site_id"], row["target"]
        candidates = daily_by_pair.get((site_id, target), [])
        if not candidates:
            continue
        worst = max(candidates, key=lambda item: max(numeric(item, "row_seam_ratio"), numeric(item, "column_seam_ratio")))
        water_year = int(row["water_year_end"])
        source_index = int(worst["target_time_index"])
        source = args.data_root / site_id / f"{site_id}-{water_year}_subdaily.nc"
        with xr.open_dataset(source, engine="h5netcdf", decode_cf=False, mask_and_scale=False) as dataset:
            count = int(dataset.sizes["time"])
            center = min(max(source_index, 1), count - 2)
            indices = (center - 1, center, center + 1)
            fields = finite_array(dataset[target].isel(time=list(indices)).values)
        filename = f"{site_id}__{target}__band_{center:03d}__{worst['target_date'].replace('-', '')}_rgb.png"
        image_path = args.output_dir / filename
        figure, axis = plt.subplots(figsize=(7, 7), constrained_layout=True)
        axis.imshow(rgb_preview(fields), interpolation="nearest")
        axis.set_axis_off()
        axis.set_title(
            f"{site_id} | {target} | RGB: bands {indices[0]}, {indices[1]}, {indices[2]}\n"
            f"selected {worst['target_date']} | site medians row={numeric(row, 'row_seam_ratio_median'):.2f}, "
            f"column={numeric(row, 'column_seam_ratio_median'):.2f}",
            fontsize=9,
        )
        figure.savefig(image_path, dpi=180, bbox_inches="tight", pad_inches=0.03)
        plt.close(figure)
        index_rows.append({
            "site_id": site_id, "water_year_end": water_year, "target": target,
            "row_seam_ratio_median": row["row_seam_ratio_median"],
            "column_seam_ratio_median": row["column_seam_ratio_median"],
            "selected_target_time_index": source_index,
            "selected_target_date": worst["target_date"],
            "selected_row_seam_ratio": worst["row_seam_ratio"],
            "selected_column_seam_ratio": worst["column_seam_ratio"],
            "rgb_target_time_indices": ";".join(map(str, indices)),
            "preview_png": filename,
        })
    index_path = args.output_dir / "tskin_seam_review_index.csv"
    with index_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]) if index_rows else ["site_id"])
        writer.writeheader()
        writer.writerows(index_rows)
    print({"selected_site_targets": len(selected), "previews_written": len(index_rows), "index": str(index_path)})


if __name__ == "__main__":
    main()
