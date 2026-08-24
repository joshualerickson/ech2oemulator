#!/usr/bin/env python3
"""Export one temporal-RGB contact sheet per site in the unresolved QA queue.

Only target channels with a non-keep artifact action are displayed.  Each panel
uses the day before, selected worst-seam day, and day after as RGB, preserving
the selected date and raw scores in both the title and index CSV.  This is a
manual-review aid; it does not alter QA decisions.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scripts.benchmark_target_artifacts import finite_array
from scripts.export_tskin_seam_review_rgb import rgb_preview


KEEP_ACTIONS = {"keep_candidate", "keep_candidate_geometry_low_confidence", "keep_manual"}
REVIEW_SITE_STATUSES = {"include_candidate_with_target_review", "hold_for_review"}


def numeric(row: dict[str, str], name: str) -> float:
    value = row.get(name, "")
    return float(value) if value not in {"", None} else float("nan")


def seam_score(row: dict[str, str]) -> float:
    """Select the strongest interior-seam orientation for visual review."""
    values = (numeric(row, "row_seam_ratio"), numeric(row, "column_seam_ratio"))
    finite = [value for value in values if np.isfinite(value)]
    return max(finite) if finite else float("-inf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-decisions", type=Path, required=True)
    parser.add_argument("--triage", type=Path, required=True)
    parser.add_argument("--daily-artifact-audit", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with args.site_decisions.open(newline="") as handle:
        site_rows = {row["site_id"]: row for row in csv.DictReader(handle)}
    site_status = {site_id: row["training_site_status"] for site_id, row in site_rows.items()}
    with args.triage.open(newline="") as handle:
        flagged = [
            row for row in csv.DictReader(handle)
            if site_status.get(row["site_id"]) in REVIEW_SITE_STATUSES and row["action"] not in KEEP_ACTIONS
        ]
    by_pair: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    with args.daily_artifact_audit.open(newline="") as handle:
        for row in csv.DictReader(handle):
            by_pair[(row["site_id"], row["target"])].append(row)
    by_site: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in flagged:
        by_site[row["site_id"]].append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, object]] = []
    for site_id, targets in sorted(by_site.items()):
        water_year = int(site_rows[site_id]["water_year_end"])
        source = args.data_root / site_id / f"{site_id}-{water_year}_subdaily.nc"
        panels = []
        with xr.open_dataset(source, engine="h5netcdf", decode_cf=False, mask_and_scale=False) as dataset:
            count = int(dataset.sizes["time"])
            for target_row in sorted(targets, key=lambda item: item["target"]):
                target = target_row["target"]
                daily = by_pair[(site_id, target)]
                worst = max(daily, key=seam_score)
                source_index = int(worst["target_time_index"])
                center = min(max(source_index, 1), count - 2)
                indices = (center - 1, center, center + 1)
                fields = finite_array(dataset[target].isel(time=list(indices)).values)
                panels.append((target_row, worst, indices, rgb_preview(fields)))
        columns = min(3, len(panels))
        rows = math.ceil(len(panels) / columns)
        figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 5 * rows), squeeze=False, constrained_layout=True)
        for axis in axes.flat:
            axis.set_axis_off()
        for axis, (target_row, worst, indices, image) in zip(axes.flat, panels):
            axis.imshow(image, interpolation="nearest")
            axis.set_axis_off()
            axis.set_title(
                f"{target_row['target']} | {target_row['action']}\n"
                f"RGB bands {indices[0]}/{indices[1]}/{indices[2]} | {worst['target_date']}\n"
                f"row={numeric(worst, 'row_seam_ratio'):.2f}, col={numeric(worst, 'column_seam_ratio'):.2f}",
                fontsize=8,
            )
        figure.suptitle(f"{site_id} | unresolved target-artifact review", fontsize=12)
        filename = f"{site_id}__review_rgb.png"
        figure.savefig(args.output_dir / filename, dpi=180, bbox_inches="tight", pad_inches=0.04)
        plt.close(figure)
        index_rows.append({
            "site_id": site_id,
            "water_year_end": water_year,
            "site_training_status": site_status[site_id],
            "flagged_targets": ";".join(panel[0]["target"] for panel in panels),
            "target_actions": ";".join(f"{panel[0]['target']}={panel[0]['action']}" for panel in panels),
            "selected_dates": ";".join(f"{panel[0]['target']}={panel[1]['target_date']}" for panel in panels),
            "contact_sheet_png": filename,
        })
    index_path = args.output_dir / "review_queue_rgb_index.csv"
    with index_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]) if index_rows else ["site_id"])
        writer.writeheader()
        writer.writerows(index_rows)
    print({"review_sites": len(by_site), "flagged_site_targets": len(flagged), "contact_sheets": len(index_rows), "index": str(index_path)})


if __name__ == "__main__":
    main()
