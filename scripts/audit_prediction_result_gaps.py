#!/usr/bin/env python3
"""Classify missing and all-NA prediction results using source support masks."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import timedelta
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import rasterio
from rasterio.transform import Affine

from scripts.predict_fire_seq90 import dated_bands, read_viirs_windows
from src.data.phase2_qc import DYNAMIC_CHANNELS
from src.data.static_contract import Grid, STATIC_CHANNELS, load_selected_static_stack


SUFFIX_240 = "_seq90_lstm_predictions.nc"
SUFFIX_30 = "_direct_30m.nc"


def source_support(site: Path, viirs_start, viirs_end, buffer_days: int) -> dict[str, object]:
    datasets: dict[str, rasterio.DatasetReader] = {}
    try:
        dates_by_channel = {}
        for channel in DYNAMIC_CHANNELS:
            matches = sorted(site.glob(f"{channel}_*.tif"))
            if len(matches) != 1:
                raise ValueError(f"expected one {channel}_*.tif; found {len(matches)}")
            dates_by_channel[channel], datasets[channel] = dated_bands(matches[0])
        dates = dates_by_channel[DYNAMIC_CHANNELS[0]]
        if any(dates_by_channel[channel] != dates for channel in DYNAMIC_CHANNELS[1:]):
            raise ValueError("forcing date sequences differ across channels")
        reference = datasets[DYNAMIC_CHANNELS[0]]
        grid = Grid(reference.width - 2, reference.height - 2, reference.transform * Affine.translation(1, 1), reference.crs)
        static = load_selected_static_stack(site / "Spatial", grid)
        static_support = np.isfinite(static).all(axis=0)
        static_counts = {name: int(np.isfinite(static[index]).sum()) for index, name in enumerate(STATIC_CHANNELS)}
        index = {current: position for position, current in enumerate(dates)}
        targets = [viirs_start - timedelta(days=buffer_days) + timedelta(days=offset)
                   for offset in range((viirs_end - viirs_start).days + 1 + buffer_days * 2)]
        eligible = [current for current in targets if current in index and current - timedelta(days=89) in index]
        if not eligible:
            return {
                "static_valid_pixels": int(static_support.sum()), "static_channel_valid_pixels": static_counts,
                "eligible_dates": 0, "max_dynamic_valid_pixels": 0, "max_combined_valid_pixels": 0,
                "support_class": "insufficient_90_day_forcing_window",
            }
        max_dynamic = max_combined = 0
        for target in eligible:
            bands = list(range(index[target] - 89 + 1, index[target] + 2))
            dynamic = np.stack([
                datasets[channel].read(bands, out_dtype="float32")[:, 1:-1, 1:-1]
                for channel in DYNAMIC_CHANNELS
            ])
            dynamic_support = np.isfinite(dynamic).all(axis=(0, 1))
            max_dynamic = max(max_dynamic, int(dynamic_support.sum()))
            max_combined = max(max_combined, int((dynamic_support & static_support).sum()))
        if not static_support.any():
            support_class = "no_complete_static_support"
        elif not max_dynamic:
            support_class = "no_complete_90_day_dynamic_support"
        elif not max_combined:
            support_class = "static_dynamic_support_disjoint"
        else:
            support_class = "support_exists_unexpected_all_na"
        return {
            "static_valid_pixels": int(static_support.sum()), "static_channel_valid_pixels": static_counts,
            "eligible_dates": len(eligible), "max_dynamic_valid_pixels": max_dynamic,
            "max_combined_valid_pixels": max_combined, "support_class": support_class,
        }
    finally:
        for dataset in datasets.values():
            dataset.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-240-dir", type=Path, required=True)
    parser.add_argument("--output-30-dir", type=Path, required=True)
    parser.add_argument("--viirs-date-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--buffer-days", type=int, default=5)
    args = parser.parse_args()

    with args.summary_csv.open(newline="", encoding="utf-8-sig") as handle:
        summary = list(csv.DictReader(handle))
    windows = read_viirs_windows(args.viirs_date_csv)
    output: list[dict[str, object]] = []
    for row_number, row in enumerate(summary, start=1):
        site_id, status = row["event_id"], row["status"]
        if status == "good":
            continue
        input_site = args.input_root / site_id
        record: dict[str, object] = {
            "site_id": site_id, "summary_status": status, "state": site_id[:2], "year": row.get("yr", ""),
            "input_folder_exists": input_site.is_dir(),
            "output_240_exists": (args.output_240_dir / f"{site_id}{SUFFIX_240}").is_file(),
            "output_30_exists": (args.output_30_dir / f"{site_id}{SUFFIX_30}").is_file(),
            "diagnosis": "",
            "static_valid_pixels": "", "eligible_dates": "", "max_dynamic_valid_pixels": "", "max_combined_valid_pixels": "",
            "static_channel_valid_pixels": "", "error": "",
        }
        if status == "missing":
            if not input_site.is_dir():
                # There is intentionally no state-prefix gate. Prediction is
                # governed by the actual site grid and raster support, so an
                # event becomes eligible once its input folder is available.
                record["diagnosis"] = "missing_input_folder"
            elif not record["output_240_exists"]:
                record["diagnosis"] = "no_240m_prediction_output"
            else:
                record["diagnosis"] = "has_240m_output_but_no_30m_output"
        else:
            if not input_site.is_dir():
                record["diagnosis"] = "all_na_but_input_folder_absent"
            elif site_id not in windows:
                record["diagnosis"] = "all_na_but_absent_from_viirs_csv"
            else:
                try:
                    support = source_support(input_site, *windows[site_id], args.buffer_days)
                    record.update({key: value for key, value in support.items() if key != "support_class"})
                    record["static_channel_valid_pixels"] = json.dumps(support["static_channel_valid_pixels"], sort_keys=True)
                    record["diagnosis"] = str(support["support_class"])
                except Exception as error:
                    record["diagnosis"] = "support_audit_error"
                    record["error"] = str(error)
        output.append(record)
        if row_number % 25 == 0:
            print({"summary_rows_examined": row_number, "diagnostic_rows": len(output)}, flush=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(output[0]) if output else ["site_id"]
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    counts: dict[str, int] = {}
    for row in output:
        counts[row["diagnosis"]] = counts.get(row["diagnosis"], 0) + 1
    print(json.dumps({"rows": len(output), "diagnosis_counts": counts, "output": str(args.output_csv)}, indent=2))


if __name__ == "__main__":
    main()
