#!/usr/bin/env python3
"""Replay a Jan--Sep stateful recurrent checkpoint on prediction-only folders.

Unlike the fixed Seq90 predictor, this carries the ConvGRU/ConvLSTM state
forward from the requested replay start date.  It reads only forcing GeoTIFFs
and selected static rasters; an ECH2O target NetCDF is neither expected nor
read.  Chunking is an inference-memory convenience: state is passed between
chunks, so it does not reset the recurrent memory.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import rasterio
import torch
from rasterio.transform import Affine

from src.data.phase2_qc import DYNAMIC_CHANNELS, TARGET_CHANNELS
from src.data.static_contract import Grid, STATIC_CHANNELS, load_selected_static_stack
from src.data.transforms import standardize
from src.models.multitask_model import load_model_state, model_from_checkpoint
from src.utils.runtime import resolve_device, to_device


def dated_bands(path: Path) -> tuple[list[date], rasterio.DatasetReader]:
    dataset = rasterio.open(path)
    try:
        days = [date.fromisoformat(f"{value[:4]}-{value[4:6]}-{value[6:8]}") for value in dataset.descriptions if value]
    except (IndexError, ValueError) as error:
        dataset.close()
        raise ValueError(f"{path} requires YYYYMMDD band descriptions") from error
    if len(days) != dataset.count or any(next_day - current != timedelta(days=1) for current, next_day in zip(days, days[1:])):
        dataset.close()
        raise ValueError(f"{path} has missing, undescribed, or non-contiguous daily forcing bands")
    return days, dataset


def write_prediction(path: Path, values: np.ndarray, grid: Grid, support: np.ndarray) -> None:
    profile = {
        "driver": "GTiff", "width": grid.width, "height": grid.height, "count": len(TARGET_CHANNELS),
        "dtype": "float32", "crs": grid.crs, "transform": grid.transform, "nodata": -9999.0,
        "compress": "deflate", "tiled": False,
    }
    with rasterio.open(path, "w", **profile) as output:
        for band, (name, field) in enumerate(zip(TARGET_CHANNELS, values), start=1):
            output.write(np.where(support, field, -9999.0).astype("float32"), band)
            output.set_band_description(band, name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--site-id", action="append", default=None, help="Repeat to select sites; default is every folder.")
    parser.add_argument("--replay-start-date", type=date.fromisoformat, required=True,
                        help="First forcing date used to initialize zero recurrent state.")
    parser.add_argument("--start-date", type=date.fromisoformat, required=True, help="First date exported after replay.")
    parser.add_argument("--end-date", type=date.fromisoformat, required=True, help="Last date exported after replay.")
    parser.add_argument("--chunk-days", type=int, default=30, help="Inference chunk size; recurrent state is retained across chunks.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.replay_start_date > args.start_date or args.start_date > args.end_date:
        parser.error("Require replay-start-date <= start-date <= end-date")
    if args.chunk_days < 0:
        parser.error("chunk-days must be non-negative")
    sites = sorted(path for path in args.input_root.iterdir() if path.is_dir() and (args.site_id is None or path.name in args.site_id))
    if not sites:
        raise ValueError("No requested input folders found")
    stats = json.loads(args.normalization.read_text())["groups"]
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = model_from_checkpoint(checkpoint).to(device)
    load_model_state(model, checkpoint)
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run = {"protocol": "stateful Jan--Sep replay", "checkpoint": str(args.checkpoint), "normalization": str(args.normalization),
           "replay_start_date": args.replay_start_date.isoformat(), "start_date": args.start_date.isoformat(),
           "end_date": args.end_date.isoformat(), "chunk_days": args.chunk_days, "sites": []}
    for site in sites:
        datasets: dict[str, rasterio.DatasetReader] = {}
        try:
            dates_by_channel: dict[str, list[date]] = {}
            for channel in DYNAMIC_CHANNELS:
                matches = sorted(site.glob(f"{channel}_*.tif"))
                if len(matches) != 1:
                    raise ValueError(f"{site.name}: expected exactly one {channel}_*.tif, found {len(matches)}")
                dates_by_channel[channel], datasets[channel] = dated_bands(matches[0])
            days = dates_by_channel[DYNAMIC_CHANNELS[0]]
            reference = datasets[DYNAMIC_CHANNELS[0]]
            signature = (reference.width, reference.height, reference.transform, reference.crs, days)
            for channel, dataset in datasets.items():
                if (dataset.width, dataset.height, dataset.transform, dataset.crs, dates_by_channel[channel]) != signature:
                    raise ValueError(f"{site.name}: forcing grids or band dates disagree across channels ({channel})")
            day_index = {day: index for index, day in enumerate(days)}
            if args.replay_start_date not in day_index or args.end_date not in day_index:
                raise ValueError(f"{site.name}: requested replay/export period is outside available forcing dates {days[0]} to {days[-1]}")
            grid = Grid(reference.width - 2, reference.height - 2, reference.transform * Affine.translation(1, 1), reference.crs)
            static = load_selected_static_stack(site / "Spatial", grid)
            start_index, end_index = day_index[args.replay_start_date], day_index[args.end_date]
            raw_dynamic = np.stack([datasets[channel].read(list(range(start_index + 1, end_index + 2)), out_dtype="float32")[:, 1:-1, 1:-1] for channel in DYNAMIC_CHANNELS], axis=1)
            static_support = np.isfinite(static).all(axis=0)
            dynamic = to_device(standardize(torch.nan_to_num(torch.from_numpy(raw_dynamic)).unsqueeze(0), stats["dynamic"], DYNAMIC_CHANNELS, 2), device)
            static_input = to_device(standardize(torch.nan_to_num(torch.from_numpy(static)).unsqueeze(0), stats["static"], STATIC_CHANNELS, 1), device)
            predictions = []
            hidden = None
            chunk = dynamic.shape[1] if args.chunk_days == 0 else args.chunk_days
            with torch.no_grad():
                for offset in range(0, dynamic.shape[1], chunk):
                    output, hidden = model.forward_stateful_chunks(dynamic[:, offset:offset + chunk], static_input, hidden)
                    predictions.append(output.cpu())
            normalized = torch.cat(predictions, dim=1)[0]
            means = torch.tensor([stats["target"][name]["mean"] for name in TARGET_CHANNELS]).view(1, -1, 1, 1)
            stds = torch.tensor([stats["target"][name]["std"] for name in TARGET_CHANNELS]).view(1, -1, 1, 1)
            values = (normalized * stds + means).numpy()
            destination = args.output_dir / site.name
            destination.mkdir(exist_ok=True)
            written = 0
            for index, day in enumerate(days[start_index:end_index + 1]):
                if args.start_date <= day <= args.end_date:
                    support = static_support & np.isfinite(raw_dynamic[index]).all(axis=0)
                    write_prediction(destination / f"{site.name}_{day.isoformat()}_prediction.tif", values[index], grid, support)
                    written += 1
            run["sites"].append({"site_id": site.name, "available_dates": [days[0].isoformat(), days[-1].isoformat()], "prediction_count": written})
            print({"site_id": site.name, "predictions": written}, flush=True)
        finally:
            for dataset in datasets.values():
                dataset.close()
    (args.output_dir / "run_metadata.json").write_text(json.dumps(run, indent=2) + "\n")


if __name__ == "__main__":
    main()
