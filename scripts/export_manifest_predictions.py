#!/usr/bin/env python3
"""Export GIS-ready predictions for selected rows of a sequence manifest.

Each site/date gets four five-band GeoTIFFs: prediction, observation,
prediction-minus-observation residual, and valid-mask.  Band order is the
canonical TARGET_CHANNELS order and descriptions are written to the TIFF.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import rasterio
import torch

from src.data.phase2_qc import DYNAMIC_CHANNELS, TARGET_CHANNELS
from src.data.sequence_dataset import SequenceDataset
from src.data.static_contract import STATIC_CHANNELS
from src.data.transforms import standardize
from src.models.multitask_model import model_from_checkpoint, load_model_state


def inverse_target_scale(values: torch.Tensor, stats: dict) -> torch.Tensor:
    mean = torch.tensor([stats[name]["mean"] for name in TARGET_CHANNELS], dtype=values.dtype).view(-1, 1, 1)
    std = torch.tensor([stats[name]["std"] for name in TARGET_CHANNELS], dtype=values.dtype).view(-1, 1, 1)
    return values * std + mean


def write_tif(path: Path, values: np.ndarray, source: str, *, nodata: float) -> None:
    """Use the target NetCDF grid as the authoritative CRS/transform template."""
    with rasterio.open(f"NETCDF:{source}:soilmoisture") as reference:
        profile = reference.profile.copy()
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    profile.update(driver="GTiff", count=len(TARGET_CHANNELS), dtype="float32", nodata=nodata, compress="deflate", tiled=False)
    with rasterio.open(path, "w", **profile) as dst:
        for band, (name, value) in enumerate(zip(TARGET_CHANNELS, values), start=1):
            dst.write(np.where(np.isfinite(value), value, nodata).astype("float32"), band)
            dst.set_band_description(band, name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="dev_spatial_test", help="Manifest split to export (default: dev_spatial_test).")
    parser.add_argument("--month", type=int, default=None, help="Only target dates in this calendar month.")
    parser.add_argument("--max-days-per-site", type=int, default=1, help="Export at most this many chronological dates per site; 0 means all.")
    parser.add_argument("--site-id", action="append", default=None, help="Optional site ID filter; repeat for several sites.")
    args = parser.parse_args()

    if args.month is not None and not 1 <= args.month <= 12:
        parser.error("--month must be in 1..12")
    if args.max_days_per_site < 0:
        parser.error("--max-days-per-site must be non-negative")

    stats = json.loads(args.normalization.read_text())["groups"]
    dataset = SequenceDataset(args.manifest, args.split, cache_site_arrays=True)
    allowed_sites = set(args.site_id) if args.site_id else None
    selected: list[int] = []
    per_site = defaultdict(int)
    for index, row in enumerate(dataset.rows):
        if allowed_sites is not None and row["site_id"] not in allowed_sites:
            continue
        if args.month is not None and int(row["target_date"][5:7]) != args.month:
            continue
        if args.max_days_per_site and per_site[row["site_id"]] >= args.max_days_per_site:
            continue
        per_site[row["site_id"]] += 1
        selected.append(index)
    if not selected:
        raise ValueError("No rows matched the manifest/split/date/site selection")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = model_from_checkpoint(checkpoint)
    load_model_state(model, checkpoint)
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for number, index in enumerate(selected, start=1):
            sample = dataset[index]
            dynamic = standardize(torch.nan_to_num(sample["dynamic_sequence"]).unsqueeze(0), stats["dynamic"], DYNAMIC_CHANNELS, 2)
            current = standardize(torch.nan_to_num(sample["current_forcing"]).unsqueeze(0), stats["dynamic"], DYNAMIC_CHANNELS, 1)
            static = standardize(torch.nan_to_num(sample["static_stack"]).unsqueeze(0), stats["static"], STATIC_CHANNELS, 1)
            prediction = inverse_target_scale(model(dynamic, current, static)[0], stats["target"]).numpy()
            observation = sample["target_stack"].numpy()
            mask = sample["valid_mask"].numpy().astype(bool)
            row = dataset.rows[index]
            destination = args.output_dir / row["site_id"]
            destination.mkdir(exist_ok=True)
            stem = f"{row['site_id']}_{row['target_date']}"
            write_tif(destination / f"{stem}_prediction.tif", np.where(mask, prediction, np.nan), row["target_source"], nodata=-9999.0)
            write_tif(destination / f"{stem}_observation.tif", observation, row["target_source"], nodata=-9999.0)
            write_tif(destination / f"{stem}_residual.tif", np.where(mask, prediction - observation, np.nan), row["target_source"], nodata=-9999.0)
            write_tif(destination / f"{stem}_valid_mask.tif", mask.astype("float32"), row["target_source"], nodata=0.0)
            print({"progress": f"{number}/{len(selected)}", "site_id": row["site_id"], "date": row["target_date"]}, flush=True)
    print({"exports": len(selected), "sites": len(per_site), "output_dir": str(args.output_dir)})


if __name__ == "__main__":
    main()
