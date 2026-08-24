#!/usr/bin/env python3
"""Evaluate validation pixels inside versus outside site channel masks.

The PCRaster ``Spatial/chanmask.map`` grid is required to be the approved
one-cell parent of the ECH2O target grid. Value 1 denotes channel pixels and
0 denotes non-channel pixels; nodata is excluded from both domains.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_validation_manifest import Moments, SiteBatchSampler
from src.data.phase2_qc import DYNAMIC_CHANNELS, TARGET_CHANNELS
from src.data.sequence_dataset import SequenceDataset, pad_collate
from src.data.static_contract import STATIC_CHANNELS
from src.data.transforms import standardize
from src.data.water_year_dataset import WaterYearDataset
from src.models.multitask_model import load_model_state, model_from_checkpoint


def channel_mask(static_source: str, target_source: str) -> tuple[np.ndarray, np.ndarray]:
    """Load the channel map and assert its one-cell parent-grid relation."""
    path = Path(static_source) / "chanmask.map"
    if not path.is_file():
        raise FileNotFoundError(f"Missing channel mask: {path}")
    with rasterio.open(path) as source, rasterio.open(f"NETCDF:{target_source}:soilmoisture") as target:
        if source.width != target.width + 2 or source.height != target.height + 2:
            raise ValueError(f"{path}: expected one-cell parent dimensions for target grid")
        if not (np.isclose(source.transform.a, target.transform.a) and np.isclose(source.transform.e, target.transform.e)
                and np.isclose(target.transform.c, source.transform.c + source.transform.a)
                and np.isclose(target.transform.f, source.transform.f + source.transform.e)):
            raise ValueError(f"{path}: transform is not the approved one-cell parent grid")
        values = source.read(1, out_dtype="float32")[1:-1, 1:-1]
        nodata = source.nodata
    valid = np.isfinite(values)
    if nodata is not None and np.isfinite(nodata):
        valid &= ~np.isclose(values, nodata)
    unexpected = np.unique(values[valid & ~np.isin(values, (0.0, 1.0))])
    if unexpected.size:
        raise ValueError(f"{path}: expected only 0/1 channel values, got {unexpected[:8]}")
    inside = np.full(values.shape, False, dtype=bool)
    inside[valid] = values[valid] == 1.0
    return inside, valid


def write(path: Path, grouped: dict[tuple[str, str], Moments]) -> None:
    fields = ("domain", "target", "count", "mae", "rmse", "bias", "correlation", "r2", "sd_ratio")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (domain, target), moments in sorted(grouped.items()):
            writer.writerow({"domain": domain, "target": target} | moments.metrics())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("fixed", "stateful"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="dev_spatial_val")
    parser.add_argument("--months", type=int, nargs="+", default=[6, 7, 8, 9])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=24)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    stats = json.loads(args.normalization.read_text())["groups"]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = model_from_checkpoint(checkpoint)
    load_model_state(model, checkpoint)
    model.eval()
    means = torch.tensor([stats["target"][name]["mean"] for name in TARGET_CHANNELS]).view(1, -1, 1, 1)
    stds = torch.tensor([stats["target"][name]["std"] for name in TARGET_CHANNELS]).view(1, -1, 1, 1)
    grouped: dict[tuple[str, str], Moments] = defaultdict(Moments)

    if args.protocol == "fixed":
        dataset = SequenceDataset(args.manifest, args.split, cache_site_arrays=True)
        sources = {row["site_id"]: (row["static_source"], row["target_source"]) for row in dataset.rows}
        masks = {site: channel_mask(*paths) for site, paths in sources.items()}
        loader = DataLoader(dataset, batch_sampler=SiteBatchSampler(dataset, args.batch_size), collate_fn=pad_collate, num_workers=0)
        with torch.no_grad():
            for batch in loader:
                sequence = standardize(torch.nan_to_num(batch["dynamic_sequence"]), stats["dynamic"], DYNAMIC_CHANNELS, 2)
                current = standardize(torch.nan_to_num(batch["current_forcing"]), stats["dynamic"], DYNAMIC_CHANNELS, 1)
                static = standardize(torch.nan_to_num(batch["static_stack"]), stats["static"], STATIC_CHANNELS, 1)
                prediction = (model(sequence, current, static) * stds + means).numpy()
                observation, valid = batch["target_stack"].numpy(), batch["valid_mask"].numpy().astype(bool)
                for item, (site, target_date) in enumerate(zip(batch["site_id"], batch["target_date"])):
                    if int(target_date[5:7]) not in args.months:
                        continue
                    site_mask, site_support = masks[site]
                    padded = np.zeros(valid.shape[-2:], dtype=bool)
                    padded_support = np.zeros(valid.shape[-2:], dtype=bool)
                    padded[:site_mask.shape[0], :site_mask.shape[1]] = site_mask
                    padded_support[:site_support.shape[0], :site_support.shape[1]] = site_support
                    for channel, target in enumerate(TARGET_CHANNELS):
                        usable = valid[item, channel] & np.isfinite(observation[item, channel])
                        for domain, domain_mask in (("inside_channel", padded), ("outside_channel", padded_support & ~padded)):
                            selected = usable & domain_mask
                            if selected.any():
                                grouped[(domain, target)].update(prediction[item, channel][selected], observation[item, channel][selected])
    else:
        dataset = WaterYearDataset(args.manifest, args.split)
        sources = {row["site_id"]: (row["static_source"], row["target_source"]) for row in dataset.rows}
        masks = {site: channel_mask(*paths) for site, paths in sources.items()}
        with torch.no_grad():
            for sample in dataset:
                sequence = standardize(torch.nan_to_num(sample["dynamic"]).unsqueeze(0), stats["dynamic"], DYNAMIC_CHANNELS, 2)
                static = standardize(torch.nan_to_num(sample["static"]).unsqueeze(0), stats["static"], STATIC_CHANNELS, 1)
                prediction, _ = model.forward_stateful_chunks(sequence, static)
                prediction = (prediction * stds.unsqueeze(1) + means.unsqueeze(1))[0].numpy()
                observation, valid = sample["target"].numpy(), sample["valid"].numpy().astype(bool)
                site_mask, site_support = masks[sample["site_id"]]
                for day_index, day in enumerate(sample["dates"]):
                    if int(day[5:7]) not in args.months or not bool(sample["loss_days"][day_index]):
                        continue
                    for channel, target in enumerate(TARGET_CHANNELS):
                        usable = valid[day_index, channel] & np.isfinite(observation[day_index, channel])
                        for domain, domain_mask in (("inside_channel", site_mask), ("outside_channel", site_support & ~site_mask)):
                            selected = usable & domain_mask
                            if selected.any():
                                grouped[(domain, target)].update(prediction[day_index, channel][selected], observation[day_index, channel][selected])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write(args.output_dir / "by_channel_domain_target.csv", grouped)
    summary = {domain: {target: moments.metrics() for (stored_domain, target), moments in grouped.items() if stored_domain == domain}
               for domain in ("inside_channel", "outside_channel")}
    (args.output_dir / "channel_mask_summary.json").write_text(json.dumps({
        "description": "Exact valid-pixel metrics inside versus outside the one-cell-cropped chanmask.map channel domain.",
        "manifest": str(args.manifest), "checkpoint": str(args.checkpoint), "checkpoint_epoch": checkpoint.get("epoch"),
        "protocol": args.protocol, "split": args.split, "months": args.months, "metrics": summary,
    }, indent=2) + "\n")
    print(json.dumps({"output_dir": str(args.output_dir), "checkpoint_epoch": checkpoint.get("epoch"), "domains": {key: {target: value["count"] for target, value in values.items()} for key, values in summary.items()}}, indent=2))


if __name__ == "__main__":
    main()
