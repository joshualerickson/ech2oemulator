#!/usr/bin/env python3
"""Exhaustively evaluate stateful Oct--Sep recurrent checkpoints by pixel."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_validation_manifest import Moments, write_table
from src.data.phase2_qc import DYNAMIC_CHANNELS, TARGET_CHANNELS
from src.data.static_contract import STATIC_CHANNELS
from src.data.transforms import standardize
from src.data.water_year_dataset import WaterYearDataset
from src.models.multitask_model import load_model_state, model_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="dev_spatial_val")
    parser.add_argument("--months", type=int, nargs="+", default=[6, 7, 8, 9])
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    stats = json.loads(args.normalization.read_text())["groups"]
    dataset = WaterYearDataset(args.manifest, args.split)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = model_from_checkpoint(checkpoint)
    load_model_state(model, checkpoint)
    model.eval()
    means = torch.tensor([stats["target"][name]["mean"] for name in TARGET_CHANNELS]).view(1, 1, -1, 1, 1)
    stds = torch.tensor([stats["target"][name]["std"] for name in TARGET_CHANNELS]).view(1, 1, -1, 1, 1)
    all_metrics: dict[tuple[str, ...], Moments] = defaultdict(Moments)
    month_metrics: dict[tuple[str, ...], Moments] = defaultdict(Moments)
    state_metrics: dict[tuple[str, ...], Moments] = defaultdict(Moments)
    month_state_metrics: dict[tuple[str, ...], Moments] = defaultdict(Moments)
    site_metrics: dict[tuple[str, ...], Moments] = defaultdict(Moments)
    with torch.no_grad():
        for index, sample in enumerate(dataset, 1):
            dynamic = standardize(torch.nan_to_num(sample["dynamic"]).unsqueeze(0), stats["dynamic"], DYNAMIC_CHANNELS, 2)
            static = standardize(torch.nan_to_num(sample["static"]).unsqueeze(0), stats["static"], STATIC_CHANNELS, 1)
            prediction, _ = model.forward_stateful_chunks(dynamic, static)
            prediction = (prediction * stds + means)[0].numpy()
            observation = sample["target"].numpy()
            valid = sample["valid"].numpy().astype(bool)
            for day_index, day in enumerate(sample["dates"]):
                month, state = int(day[5:7]), sample["site_id"][:2]
                if month not in args.months or not bool(sample["loss_days"][day_index]):
                    continue
                for channel, target in enumerate(TARGET_CHANNELS):
                    mask = valid[day_index, channel] & np.isfinite(observation[day_index, channel])
                    if not mask.any():
                        continue
                    p, y = prediction[day_index, channel][mask], observation[day_index, channel][mask]
                    all_metrics[(target,)].update(p, y)
                    month_metrics[(str(month), target)].update(p, y)
                    state_metrics[(state, target)].update(p, y)
                    month_state_metrics[(str(month), state, target)].update(p, y)
                    site_metrics[(sample["site_id"], target)].update(p, y)
            print(f"evaluated site {index}/{len(dataset)}: {sample['site_id']}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_table(args.output_dir / "by_target.csv", all_metrics, ())
    write_table(args.output_dir / "by_month_target.csv", month_metrics, ("month",))
    write_table(args.output_dir / "by_state_target.csv", state_metrics, ("state",))
    write_table(args.output_dir / "by_month_state_target.csv", month_state_metrics, ("month", "state"))
    write_table(args.output_dir / "by_site_target.csv", site_metrics, ("site_id",))
    summary = {"manifest": str(args.manifest), "checkpoint": str(args.checkpoint), "split": args.split,
               "months": args.months, "checkpoint_epoch": checkpoint.get("epoch"),
               "metrics": {target: moment.metrics() for (target,), moment in sorted(all_metrics.items())}}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
