#!/usr/bin/env python3
"""Write reproducible aggregate, state, and site metrics for a fixed manifest split."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.phase2_qc import DYNAMIC_CHANNELS, TARGET_CHANNELS
from src.data.sequence_dataset import SequenceDataset, pad_collate
from src.data.static_contract import STATIC_CHANNELS
from src.data.transforms import standardize
from src.models.multitask_model import load_model_state, model_from_checkpoint


METRIC_COLUMNS = ("count", "mae", "rmse", "bias", "correlation", "r2", "sd_ratio")


class SiteBatchSampler(Sampler[list[int]]):
    """Batch only same-site dates, preserving their common spatial dimensions."""

    def __init__(self, dataset: SequenceDataset, batch_size: int) -> None:
        self.batch_size = batch_size
        self.groups: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(dataset.rows):
            self.groups[row["site_id"]].append(index)

    def __iter__(self):
        for site_id in sorted(self.groups):
            indices = self.groups[site_id]
            yield from (indices[start:start + self.batch_size] for start in range(0, len(indices), self.batch_size))

    def __len__(self) -> int:
        return sum((len(indices) + self.batch_size - 1) // self.batch_size for indices in self.groups.values())


class Moments:
    def __init__(self) -> None:
        self.count = 0
        self.abs_error = self.squared_error = self.error = 0.0
        self.prediction = self.prediction_sq = self.observation = self.observation_sq = self.cross = 0.0

    def update(self, prediction: np.ndarray, observation: np.ndarray) -> None:
        residual = prediction - observation
        self.count += prediction.size
        self.abs_error += float(np.abs(residual).sum())
        self.squared_error += float(np.square(residual).sum())
        self.error += float(residual.sum())
        self.prediction += float(prediction.sum())
        self.prediction_sq += float(np.square(prediction).sum())
        self.observation += float(observation.sum())
        self.observation_sq += float(np.square(observation).sum())
        self.cross += float((prediction * observation).sum())

    def metrics(self) -> dict[str, float | int]:
        n = max(self.count, 1)
        pred_mean, obs_mean = self.prediction / n, self.observation / n
        pred_var = max(self.prediction_sq / n - pred_mean**2, 0.0)
        obs_var = max(self.observation_sq / n - obs_mean**2, 0.0)
        covariance = self.cross / n - pred_mean * obs_mean
        correlation = covariance / max((pred_var * obs_var) ** 0.5, 1e-12)
        return {
            "count": self.count,
            "mae": self.abs_error / n,
            "rmse": (self.squared_error / n) ** 0.5,
            "bias": self.error / n,
            "correlation": correlation,
            "r2": 1.0 - self.squared_error / max(self.observation_sq - n * obs_mean**2, 1e-12),
            "sd_ratio": pred_var**0.5 / max(obs_var**0.5, 1e-12),
        }


def write_table(path: Path, grouped: dict[tuple[str, ...], Moments], group_names: tuple[str, ...]) -> None:
    fields = [*group_names, "target", *METRIC_COLUMNS]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, moment in sorted(grouped.items()):
            *groups, target = key
            writer.writerow(dict(zip(group_names, groups)) | {"target": target} | moment.metrics())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="dev_spatial_val")
    parser.add_argument("--months", type=int, nargs="+", default=[6, 7, 8, 9])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    stats = json.loads(args.normalization.read_text())["groups"]
    dataset = SequenceDataset(args.manifest, args.split, cache_site_arrays=True)
    loader = DataLoader(dataset, batch_sampler=SiteBatchSampler(dataset, args.batch_size), collate_fn=pad_collate, num_workers=0)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = model_from_checkpoint(checkpoint)
    load_model_state(model, checkpoint)
    model.eval()
    means = torch.tensor([stats["target"][name]["mean"] for name in TARGET_CHANNELS]).view(1, -1, 1, 1)
    stds = torch.tensor([stats["target"][name]["std"] for name in TARGET_CHANNELS]).view(1, -1, 1, 1)
    all_metrics: dict[tuple[str, ...], Moments] = defaultdict(Moments)
    month_metrics: dict[tuple[str, ...], Moments] = defaultdict(Moments)
    state_metrics: dict[tuple[str, ...], Moments] = defaultdict(Moments)
    month_state_metrics: dict[tuple[str, ...], Moments] = defaultdict(Moments)
    site_metrics: dict[tuple[str, ...], Moments] = defaultdict(Moments)
    with torch.no_grad():
        for batch in loader:
            sequence = standardize(torch.nan_to_num(batch["dynamic_sequence"]), stats["dynamic"], DYNAMIC_CHANNELS, 2)
            current = standardize(torch.nan_to_num(batch["current_forcing"]), stats["dynamic"], DYNAMIC_CHANNELS, 1)
            static = standardize(torch.nan_to_num(batch["static_stack"]), stats["static"], STATIC_CHANNELS, 1)
            prediction = (model(sequence, current, static) * stds + means).numpy()
            observation, valid = batch["target_stack"].numpy(), batch["valid_mask"].numpy().astype(bool)
            for item, (site_id, target_date) in enumerate(zip(batch["site_id"], batch["target_date"])):
                month, state = int(target_date[5:7]), site_id[:2]
                if month not in args.months:
                    continue
                for channel, target in enumerate(TARGET_CHANNELS):
                    mask = valid[item, channel] & np.isfinite(observation[item, channel])
                    if not mask.any():
                        continue
                    p, y = prediction[item, channel][mask], observation[item, channel][mask]
                    all_metrics[(target,)].update(p, y)
                    month_metrics[(str(month), target)].update(p, y)
                    state_metrics[(state, target)].update(p, y)
                    month_state_metrics[(str(month), state, target)].update(p, y)
                    site_metrics[(site_id, target)].update(p, y)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_table(args.output_dir / "by_target.csv", all_metrics, ())
    write_table(args.output_dir / "by_month_target.csv", month_metrics, ("month",))
    write_table(args.output_dir / "by_state_target.csv", state_metrics, ("state",))
    write_table(args.output_dir / "by_month_state_target.csv", month_state_metrics, ("month", "state"))
    write_table(args.output_dir / "by_site_target.csv", site_metrics, ("site_id",))
    summary = {
        "manifest": str(args.manifest), "checkpoint": str(args.checkpoint), "split": args.split,
        "months": args.months, "checkpoint_epoch": checkpoint.get("epoch"),
        "metrics": {target: moment.metrics() for (target,), moment in sorted(all_metrics.items())},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
