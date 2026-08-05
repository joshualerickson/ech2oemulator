#!/usr/bin/env python3
"""Evaluate every valid validation pixel in fixed terrain/climate quantile strata.

The strata are computed once from the selected split's static pixels and then
applied identically to every target and model report.  This is an exact masked
pixel aggregation, not a random pixel sample.
"""
from __future__ import annotations

import argparse
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


CONTEXTS = ("dem", "tpi", "twi", "wbdef")
CONTEXT_LABELS = {
    "dem": "Elevation", "tpi": "Topographic position index", "twi": "Topographic wetness index", "wbdef": "Climatic water deficit",
}


class SiteBatchSampler(Sampler[list[int]]):
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
        error = prediction - observation
        self.count += prediction.size
        self.abs_error += float(np.abs(error).sum())
        self.squared_error += float(np.square(error).sum())
        self.error += float(error.sum())
        self.prediction += float(prediction.sum())
        self.prediction_sq += float(np.square(prediction).sum())
        self.observation += float(observation.sum())
        self.observation_sq += float(np.square(observation).sum())
        self.cross += float((prediction * observation).sum())

    def result(self) -> dict[str, float | int]:
        n = max(self.count, 1)
        p_mean, y_mean = self.prediction / n, self.observation / n
        p_var = max(self.prediction_sq / n - p_mean**2, 0.0)
        y_var = max(self.observation_sq / n - y_mean**2, 0.0)
        covariance = self.cross / n - p_mean * y_mean
        return {
            "count": self.count,
            "mae": self.abs_error / n,
            "rmse": (self.squared_error / n) ** 0.5,
            "bias": self.error / n,
            "correlation": covariance / max((p_var * y_var) ** 0.5, 1e-12),
            "r2": 1.0 - self.squared_error / max(self.observation_sq - n * y_mean**2, 1e-12),
            "sd_ratio": p_var**0.5 / max(y_var**0.5, 1e-12),
        }


def static_edges(dataset: SequenceDataset, bins: int, sample_per_site: int) -> dict[str, np.ndarray]:
    """Derive deterministic split-wide static quantiles without target values."""
    channel_index = {name: index for index, name in enumerate(STATIC_CHANNELS)}
    values: dict[str, list[np.ndarray]] = {name: [] for name in CONTEXTS}
    seen: set[str] = set()
    for row in dataset.rows:
        site_id = row["site_id"]
        if site_id in seen:
            continue
        seen.add(site_id)
        static = dataset._static(row).numpy()  # one cached static stack per site
        for name in CONTEXTS:
            finite = static[channel_index[name]][np.isfinite(static[channel_index[name]])]
            if finite.size > sample_per_site:
                finite = finite[np.linspace(0, finite.size - 1, sample_per_site, dtype=int)]
            values[name].append(finite)
    return {name: np.quantile(np.concatenate(parts), np.linspace(0, 1, bins + 1)) for name, parts in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="dev_spatial_val")
    parser.add_argument("--months", type=int, nargs="+", default=[6, 7, 8, 9])
    parser.add_argument("--bins", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--static-sample-per-site", type=int, default=100_000)
    args = parser.parse_args()
    if args.bins < 2 or args.batch_size < 1:
        parser.error("--bins must be at least 2 and --batch-size must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    stats = json.loads(args.normalization.read_text())["groups"]
    dataset = SequenceDataset(args.manifest, args.split, cache_site_arrays=True)
    edges = static_edges(dataset, args.bins, args.static_sample_per_site)
    loader = DataLoader(dataset, batch_sampler=SiteBatchSampler(dataset, args.batch_size), collate_fn=pad_collate, num_workers=0)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = model_from_checkpoint(state)
    load_model_state(model, state)
    model.eval()
    target_mean = torch.tensor([stats["target"][name]["mean"] for name in TARGET_CHANNELS]).view(1, -1, 1, 1)
    target_std = torch.tensor([stats["target"][name]["std"] for name in TARGET_CHANNELS]).view(1, -1, 1, 1)
    context_index = {name: STATIC_CHANNELS.index(name) for name in CONTEXTS}
    grouped: dict[tuple[str, int, str], Moments] = defaultdict(Moments)
    by_month: dict[tuple[str, str, int, str], Moments] = defaultdict(Moments)
    by_state: dict[tuple[str, str, int, str], Moments] = defaultdict(Moments)
    by_month_state: dict[tuple[str, str, str, int, str], Moments] = defaultdict(Moments)
    seen_months: set[str] = set()
    seen_states: set[str] = set()
    seen_month_states: set[tuple[str, str]] = set()
    with torch.no_grad():
        for batch in loader:
            sequence = standardize(torch.nan_to_num(batch["dynamic_sequence"]), stats["dynamic"], DYNAMIC_CHANNELS, 2)
            current = standardize(torch.nan_to_num(batch["current_forcing"]), stats["dynamic"], DYNAMIC_CHANNELS, 1)
            static = standardize(torch.nan_to_num(batch["static_stack"]), stats["static"], STATIC_CHANNELS, 1)
            prediction = (model(sequence, current, static) * target_std + target_mean).numpy()
            observation, valid, raw_static = batch["target_stack"].numpy(), batch["valid_mask"].numpy().astype(bool), batch["static_stack"].numpy()
            for item, (site_id, target_date) in enumerate(zip(batch["site_id"], batch["target_date"])):
                month = str(int(target_date[5:7]))
                if int(month) not in args.months:
                    continue
                site_state = site_id[:2]
                seen_months.add(month)
                seen_states.add(site_state)
                seen_month_states.add((month, site_state))
                for context in CONTEXTS:
                    values = raw_static[item, context_index[context]]
                    bin_id = np.digitize(values, edges[context][1:-1], right=False)
                    for channel, target in enumerate(TARGET_CHANNELS):
                        usable = valid[item, channel] & np.isfinite(observation[item, channel]) & np.isfinite(values)
                        for index in range(args.bins):
                            mask = usable & (bin_id == index)
                            if mask.any():
                                p, y = prediction[item, channel][mask], observation[item, channel][mask]
                                grouped[(context, index, target)].update(p, y)
                                by_month[(month, context, index, target)].update(p, y)
                                by_state[(site_state, context, index, target)].update(p, y)
                                by_month_state[(month, site_state, context, index, target)].update(p, y)

    def context_payload(source: dict, prefix: tuple[str, ...]) -> dict[str, object]:
        return {
            context: {
                "label": CONTEXT_LABELS[context], "edges": [float(value) for value in edges[context]],
                "bins": {f"Q{index + 1}": {target: source[prefix + (context, index, target)].result() for target in TARGET_CHANNELS} for index in range(args.bins)},
            }
            for context in CONTEXTS
        }
    result = {
        "description": "Exact masked pixel metrics, stratified by split-wide static-pixel quantiles with optional month/state grouping.",
        "manifest": str(args.manifest), "checkpoint": str(args.checkpoint), "checkpoint_epoch": state.get("epoch"),
        "split": args.split, "months": args.months, "bins": args.bins,
        "contexts": context_payload(grouped, ()),
        "by_month": {month: {"contexts": context_payload(by_month, (month,))} for month in sorted(seen_months)},
        "by_state": {site_state: {"contexts": context_payload(by_state, (site_state,))} for site_state in sorted(seen_states)},
        "by_month_state": {
            month: {site_state: {"contexts": context_payload(by_month_state, (month, site_state))} for m, site_state in sorted(seen_month_states) if m == month}
            for month in sorted(seen_months)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "contexts": list(result["contexts"]), "bins": args.bins}, indent=2))


if __name__ == "__main__":
    main()
