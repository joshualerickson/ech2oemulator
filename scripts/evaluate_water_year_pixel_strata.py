#!/usr/bin/env python3
"""Exact pixel-level terrain/climate strata for a stateful water-year checkpoint."""
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

from scripts.evaluate_pixel_strata import (CONTEXTS, CONTEXT_LABELS, JOINT_CONTEXTS,
                                            Moments, contribution)
from src.data.phase2_qc import DYNAMIC_CHANNELS, TARGET_CHANNELS
from src.data.static_contract import STATIC_CHANNELS
from src.data.transforms import standardize
from src.data.water_year_dataset import WaterYearDataset
from src.models.multitask_model import load_model_state, model_from_checkpoint


def static_edges(dataset: WaterYearDataset, bins: int, sample_per_site: int) -> dict[str, np.ndarray]:
    index = {name: STATIC_CHANNELS.index(name) for name in CONTEXTS}
    values: dict[str, list[np.ndarray]] = {name: [] for name in CONTEXTS}
    for sample in dataset:
        static = sample["static"].numpy()
        for name in CONTEXTS:
            finite = static[index[name]][np.isfinite(static[index[name]])]
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
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--static-sample-per-site", type=int, default=100_000)
    args = parser.parse_args()
    if args.bins < 2:
        parser.error("--bins must be at least 2")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    stats = json.loads(args.normalization.read_text())["groups"]
    dataset = WaterYearDataset(args.manifest, args.split)
    edges = static_edges(dataset, args.bins, args.static_sample_per_site)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = model_from_checkpoint(checkpoint)
    load_model_state(model, checkpoint)
    model.eval()
    means = torch.tensor([stats["target"][name]["mean"] for name in TARGET_CHANNELS]).view(1, 1, -1, 1, 1)
    stds = torch.tensor([stats["target"][name]["std"] for name in TARGET_CHANNELS]).view(1, 1, -1, 1, 1)
    context_index = {name: STATIC_CHANNELS.index(name) for name in CONTEXTS}
    grouped: dict[tuple[str, int, str], Moments] = defaultdict(Moments)
    by_month: dict[tuple[str, str, int, str], Moments] = defaultdict(Moments)
    by_state: dict[tuple[str, str, int, str], Moments] = defaultdict(Moments)
    by_month_state: dict[tuple[str, str, str, int, str], Moments] = defaultdict(Moments)
    joint: dict[tuple[str, str, int, int, str], Moments] = defaultdict(Moments)
    joint_by_month: dict[tuple[str, str, str, int, int, str], Moments] = defaultdict(Moments)
    joint_by_state: dict[tuple[str, str, str, int, int, str], Moments] = defaultdict(Moments)
    joint_by_month_state: dict[tuple[str, str, str, str, int, int, str], Moments] = defaultdict(Moments)
    seen_months: set[str] = set()
    seen_states: set[str] = set()
    seen_month_states: set[tuple[str, str]] = set()
    with torch.no_grad():
        for sample_index, sample in enumerate(dataset, 1):
            sequence = standardize(torch.nan_to_num(sample["dynamic"]).unsqueeze(0), stats["dynamic"], DYNAMIC_CHANNELS, 2)
            static = standardize(torch.nan_to_num(sample["static"]).unsqueeze(0), stats["static"], STATIC_CHANNELS, 1)
            prediction, _ = model.forward_stateful_chunks(sequence, static)
            prediction = (prediction * stds + means)[0].numpy()
            observation = sample["target"].numpy()
            valid = sample["valid"].numpy().astype(bool)
            raw_static = sample["static"].numpy()
            bins = {name: np.digitize(raw_static[context_index[name]], edges[name][1:-1], right=False) for name in CONTEXTS}
            state = sample["site_id"][:2]
            for day_index, day in enumerate(sample["dates"]):
                month = int(day[5:7])
                if month not in args.months or not bool(sample["loss_days"][day_index]):
                    continue
                month_text = str(month)
                seen_months.add(month_text); seen_states.add(state); seen_month_states.add((month_text, state))
                for context in CONTEXTS:
                    values, bin_id = raw_static[context_index[context]], bins[context]
                    for channel, target in enumerate(TARGET_CHANNELS):
                        usable = valid[day_index, channel] & np.isfinite(observation[day_index, channel]) & np.isfinite(values)
                        for bin_number in range(args.bins):
                            mask = usable & (bin_id == bin_number)
                            if mask.any():
                                part = contribution(prediction[day_index, channel][mask], observation[day_index, channel][mask])
                                grouped[(context, bin_number, target)].merge(part)
                                by_month[(month_text, context, bin_number, target)].merge(part)
                                by_state[(state, context, bin_number, target)].merge(part)
                                by_month_state[(month_text, state, context, bin_number, target)].merge(part)
                for x_context, y_context in JOINT_CONTEXTS:
                    x_values, y_values = raw_static[context_index[x_context]], raw_static[context_index[y_context]]
                    for channel, target in enumerate(TARGET_CHANNELS):
                        usable = valid[day_index, channel] & np.isfinite(observation[day_index, channel]) & np.isfinite(x_values) & np.isfinite(y_values)
                        for x_bin in range(args.bins):
                            for y_bin in range(args.bins):
                                mask = usable & (bins[x_context] == x_bin) & (bins[y_context] == y_bin)
                                if mask.any():
                                    part = contribution(prediction[day_index, channel][mask], observation[day_index, channel][mask])
                                    joint[(x_context, y_context, x_bin, y_bin, target)].merge(part)
                                    joint_by_month[(month_text, x_context, y_context, x_bin, y_bin, target)].merge(part)
                                    joint_by_state[(state, x_context, y_context, x_bin, y_bin, target)].merge(part)
                                    joint_by_month_state[(month_text, state, x_context, y_context, x_bin, y_bin, target)].merge(part)
            print(f"evaluated site {sample_index}/{len(dataset)}: {sample['site_id']}", flush=True)

    def context_payload(source: dict, prefix: tuple[str, ...]) -> dict[str, object]:
        return {context: {"label": CONTEXT_LABELS[context], "edges": [float(value) for value in edges[context]],
                          "bins": {f"Q{index + 1}": {target: source[prefix + (context, index, target)].result() for target in TARGET_CHANNELS} for index in range(args.bins)}}
                for context in CONTEXTS}

    def joint_payload(source: dict, prefix: tuple[str, ...]) -> dict[str, object]:
        result: dict[str, object] = {}
        for x_context, y_context in JOINT_CONTEXTS:
            result[f"{x_context}_x_{y_context}"] = {
                "x_context": x_context, "y_context": y_context, "x_label": CONTEXT_LABELS[x_context], "y_label": CONTEXT_LABELS[y_context],
                "x_edges": [float(value) for value in edges[x_context]], "y_edges": [float(value) for value in edges[y_context]],
                "cells": {f"Q{x_bin + 1}_Q{y_bin + 1}": {target: source[prefix + (x_context, y_context, x_bin, y_bin, target)].result() for target in TARGET_CHANNELS}
                          for y_bin in range(args.bins) for x_bin in range(args.bins)},
            }
        return result

    result = {
        "description": "Exact valid pixel metrics for full-water-year BPTT, stratified by static-pixel quantiles with month/state grouping.",
        "manifest": str(args.manifest), "checkpoint": str(args.checkpoint), "checkpoint_epoch": checkpoint.get("epoch"),
        "split": args.split, "months": args.months, "bins": args.bins, "contexts": context_payload(grouped, ()),
        "by_month": {month: {"contexts": context_payload(by_month, (month,))} for month in sorted(seen_months)},
        "by_state": {state: {"contexts": context_payload(by_state, (state,))} for state in sorted(seen_states)},
        "by_month_state": {month: {state: {"contexts": context_payload(by_month_state, (month, state))} for item_month, state in sorted(seen_month_states) if item_month == month} for month in sorted(seen_months)},
        "joint_contexts": joint_payload(joint, ()),
        "joint_by_month": {month: {"joint_contexts": joint_payload(joint_by_month, (month,))} for month in sorted(seen_months)},
        "joint_by_state": {state: {"joint_contexts": joint_payload(joint_by_state, (state,))} for state in sorted(seen_states)},
        "joint_by_month_state": {month: {state: {"joint_contexts": joint_payload(joint_by_month_state, (month, state))} for item_month, state in sorted(seen_month_states) if item_month == month} for month in sorted(seen_months)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "contexts": list(result["contexts"]), "bins": args.bins}, indent=2))


if __name__ == "__main__":
    main()
