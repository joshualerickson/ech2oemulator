#!/usr/bin/env python3
"""Estimate model counterfactual sensitivity to selected static features.

This is a model-explanation diagnostic, not a causal treatment-effect analysis.
For each sampled site/date, the 90-day forcing sequence is held fixed while one
static raster channel is shifted by a requested number of *training* standard
deviations.  The script writes per-site/date summaries and an aggregate table
of the pixelwise changes in predicted ECH2O targets.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.data.phase2_qc import DYNAMIC_CHANNELS, TARGET_CHANNELS
from src.data.sequence_dataset import SequenceDataset
from src.data.static_contract import STATIC_CHANNELS
from src.data.transforms import standardize
from src.models.multitask_model import load_model_state, model_from_checkpoint
from src.utils.runtime import resolve_device, to_device


# The baseline channels are physical fields, so constrain only changes whose
# admissible ranges are unambiguous.  Other channels remain available, but must
# be explicitly requested and are left unbounded.
PHYSICAL_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "vcf": (0.0, 1.0),
    "lai": (0.0, None),
}


def select_indices(dataset: SequenceDataset, maximum: int, seed: int) -> list[int]:
    """Select samples evenly across sites and June--September months."""
    by_site_month: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(dataset.rows):
        month = int(row["target_date"][5:7])
        by_site_month[(row["site_id"], month)].append(index)
    generator = np.random.default_rng(seed)
    sites = sorted({site for site, _ in by_site_month})
    generator.shuffle(sites)
    selected: list[int] = []
    round_number = 0
    months = (6, 7, 8, 9)
    while len(selected) < maximum:
        added = False
        for site_number, site in enumerate(sites):
            # Rotate assigned month by site so one pass has near-equal months.
            month = months[(site_number + round_number) % len(months)]
            rows = by_site_month.get((site, month), [])
            if rows:
                # Deterministically spread repeated visits across that month.
                position = round_number % len(rows)
                selected.append(rows[position])
                added = True
                if len(selected) == maximum:
                    break
        if not added:
            break
        round_number += 1
    return selected


def inverse_prediction(prediction: torch.Tensor, stats: dict[str, dict[str, float | int]]) -> np.ndarray:
    means = torch.tensor([stats[name]["mean"] for name in TARGET_CHANNELS], dtype=prediction.dtype).view(-1, 1, 1)
    stds = torch.tensor([stats[name]["std"] for name in TARGET_CHANNELS], dtype=prediction.dtype).view(-1, 1, 1)
    return (prediction.cpu() * stds + means).numpy()


def perturb(values: torch.Tensor, channel: int, raw_delta: float, bounds: tuple[float | None, float | None] | None) -> torch.Tensor:
    changed = values.clone()
    candidate = changed[channel] + raw_delta
    if bounds is not None:
        lower, upper = bounds
        if lower is not None:
            candidate = candidate.clamp_min(lower)
        if upper is not None:
            candidate = candidate.clamp_max(upper)
    changed[channel] = torch.where(torch.isfinite(changed[channel]), candidate, changed[channel])
    return changed


def summarize(delta: np.ndarray, support: np.ndarray) -> dict[str, float | int]:
    values = delta[support]
    if not len(values):
        return {"pixel_count": 0, "mean_prediction_change": np.nan, "median_prediction_change": np.nan,
                "p10_prediction_change": np.nan, "p90_prediction_change": np.nan}
    return {
        "pixel_count": int(len(values)),
        "mean_prediction_change": float(np.mean(values)),
        "median_prediction_change": float(np.median(values)),
        "p10_prediction_change": float(np.quantile(values, 0.10)),
        "p90_prediction_change": float(np.quantile(values, 0.90)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--features", nargs="+", default=("vcf", "lai"))
    parser.add_argument("--delta-z", type=float, nargs="+", default=(-1.0, 1.0),
                        help="Raw-static shift expressed in training standard deviations.")
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.max_samples < 1 or args.threads < 1 or not args.delta_z:
        parser.error("max-samples, threads, and at least one delta-z must be positive/non-empty")
    unknown = sorted(set(args.features).difference(STATIC_CHANNELS))
    if unknown:
        parser.error(f"Requested static features are not in the trained contract: {unknown}")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    stats = json.loads(args.normalization.read_text(encoding="utf-8"))["groups"]
    dataset = SequenceDataset(args.manifest, args.split, cache_site_arrays=True)
    indices = select_indices(dataset, args.max_samples, args.seed)
    if not indices:
        raise ValueError("No samples selected")
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = model_from_checkpoint(checkpoint).to(device)
    load_model_state(model, checkpoint)
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    channel_indices = {name: STATIC_CHANNELS.index(name) for name in args.features}
    with torch.inference_mode():
        for number, index in enumerate(indices, start=1):
            sample = dataset[index]
            dynamic_raw = sample["dynamic_sequence"]
            static_raw = sample["static_stack"]
            support = (
                torch.isfinite(static_raw).all(dim=0)
                & torch.isfinite(dynamic_raw).all(dim=(0, 1))
            ).numpy()
            dynamic = to_device(
                standardize(torch.nan_to_num(dynamic_raw).unsqueeze(0), stats["dynamic"], DYNAMIC_CHANNELS, 2), device,
            )
            static = to_device(
                standardize(torch.nan_to_num(static_raw).unsqueeze(0), stats["static"], STATIC_CHANNELS, 1), device,
            )
            baseline = inverse_prediction(model(dynamic, dynamic[:, -1], static)[0], stats["target"])
            for feature in args.features:
                channel = channel_indices[feature]
                bounds = PHYSICAL_BOUNDS.get(feature)
                for delta_z in args.delta_z:
                    raw_delta = delta_z * float(stats["static"][feature]["std"])
                    altered_raw = perturb(static_raw, channel, raw_delta, bounds)
                    altered = to_device(
                        standardize(torch.nan_to_num(altered_raw).unsqueeze(0), stats["static"], STATIC_CHANNELS, 1), device,
                    )
                    prediction = inverse_prediction(model(dynamic, dynamic[:, -1], altered)[0], stats["target"])
                    effective = (altered_raw[channel] - static_raw[channel]).numpy()[support]
                    for target, baseline_values, changed_values in zip(TARGET_CHANNELS, baseline, prediction):
                        record: dict[str, object] = {
                            "site_id": sample["site_id"], "target_date": sample["target_date"],
                            "sample_number": number, "feature": feature, "delta_z": delta_z,
                            "requested_raw_delta": raw_delta,
                            "mean_effective_raw_delta": float(np.mean(effective)) if len(effective) else np.nan,
                            "target": target,
                            "baseline_mean_prediction": float(np.mean(baseline_values[support])) if support.any() else np.nan,
                        }
                        record.update(summarize(changed_values - baseline_values, support))
                        rows.append(record)
            print({"sample": number, "total": len(indices), "site_id": sample["site_id"], "target_date": sample["target_date"]}, flush=True)

    sample_path = args.output_dir / "per_sample_effects.csv"
    fields = list(rows[0])
    with sample_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(row["feature"], row["delta_z"], row["target"])].append(row)
    summary_rows = []
    for (feature, delta_z, target), values in sorted(groups.items()):
        effects = np.asarray([float(row["mean_prediction_change"]) for row in values], dtype=float)
        summary_rows.append({
            "feature": feature, "delta_z": delta_z, "target": target, "sample_count": len(values),
            "mean_of_sample_mean_changes": float(np.nanmean(effects)),
            "median_of_sample_mean_changes": float(np.nanmedian(effects)),
            "p10_of_sample_mean_changes": float(np.nanquantile(effects, 0.10)),
            "p90_of_sample_mean_changes": float(np.nanquantile(effects, 0.90)),
        })
    summary_path = args.output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    (args.output_dir / "run.json").write_text(json.dumps({
        "manifest": str(args.manifest), "split": args.split, "checkpoint": str(args.checkpoint),
        "normalization": str(args.normalization), "features": args.features, "delta_z": args.delta_z,
        "max_samples": args.max_samples, "selected_samples": len(indices), "seed": args.seed,
        "interpretation": "Counterfactual model sensitivity, not a causal effect. Dynamic forcings are held fixed; one static channel changes at a time.",
    }, indent=2) + "\n", encoding="utf-8")
    print({"status": "complete", "samples": len(indices), "per_sample": str(sample_path), "summary": str(summary_path)})


if __name__ == "__main__":
    main()
