#!/usr/bin/env python3
"""Run joint VCF--LAI counterfactuals, stratified by season, climate, terrain."""
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

from scripts.explore_static_sensitivity import PHYSICAL_BOUNDS, inverse_prediction, perturb, select_indices, summarize
from src.data.phase2_qc import DYNAMIC_CHANNELS, TARGET_CHANNELS
from src.data.sequence_dataset import SequenceDataset
from src.data.static_contract import STATIC_CHANNELS
from src.data.transforms import standardize
from src.models.multitask_model import load_model_state, model_from_checkpoint
from src.utils.runtime import resolve_device, to_device


SCENARIOS = {
    "baseline": (0.0, 0.0),
    "coupled_low": (-1.0, -1.0),
    "coupled_high": (1.0, 1.0),
    "vcf_high_lai_low": (1.0, -1.0),
    "vcf_low_lai_high": (-1.0, 1.0),
}


def quartile_labels(values: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    edges = np.asarray([thresholds["q25"], thresholds["q50"], thresholds["q75"]])
    return np.digitize(values, edges, right=True) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--strata-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.max_samples < 1 or args.threads < 1:
        parser.error("max-samples and threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    stats = json.loads(args.normalization.read_text())["groups"]
    strata = json.loads(args.strata_json.read_text())["quantiles"]
    required_strata = ("wbdef", "tpi", "twi", "dem")
    if set(required_strata).difference(strata):
        parser.error(f"strata-json must include {required_strata}")
    dataset = SequenceDataset(args.manifest, args.split, cache_site_arrays=True)
    indices = select_indices(dataset, args.max_samples, args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = resolve_device(args.device)
    model = model_from_checkpoint(checkpoint).to(device)
    load_model_state(model, checkpoint)
    model.eval()
    vcf_index, lai_index = STATIC_CHANNELS.index("vcf"), STATIC_CHANNELS.index("lai")
    rows: list[dict[str, object]] = []

    with torch.inference_mode():
        for number, index in enumerate(indices, start=1):
            sample = dataset[index]
            dynamic_raw, static_raw = sample["dynamic_sequence"], sample["static_stack"]
            support = (torch.isfinite(static_raw).all(0) & torch.isfinite(dynamic_raw).all((0, 1))).numpy()
            dynamic = to_device(standardize(torch.nan_to_num(dynamic_raw).unsqueeze(0), stats["dynamic"], DYNAMIC_CHANNELS, 2), device)
            baseline_static = to_device(standardize(torch.nan_to_num(static_raw).unsqueeze(0), stats["static"], STATIC_CHANNELS, 1), device)
            baseline = inverse_prediction(model(dynamic, dynamic[:, -1], baseline_static)[0], stats["target"])
            month = int(sample["target_date"][5:7])
            memberships = {"all": np.ones_like(support, dtype=bool)}
            for channel in required_strata:
                raw = static_raw[STATIC_CHANNELS.index(channel)].numpy()
                labels = quartile_labels(raw, strata[channel])
                for quartile in range(1, 5):
                    memberships[f"{channel}_q{quartile}"] = support & np.isfinite(raw) & (labels == quartile)
            for scenario, (vcf_z, lai_z) in SCENARIOS.items():
                changed = perturb(static_raw, vcf_index, vcf_z * float(stats["static"]["vcf"]["std"]), PHYSICAL_BOUNDS["vcf"])
                changed = perturb(changed, lai_index, lai_z * float(stats["static"]["lai"]["std"]), PHYSICAL_BOUNDS["lai"])
                altered = to_device(standardize(torch.nan_to_num(changed).unsqueeze(0), stats["static"], STATIC_CHANNELS, 1), device)
                predicted = inverse_prediction(model(dynamic, dynamic[:, -1], altered)[0], stats["target"])
                effective_vcf = (changed[vcf_index] - static_raw[vcf_index]).numpy()
                effective_lai = (changed[lai_index] - static_raw[lai_index]).numpy()
                for target, base, counterfactual in zip(TARGET_CHANNELS, baseline, predicted):
                    delta = counterfactual - base
                    for label, mask in memberships.items():
                        record: dict[str, object] = {
                            "site_id": sample["site_id"], "target_date": sample["target_date"], "month": month,
                            "sample_number": number, "scenario": scenario, "vcf_delta_z": vcf_z, "lai_delta_z": lai_z,
                            "stratum": label, "target": target,
                            "mean_effective_vcf_change": float(np.mean(effective_vcf[mask])) if mask.any() else np.nan,
                            "mean_effective_lai_change": float(np.mean(effective_lai[mask])) if mask.any() else np.nan,
                            "baseline_mean_prediction": float(np.mean(base[mask])) if mask.any() else np.nan,
                        }
                        record.update(summarize(delta, mask))
                        rows.append(record)
            print({"sample": number, "total": len(indices), "site_id": sample["site_id"], "date": sample["target_date"]}, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_sample = args.output_dir / "joint_canopy_per_sample_strata.csv"
    with per_sample.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(row["scenario"], row["target"], row["month"], row["stratum"])].append(row)
    summary = []
    for (scenario, target, month, stratum), records in sorted(grouped.items()):
        effects = np.asarray([float(record["mean_prediction_change"]) for record in records])
        summary.append({"scenario": scenario, "target": target, "month": month, "stratum": stratum,
                        "sample_count": len(records), "mean_of_sample_means": float(np.nanmean(effects)),
                        "median_of_sample_means": float(np.nanmedian(effects)),
                        "p10_of_sample_means": float(np.nanquantile(effects, .1)),
                        "p90_of_sample_means": float(np.nanquantile(effects, .9))})
    summary_path = args.output_dir / "joint_canopy_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    (args.output_dir / "run.json").write_text(json.dumps({"scenarios": SCENARIOS, "samples": len(indices), "seed": args.seed,
        "interpretation": "Model counterfactual sensitivity, not a causal canopy treatment effect. Cross scenarios may be off the observed VCF--LAI correlation manifold."}, indent=2) + "\n")
    print({"status": "complete", "samples": len(indices), "per_sample": str(per_sample), "summary": str(summary_path)})


if __name__ == "__main__":
    main()
