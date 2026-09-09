#!/usr/bin/env python3
"""Estimate the observed VCF--LAI association from a reproducible training-pixel sample.

The sample is drawn across valid pixels in unique sites of one persisted manifest
split.  It reports correlation *and* joint quartile occupancy, which is more
useful than correlation alone for judging whether crossed VCF/LAI sensitivity
scenarios are well supported by the observed covariate distribution.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.stats import rankdata

from src.data.sequence_dataset import SequenceDataset
from src.data.static_contract import read_ascii_on_target_grid, resolve_ascii_static


def reservoir_update(
    reservoir: np.ndarray, values: np.ndarray, seen: int, rng: np.random.Generator
) -> int:
    """Update a fixed-size reservoir so every valid source pixel is equally likely."""
    capacity = len(reservoir)
    for value in values:
        seen += 1
        if seen <= capacity:
            reservoir[seen - 1] = value
        else:
            index = int(rng.integers(0, seen))
            if index < capacity:
                reservoir[index] = value
    return seen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--sample-size", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.sample_size < 100:
        parser.error("--sample-size must be at least 100")

    dataset = SequenceDataset(args.manifest, args.split)
    sites = {row["site_id"]: row for row in dataset.rows}
    rng = np.random.default_rng(args.seed)
    reservoir = np.empty((args.sample_size, 2), dtype=np.float32)
    seen = 0
    used_sites = 0
    for site_id, row in sorted(sites.items()):
        grid = dataset._grid(row)
        spatial = Path(row["static_source"])
        vcf = read_ascii_on_target_grid(resolve_ascii_static(spatial, "vcf"), grid)
        lai = read_ascii_on_target_grid(resolve_ascii_static(spatial, "lai"), grid)
        valid = np.isfinite(vcf) & np.isfinite(lai)
        values = np.column_stack((vcf[valid], lai[valid])).astype(np.float32, copy=False)
        if values.size:
            seen = reservoir_update(reservoir, values, seen, rng)
            used_sites += 1
        print({"site_id": site_id, "valid_pixels": int(values.shape[0])}, flush=True)

    sample = reservoir[: min(seen, args.sample_size)]
    vcf, lai = sample[:, 0], sample[:, 1]
    pearson = float(np.corrcoef(vcf, lai)[0, 1])
    spearman = float(np.corrcoef(rankdata(vcf), rankdata(lai))[0, 1])
    vcf_q = np.quantile(vcf, (0.25, 0.5, 0.75))
    lai_q = np.quantile(lai, (0.25, 0.5, 0.75))
    vcf_label = np.digitize(vcf, vcf_q, right=True) + 1
    lai_label = np.digitize(lai, lai_q, right=True) + 1
    joint = {
        f"vcf_q{i}_lai_q{j}": {
            "count": int(np.sum((vcf_label == i) & (lai_label == j))),
            "fraction": float(np.mean((vcf_label == i) & (lai_label == j))),
        }
        for i in range(1, 5) for j in range(1, 5)
    }
    scenarios = {
        "coupled_low__vcf_q1_lai_q1": joint["vcf_q1_lai_q1"],
        "coupled_high__vcf_q4_lai_q4": joint["vcf_q4_lai_q4"],
        "crossed_vcf_high_lai_low__vcf_q4_lai_q1": joint["vcf_q4_lai_q1"],
        "crossed_vcf_low_lai_high__vcf_q1_lai_q4": joint["vcf_q1_lai_q4"],
    }
    output = {
        "manifest": str(args.manifest), "split": args.split, "seed": args.seed,
        "unique_sites": len(sites), "sites_with_valid_vcf_lai": used_sites,
        "source_valid_pixel_count": seen, "reservoir_sample_size": int(len(sample)),
        "correlation": {"pearson": pearson, "spearman_rank": spearman},
        "sample_quantiles": {
            "vcf": {"q25": float(vcf_q[0]), "q50": float(vcf_q[1]), "q75": float(vcf_q[2])},
            "lai": {"q25": float(lai_q[0]), "q50": float(lai_q[1]), "q75": float(lai_q[2])},
        },
        "joint_quartile_occupancy": joint,
        "key_scenario_support": scenarios,
        "interpretation": (
            "Quartiles and occupancies are calculated from the pooled valid training-pixel reservoir. "
            "Crossed q4/q1 and q1/q4 cells quantify support for opposite-direction VCF/LAI combinations; "
            "they are not the same magnitude as the dashboard's global-plus/minus-one-SD perturbations."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
