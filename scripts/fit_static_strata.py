#!/usr/bin/env python3
"""Fit training-reference quartile thresholds for static sensitivity strata."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data.sequence_dataset import SequenceDataset
from src.data.static_contract import STATIC_CHANNELS, load_selected_static_stack


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--channels", nargs="+", default=("wbdef", "tpi", "twi", "dem"))
    args = parser.parse_args()
    unknown = set(args.channels).difference(STATIC_CHANNELS)
    if unknown:
        parser.error(f"Unknown static channels: {sorted(unknown)}")
    dataset = SequenceDataset(args.manifest, args.split)
    unique = {row["site_id"]: row for row in dataset.rows}
    values = {channel: [] for channel in args.channels}
    for site, row in sorted(unique.items()):
        stack = load_selected_static_stack(Path(row["static_source"]), dataset._grid(row))
        for channel in args.channels:
            field = stack[STATIC_CHANNELS.index(channel)]
            values[channel].append(field[np.isfinite(field)])
        print({"site_id": site}, flush=True)
    output = {
        "manifest": str(args.manifest), "split": args.split, "site_count": len(unique), "quantiles": {},
    }
    for channel, pieces in values.items():
        combined = np.concatenate(pieces)
        output["quantiles"][channel] = {
            "count": int(len(combined)),
            "q25": float(np.quantile(combined, 0.25)), "q50": float(np.quantile(combined, 0.50)),
            "q75": float(np.quantile(combined, 0.75)),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print({"status": "complete", "output": str(args.output), "site_count": len(unique)})


if __name__ == "__main__":
    main()
