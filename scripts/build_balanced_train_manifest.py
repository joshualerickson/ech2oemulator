#!/usr/bin/env python3
"""Build a state-balanced seasonal training manifest, excluding an evaluation panel."""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.temporal_contract import TEMPORAL_CONTRACT


FIELDS = (
    "split", "site_id", "bbox_id", "water_year_end", "temporal_contract", "start_date", "end_date", "target_date",
    "sequence_length", "target_time_index", "forcing_start_index", "forcing_end_index", "dynamic_sequence_source", "static_source", "target_source",
)


def screen_path(daily_dir: Path, site_id: str) -> Path:
    matches = list(daily_dir.glob(f"{site_id}_*_daily_screen.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one daily screen for {site_id}; found {len(matches)}")
    return matches[0]


def eligible_rows(site_id: str, daily_dir: Path, data_root: Path, months: set[int], sequence_length: int) -> list[dict[str, object]]:
    with screen_path(daily_dir, site_id).open(newline="") as handle:
        daily = list(csv.DictReader(handle))
    by_date = {date.fromisoformat(row["date"]): row for row in daily}
    water_year_end = int(daily[0]["water_year_end"])
    root = data_root / site_id
    output = []
    for target_date, target in sorted(by_date.items()):
        if target_date.month not in months or target["target_qa_valid"] != "True":
            continue
        history = [by_date.get(target_date - timedelta(days=offset)) for offset in range(sequence_length - 1, -1, -1)]
        if any(day is None or day["forcing_valid"] != "True" for day in history):
            continue
        output.append({
            "split": "dev_train", "site_id": site_id, "bbox_id": site_id,
            "water_year_end": water_year_end, "temporal_contract": TEMPORAL_CONTRACT, "start_date": history[0]["date"],
            "end_date": target["date"], "target_date": target["date"],
            "sequence_length": sequence_length, "target_time_index": target["target_time_index"], "forcing_start_index": history[0]["forcing_time_index"], "forcing_end_index": target["forcing_time_index"], "dynamic_sequence_source": str(root),
            "static_source": str(root / "Spatial"),
            "target_source": str(root / f"{site_id}-{water_year_end}_subdaily.nc"),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True, help="All sites in this manifest are held out.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--site-count", type=int, default=50)
    parser.add_argument("--months", type=int, nargs="+", default=[6, 7, 8, 9])
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    if args.site_count < 1 or args.sequence_length < 1:
        parser.error("--site-count and --sequence-length must be positive")

    with args.external_manifest.open(newline="") as handle:
        held_out = {row["site_id"] for row in csv.DictReader(handle)}
    candidates = sorted({path.name.split("_")[0] for path in args.daily_dir.glob("*_daily_screen.csv")} - held_out)
    # Eligibility is evaluated before sampling: a completed daily screen does
    # not necessarily contain a June--September target with a full 30-day past.
    candidate_rows = {
        site_id: eligible_rows(site_id, args.daily_dir, args.data_root, set(args.months), args.sequence_length)
        for site_id in candidates
    }
    candidates = [site_id for site_id in candidates if candidate_rows[site_id]]
    by_state: dict[str, list[str]] = defaultdict(list)
    for site_id in candidates:
        by_state[site_id[:2]].append(site_id)
    if args.site_count < len(by_state):
        parser.error(f"Cannot balance {args.site_count} sites across {len(by_state)} states")

    # Deterministic shuffle prevents ordering in filenames from choosing the sites.
    rng = random.Random(args.seed)
    for sites in by_state.values():
        rng.shuffle(sites)

    # Give every state an equal base allocation, then assign the small remainder
    # to states with the largest eligible pool.  This is intentionally site-, not
    # sequence-, balanced so large bboxes/states do not dominate selection.
    states = sorted(by_state)
    base, remainder = divmod(args.site_count, len(states))
    capacity_order = sorted(states, key=lambda state: (-len(by_state[state]), state))
    quotas = {state: base + (1 if state in capacity_order[:remainder] else 0) for state in states}
    short = {state: quota for state, quota in quotas.items() if quota > len(by_state[state])}
    if short:
        parser.error(f"Not enough candidate sites for allocation: {short}")
    selected = [site for state in states for site in by_state[state][:quotas[state]]]

    all_rows = [row for site_id in selected for row in candidate_rows[site_id]]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "train_manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    summary = {
        "purpose": "50-site state-balanced ConvGRU magnitude/generalization test",
        "months": args.months,
        "sequence_length": args.sequence_length,
        "seed": args.seed,
        "external_held_out_sites": sorted(held_out),
        "selected_sites": sorted(selected),
        "sites_by_state": {state: sorted(site for site in selected if site.startswith(state)) for state in states},
        "site_counts_by_state": dict(sorted(Counter(site[:2] for site in selected).items())),
        "eligible_sequences": len(all_rows),
        "sequences_by_month": dict(sorted(Counter(int(row["target_date"][5:7]) for row in all_rows).items())),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
