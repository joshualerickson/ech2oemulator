#!/usr/bin/env python3
"""Combine the selected training/validation cohort and external panel for final fitting.

Model selection must happen before this step.  The resulting manifest contains
only ``final_fit_train`` rows, so it cannot be used for an unbiased validation
metric.  It is the deployment-training cohort for the chosen architecture and
lookback after the spatial validation and external panel have done their jobs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")
        return reader.fieldnames, list(reader)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-manifest", type=Path, required=True,
                        help="Persisted train/validation Seq90 model-selection manifest.")
    parser.add_argument("--external-manifest", type=Path, required=True,
                        help="Corrected Seq90 external panel to incorporate after evaluation.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--final-split", default="final_fit_train")
    parser.add_argument("--months", type=int, nargs="+", default=[6, 7, 8, 9],
                        help="Target months retained for the final fit (default: June--September).")
    args = parser.parse_args()

    fields, selected_all = read_manifest(args.selected_manifest)
    external_fields, external_all = read_manifest(args.external_manifest)
    if fields != external_fields:
        raise ValueError("Selected and external manifests have different schema/order")
    required = {"site_id", "split", "sequence_length", "temporal_contract", "target_date"}
    missing = required - set(fields)
    if missing:
        raise ValueError(f"Manifest missing required fields: {sorted(missing)}")
    allowed_months = set(args.months)
    if not allowed_months or not allowed_months <= set(range(1, 13)):
        raise ValueError("--months must contain calendar months in 1..12")
    selected = [row for row in selected_all if int(row["target_date"][5:7]) in allowed_months]
    external = [row for row in external_all if int(row["target_date"][5:7]) in allowed_months]
    if not selected or not external:
        raise ValueError("Month filtering removed an input cohort; check --months and manifests")
    lengths = {row["sequence_length"] for row in selected + external}
    contracts = {row["temporal_contract"] for row in selected + external}
    if len(lengths) != 1 or len(contracts) != 1:
        raise ValueError(f"Inputs disagree on sequence length or temporal contract: {lengths=}, {contracts=}")

    selected_sites = {row["site_id"] for row in selected}
    external_sites = {row["site_id"] for row in external}
    overlap = selected_sites & external_sites
    if overlap:
        raise ValueError(f"External panel overlaps selected cohort: {sorted(overlap)}")
    rows = [{**row, "split": args.final_split} for row in selected + external]
    if not rows:
        raise ValueError("No rows supplied")
    rows.sort(key=lambda row: (row["site_id"], row["target_date"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "purpose": "final deployment fit after Seq90 model selection",
        "warning": "All rows are training rows; do not report metrics from this manifest as validation performance.",
        "selected_manifest": str(args.selected_manifest),
        "external_manifest": str(args.external_manifest),
        "final_split": args.final_split,
        "target_months": sorted(allowed_months),
        "sequence_length": int(next(iter(lengths))),
        "temporal_contract": next(iter(contracts)),
        "site_count": len(selected_sites | external_sites),
        "selected_cohort_site_count": len(selected_sites),
        "external_panel_site_count": len(external_sites),
        "row_count": len(rows),
        "rows_by_state": dict(sorted(Counter(row["site_id"][:2] for row in rows).items())),
        "rows_by_month": dict(sorted(Counter(row["target_date"][5:7] for row in rows).items())),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
