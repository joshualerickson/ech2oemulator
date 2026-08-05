#!/usr/bin/env python3
"""Convert validation metric CSV tables into a compact month/state JSON report."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    converted = []
    for row in rows:
        converted.append({key: int(value) if key == "count" else float(value) if key not in {"month", "state", "site_id", "target"} else value for key, value in row.items()})
    return converted


def metrics_only(row: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in row.items() if key not in {"month", "state", "site_id", "target"}}


def read_site_audit(path: Path | None) -> dict[str, dict[str, object]]:
    """Read the immutable split covariates used to construct validation strata."""
    if path is None:
        return {}
    fields = {
        "state", "cell_count", "wbdef_median", "twi_median", "tpi_median",
        "wbdef_median_bin", "cell_count_bin", "tpi_median_bin", "twi_median_bin",
    }
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, dict[str, object]] = {}
    for row in rows:
        result[row["site_id"]] = {
            name: (int(row[name]) if name.endswith("_bin") or name == "cell_count"
                   else float(row[name]) if name.endswith("_median") else row[name])
            for name in fields
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--site-audit", type=Path, default=None,
                        help="Persisted site split audit with climate/terrain/bbox covariates.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    destination = args.output or args.metrics_dir / "month_state_summary.json"
    overall = read_rows(args.metrics_dir / "by_target.csv")
    by_state = read_rows(args.metrics_dir / "by_state_target.csv")
    by_month = read_rows(args.metrics_dir / "by_month_target.csv")
    by_month_state = read_rows(args.metrics_dir / "by_month_state_target.csv")
    by_site = read_rows(args.metrics_dir / "by_site_target.csv")
    audit = read_site_audit(args.site_audit)
    payload: dict[str, object] = {
        "description": "Pixel-weighted validation metrics; nested by calendar month, state, and target.",
        "overall_by_target": {str(row["target"]): metrics_only(row) for row in overall},
        "by_state": {},
        "by_month": {},
        "site_diagnostics": [],
    }
    states_overall: dict[str, object] = payload["by_state"]  # type: ignore[assignment]
    for row in by_state:
        state, target = str(row["state"]), str(row["target"])
        states_overall.setdefault(state, {})[target] = metrics_only(row)  # type: ignore[index]
    months: dict[str, dict[str, object]] = payload["by_month"]  # type: ignore[assignment]
    for row in by_month:
        month = str(row["month"])
        months.setdefault(month, {"overall_by_target": {}, "by_state": {}})
        months[month]["overall_by_target"][str(row["target"])] = metrics_only(row)  # type: ignore[index]
    for row in by_month_state:
        month, state, target = str(row["month"]), str(row["state"]), str(row["target"])
        states: dict[str, object] = months[month]["by_state"]  # type: ignore[index,assignment]
        states.setdefault(state, {})[target] = metrics_only(row)  # type: ignore[index]
    diagnostics: list[dict[str, object]] = payload["site_diagnostics"]  # type: ignore[assignment]
    for row in by_site:
        site_id = str(row["site_id"])
        if site_id not in audit:
            continue
        diagnostics.append({
            "site_id": site_id,
            "target": row["target"],
            "metrics": metrics_only(row),
            "context": audit[site_id],
        })
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
