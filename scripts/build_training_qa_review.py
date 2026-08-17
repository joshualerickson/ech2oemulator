#!/usr/bin/env python3
"""Create compact include/exclude/review lists from training QA decisions."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-decisions", type=Path, required=True)
    parser.add_argument("--target-decisions", type=Path, required=True)
    parser.add_argument("--triage", type=Path, required=True, help="Artifact triage CSV containing scores and date previews.")
    parser.add_argument("--daily-artifact-audit", type=Path, required=True, help="Daily artifact CSV; supplies all metric distributions by site and target.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with args.site_decisions.open(newline="") as handle:
        sites = list(csv.DictReader(handle))
    with args.target_decisions.open(newline="") as handle:
        targets = list(csv.DictReader(handle))
    with args.triage.open(newline="") as handle:
        triage = list(csv.DictReader(handle))
    with args.daily_artifact_audit.open(newline="") as handle:
        daily_artifacts = list(csv.DictReader(handle))
    by_site: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in targets:
        by_site[row["site_id"]].append(row)
    triage_by_site: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in triage:
        triage_by_site[row["site_id"]].append(row)
    metric_names = ("legacy_blockiness", "row_seam_ratio", "column_seam_ratio", "edge_gradient_ratio", "corner_jump_ratio", "high_frequency_ratio")
    metric_values: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in daily_artifacts:
        for metric in metric_names:
            value = row.get(metric, "")
            if value not in {"", None}:
                metric_values[(row["site_id"], row["target"])][metric].append(float(value))

    def summary(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"median": None, "p95": None, "max": None}
        array = np.asarray(values, dtype=float)
        return {"median": float(np.median(array)), "p95": float(np.percentile(array, 95)), "max": float(np.max(array))}

    review_rows, include_rows, exclude_rows, target_stat_rows = [], [], [], []
    for row in sites:
        target_rows = by_site.get(row["site_id"], [])
        artifact_rows = triage_by_site.get(row["site_id"], [])
        per_target_stats = {
            target: {metric: summary(metric_values[(row["site_id"], target)][metric]) for metric in metric_names}
            for target in sorted({artifact["target"] for artifact in artifact_rows})
        }
        target_decision_by_name = {target_row["target"]: target_row for target_row in target_rows}
        triage_by_target = {artifact["target"]: artifact for artifact in artifact_rows}
        for target, metrics in per_target_stats.items():
            decision = target_decision_by_name.get(target, {})
            artifact = triage_by_target.get(target, {})
            target_stat_rows.append({
                "site_id": row["site_id"], "water_year_end": row["water_year_end"],
                "site_status": row["training_site_status"], "target": target,
                "target_artifact_action": artifact.get("action", ""),
                "target_automated_action": artifact.get("automated_action", ""),
                "candidate_day_count": artifact.get("candidate_day_count", ""),
                "candidate_index_date_preview": artifact.get("candidate_index_date_preview", ""),
                **{f"{metric}_{statistic}": metrics[metric][statistic] for metric in metric_names for statistic in ("median", "p95", "max")},
            })
        def site_metric_max(metric: str, statistic: str) -> float | str:
            values = [stats[metric][statistic] for stats in per_target_stats.values() if stats[metric][statistic] is not None]
            return max(values) if values else ""
        target_action_summary = "; ".join(
            f"{target['target']}={target['target_training_status']}"
            for target in sorted(target_rows, key=lambda item: item["target"])
            if target["target_training_status"] not in {"keep_candidate", "keep_manual", "keep_candidate_geometry_low_confidence"}
        )
        compact = {
            "site_id": row["site_id"],
            "water_year_end": row["water_year_end"],
            "site_status": row["training_site_status"],
            "reason": row["reason"],
            "target_actions_requiring_attention": target_action_summary,
            "artifact_target_actions_all": "; ".join(
                f"{artifact['target']}={artifact['action']}" for artifact in sorted(artifact_rows, key=lambda item: item["target"])
            ),
            "artifact_targets_requiring_attention": "; ".join(
                f"{artifact['target']}={artifact['action']}" for artifact in sorted(artifact_rows, key=lambda item: item["target"])
                if artifact["action"] not in {"keep_candidate", "keep_manual", "keep_candidate_geometry_low_confidence"}
            ),
            "artifact_candidate_days_total": sum(int(artifact["candidate_day_count"]) for artifact in artifact_rows),
            "artifact_candidate_days_max_target": max((int(artifact["candidate_day_count"]) for artifact in artifact_rows), default=0),
            "artifact_max_row_seam_p95": max((float(artifact["row_seam_p95"]) for artifact in artifact_rows if artifact.get("row_seam_p95") not in {"", None}), default=""),
            "artifact_date_preview": " | ".join(
                f"{artifact['target']}:{artifact['candidate_index_date_preview']}"
                for artifact in sorted(artifact_rows, key=lambda item: item["target"])
                if artifact.get("candidate_index_date_preview")
            ),
            "artifact_stats_by_target_json": json.dumps(per_target_stats, separators=(",", ":")),
            **{
                f"artifact_{metric}_{statistic}_max": site_metric_max(metric, statistic)
                for metric in metric_names for statistic in ("median", "p95", "max")
            },
            "forcing_qa_status": row.get("forcing_daily_status", "pending_daily_forcing_qa"),
        }
        if row["training_site_status"].startswith("exclude"):
            exclude_rows.append(compact)
        elif row["training_site_status"] == "include_candidate":
            include_rows.append(compact)
        else:
            review_rows.append(compact)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        ("training_qa_exclude_sites.csv", exclude_rows),
        ("training_qa_review_sites.csv", review_rows),
        ("training_qa_include_candidates.csv", include_rows),
    ):
        with (args.output_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=(list(rows[0]) if rows else ["site_id", "site_status"]))
            writer.writeheader()
            writer.writerows(rows)
    with (args.output_dir / "training_qa_site_target_artifact_stats.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(target_stat_rows[0]))
        writer.writeheader()
        writer.writerows(target_stat_rows)
    print({"exclude_sites": len(exclude_rows), "review_sites": len(review_rows), "include_candidates": len(include_rows), "site_status_counts": dict(Counter(row["training_site_status"] for row in sites))})


if __name__ == "__main__":
    main()
