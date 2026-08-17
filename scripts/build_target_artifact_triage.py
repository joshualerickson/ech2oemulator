#!/usr/bin/env python3
"""Convert a raw target-artifact audit into a conservative review queue.

The resulting CSV is intentionally a triage aid, not an automatic data-deletion
tool. Tskin uses the curated severe-site seam reference. Other targets require
two extreme score signals before review because low-gradient fields can inflate
ratios; none are proposed for automatic exclusion here.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def numeric(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", None} else float("nan")


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else float("inf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--site-geometry", type=Path, default=None, help="Optional site_split_audit.csv; omit when daily audit already contains target dimensions.")
    parser.add_argument("--overrides", type=Path, default=None, help="Optional reviewed site/target overrides; target '*' applies to all targets at a site.")
    parser.add_argument("--narrow-min-dimension", type=int, default=15, help="At or below this target-grid dimension, suppress raster-wide seam decisions.")
    parser.add_argument("--geometry-aspect-ratio", type=float, default=1.5, help="Minimum long/short ratio for the narrow-grid low-confidence flag.")
    parser.add_argument("--timeline-preview-days", type=int, default=10, help="Candidate index/date pairs retained directly in the main triage CSV.")
    args = parser.parse_args()
    with args.daily_audit.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Daily audit is empty")
    geometry = {}
    if args.site_geometry:
        with args.site_geometry.open(newline="") as handle:
            geometry = {row["site_id"]: row for row in csv.DictReader(handle)}
    overrides: dict[tuple[str, str], dict[str, str]] = {}
    if args.overrides:
        with args.overrides.open(newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row["site_id"], row["target"])
                if key in overrides:
                    raise ValueError(f"Duplicate manual override for {key}")
                if row["override_action"] not in {"exclude_site_all_targets", "keep_site_all_targets", "exclude_target", "keep_target"}:
                    raise ValueError(f"Unsupported override action: {row['override_action']}")
                overrides[key] = row
    if args.narrow_min_dimension < 2 or args.geometry_aspect_ratio < 1:
        parser.error("geometry thresholds must be sensible positive values")

    # Global target-specific thresholds only rank the strongest non-Tskin cases
    # for visual inspection. They are not scientific cutoffs or exclusions.
    thresholds: dict[str, dict[str, float]] = {}
    for target in sorted({row["target"] for row in rows}):
        subset = [row for row in rows if row["target"] == target]
        thresholds[target] = {
            "row_seam_p99_5": percentile([numeric(row, "row_seam_ratio") for row in subset if np.isfinite(numeric(row, "row_seam_ratio"))], 99.5),
            "edge_p99": percentile([numeric(row, "edge_gradient_ratio") for row in subset if np.isfinite(numeric(row, "edge_gradient_ratio"))], 99),
            "corner_p99_9": percentile([numeric(row, "corner_jump_ratio") for row in subset if np.isfinite(numeric(row, "corner_jump_ratio"))], 99.9),
        }

    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["site_id"], row["target"])].append(row)
    triage = []
    candidate_details = []
    for (split, site_id, target), values in sorted(grouped.items()):
        values.sort(key=lambda row: int(row["target_time_index"]))
        threshold = thresholds[target]
        seam_candidates: list[dict[str, str]] = []
        corner_candidates: list[dict[str, str]] = []
        for row in values:
            seam, edge, corner = (numeric(row, name) for name in ("row_seam_ratio", "edge_gradient_ratio", "corner_jump_ratio"))
            if target.startswith("tskin"):
                if seam >= 8.0:
                    seam_candidates.append(row)
            elif seam >= threshold["row_seam_p99_5"] and edge >= threshold["edge_p99"]:
                seam_candidates.append(row)
            if corner >= threshold["corner_p99_9"]:
                corner_candidates.append(row)
        seams = [numeric(row, "row_seam_ratio") for row in values if np.isfinite(numeric(row, "row_seam_ratio"))]
        p50, p95 = percentile(seams, 50), percentile(seams, 95)
        # The persistent-Tskin condition mirrors the known severe reference and
        # is still a *candidate* awaiting visual confirmation.
        if target.startswith("tskin") and len(seam_candidates) >= 80 and p50 >= 6.0 and p95 >= 12.0:
            action = "exclude_candidate_persistent_tskin"
            reason = "persistent severe row-seam pattern; visually confirm before exclusion"
        elif len(seam_candidates) > 2:
            action = "review_timeline"
            reason = "more than two candidate days; inspect temporal sequence rather than isolated maps"
        elif len(seam_candidates):
            action = "review_days"
            reason = "one or two candidate days"
        elif len(corner_candidates):
            action = "review_corner_pixels"
            reason = "localized corner signal; prefer a pixel mask if visually confirmed"
        else:
            action = "keep_candidate"
            reason = "no candidate signal under this provisional triage"
        selected = seam_candidates if seam_candidates else corner_candidates
        grid = geometry.get(site_id)
        if grid is not None:
            height, width = int(grid["height"]), int(grid["width"])
        else:
            if not values[0].get("target_height") or not values[0].get("target_width"):
                raise ValueError(f"{site_id}: no geometry in daily audit and no --site-geometry supplied")
            height, width = int(values[0]["target_height"]), int(values[0]["target_width"])
        min_dimension, aspect_ratio = min(height, width), max(height, width) / min(height, width)
        geometry_low_confidence = min_dimension <= args.narrow_min_dimension and aspect_ratio >= args.geometry_aspect_ratio
        original_action = action
        if geometry_low_confidence and action != "keep_candidate":
            action = "keep_candidate_geometry_low_confidence"
            reason = "narrow/asymmetric target grid can inflate raster-wide seam scores; inspect only if independently indicated"
        preview = selected[:args.timeline_preview_days]
        for row in selected:
            candidate_details.append({
                "split": split, "site_id": site_id, "target": target,
                "candidate_source": "seam" if seam_candidates else "corner",
                "target_time_index": row["target_time_index"], "target_date": row["target_date"],
                "row_seam_ratio": row["row_seam_ratio"], "column_seam_ratio": row["column_seam_ratio"],
                "edge_gradient_ratio": row["edge_gradient_ratio"], "corner_jump_ratio": row["corner_jump_ratio"],
                "geometry_low_confidence": geometry_low_confidence,
            })
        override = overrides.get((site_id, target)) or overrides.get((site_id, "*"))
        effective_action, effective_reason = action, reason
        if override:
            if override["override_action"] in {"exclude_site_all_targets", "exclude_target"}:
                effective_action = "exclude_manual"
            else:
                effective_action = "keep_manual"
            effective_reason = override["reason"]
        triage.append({
            "split": split, "site_id": site_id, "target": target, "action": effective_action, "reason": effective_reason,
            "automated_action": action, "manual_override_action": override["override_action"] if override else "", "manual_override_note": override["reviewer_note"] if override else "",
            "pre_geometry_action": original_action,
            "target_height": height, "target_width": width, "min_target_dimension": min_dimension, "aspect_ratio": aspect_ratio,
            "geometry_low_confidence": geometry_low_confidence,
            "days_scored": len(values), "candidate_day_count": len(selected),
            "candidate_target_time_indices_preview": ";".join(row["target_time_index"] for row in preview),
            "candidate_dates_preview": ";".join(row["target_date"] for row in preview),
            "candidate_index_date_preview": ";".join(f"{row['target_time_index']}={row['target_date']}" for row in preview),
            "candidate_detail_csv": args.output.with_name("target_artifact_candidate_days.csv").name if selected else "",
            "row_seam_p50": p50, "row_seam_p95": p95,
            "target_row_seam_review_threshold": 8.0 if target.startswith("tskin") else threshold["row_seam_p99_5"],
            "target_edge_review_threshold": threshold["edge_p99"],
            "target_corner_review_threshold": threshold["corner_p99_9"],
            "policy_version": "provisional_v1_no_automatic_exclusion",
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(triage[0]))
        writer.writeheader()
        writer.writerows(triage)
    detail_path = args.output.with_name("target_artifact_candidate_days.csv")
    with detail_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidate_details[0]) if candidate_details else ("site_id",))
        writer.writeheader()
        writer.writerows(candidate_details)
    print({"output": str(args.output), "candidate_detail_output": str(detail_path), "rows": len(triage), "candidate_days": len(candidate_details), "overrides": str(args.overrides) if args.overrides else None, "actions": {action: sum(row["action"] == action for row in triage) for action in sorted({row["action"] for row in triage})}})


if __name__ == "__main__":
    main()
