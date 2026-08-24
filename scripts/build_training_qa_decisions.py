#!/usr/bin/env python3
"""Create training-ready site and target QA decisions from source and artifact QA.

The site CSV is for split builders. The target CSV is for a future loader/mask
policy, since an artifact can invalidate Tskin while leaving soil moisture and
PLC usable. No automated artifact candidate is silently converted into a final
whole-site exclusion.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def targets_with_missing_corner_median(path: Path | None) -> dict[str, list[str]]:
    """Return targets whose corner-jump statistic is undefined for every day.

    This is distinct from a large corner jump: an absent median means that no
    valid corner-to-interior comparison exists across the audited period.  It
    is a persistent target-support failure for full-bbox training.
    """
    if path is None:
        return {}
    values: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in read_csv(path):
        values[(row["site_id"], row["target"])].append(row.get("corner_jump_ratio", ""))
    flagged: dict[str, list[str]] = defaultdict(list)
    for (site_id, target), scores in values.items():
        if scores and not any(value not in {"", None} for value in scores):
            flagged[site_id].append(target)
    return {site_id: sorted(targets) for site_id, targets in flagged.items()}


def forcing_summary(daily_dir: Path | None, site_id: str) -> dict[str, int | str]:
    if daily_dir is None:
        return {"forcing_daily_status": "pending_daily_forcing_qa", "jan_sep_forcing_unavailable_days": 0, "jan_sep_forcing_invalid_days": 0, "jan_sep_forcing_edge_issue_days": 0}
    paths = list(daily_dir.glob(f"{site_id}_*_daily_screen.csv"))
    if len(paths) != 1:
        return {"forcing_daily_status": "missing_daily_forcing_qa", "jan_sep_forcing_unavailable_days": 0, "jan_sep_forcing_invalid_days": 0, "jan_sep_forcing_edge_issue_days": 0}
    rows = [row for row in read_csv(paths[0]) if date.fromisoformat(row["date"]).month in range(1, 10)]
    unavailable = sum(row.get("forcing_available") != "True" for row in rows)
    invalid = sum(row.get("forcing_available") == "True" and row.get("forcing_valid") != "True" for row in rows)
    edge = sum(int(row.get("forcing_edge_invalid_pixel_count") or 0) > 0 for row in rows)
    status = "forcing_clean" if not unavailable and not invalid else "forcing_day_masks_required"
    return {"forcing_daily_status": status, "jan_sep_forcing_unavailable_days": unavailable, "jan_sep_forcing_invalid_days": invalid, "jan_sep_forcing_edge_issue_days": edge}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema-report", type=Path, required=True)
    parser.add_argument("--triage", type=Path, required=True)
    parser.add_argument("--daily-dir", type=Path, default=None, help="Optional v2 daily screens; adds Jan--Sep forcing QA counts to the site CSV.")
    parser.add_argument("--daily-artifact-audit", type=Path, default=None, help="Raw daily artifact audit; sites with an undefined per-target corner-jump median are excluded for full-bbox training.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    schema = json.loads(args.schema_report.read_text())
    triage = read_csv(args.triage)
    by_site: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in triage:
        by_site[row["site_id"]].append(row)
    missing_corner_targets = targets_with_missing_corner_median(args.daily_artifact_audit)

    site_rows = []
    target_rows = []
    for source in sorted(schema["sites"], key=lambda row: row["site_id"]):
        site_id = source["site_id"]
        issues = source.get("issues", [])
        target_decisions = by_site.get(site_id, [])
        corner_targets = missing_corner_targets.get(site_id, [])
        if issues:
            status, reason = "exclude_source_contract", "; ".join(issues)
        elif corner_targets:
            status, reason = "exclude_target_corner_support_missing", f"undefined corner_jump_ratio median for: {', '.join(corner_targets)}"
        elif any(row["action"] == "exclude_manual" for row in target_decisions):
            status, reason = "exclude_manual", "reviewed manual whole-site exclusion"
        elif any(row["action"] == "exclude_candidate_persistent_tskin" for row in target_decisions):
            status, reason = "hold_for_review", "persistent automated Tskin candidate remains unreviewed"
        elif any(row["action"] in {"review_days", "review_timeline"} for row in target_decisions):
            status, reason = "include_candidate_with_target_review", "site is source-valid; one or more target channels need artifact review"
        elif target_decisions:
            status, reason = "include_candidate", "source-valid and no site-level artifact concern"
        else:
            status, reason = "pending_artifact_audit", "source-valid but no artifact triage row"
        forcing = forcing_summary(args.daily_dir, site_id)
        site_rows.append({
            "site_id": site_id,
            "water_year_end": source.get("water_year_end", ""),
            "training_site_status": status,
            "reason": reason,
            "source_contract_issues": ";".join(issues),
            "target_rows": len(target_decisions),
            "target_actions": ";".join(sorted({row["action"] for row in target_decisions})),
            "targets_with_missing_corner_jump_median": ";".join(corner_targets),
            **forcing,
        })
        for row in target_decisions:
            effective_status = (
                status if status.startswith("exclude") or status == "include_candidate"
                else row["action"]
            )
            target_rows.append({
                "site_id": site_id,
                "water_year_end": source.get("water_year_end", ""),
                "target": row["target"],
                "target_training_status": row["action"],
                "effective_full_bbox_training_status": effective_status,
                "effective_full_bbox_training_reason": reason if effective_status == status else row["reason"],
                "reason": row["reason"],
                "automated_action": row["automated_action"],
                "manual_override_action": row["manual_override_action"],
                "candidate_day_count": row["candidate_day_count"],
                "candidate_index_date_preview": row["candidate_index_date_preview"],
            })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("training_qa_site_decisions.csv", site_rows), ("training_qa_target_decisions.csv", target_rows)):
        with (args.output_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "purpose": "pre-split training/validation eligibility from source-contract and artifact QA",
        "site_status_counts": dict(sorted(Counter(row["training_site_status"] for row in site_rows).items())),
        "target_status_counts": dict(sorted(Counter(row["target_training_status"] for row in target_rows).items())),
        "site_csv": "training_qa_site_decisions.csv",
        "target_csv": "training_qa_target_decisions.csv",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
