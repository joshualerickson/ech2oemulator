#!/usr/bin/env python3
"""Resolve a completed RGB review packet into a versioned QA override policy.

The review index defines the exact cohort the reviewer inspected.  Sites in
the bad-site list become whole-site exclusions; every other site in that index
becomes an explicit whole-site keep. Existing manual overrides are preserved.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def site_ids(path: Path) -> set[str]:
    return {
        line.strip() for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-overrides", type=Path, required=True)
    parser.add_argument("--review-index", type=Path, required=True)
    parser.add_argument("--bad-sites", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.base_overrides.open(newline="") as handle:
        base = list(csv.DictReader(handle))
    with args.review_index.open(newline="") as handle:
        reviewed = {row["site_id"] for row in csv.DictReader(handle)}
    bad = site_ids(args.bad_sites)
    unknown_bad = bad - reviewed
    if unknown_bad:
        parser.error(f"bad-site IDs are absent from the reviewed packet: {sorted(unknown_bad)}")
    by_site = {row["site_id"]: row for row in base if row["target"] == "*"}
    rows = list(base)
    for site_id in sorted(reviewed):
        action = "exclude_site_all_targets" if site_id in bad else "keep_site_all_targets"
        review_row = {
            "site_id": site_id,
            "target": "*",
            "override_action": action,
            "reason": "reviewed RGB artifact" if site_id in bad else "approved RGB artifact review",
            "reviewer_note": "Excluded from v3 full review queue" if site_id in bad else "No visually disqualifying artifact in v3 full review queue",
        }
        existing = by_site.get(site_id)
        if existing is not None:
            # Existing deliberate exclusions take precedence unless this exact
            # review explicitly rejects the site as well.
            if existing["override_action"] == "exclude_site_all_targets" and site_id not in bad:
                continue
            rows = [row for row in rows if not (row["site_id"] == site_id and row["target"] == "*")]
        rows.append(review_row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["site_id", "target", "override_action", "reason", "reviewer_note"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["site_id"], row["target"])))
    print({"reviewed_sites": len(reviewed), "approved": len(reviewed - bad), "excluded": len(bad), "base_override_rows": len(base), "output": str(args.output)})


if __name__ == "__main__":
    main()
