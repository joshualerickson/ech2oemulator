#!/usr/bin/env python3
"""Merge non-overlapping Phase 2 manifest chunks with integrity checks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=30)
    args = parser.parse_args()
    prefix = f"sequence_{args.sequence_length}_sites_"
    summaries = sorted(args.chunk_dir.glob(f"{prefix}*_summary.json"))
    if not summaries:
        raise FileNotFoundError(f"No chunk summaries in {args.chunk_dir}")

    daily_rows: list[dict[str, str]] = []
    manifest_rows: list[dict[str, str]] = []
    selected_site_count = 0
    for summary_path in summaries:
        stem = summary_path.name.removesuffix("_summary.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        daily_rows.extend(read_rows(args.chunk_dir / f"{stem}_daily_screen.csv"))
        manifest_rows.extend(read_rows(args.chunk_dir / f"{stem}_manifest.csv"))
        selected_site_count += int(summary["source_valid_sites_selected"])

    daily_keys = [(row["site_id"], row["date"]) for row in daily_rows]
    manifest_keys = [(row["site_id"], row["target_date"]) for row in manifest_rows]
    if len(daily_keys) != len(set(daily_keys)):
        raise ValueError("Duplicate site/date rows across daily-screen chunks")
    if len(manifest_keys) != len(set(manifest_keys)):
        raise ValueError("Duplicate site/target-date rows across manifest chunks")
    daily_rows.sort(key=lambda row: (row["site_id"], row["date"]))
    manifest_rows.sort(key=lambda row: (row["site_id"], row["target_date"]))
    write_rows(args.output_dir / "daily_screen.csv", daily_rows)
    write_rows(args.output_dir / "manifest.csv", manifest_rows)
    summary = {
        "chunk_count": len(summaries),
        "source_valid_sites_selected": selected_site_count,
        "daily_row_count": len(daily_rows),
        "manifest_row_count": len(manifest_rows),
        "daily_forcing_invalid_count": sum(row["forcing_valid"] == "False" for row in daily_rows),
        "daily_target_qa_invalid_count": sum(row["target_qa_valid"] == "False" for row in daily_rows),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
