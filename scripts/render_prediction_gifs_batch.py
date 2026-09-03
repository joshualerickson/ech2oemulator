#!/usr/bin/env python3
"""Select completed 30 m predictions by state and render resumable GIF batches."""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path


SUFFIX = "_direct_30m.nc"


def select_sites(input_dir: Path, per_state: int, seed: int) -> list[tuple[str, Path]]:
    groups: dict[str, list[Path]] = {}
    for path in sorted(input_dir.glob(f"*{SUFFIX}")):
        site_id = path.name[: -len(SUFFIX)]
        if len(site_id) < 2 or not site_id[:2].isalpha():
            continue
        groups.setdefault(site_id[:2], []).append(path)
    insufficient = {state: len(paths) for state, paths in groups.items() if len(paths) < per_state}
    if insufficient:
        raise ValueError(f"States with fewer than {per_state} completed direct outputs: {insufficient}")
    generator = random.Random(seed)
    selected: list[tuple[str, Path]] = []
    for state in sorted(groups):
        selected.extend((state, path) for path in sorted(generator.sample(groups[state], per_state)))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-state", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.per_state < 1 or args.workers < 1:
        parser.error("per-state and workers must be positive")

    selected = select_sites(args.input_dir, args.per_state, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / f"selection_{args.per_state}_per_state_seed_{args.seed}.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("state", "site_id", "input_netcdf", "output_dir"))
        writer.writeheader()
        for state, path in selected:
            site_id = path.name[: -len(SUFFIX)]
            writer.writerow({"state": state, "site_id": site_id, "input_netcdf": str(path), "output_dir": str(args.output_dir / state / site_id)})

    renderer = Path(__file__).with_name("render_prediction_gifs.py")
    status_path = args.output_dir / f"render_status_{args.per_state}_per_state_seed_{args.seed}.jsonl"

    def render(state: str, path: Path) -> dict[str, object]:
        site_id = path.name[: -len(SUFFIX)]
        command = [
            sys.executable, "-u", str(renderer), "--input", str(path),
            "--output-dir", str(args.output_dir / state / site_id),
            "--fps", str(args.fps), "--dpi", str(args.dpi),
        ]
        if args.overwrite:
            command.append("--overwrite")
        completed = subprocess.run(command, text=True, capture_output=True)
        return {
            "state": state, "site_id": site_id, "input_netcdf": str(path),
            "returncode": completed.returncode,
            "status": "completed" if completed.returncode == 0 else "failed",
            "stdout": completed.stdout if completed.returncode else "",
            "stderr": completed.stderr if completed.returncode else "",
        }

    print(json.dumps({"selected_sites": len(selected), "states": sorted({state for state, _ in selected}), "manifest": str(manifest)}), flush=True)
    completed_count = failed_count = 0
    with status_path.open("w", encoding="utf-8") as status_handle, ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(render, state, path): (state, path) for state, path in selected}
        remaining = set(futures)
        while remaining:
            done, remaining = wait(remaining, return_when=FIRST_COMPLETED)
            for future in done:
                record = future.result()
                completed_count += 1
                failed_count += record["status"] == "failed"
                status_handle.write(json.dumps(record) + "\n")
                status_handle.flush()
                print({"finished": completed_count, "total": len(selected), "failed": failed_count,
                       "site_id": record["site_id"], "status": record["status"]}, flush=True)
    if failed_count:
        raise SystemExit(f"{failed_count} site render(s) failed; inspect {status_path}")


if __name__ == "__main__":
    main()
