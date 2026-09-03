#!/usr/bin/env python3
"""Run resumable, independent 30 m fire predictions across CPU workers.

The successful 240 m output directory is the authoritative eligibility list.
Each task launches a fresh process so memory from unusually large fires is
returned to the operating system before the worker accepts another site.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import date
from pathlib import Path

import h5py
import numpy as np


OUTPUT_SUFFIX_240M = "_seq90_lstm_predictions.nc"


def audit_240m_output(path: Path) -> dict[str, object]:
    site = path.name[: -len(OUTPUT_SUFFIX_240M)]
    record: dict[str, object] = {"site_id": site, "path": str(path)}
    try:
        with h5py.File(path, "r") as dataset:
            variable = dataset["soilmoisture"]
            fill = variable.attrs.get("_FillValue", -9999.0)
            maximum = 0
            total_pixels = int(variable.shape[-2] * variable.shape[-1])
            # Support can vary by forcing day, so inspect every time slice but
            # retain only counts rather than the full time cube.
            for time_index in range(variable.shape[0]):
                values = variable[time_index]
                count = int(np.count_nonzero(np.isfinite(values) & (values != fill)))
                maximum = max(maximum, count)
        record.update({
            "status": "valid" if maximum else "all_na",
            "max_valid_pixels_per_day": maximum,
            "total_spatial_pixels": total_pixels,
            "max_valid_fraction": maximum / total_pixels if total_pixels else 0.0,
        })
    except Exception as error:
        record.update({"status": "unreadable", "error": str(error)})
    return record


def successful_sites(path: Path, requested: set[str] | None = None) -> tuple[list[str], list[dict[str, object]]]:
    """Audit all 240 m outputs, or only an explicit recovery subset."""
    outputs = sorted(
        item for item in path.glob(f"*{OUTPUT_SUFFIX_240M}")
        if item.is_file()
        and item.stat().st_size > 0
        and (requested is None or item.name[: -len(OUTPUT_SUFFIX_240M)] in requested)
    )
    with ThreadPoolExecutor(max_workers=min(32, max(1, len(outputs)))) as executor:
        audit = list(executor.map(audit_240m_output, outputs))
    # Old 240 m all-NA results are reported but not excluded: that prediction
    # reader had a one-day sequence-index error, and native 30 m support can
    # also recover pixels absent from the coarse intersection. The corrected
    # 30 m run performs its own support check before writing.
    sites = sorted(str(record["site_id"]) for record in audit if record["status"] != "unreadable")
    return sites, audit


def prediction_day_counts(path: Path, buffer_days: int) -> dict[str, int]:
    result: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            try:
                start = date.fromisoformat(row["viirs_start_date"])
                end = date.fromisoformat(row["viirs_end_date"])
                result[row["event_id"]] = (end - start).days + 1 + buffer_days * 2
            except (KeyError, ValueError):
                continue
    return result


def workload(site_dir: Path, days: int) -> int:
    """Return a relative fine-grid pixel-day workload for queue ordering."""
    try:
        with (site_dir / "Spatial" / "dem.asc").open() as handle:
            width = int(handle.readline().split()[-1]) - 2
            height = int(handle.readline().split()[-1]) - 2
        return max(width, 0) * max(height, 0) * 64 * days
    except (FileNotFoundError, IndexError, ValueError):
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--native-static-root", type=Path, required=True)
    parser.add_argument("--eligible-240m-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--viirs-date-csv", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--site-list", type=Path,
        help="Optional newline-delimited subset of successful 240 m site IDs. "
             "Use this to run a recovery cohort without scheduling every 240 m output.",
    )
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--threads-per-worker", type=int, default=24)
    parser.add_argument("--reproject-threads-per-worker", type=int, default=2)
    parser.add_argument("--forcing-channel-workers", type=int, default=3)
    parser.add_argument("--date-buffer-days", type=int, default=5)
    parser.add_argument("--resolution", type=float, default=30.0)
    parser.add_argument("--tpi-radius-cells", type=int, default=40)
    parser.add_argument("--methods", nargs="+", default=("direct_30m",),
                        choices=("direct_30m", "spline_30m", "static_guided_30m"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--heartbeat-seconds", type=float, default=60.0,
                        help="Report active sites when no fire has finished within this interval.")
    args = parser.parse_args()
    if any(value < 1 for value in (
        args.workers, args.threads_per_worker, args.reproject_threads_per_worker,
        args.forcing_channel_workers, args.progress_every, args.heartbeat_seconds,
    )):
        parser.error("worker and thread counts and progress-every must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")

    methods = tuple(dict.fromkeys(args.methods))
    requested: list[str] | None = None
    if args.site_list:
        requested = list(dict.fromkeys(
            line.strip() for line in args.site_list.read_text(encoding="utf-8").splitlines() if line.strip()
        ))
    sites, support_audit = successful_sites(
        args.eligible_240m_dir, set(requested) if requested is not None else None,
    )
    if requested is not None:
        available = set(sites)
        missing_240m = [site for site in requested if site not in available]
        if missing_240m:
            parser.error(
                f"{len(missing_240m)} requested sites lack a readable 240 m prediction; "
                f"first: {missing_240m[:5]}"
            )
        sites = sorted(set(requested))
    all_na = [record for record in support_audit if record["status"] == "all_na"]
    unreadable = [record for record in support_audit if record["status"] == "unreadable"]
    absent = [site for site in sites if not (args.input_root / site).is_dir()]
    if absent:
        parser.error(f"{len(absent)} eligible sites are absent from NVMe input root; first: {absent[:5]}")
    days = prediction_day_counts(args.viirs_date_csv, args.date_buffer_days)
    missing_dates = [site for site in sites if site not in days]
    if missing_dates:
        parser.error(f"{len(missing_dates)} eligible sites lack VIIRS dates; first: {missing_dates[:5]}")

    expected = lambda site, method: args.output_dir / f"{site}_{method}.nc"
    skipped = [site for site in sites if not args.overwrite and all(expected(site, method).is_file() and expected(site, method).stat().st_size > 0 for method in methods)]
    pending = [site for site in sites if site not in set(skipped)]
    ordered = sorted(pending, key=lambda site: workload(args.input_root / site, days[site]), reverse=True)
    # Mix large and small fires instead of starting all largest jobs together;
    # this reduces peak RAM while retaining longest-processing-time scheduling.
    pending = []
    left, right = 0, len(ordered) - 1
    while left <= right:
        pending.append(ordered[left])
        left += 1
        if left <= right:
            pending.append(ordered[right])
            right -= 1
    if args.limit is not None:
        pending = pending[: args.limit]
    summary = {
        "written_240m_sites": len(support_audit), "eligible_240m_sites": len(sites),
        "observed_all_na_240m": len(all_na), "excluded_unreadable_240m": len(unreadable),
        "skipped_existing": len(skipped),
        "pending_sites": len(pending), "workers": args.workers,
        "threads_per_worker": args.threads_per_worker, "methods": list(methods),
        "reproject_threads_per_worker": args.reproject_threads_per_worker,
        "forcing_channel_workers": args.forcing_channel_workers,
    }
    print(summary, flush=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.run_dir / "support_audit_240m.csv"
    audit_fields = ("site_id", "status", "max_valid_pixels_per_day", "total_spatial_pixels", "max_valid_fraction", "path", "error")
    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(support_audit)
    if args.dry_run or not pending:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    failure_dir = args.run_dir / "failures"
    failure_dir.mkdir(exist_ok=True)
    metadata_dir = args.run_dir / "site_metadata"
    metadata_dir.mkdir(exist_ok=True)
    status_path = args.run_dir / "status.jsonl"
    repository = Path(__file__).resolve().parents[1]
    lock = threading.Lock()
    active_sites: set[str] = set()
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    def run_site(site: str) -> dict[str, object]:
        command = [
            sys.executable, "-u", "scripts/benchmark_30m_prediction_methods.py",
            "--input-root", str(args.input_root),
            "--native-static-root", str(args.native_static_root),
            "--site-id", site,
            "--checkpoint", str(args.checkpoint),
            "--normalization", str(args.normalization),
            "--viirs-date-csv", str(args.viirs_date_csv),
            "--output-dir", str(args.output_dir),
            "--metadata-dir", str(metadata_dir),
            "--date-buffer-days", str(args.date_buffer_days),
            "--resolution", str(args.resolution),
            "--tpi-radius-cells", str(args.tpi_radius_cells),
            "--threads", str(args.threads_per_worker),
            "--reproject-threads", str(args.reproject_threads_per_worker),
            "--forcing-channel-workers", str(args.forcing_channel_workers),
            "--device", args.device,
            "--reject-all-na",
            "--methods", *methods,
        ]
        with lock:
            active_sites.add(site)
        try:
            completed = subprocess.run(command, cwd=repository, env=environment, capture_output=True, text=True)
        finally:
            with lock:
                active_sites.discard(site)
        record: dict[str, object] = {"site_id": site, "returncode": completed.returncode}
        if completed.returncode == 0:
            record["status"] = "completed"
        elif "No valid 30 m support remains" in completed.stdout + completed.stderr:
            record["status"] = "skipped_no_valid_support"
        else:
            record["status"] = "failed"
            failure_log = failure_dir / f"{site}.log"
            failure_log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
            record["failure_log"] = str(failure_log)
        with lock:
            with status_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        return record

    completed_count = failed_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_site, site): site for site in pending}
        remaining = set(futures)
        while remaining:
            finished, remaining = wait(
                remaining, timeout=args.heartbeat_seconds, return_when=FIRST_COMPLETED,
            )
            if not finished:
                with lock:
                    active = sorted(active_sites)
                print({"status": "running", "finished": completed_count, "total": len(pending),
                       "active_count": len(active), "active_sites": active}, flush=True)
                continue
            for future in finished:
                record = future.result()
                completed_count += 1
                failed_count += record["status"] == "failed"
                if completed_count % args.progress_every == 0 or record["status"] == "failed" or completed_count == len(pending):
                    print({"finished": completed_count, "total": len(pending), "failed": failed_count,
                           "last_site": record["site_id"], "last_status": record["status"]}, flush=True)
    if failed_count:
        raise SystemExit(f"Completed queue with {failed_count} failed sites; inspect {failure_dir}")
    print({"status": "complete", "completed": completed_count, "output_dir": str(args.output_dir)}, flush=True)


if __name__ == "__main__":
    main()
