#!/usr/bin/env python3
"""Launch independent, resumable Seq90 prediction workers for fire folders.

Fire grids have variable sizes, so workers partition fires rather than trying
to spatially pad unrelated sites into one batch. Each worker retains normal
within-fire target-date batching. Output file names are site-specific, making
concurrent writes safe.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--netcdf-output-dir", type=Path, required=True)
    parser.add_argument("--viirs-date-csv", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Workspace directory for worker site lists, logs, and merged metadata.")
    parser.add_argument("--site-list", type=Path,
                        help="Optional newline-delimited subset of fire IDs to shard across workers.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads-per-worker", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--date-buffer-days", type=int, default=5)
    parser.add_argument("--overwrite-results-netcdf", action="store_true")
    parser.add_argument("--skip-existing-results-netcdf", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.workers < 1 or args.threads_per_worker < 1 or args.batch_size < 1:
        parser.error("workers, threads-per-worker, and batch-size must be positive")
    if args.overwrite_results_netcdf and args.skip_existing_results_netcdf:
        parser.error("Choose either overwrite or skip existing NetCDFs")

    available = {path.name for path in args.input_root.iterdir() if path.is_dir()}
    if args.site_list:
        sites = list(dict.fromkeys(line.strip() for line in args.site_list.read_text().splitlines() if line.strip()))
        missing = [site for site in sites if site not in available]
        if missing:
            parser.error(f"{len(missing)} requested sites are absent from input-root; first: {missing[:5]}")
    else:
        sites = sorted(available)
    if not sites:
        parser.error("No fire folders found")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    list_dir = args.run_dir / "site_lists"
    log_dir = args.run_dir / "worker_logs"
    list_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    shards = [sites[index::args.workers] for index in range(args.workers)]
    processes: list[subprocess.Popen[bytes]] = []
    for index, shard in enumerate(shards):
        site_list = list_dir / f"worker_{index:02d}.txt"
        site_list.write_text("\n".join(shard) + "\n", encoding="utf-8")
        command = [
            sys.executable, "-u", "scripts/predict_fire_seq90.py",
            "--input-root", str(args.input_root), "--site-list", str(site_list),
            "--checkpoint", str(args.checkpoint), "--normalization", str(args.normalization),
            "--output-dir", str(args.run_dir / f"worker_{index:02d}_metadata"),
            "--write-results-netcdf", "--netcdf-output-dir", str(args.netcdf_output_dir),
            "--viirs-date-csv", str(args.viirs_date_csv), "--date-buffer-days", str(args.date_buffer_days),
            "--threads", str(args.threads_per_worker), "--interop-threads", "1",
            "--batch-size", str(args.batch_size), "--device", args.device,
        ]
        if args.overwrite_results_netcdf:
            command.append("--overwrite-results-netcdf")
        if args.skip_existing_results_netcdf:
            command.append("--skip-existing-results-netcdf")
        log = (log_dir / f"worker_{index:02d}.log").open("wb")
        processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
        print({"worker": index, "site_count": len(shard), "pid": processes[-1].pid}, flush=True)
    exit_codes = [process.wait() for process in processes]
    if any(code != 0 for code in exit_codes):
        raise SystemExit(f"One or more workers failed: {exit_codes}")
    print({"status": "complete", "workers": args.workers, "site_count": len(sites)}, flush=True)


if __name__ == "__main__":
    main()
