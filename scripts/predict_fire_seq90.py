#!/usr/bin/env python3
"""Predict five daily ECH2O fields for fire folders using a fixed-window model.

Forcing GeoTIFF band descriptions must be ISO-like YYYYMMDD dates.  This is a
prediction-only workflow: no ECH2O target NetCDF is read or required.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import h5netcdf
import rasterio
import torch
import xarray as xr
from rasterio.transform import Affine

from src.data.phase2_qc import DYNAMIC_CHANNELS, TARGET_CHANNELS
from src.data.static_contract import Grid, STATIC_CHANNELS, load_selected_static_stack
from src.data.transforms import standardize
from src.models.multitask_model import load_model_state, model_from_checkpoint
from src.utils.runtime import resolve_device, to_device


def spatial_ref_attrs(grid: Grid) -> dict[str, object]:
    """Return the shared CF grid-mapping attributes for prediction NetCDFs."""
    return {
        "grid_mapping_name": "albers_conical_equal_area",
        "standard_parallel": np.asarray([29.5, 45.5], dtype="float64"),
        "latitude_of_projection_origin": 23.0,
        "longitude_of_central_meridian": -96.0,
        "false_easting": 0.0,
        "false_northing": 0.0,
        "semi_major_axis": 6378137.0,
        "inverse_flattening": 298.257222101,
        "longitude_of_prime_meridian": 0.0,
        "spatial_ref": grid.crs.to_wkt(),
        "crs_wkt": grid.crs.to_wkt(),
        "epsg_code": "EPSG:5070",
        "GeoTransform": " ".join(str(value) for value in (
            grid.transform.c, grid.transform.a, grid.transform.b,
            grid.transform.f, grid.transform.d, grid.transform.e,
        )),
    }


class ResultsNetCDFWriter:
    """Incrementally write a prediction cube, then atomically publish it."""

    def __init__(self, path: Path, days: list[date], grid: Grid) -> None:
        if not days:
            raise ValueError("Cannot create a prediction NetCDF without dates")
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.stem}.", suffix=".tmp.nc", dir=path.parent,
        )
        os.close(descriptor)
        self.temporary_path = Path(temporary_name)
        self.dataset = h5netcdf.File(self.temporary_path, "w")
        self.dataset.dimensions = {"time": len(days), "y": grid.height, "x": grid.width}
        self.dataset.attrs.update({
            "title": "ECH2O emulator Seq90 predictions",
            "prediction_only": "true; no ECH2O target NetCDF was read",
            "crs_wkt": grid.crs.to_wkt(),
            "spatial_resolution_x": grid.transform.a,
            "spatial_resolution_y": grid.transform.e,
            "sequence_length_days": 90,
            "temporal_protocol": "causal fixed window ending on each output date",
        })
        time = self.dataset.create_variable("time", ("time",), dtype="int64")
        # Match the established 240 m xarray serialization exactly. Several
        # GIS clients expose raw coordinate values, so a Unix-epoch origin
        # would appear as 15,000--20,000 instead of the expected 0, 1, 2, ... .
        origin = days[0]
        time[:] = np.asarray([(day - origin).days for day in days], dtype="int64")
        time.attrs.update({
            "units": f"days since {origin.isoformat()} 00:00:00",
            "calendar": "proleptic_gregorian",
        })
        y = self.dataset.create_variable("y", ("y",), dtype="float64")
        y[:] = grid.transform.f + grid.transform.e * (np.arange(grid.height) + 0.5)
        y.attrs.update({"standard_name": "projection_y_coordinate", "long_name": "y coordinate of projection", "units": "m"})
        x = self.dataset.create_variable("x", ("x",), dtype="float64")
        x[:] = grid.transform.c + grid.transform.a * (np.arange(grid.width) + 0.5)
        x.attrs.update({"standard_name": "projection_x_coordinate", "long_name": "x coordinate of projection", "units": "m"})
        spatial_ref = self.dataset.create_variable("spatial_ref", (), dtype="int32")
        spatial_ref[...] = np.int32(0)
        spatial_ref.attrs.update(spatial_ref_attrs(grid))
        chunks = (1, min(grid.height, 256), min(grid.width, 256))
        self.variables = {}
        for name in TARGET_CHANNELS:
            variable = self.dataset.create_variable(
                name, ("time", "y", "x"), dtype="float32", fillvalue=-9999.0,
                chunks=chunks, compression="gzip", compression_opts=4,
            )
            variable.attrs["grid_mapping"] = "spatial_ref"
            self.variables[name] = variable
        self.closed = False

    def write(self, time_index: int, values: np.ndarray, support: np.ndarray) -> None:
        for channel, name in enumerate(TARGET_CHANNELS):
            self.variables[name][time_index] = np.where(
                support, values[channel], -9999.0,
            ).astype("float32")

    def close(self, publish: bool = True) -> None:
        if self.closed:
            return
        self.dataset.close()
        self.closed = True
        if publish:
            os.replace(self.temporary_path, self.path)
            # Prediction products are intentionally collaborative artifacts:
            # preserve open read/write access after atomic publication rather
            # than retaining the creator-only mode from mkstemp().
            self.path.chmod(0o666)
            # GDAL/QGIS PAM sidecars cache coordinate metadata and statistics.
            # An older sidecar can override a corrected NetCDF time axis.
            Path(f"{self.path}.aux.xml").unlink(missing_ok=True)
        else:
            self.temporary_path.unlink(missing_ok=True)


def dated_bands(path: Path) -> tuple[list[date], rasterio.DatasetReader]:
    dataset = rasterio.open(path)
    descriptions = dataset.descriptions
    try:
        dates = [date.fromisoformat(f"{value[:4]}-{value[4:6]}-{value[6:8]}") for value in descriptions if value]
    except (IndexError, ValueError) as error:
        dataset.close()
        raise ValueError(f"{path} requires YYYYMMDD band descriptions") from error
    if len(dates) != dataset.count or any(next_day - current != timedelta(days=1) for current, next_day in zip(dates, dates[1:])):
        dataset.close()
        raise ValueError(f"{path} has missing, undescribed, or non-contiguous daily forcing bands")
    return dates, dataset


def write_prediction(path: Path, values: np.ndarray, grid: Grid, support: np.ndarray) -> None:
    profile = {
        "driver": "GTiff", "width": grid.width, "height": grid.height, "count": len(TARGET_CHANNELS),
        "dtype": "float32", "crs": grid.crs, "transform": grid.transform, "nodata": -9999.0,
        "compress": "deflate", "tiled": False,
    }
    with rasterio.open(path, "w", **profile) as output:
        for band, (name, field) in enumerate(zip(TARGET_CHANNELS, values), start=1):
            output.write(np.where(support, field, -9999.0).astype("float32"), band)
            output.set_band_description(band, name)


def write_results_netcdf(path: Path, values: np.ndarray, days: list[date], grid: Grid) -> None:
    """Write prediction-only fields using a compact CF-style spatial layout."""
    y = grid.transform.f + grid.transform.e * (np.arange(grid.height) + 0.5)
    x = grid.transform.c + grid.transform.a * (np.arange(grid.width) + 0.5)
    crs = grid.crs
    spatial_ref = xr.DataArray(np.int32(0), attrs=spatial_ref_attrs(grid))
    dataset = xr.Dataset(
        data_vars={
            name: xr.DataArray(
                values[:, channel].astype("float32"),
                dims=("time", "y", "x"),
                attrs={"grid_mapping": "spatial_ref"},
            )
            for channel, name in enumerate(TARGET_CHANNELS)
        },
        coords={
            "time": np.asarray(days, dtype="datetime64[ns]"),
            "y": xr.DataArray(y, dims="y", attrs={"standard_name": "projection_y_coordinate", "long_name": "y coordinate of projection", "units": "m"}),
            "x": xr.DataArray(x, dims="x", attrs={"standard_name": "projection_x_coordinate", "long_name": "x coordinate of projection", "units": "m"}),
        },
        attrs={
            "title": "ECH2O emulator Seq90 predictions",
            "prediction_only": "true; no ECH2O target NetCDF was read",
            "crs_wkt": grid.crs.to_wkt(),
            "spatial_resolution_x": grid.transform.a,
            "spatial_resolution_y": grid.transform.e,
            "sequence_length_days": 90,
            "temporal_protocol": "causal fixed window ending on each output date",
        },
    )
    dataset["spatial_ref"] = spatial_ref
    encoding = {name: {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": -9999.0} for name in TARGET_CHANNELS}
    # HDF5 can refuse to truncate ``path`` when a GIS client has the existing
    # NetCDF open.  Write beside it and atomically replace it only after the
    # new file is complete; this also prevents a failed write leaving a
    # zero-byte or partially written prediction at the final path.
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp.nc", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        dataset.to_netcdf(temporary_path, engine="h5netcdf", encoding=encoding)
        os.replace(temporary_path, path)
        # ``mkstemp`` creates a 0600 file.  The final prediction directory is
        # deliberately shared, so publish NetCDFs with group/other read-write
        # access as well.
        path.chmod(0o666)
    finally:
        temporary_path.unlink(missing_ok=True)


def read_viirs_windows(path: Path) -> dict[str, tuple[date, date]]:
    """Return one authoritative inclusive VIIRS interval per MTBS event/site."""
    windows: dict[str, tuple[date, date]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            site_id = row.get("event_id", "")
            start_text, end_text = row.get("viirs_start_date", ""), row.get("viirs_end_date", "")
            if not site_id or not start_text or not end_text:
                continue
            start, end = date.fromisoformat(start_text), date.fromisoformat(end_text)
            if end < start:
                raise ValueError(f"{path}: VIIRS end precedes start for {site_id}")
            previous = windows.setdefault(site_id, (start, end))
            if previous != (start, end):
                raise ValueError(f"{path}: conflicting VIIRS date intervals for {site_id}")
    return windows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--write-results-netcdf", action="store_true",
                        help="Write one prediction-only NetCDF to each selected site's Results/ folder instead of per-day TIFFs.")
    parser.add_argument("--results-root", type=Path, default=None,
                        help="Optional writable mirror root for NetCDF Results output; uses <root>/<site_id>/Results/.")
    parser.add_argument("--netcdf-output-dir", type=Path, default=None,
                        help="Flat directory for prediction NetCDFs; takes precedence over Results-folder output.")
    parser.add_argument("--overwrite-results-netcdf", action="store_true",
                        help="Allow replacement of an existing model-prediction NetCDF in Results/.")
    parser.add_argument("--skip-existing-results-netcdf", action="store_true",
                        help="Leave existing model-prediction NetCDFs untouched and resume remaining sites.")
    parser.add_argument("--site-id", action="append", default=None, help="Repeat to select sites; default is every folder.")
    parser.add_argument("--site-list", type=Path, default=None,
                        help="Optional newline-delimited site IDs; enables parallel launcher shards.")
    parser.add_argument("--start-date", type=date.fromisoformat, default=None)
    parser.add_argument("--end-date", type=date.fromisoformat, default=None)
    parser.add_argument("--viirs-date-csv", type=Path, default=None,
                        help="MTBS/VIIRS table with event_id, viirs_start_date, and viirs_end_date.")
    parser.add_argument("--date-buffer-days", type=int, default=5,
                        help="Inclusive days predicted before VIIRS start and after VIIRS end.")
    parser.add_argument("--sequence-length", type=int, default=90)
    parser.add_argument("--months", type=int, nargs="+", default=[6, 7, 8, 9], help="Permitted target months; default matches final model training.")
    parser.add_argument("--threads", type=int, default=32, help="PyTorch CPU compute threads; use 0 to keep the runtime default.")
    parser.add_argument("--interop-threads", type=int, default=1, help="PyTorch CPU inter-op threads.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Same-site target dates evaluated together; does not alter their 90-day causal windows.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.sequence_length != 90:
        parser.error("This final prediction entry point is intentionally locked to the selected Seq90 protocol.")
    if args.end_date and args.start_date and args.end_date < args.start_date:
        parser.error("--end-date precedes --start-date")
    if args.threads < 0 or args.interop_threads < 1 or args.batch_size < 1 or args.date_buffer_days < 0:
        parser.error("threads and date-buffer-days must be non-negative; interop-threads and batch-size must be positive")
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.interop_threads)
    if args.site_id and args.site_list:
        parser.error("Use either --site-id or --site-list, not both")
    selected_sites = None
    if args.site_list:
        selected_sites = {line.strip() for line in args.site_list.read_text(encoding="utf-8").splitlines() if line.strip()}
    elif args.site_id:
        selected_sites = set(args.site_id)
    sites = sorted(path for path in args.input_root.iterdir() if path.is_dir() and (selected_sites is None or path.name in selected_sites))
    if not sites:
        raise ValueError("No requested fire folders found")
    stats = json.loads(args.normalization.read_text())["groups"]
    viirs_windows = read_viirs_windows(args.viirs_date_csv) if args.viirs_date_csv else {}
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = model_from_checkpoint(checkpoint).to(device)
    load_model_state(model, checkpoint)
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run = {"checkpoint": str(args.checkpoint), "normalization": str(args.normalization), "sequence_length": 90, "target_months": args.months,
           "viirs_date_csv": str(args.viirs_date_csv) if args.viirs_date_csv else None, "date_buffer_days": args.date_buffer_days, "sites": []}
    for site in sites:
        datasets: dict[str, rasterio.DatasetReader] = {}
        try:
            days_by_channel = {}
            for channel in DYNAMIC_CHANNELS:
                matches = sorted(site.glob(f"{channel}_*.tif"))
                if len(matches) != 1:
                    raise ValueError(f"{site.name}: expected exactly one {channel}_*.tif, found {len(matches)}")
                days, dataset = dated_bands(matches[0])
                datasets[channel] = dataset
                days_by_channel[channel] = days
            days = days_by_channel[DYNAMIC_CHANNELS[0]]
            reference = datasets[DYNAMIC_CHANNELS[0]]
            signature = (reference.width, reference.height, reference.transform, reference.crs, days)
            for channel, dataset in datasets.items():
                if (dataset.width, dataset.height, dataset.transform, dataset.crs, days_by_channel[channel]) != signature:
                    raise ValueError(f"{site.name}: forcing grids or band dates disagree across channels ({channel})")
            if reference.width < 3 or reference.height < 3:
                raise ValueError(f"{site.name}: forcing grid is too small for the required one-cell inset")
            grid = Grid(reference.width - 2, reference.height - 2, reference.transform * Affine.translation(1, 1), reference.crs)
            static = load_selected_static_stack(site / "Spatial", grid)
            day_index = {value: index for index, value in enumerate(days)}
            if args.viirs_date_csv:
                if site.name not in viirs_windows:
                    run["sites"].append({"site_id": site.name, "prediction_count": 0, "status": "skipped_absent_from_viirs_csv"})
                    print({"site_id": site.name, "status": "skipped_absent_from_viirs_csv"}, flush=True)
                    continue
                viirs_start, viirs_end = viirs_windows[site.name]
                requested_start = viirs_start - timedelta(days=args.date_buffer_days)
                requested_end = viirs_end + timedelta(days=args.date_buffer_days)
                required_start_history = requested_start - timedelta(days=args.sequence_length - 1)
                endpoint_history = requested_end - timedelta(days=args.sequence_length - 1)
                required_dates = (required_start_history, requested_start, endpoint_history, requested_end)
                if any(value not in day_index for value in required_dates):
                    run["sites"].append({
                        "site_id": site.name, "prediction_count": 0, "status": "skipped_insufficient_90_day_forcing_window",
                        "requested_prediction_dates": [requested_start.isoformat(), requested_end.isoformat()],
                        "required_forcing_dates": [required_start_history.isoformat(), requested_end.isoformat()],
                        "available_forcing_dates": [days[0].isoformat(), days[-1].isoformat()],
                    })
                    print({"site_id": site.name, "status": "skipped_insufficient_90_day_forcing_window"}, flush=True)
                    continue
                requested = [value for value in days if requested_start <= value <= requested_end]
                expected_count = (requested_end - requested_start).days + 1
                if len(requested) != expected_count:
                    raise ValueError(f"{site.name}: forcing dates are not continuous across the requested VIIRS prediction window")
                eligible = requested
            else:
                requested = [value for value in days if value.month in set(args.months) and (args.start_date is None or value >= args.start_date) and (args.end_date is None or value <= args.end_date)]
                eligible = [value for value in requested if value - timedelta(days=args.sequence_length - 1) in day_index]
            if not eligible:
                raise ValueError(f"{site.name}: no eligible prediction dates. Available forcing dates are {days[0]} to {days[-1]}; Seq90 requires 90 contiguous days and targets in months {args.months}.")
            netcdf_path = None
            if args.write_results_netcdf:
                result_dir = args.netcdf_output_dir if args.netcdf_output_dir else ((args.results_root / site.name / "Results") if args.results_root else (site / "Results"))
                result_dir.mkdir(parents=True, exist_ok=True)
                netcdf_path = result_dir / f"{site.name}_seq90_lstm_predictions.nc"
                if netcdf_path.exists():
                    if args.skip_existing_results_netcdf:
                        run["sites"].append({"site_id": site.name, "prediction_count": 0, "status": "existing_results_netcdf_skipped"})
                        print({"site_id": site.name, "status": "existing_results_netcdf_skipped"}, flush=True)
                        continue
                    if not args.overwrite_results_netcdf:
                        raise FileExistsError(f"Refusing to replace {netcdf_path}; pass --overwrite-results-netcdf to replace it")
            destination = args.output_dir / site.name
            if not args.write_results_netcdf:
                destination.mkdir(exist_ok=True)
            written = 0
            netcdf_values: list[np.ndarray] = []
            static_support = np.isfinite(static).all(axis=0)
            static_input = to_device(
                standardize(torch.nan_to_num(torch.from_numpy(static)).unsqueeze(0), stats["static"], STATIC_CHANNELS, 1),
                device,
            )
            means = torch.tensor([stats["target"][name]["mean"] for name in TARGET_CHANNELS]).view(-1, 1, 1)
            stds = torch.tensor([stats["target"][name]["std"] for name in TARGET_CHANNELS]).view(-1, 1, 1)
            for offset in range(0, len(eligible), args.batch_size):
                target_days = eligible[offset:offset + args.batch_size]
                # Ninety causal forcing days: target_day - 89 through target_day.
                dynamic = np.stack([
                    np.stack([
                        datasets[channel].read(
                            [value + 1 for value in range(day_index[target_day] - 89, day_index[target_day] + 1)],
                            out_dtype="float32",
                        )[:, 1:-1, 1:-1]
                        for channel in DYNAMIC_CHANNELS
                    ], axis=1)
                    for target_day in target_days
                ])
                support = static_support[None] & np.isfinite(dynamic).all(axis=(1, 2))
                sequence = to_device(standardize(torch.nan_to_num(torch.from_numpy(dynamic)), stats["dynamic"], DYNAMIC_CHANNELS, 2), device)
                current = sequence[:, -1]
                with torch.no_grad():
                    prediction = model(sequence, current, static_input.expand(len(target_days), -1, -1, -1)).cpu()
                values = (prediction * stds + means).numpy()
                for index, target_day in enumerate(target_days):
                    if args.write_results_netcdf:
                        netcdf_values.append(np.where(support[index], values[index], np.nan))
                    else:
                        write_prediction(destination / f"{site.name}_{target_day.isoformat()}_prediction.tif", values[index], grid, support[index])
                    written += 1
            if args.write_results_netcdf:
                assert netcdf_path is not None
                write_results_netcdf(netcdf_path, np.stack(netcdf_values), eligible, grid)
            run["sites"].append({"site_id": site.name, "available_dates": [days[0].isoformat(), days[-1].isoformat()],
                                 "prediction_dates": [eligible[0].isoformat(), eligible[-1].isoformat()], "prediction_count": written,
                                 "viirs_dates": [viirs_start.isoformat(), viirs_end.isoformat()] if args.viirs_date_csv else None})
            print({"site_id": site.name, "predictions": written}, flush=True)
        except Exception as error:
            run["sites"].append({"site_id": site.name, "prediction_count": 0, "status": "skipped_input_or_prediction_error", "error": str(error)})
            print({"site_id": site.name, "status": "skipped_input_or_prediction_error", "error": str(error)}, flush=True)
        finally:
            for dataset in datasets.values():
                dataset.close()
    (args.output_dir / "run_metadata.json").write_text(json.dumps(run, indent=2) + "\n")


if __name__ == "__main__":
    main()
