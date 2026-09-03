#!/usr/bin/env python3
"""Benchmark experimental 30 m prediction/downscaling methods for one fire.

Methods
-------
direct_30m
    Bilinearly resample the six 240 m forcings to a 30 m aligned grid, retain
    native 30 m TWI/FAC/TPI/PSST/CWD, and run the 240 m Seq90 ConvLSTM fully
    convolutionally.  This changes the physical receptive field and is an
    experiment, not a resolution-valid model.

spline_30m
    Predict on the trained 240 m grid, then cubic-resample each target field.

static_guided_30m
    Fit a per-date, per-target ridge relation between coarse prediction and
    aggregate native 30 m context.  Apply only a smooth, bounded within-cell
    context anomaly to the cubic field.  This deliberately does *not* force
    independent 8 x 8 corrections, since those create visible parent-cell
    seams.  It is a conservative disaggregation heuristic, not supervised
    30 m accuracy.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from time import perf_counter

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import rasterio
import torch
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject
from scipy.ndimage import gaussian_filter, uniform_filter

from scripts.predict_fire_seq90 import ResultsNetCDFWriter, dated_bands, read_viirs_windows, write_results_netcdf
from src.data.phase2_qc import DYNAMIC_CHANNELS, TARGET_CHANNELS
from src.data.static_contract import (
    EXTERNAL_STATIC_SOURCES,
    Grid,
    STATIC_CHANNELS,
    load_selected_static_stack,
    read_ascii_on_target_grid,
    resolve_ascii_static,
)
from src.data.transforms import standardize
from src.models.multitask_model import load_model_state, model_from_checkpoint
from src.utils.runtime import resolve_device, to_device


NATIVE_DEM = Path("/mnt/DataDrive1/data/watershed_modeling/spatial_inputs/LandFireDEM/dem_epsg4326.compressed.tiled.oviews.tif")
NATIVE_CONTEXT = ("twi", "fac", "tpi", "psst", "wbdef")
NVME_STATIC_FILENAMES = {
    "twi": "CONUS_TWI_epsg5072_30m_unmasked.tif",
    "fac": "fac.vrt",
    "psst": "psst_q95_1982-2018_30m.tif",
    "wbdef": "def_9221wyr_pansharpened8x_bilinear_def2050_vs_dem30_gsr30_bw200_strength42_preds.tif",
    "dem": "dem_epsg4326.compressed.tiled.oviews.tif",
}


def resample_array(
    values: np.ndarray,
    source: Grid,
    target: Grid,
    method: Resampling,
    num_threads: int = 1,
) -> np.ndarray:
    destination = np.full((target.height, target.width), np.nan, dtype=np.float32)
    reproject(
        source=values, destination=destination,
        src_transform=source.transform, src_crs=source.crs,
        dst_transform=target.transform, dst_crs=target.crs,
        src_nodata=np.nan, dst_nodata=np.nan, resampling=method,
        num_threads=num_threads,
    )
    return destination


def read_native(
    path: Path,
    grid: Grid,
    method: Resampling = Resampling.bilinear,
    num_threads: int = 1,
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing native 30 m context source: {path}")
    destination = np.full((grid.height, grid.width), np.nan, dtype=np.float32)
    with rasterio.open(path) as source:
        reproject(
            source=rasterio.band(source, 1), destination=destination,
            src_transform=source.transform, src_crs=source.crs, src_nodata=source.nodata,
            dst_transform=grid.transform, dst_crs=grid.crs, dst_nodata=np.nan,
            resampling=method,
            num_threads=num_threads,
        )
    return destination


def native_tpi(
    grid: Grid,
    radius_cells: int,
    dem_path: Path = NATIVE_DEM,
    num_threads: int = 1,
) -> np.ndarray:
    """TPI on the 30 m grid, retaining the 1,200 m model-scale radius."""
    dem = read_native(dem_path, grid, num_threads=num_threads)
    valid = np.isfinite(dem)
    size = radius_cells * 2 + 1
    values = np.where(valid, dem, 0.0)
    count = uniform_filter(valid.astype(np.float64), size=size, mode="constant", cval=0.0) * size**2
    mean = uniform_filter(values.astype(np.float64), size=size, mode="constant", cval=0.0) * size**2
    mean = np.divide(mean, count, out=np.full_like(mean, np.nan), where=count > 0)
    return np.where(valid, dem - mean, np.nan).astype(np.float32)


def fine_grid(coarse: Grid, resolution: float) -> tuple[Grid, int]:
    ratio = coarse.transform.a / resolution
    if not np.isclose(ratio, round(ratio)):
        raise ValueError(f"Cannot align {resolution:g} m grid to {coarse.transform.a:g} m grid by an integer factor")
    factor = int(round(ratio))
    return Grid(coarse.width * factor, coarse.height * factor, coarse.transform * Affine.scale(1 / factor, 1 / factor), coarse.crs), factor


def build_static_30m(
    site: Path,
    coarse: Grid,
    fine: Grid,
    tpi_radius_cells: int,
    native_sources: dict[str, Path],
    reproject_threads: int = 1,
) -> np.ndarray:
    """Upsample site-local channels and replace context with native 30 m data."""
    result = np.full((len(STATIC_CHANNELS), fine.height, fine.width), np.nan, dtype=np.float32)
    for name in STATIC_CHANNELS:
        if name in NATIVE_CONTEXT:
            continue
        coarse_values = read_ascii_on_target_grid(resolve_ascii_static(site / "Spatial", name), coarse)
        result[STATIC_CHANNELS.index(name)] = resample_array(
            coarse_values, coarse, fine, Resampling.bilinear, reproject_threads,
        )
    context = {
        "twi": read_native(native_sources["twi"], fine, num_threads=reproject_threads),
        "fac": read_native(native_sources["fac"], fine, num_threads=reproject_threads),
        "tpi": native_tpi(fine, tpi_radius_cells, native_sources["dem"], reproject_threads),
        "psst": read_native(native_sources["psst"], fine, num_threads=reproject_threads),
        "wbdef": read_native(native_sources["wbdef"], fine, num_threads=reproject_threads),
    }
    for name, values in context.items():
        result[STATIC_CHANNELS.index(name)] = values
    return result


def standard_static(values: np.ndarray, stats: dict[str, dict[str, float | int]], device: torch.device) -> torch.Tensor:
    return to_device(standardize(torch.nan_to_num(torch.from_numpy(values)).unsqueeze(0), stats, STATIC_CHANNELS, 1), device)


def predict_one(
    model: torch.nn.Module, dynamic: np.ndarray, static: torch.Tensor,
    dynamic_stats: dict[str, dict[str, float | int]], target_stats: dict[str, dict[str, float | int]], device: torch.device,
) -> np.ndarray:
    sequence = to_device(standardize(torch.nan_to_num(torch.from_numpy(dynamic)).unsqueeze(0), dynamic_stats, DYNAMIC_CHANNELS, 2), device)
    with torch.no_grad():
        normalized = model(sequence, sequence[:, -1], static)[0].cpu()
    means = torch.tensor([target_stats[name]["mean"] for name in TARGET_CHANNELS]).view(-1, 1, 1)
    stds = torch.tensor([target_stats[name]["std"] for name in TARGET_CHANNELS]).view(-1, 1, 1)
    return (normalized * stds + means).numpy()


class FineDynamicWindow:
    """Cache the 90 resampled forcing days shared by adjacent predictions.

    Prediction windows remain scientifically identical: the recurrent state is
    reset and all 90 chronological steps are evaluated for every target day.
    Only the deterministic 240-to-30 m raster reprojection is reused.
    """

    def __init__(
        self,
        datasets: dict[str, rasterio.DatasetReader],
        source: Grid,
        target: Grid,
        dynamic_stats: dict[str, dict[str, float | int]],
        reproject_threads: int,
        channel_workers: int,
    ) -> None:
        self.datasets = datasets
        self.source = source
        self.target = target
        self.dynamic_stats = dynamic_stats
        self.reproject_threads = reproject_threads
        self.executor = ThreadPoolExecutor(max_workers=channel_workers)
        self.cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self.reprojected_days = 0

    def close(self) -> None:
        self.executor.shutdown(wait=True)

    def _channel(self, name: str, index: int) -> np.ndarray:
        values = self.datasets[name].read(index + 1, out_dtype="float32")[1:-1, 1:-1]
        return resample_array(
            values,
            self.source,
            self.target,
            Resampling.bilinear,
            self.reproject_threads,
        )

    def _load(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        channels = list(self.executor.map(lambda name: self._channel(name, index), DYNAMIC_CHANNELS))
        values = np.stack(channels)
        support = np.isfinite(values).all(axis=0)
        # Cache normalized CPU arrays so overlapping windows also avoid doing
        # the same normalization 89 more times.
        normalized = standardize(
            torch.nan_to_num(torch.from_numpy(values)).unsqueeze(0),
            self.dynamic_stats,
            DYNAMIC_CHANNELS,
            1,
        )[0].numpy()
        self.reprojected_days += 1
        return normalized, support

    def window(self, indices: list[int]) -> list[tuple[np.ndarray, np.ndarray]]:
        wanted = set(indices)
        for old_index in list(self.cache):
            if old_index not in wanted:
                del self.cache[old_index]
        for index in indices:
            if index not in self.cache:
                self.cache[index] = self._load(index)
        return [self.cache[index] for index in indices]


def predict_one_cached_window(
    model: torch.nn.Module,
    window: list[tuple[np.ndarray, np.ndarray]],
    static: torch.Tensor,
    target_stats: dict[str, dict[str, float | int]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate an exact fixed 90-day window without materializing a 5-D cube."""
    state = None
    support = np.ones(window[0][1].shape, dtype=bool)
    current: torch.Tensor | None = None
    hidden: torch.Tensor | None = None
    with torch.no_grad():
        for normalized, day_support in window:
            support &= day_support
            current = to_device(torch.from_numpy(normalized).unsqueeze(0), device)
            hidden, state = model.encoder.forward_stateful(current.unsqueeze(1), state)
        if current is None or hidden is None:
            raise AssertionError("A recurrent prediction window cannot be empty")
        normalized_prediction = model.decode(hidden, current, static)[0].cpu()
    means = torch.tensor([target_stats[name]["mean"] for name in TARGET_CHANNELS]).view(-1, 1, 1)
    stds = torch.tensor([target_stats[name]["std"] for name in TARGET_CHANNELS]).view(-1, 1, 1)
    return (normalized_prediction * stds + means).numpy(), support


def cubic_prediction(values: np.ndarray, source: Grid, target: Grid) -> np.ndarray:
    return np.stack([resample_array(layer, source, target, Resampling.cubic) for layer in values])


def static_guided(
    coarse_prediction: np.ndarray,
    spline: np.ndarray,
    context: np.ndarray,
    coarse: Grid,
    fine: Grid,
    factor: int,
) -> np.ndarray:
    """Inject continuous, bounded native-context detail into a cubic field.

    A 240 m-trained model is not entitled to react to every native 30 m pixel.
    We therefore estimate a coarse-scale context relation, compare its smooth
    30 m response with a cubic expansion of the same relation, and inject only
    the residual texture.  Unlike the original implementation, no separate
    parent-cell offsets are applied: those offsets were the source of obvious
    8 x 8 seams.
    """
    height, width = coarse_prediction.shape[1:]
    if context.shape[1:] != (height * factor, width * factor):
        raise ValueError("Native context shape does not align with the coarse prediction grid")

    # FAC is extremely skewed.  A log transform and robust clipping prevent a
    # handful of stream pixels from dominating the small per-fire regression.
    prepared = context.astype(np.float64, copy=True)
    fac_index = NATIVE_CONTEXT.index("fac")
    prepared[fac_index] = np.log1p(np.maximum(prepared[fac_index], 0.0))
    for feature_index in range(prepared.shape[0]):
        values = prepared[feature_index]
        finite = np.isfinite(values)
        if not finite.any():
            continue
        low, high = np.nanpercentile(values[finite], (1.0, 99.0))
        prepared[feature_index] = np.where(finite, np.clip(values, low, high), np.nan)
        # Remove pixel-scale raster texture that was never present at model
        # training resolution, while retaining terrain-scale variation.
        fill = float(np.nanmedian(prepared[feature_index]))
        prepared[feature_index] = gaussian_filter(
            np.where(np.isfinite(prepared[feature_index]), prepared[feature_index], fill),
            sigma=1.25,
            mode="nearest",
        )

    features = prepared.reshape(len(NATIVE_CONTEXT), height, factor, width, factor).mean(axis=(2, 4))
    output = np.empty_like(spline)
    fine_features = prepared.reshape(len(NATIVE_CONTEXT), -1).T
    fine_valid = np.isfinite(fine_features).all(axis=1)
    for channel in range(len(TARGET_CHANNELS)):
        y = coarse_prediction[channel].ravel()
        x = features.reshape(len(NATIVE_CONTEXT), -1).T
        good = np.isfinite(y) & np.isfinite(x).all(axis=1)
        if good.sum() <= len(NATIVE_CONTEXT):
            output[channel] = spline[channel]
            continue
        mean, std = x[good].mean(axis=0), x[good].std(axis=0)
        std[std < 1e-6] = 1.0
        design = np.column_stack((np.ones(good.sum()), (x[good] - mean) / std))
        penalty = np.eye(design.shape[1]) * 1e-3
        penalty[0, 0] = 0.0
        beta = np.linalg.solve(design.T @ design + penalty, design.T @ y[good])
        coarse_response = (np.column_stack((np.ones(x.shape[0]), (x - mean) / std)) @ beta).reshape(height, width)
        fine_response = np.full(fine_features.shape[0], np.nan, dtype=np.float64)
        fine_response[fine_valid] = (
            np.column_stack((np.ones(fine_valid.sum()), (fine_features[fine_valid] - mean) / std)) @ beta
        )
        fine_response = fine_response.reshape(height * factor, width * factor)
        response_baseline = resample_array(
            coarse_response.astype(np.float32),
            coarse,
            fine,
            Resampling.cubic,
        )
        residual = fine_response - response_baseline
        residual = gaussian_filter(np.nan_to_num(residual, nan=0.0), sigma=1.0, mode="nearest")
        # Limit injected texture to a modest fraction of the prediction's
        # coarse spatial range.  This keeps 30 m static outliers from changing
        # the learned 240 m magnitude or creating artificial edges.
        spread = float(np.nanpercentile(y[good], 75) - np.nanpercentile(y[good], 25))
        cap = max(spread * 0.35, 1e-6)
        residual = np.clip(residual, -cap, cap)
        output[channel] = spline[channel] + 0.35 * residual.astype(np.float32)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--site-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--viirs-date-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path,
                        help="Optional directory for the per-site timing JSON; defaults to output-dir.")
    parser.add_argument(
        "--native-static-root", type=Path,
        help="Directory containing the staged native-static filenames; defaults to their original source paths.",
    )
    parser.add_argument("--date-buffer-days", type=int, default=5)
    parser.add_argument("--resolution", type=float, default=30.0)
    parser.add_argument("--tpi-radius-cells", type=int, default=40)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--reproject-threads", type=int, default=2,
                        help="GDAL warp threads used by each forcing-channel task.")
    parser.add_argument("--forcing-channel-workers", type=int, default=3,
                        help="Forcing channels reprojected concurrently within a fire worker.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reject-all-na", action="store_true",
                        help="Fail before writing when a selected method has no valid pixels on any date.")
    parser.add_argument(
        "--methods", nargs="+", choices=("direct_30m", "spline_30m", "static_guided_30m"),
        default=("direct_30m", "spline_30m", "static_guided_30m"),
        help="Prediction products to create; production runs normally use direct_30m only.",
    )
    args = parser.parse_args()
    if args.threads < 1 or args.reproject_threads < 1 or args.forcing_channel_workers < 1:
        parser.error("threads, reproject-threads, and forcing-channel-workers must be positive")
    methods = tuple(dict.fromkeys(args.methods))
    native_sources = {
        "twi": EXTERNAL_STATIC_SOURCES["twi"],
        "fac": EXTERNAL_STATIC_SOURCES["fac"],
        "psst": EXTERNAL_STATIC_SOURCES["psst"],
        "wbdef": EXTERNAL_STATIC_SOURCES["wbdef"],
        "dem": NATIVE_DEM,
    }
    if args.native_static_root is not None:
        native_sources = {name: args.native_static_root / filename for name, filename in NVME_STATIC_FILENAMES.items()}
        missing = [str(path) for path in native_sources.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Native static staging directory is incomplete: {missing}")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    site = args.input_root / args.site_id
    windows = read_viirs_windows(args.viirs_date_csv)
    if args.site_id not in windows:
        raise ValueError(f"{args.site_id} is absent from {args.viirs_date_csv}")
    viirs_start, viirs_end = windows[args.site_id]
    prediction_days = [viirs_start - timedelta(days=args.date_buffer_days) + timedelta(days=index)
                       for index in range((viirs_end - viirs_start).days + 1 + args.date_buffer_days * 2)]
    stats = json.loads(args.normalization.read_text())["groups"]
    device = resolve_device(args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = model_from_checkpoint(state).to(device)
    load_model_state(model, state)
    model.eval()
    datasets: dict[str, rasterio.DatasetReader] = {}
    try:
        days_by_channel: dict[str, list[date]] = {}
        for name in DYNAMIC_CHANNELS:
            matches = sorted(site.glob(f"{name}_*.tif"))
            if len(matches) != 1:
                raise ValueError(f"{site}: expected exactly one {name}_*.tif")
            days_by_channel[name], datasets[name] = dated_bands(matches[0])
        days = days_by_channel[DYNAMIC_CHANNELS[0]]
        if any(days_by_channel[name] != days for name in DYNAMIC_CHANNELS[1:]):
            raise ValueError("Forcing dates differ across channels")
        index = {day: number for number, day in enumerate(days)}
        if any(day not in index or day - timedelta(days=89) not in index for day in (prediction_days[0], prediction_days[-1])):
            raise ValueError("Test site lacks a complete causal 90-day sequence at the requested prediction-window endpoints")
        reference = datasets[DYNAMIC_CHANNELS[0]]
        coarse = Grid(reference.width - 2, reference.height - 2, reference.transform * Affine.translation(1, 1), reference.crs)
        fine, factor = fine_grid(coarse, args.resolution)
        need_coarse = "spline_30m" in methods or "static_guided_30m" in methods
        started = perf_counter()
        static_fine = build_static_30m(
            site, coarse, fine, args.tpi_radius_cells, native_sources, args.reproject_threads,
        )
        static_coarse = load_selected_static_stack(site / "Spatial", coarse) if need_coarse else None
        static_prep_seconds = perf_counter() - started
        coarse_input = standard_static(static_coarse, stats["static"], device) if static_coarse is not None else None
        fine_input = standard_static(static_fine, stats["static"], device)
        coarse_predictions: list[np.ndarray] = []
        direct_predictions: list[np.ndarray] = []
        spline_predictions: list[np.ndarray] = []
        guided_predictions: list[np.ndarray] = []
        direct_supports: list[np.ndarray] = []
        coarse_supports: list[np.ndarray] = []
        coarse_seconds = direct_seconds = fine_forcing_seconds = 0.0
        context = np.stack([static_fine[STATIC_CHANNELS.index(name)] for name in NATIVE_CONTEXT])
        fine_window = FineDynamicWindow(
            datasets, coarse, fine, stats["dynamic"],
            args.reproject_threads, args.forcing_channel_workers,
        ) if "direct_30m" in methods else None
        # The production path writes each day immediately, bounding output RAM
        # independently of fire duration. Multi-method experiments retain the
        # older in-memory lists because they are intended for one-site tests.
        stream_direct = methods == ("direct_30m",)
        direct_writer = ResultsNetCDFWriter(
            args.output_dir / f"{args.site_id}_direct_30m.nc", prediction_days, fine,
        ) if stream_direct else None
        direct_has_support = False
        direct_complete = False
        static_support = np.isfinite(static_fine).all(axis=0)
        try:
            for day_number, target_day in enumerate(prediction_days, start=1):
                indices = list(range(index[target_day] - 89, index[target_day] + 1))
                if need_coarse:
                    coarse_dynamic = np.stack([datasets[name].read([value + 1 for value in indices], out_dtype="float32")[:, 1:-1, 1:-1] for name in DYNAMIC_CHANNELS], axis=1)
                    coarse_supports.append(np.isfinite(coarse_dynamic).all(axis=(0, 1)))
                    started = perf_counter()
                    if coarse_input is None:
                        raise AssertionError("Coarse static input was not prepared")
                    coarse_prediction = predict_one(model, coarse_dynamic, coarse_input, stats["dynamic"], stats["target"], device)
                    coarse_seconds += perf_counter() - started
                    coarse_predictions.append(coarse_prediction)
                    spline_prediction = cubic_prediction(coarse_prediction, coarse, fine)
                    if "spline_30m" in methods:
                        spline_predictions.append(spline_prediction)
                    if "static_guided_30m" in methods:
                        guided_predictions.append(static_guided(
                            coarse_prediction, spline_prediction, context, coarse, fine, factor,
                        ))
                if fine_window is not None:
                    started = perf_counter()
                    cached_window = fine_window.window(indices)
                    fine_forcing_seconds += perf_counter() - started
                    started = perf_counter()
                    prediction, direct_support = predict_one_cached_window(
                        model, cached_window, fine_input, stats["target"], device,
                    )
                    direct_predictions.append(prediction)
                    direct_supports.append(direct_support)
                    method_support = static_support & direct_support
                    direct_has_support |= bool(method_support.any())
                    if direct_writer is not None:
                        direct_writer.write(day_number - 1, prediction, method_support)
                        direct_predictions.clear()
                        direct_supports.clear()
                    direct_seconds += perf_counter() - started
                if day_number == 1 or day_number % 10 == 0 or day_number == len(prediction_days):
                    print({"site_id": args.site_id, "prediction_days_finished": day_number,
                           "prediction_days_total": len(prediction_days),
                           "unique_fine_forcing_days": fine_window.reprojected_days if fine_window else 0}, flush=True)
            direct_complete = True
        finally:
            if fine_window is not None:
                fine_window.close()
            if direct_writer is not None:
                direct_writer.close(publish=direct_complete and (direct_has_support or not args.reject_all_na))
        if direct_writer is not None and args.reject_all_na and not direct_has_support:
            raise ValueError(f"No valid 30 m support remains for {args.site_id} using direct_30m")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        support = static_support
        metadata = {"site_id": args.site_id, "viirs_dates": [viirs_start.isoformat(), viirs_end.isoformat()],
                    "prediction_dates": [prediction_days[0].isoformat(), prediction_days[-1].isoformat()], "prediction_day_count": len(prediction_days),
                    "coarse_resolution_m": coarse.transform.a, "fine_resolution_m": args.resolution, "scale_factor": factor,
                    "native_context_channels": list(NATIVE_CONTEXT), "tpi_radius_cells": args.tpi_radius_cells,
                    "tpi_physical_radius_m": args.tpi_radius_cells * args.resolution,
                    "methods": list(methods),
                    "torch_threads": args.threads, "reproject_threads": args.reproject_threads,
                    "forcing_channel_workers": args.forcing_channel_workers,
                    "unique_fine_forcing_days": fine_window.reprojected_days if fine_window else 0,
                    "static_preparation_seconds": static_prep_seconds, "coarse_prediction_seconds": coarse_seconds,
                    "fine_forcing_preparation_seconds": fine_forcing_seconds,
                    "direct_30m_prediction_seconds": direct_seconds, "direct_vs_coarse_compute_ratio": direct_seconds / coarse_seconds if coarse_seconds else None,
                    "notes": {
                        "direct_30m": "experimental resolution transfer; 3x3 receptive field contracts in physical distance",
                        "spline_30m": "production-safe resolution transfer: cubic interpolation of the 240 m prediction",
                        "static_guided_30m": "experimental smooth, bounded native-static texture; does not impose parent-cell offsets",
                    }}
        values_by_method = {
            "direct_30m": direct_predictions,
            "spline_30m": spline_predictions,
            "static_guided_30m": guided_predictions,
        }
        support_by_method = {
            "direct_30m": np.stack(direct_supports) if direct_supports else None,
            "spline_30m": np.repeat(np.repeat(np.stack(coarse_supports), factor, axis=1), factor, axis=2) if coarse_supports else None,
            "static_guided_30m": np.repeat(np.repeat(np.stack(coarse_supports), factor, axis=1), factor, axis=2) if coarse_supports else None,
        }
        for method in (() if stream_direct else methods):
            values = values_by_method[method]
            dynamic_support = support_by_method[method]
            if dynamic_support is None:
                raise AssertionError(f"No support stack was collected for {method}")
            method_support = support[None] & dynamic_support
            output_values = np.where(method_support[:, None], np.stack(values), np.nan)
            if args.reject_all_na and not np.isfinite(output_values).any():
                raise ValueError(f"No valid 30 m support remains for {args.site_id} using {method}")
            write_results_netcdf(args.output_dir / f"{args.site_id}_{method}.nc", output_values, prediction_days, fine)
        metadata_dir = args.metadata_dir or args.output_dir
        metadata_dir.mkdir(parents=True, exist_ok=True)
        (metadata_dir / f"{args.site_id}_benchmark.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata, indent=2))
    finally:
        for dataset in datasets.values():
            dataset.close()


if __name__ == "__main__":
    main()
