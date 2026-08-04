"""Daily source screening used to construct the canonical sequence manifest."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Iterator

import numpy as np
import rasterio
import xarray as xr

from src.data.target_qc import TARGET_PLAUSIBILITY_RANGES, plausibility_mask


DYNAMIC_CHANNELS = ("prcp", "srad", "tmin", "tmax", "rmin", "rmax")
TARGET_CHANNELS = ("soilmoisture", "tskin_am", "tskin_pm", "plc_am", "plc_pm")


def water_year_dates(water_year_end: int, count: int) -> list[date]:
    """Return source dates under the approved Oct-1 water-year convention."""
    start = date(water_year_end - 1, 10, 1)
    return [start + timedelta(days=index) for index in range(count)]


def target_grid_crop(array: np.ndarray) -> np.ndarray:
    """Crop a forcing bbox to its validated target-support one-cell inset."""
    if array.ndim != 3 or array.shape[1] < 3 or array.shape[2] < 3:
        raise ValueError(f"Expected forcing [T, H, W] with a crop border, got {array.shape}")
    return array[:, 1:-1, 1:-1]


def iter_daily_screen_rows(
    site_dir: Path,
    water_year_end: int,
    time_start: int = 0,
    time_stop: int | None = None,
) -> Iterator[dict[str, object]]:
    """Yield one screen record per site/day, with target violations unmodified."""
    nc_path = site_dir / f"{site_dir.name}-{water_year_end}_subdaily.nc"
    if not nc_path.is_file():
        raise FileNotFoundError(nc_path)

    if time_start < 0:
        raise ValueError("time_start must be non-negative")
    forcing_valid: np.ndarray | None = None
    forcing_invalid_pixels: np.ndarray | None = None
    for channel in DYNAMIC_CHANNELS:
        path = site_dir / f"{channel}_{water_year_end}.tif"
        with rasterio.open(path) as dataset:
            stop = dataset.count if time_stop is None else min(time_stop, dataset.count)
            if time_start >= stop:
                raise ValueError(f"Empty forcing time slice [{time_start}, {stop}) for {path}")
            valid = target_grid_crop(
                dataset.read_masks(indexes=list(range(time_start + 1, stop + 1))) > 0
            )
        channel_valid = np.all(valid, axis=(1, 2))
        invalid_pixels = np.count_nonzero(~valid, axis=(1, 2))
        forcing_valid = channel_valid if forcing_valid is None else forcing_valid & channel_valid
        forcing_invalid_pixels = (
            invalid_pixels
            if forcing_invalid_pixels is None
            else forcing_invalid_pixels + invalid_pixels
        )
    assert forcing_valid is not None and forcing_invalid_pixels is not None

    target_violations: dict[str, np.ndarray] = {}
    target_valid_counts: dict[str, np.ndarray] = {}
    target_min: dict[str, np.ndarray] = {}
    target_max: dict[str, np.ndarray] = {}
    count: int | None = None
    total_count: int | None = None
    # h5netcdf loads all target variables from a single file handle. This is
    # materially faster than opening five GDAL NetCDF subdatasets and avoids
    # changing the source values or date semantics.
    with xr.open_dataset(nc_path, engine="h5netcdf", decode_cf=False, mask_and_scale=False) as dataset:
        for target in TARGET_CHANNELS:
            target_total_count = int(dataset.sizes["time"])
            stop = target_total_count if time_stop is None else min(time_stop, target_total_count)
            if time_start >= stop:
                raise ValueError(f"Empty target time slice [{time_start}, {stop}) for {target}")
            values = np.asarray(
                dataset[target].isel(time=slice(time_start, stop)).values,
                dtype=np.float32,
            )
            if values.ndim != 3:
                raise ValueError(f"Expected {target} [time, y, x], got {values.shape} in {nc_path}")
            if count is None:
                count = values.shape[0]
                total_count = target_total_count
            elif count != values.shape[0]:
                raise ValueError(f"Target count mismatch in {nc_path}: {target}")
            finite = np.isfinite(values) & (values != np.float32(-1.175494e38))
            valid = plausibility_mask(values, target)
            target_violations[target] = np.count_nonzero(finite & ~valid, axis=(1, 2))
            target_valid_counts[target] = np.count_nonzero(valid, axis=(1, 2))
            masked_values = np.where(finite, values, np.nan)
            target_min[target] = np.nanmin(masked_values, axis=(1, 2))
            target_max[target] = np.nanmax(masked_values, axis=(1, 2))
    assert count is not None
    if forcing_valid.shape[0] != count:
        raise ValueError(f"Forcing/target time count mismatch in {site_dir}")

    assert total_count is not None
    for index, current_date in enumerate(water_year_dates(water_year_end, total_count)[time_start : time_start + count]):
        violations = {target: int(values[index]) for target, values in target_violations.items()}
        yield {
            "site_id": site_dir.name,
            "bbox_id": site_dir.name,
            "water_year_end": water_year_end,
            "date": current_date.isoformat(),
            "forcing_valid": bool(forcing_valid[index]),
            "forcing_invalid_pixel_count": int(forcing_invalid_pixels[index]),
            "target_plausibility_violation_count": sum(violations.values()),
            "target_qa_valid": not any(violations.values()),
            **{f"{target}_plausibility_violation_count": value for target, value in violations.items()},
            **{f"{target}_valid_pixel_count": int(values[index]) for target, values in target_valid_counts.items()},
            **{f"{target}_raw_min": float(values[index]) for target, values in target_min.items()},
            **{f"{target}_raw_max": float(values[index]) for target, values in target_max.items()},
        }
