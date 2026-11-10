"""Read target NetCDFs with either `time` or terra's per-variable Z axes.

The temporal meaning of an index is supplied by the producer-confirmed
water-year contract: target band zero is October 1 of the preceding calendar
year. A Z coordinate numbered 1..365 is not itself a calendar date.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def target_time_dim(dataset: xr.Dataset, channel: str) -> str:
    if channel not in dataset:
        raise ValueError(f"Missing target variable {channel!r}")
    variable = dataset[channel]
    if variable.ndim != 3 or variable.dims[1:] != ("northing", "easting"):
        raise ValueError(
            f"Expected {channel} [band, northing, easting], got {variable.dims} {variable.shape}"
        )
    dimension = variable.dims[0]
    if dimension != "time" and not dimension.startswith("Z"):
        raise ValueError(f"Unrecognized target band axis {dimension!r} for {channel}")
    return dimension


def target_time_count(dataset: xr.Dataset, channel: str) -> int:
    return int(dataset.sizes[target_time_dim(dataset, channel)])


def read_target(
    dataset: xr.Dataset, channel: str, indices: slice | list[int] | None = None
) -> np.ndarray:
    dimension = target_time_dim(dataset, channel)
    variable = dataset[channel] if indices is None else dataset[channel].isel({dimension: indices})
    values = np.asarray(variable.values, dtype=np.float32)
    nodata = dataset[channel].attrs.get("_FillValue")
    if nodata is not None:
        values[values == np.float32(nodata)] = np.nan
    values[~np.isfinite(values)] = np.nan
    return values
