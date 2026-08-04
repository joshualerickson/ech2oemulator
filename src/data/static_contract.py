"""Static-source contract for ECH2O bbox samples.

This deliberately small module adopts only the selected static inputs and
geospatial handling from the legacy U-Net project. It does not reuse legacy
clipping or model-preparation code.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject


CANONICAL_CRS = CRS.from_epsg(5070)
STATIC_CHANNELS = (
    "vcf",
    "theta_r",
    "BClambda",
    "poros",
    "psi_ae",
    "dem",
    "keff",
    "lai",
    "twi",
    "fac",
    "tpi",
    "psst",
    "wbdef",
)
STATIC_ALIASES = {"keff": ("keff", "ksat")}
EXTERNAL_STATIC_SOURCES = {
    "twi": Path(os.environ.get("ECH2O_TWI_PATH", "/mnt/DataDrive1/data/raster_data/CONUS_TWI_epsg5072_30m_unmasked.tif")),
    "fac": Path(os.environ.get("ECH2O_FAC_PATH", "/mnt/DataDrive1/data/raster_data/fac.vrt")),
    "psst": Path(os.environ.get("ECH2O_PSST_PATH", "/mnt/DataDrive1/data/climate_data/PSST/psst_q95_1982-2018_30m.tif")),
    "wbdef": Path(os.environ.get("ECH2O_WBDEF_PATH", "/mnt/DataDrive1/data/climate_data/waterbalance/normal_30m/def_9221wyr_pansharpened8x_bilinear_def2050_vs_dem30_gsr30_bw200_strength42_preds.tif")),
}


@dataclass(frozen=True)
class Grid:
    """One target-support grid, expressed in the approved EPSG:5070 CRS."""

    width: int
    height: int
    transform: Affine
    crs: CRS = CANONICAL_CRS


def resolve_ascii_static(spatial_dir: Path, channel: str) -> Path:
    """Resolve a selected site-local static source without filename inference."""
    for name in STATIC_ALIASES.get(channel, (channel,)):
        path = spatial_dir / f"{name}.asc"
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing selected static {channel!r} under {spatial_dir}")


def _is_one_cell_inset(source: Grid, target: Grid) -> bool:
    return (
        source.width == target.width + 2
        and source.height == target.height + 2
        and source.crs == target.crs
        and np.isclose(source.transform.a, target.transform.a)
        and np.isclose(source.transform.e, target.transform.e)
        and np.isclose(target.transform.c, source.transform.c + source.transform.a)
        and np.isclose(target.transform.f, source.transform.f + source.transform.e)
    )


def read_ascii_on_target_grid(path: Path, target_grid: Grid) -> np.ndarray:
    """Read an ASCII static after asserting its approved one-cell crop relation.

    ESRI ASCII does not carry a CRS. Its coordinates are explicitly interpreted
    as EPSG:5070, then validated against the target grid; no reprojection or
    interpolation is permitted for these site-local rasters.
    """
    with rasterio.open(path) as source:
        source_grid = Grid(source.width, source.height, source.transform, CANONICAL_CRS)
        if not _is_one_cell_inset(source_grid, target_grid):
            raise ValueError(
                f"{path} is not the approved one-cell EPSG:5070 inset parent of the target grid"
            )
        data = source.read(1, out_dtype="float32")
        nodata = source.nodata
    if nodata is not None and np.isfinite(nodata):
        data[data == nodata] = np.nan
    data[~np.isfinite(data)] = np.nan
    return data[1:-1, 1:-1]


def read_external_on_target_grid(channel: str, target_grid: Grid) -> np.ndarray:
    """Reproject a documented regional static source to the target support."""
    source_path = EXTERNAL_STATIC_SOURCES[channel]
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing external static source for {channel}: {source_path}")
    destination = np.full((target_grid.height, target_grid.width), np.nan, dtype=np.float32)
    with rasterio.open(source_path) as source:
        reproject(
            source=rasterio.band(source, 1),
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=target_grid.transform,
            dst_crs=target_grid.crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    return destination


def derive_tpi(dem: np.ndarray, radius_cells: int = 5) -> np.ndarray:
    """Derive target-grid TPI as elevation minus local finite-neighbour mean."""
    if dem.ndim != 2:
        raise ValueError(f"DEM must be [H, W], got {dem.shape}")
    result = np.full_like(dem, np.nan, dtype=np.float32)
    for row in range(dem.shape[0]):
        for column in range(dem.shape[1]):
            window = dem[
                max(0, row - radius_cells) : min(dem.shape[0], row + radius_cells + 1),
                max(0, column - radius_cells) : min(dem.shape[1], column + radius_cells + 1),
            ]
            if np.isfinite(dem[row, column]) and np.any(np.isfinite(window)):
                result[row, column] = dem[row, column] - np.nanmean(window)
    return result


def load_selected_static_stack(spatial_dir: Path, target_grid: Grid) -> np.ndarray:
    """Load the approved 13 static channels in their fixed canonical order."""
    channels: dict[str, np.ndarray] = {}
    for channel in STATIC_CHANNELS:
        if channel in EXTERNAL_STATIC_SOURCES or channel == "tpi":
            continue
        channels[channel] = read_ascii_on_target_grid(
            resolve_ascii_static(spatial_dir, channel), target_grid
        )
    for channel in EXTERNAL_STATIC_SOURCES:
        channels[channel] = read_external_on_target_grid(channel, target_grid)
    channels["tpi"] = derive_tpi(channels["dem"])
    stack = np.stack([channels[channel] for channel in STATIC_CHANNELS], axis=0)
    expected_shape = (len(STATIC_CHANNELS), target_grid.height, target_grid.width)
    if stack.shape != expected_shape:
        raise AssertionError(f"Static stack shape {stack.shape} does not match {expected_shape}")
    return stack
