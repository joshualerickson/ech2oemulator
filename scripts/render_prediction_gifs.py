#!/usr/bin/env python3
"""Render one date-labelled GIF per target from a prediction NetCDF.

Each target uses one robust, fixed colour scale across all of its frames, so
changes through time are visually comparable.  Nodata is transparent.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from src.data.phase2_qc import TARGET_CHANNELS


DISPLAY_NAMES = {
    "soilmoisture": "Soil moisture",
    "tskin_am": "Skin temperature (AM)",
    "tskin_pm": "Skin temperature (PM)",
    "plc_am": "PLC (AM)",
    "plc_pm": "PLC (PM)",
}


def colour_limits(values: xr.DataArray) -> tuple[float, float]:
    """Return a stable display range while ignoring fill values and outliers."""
    array = values.values.astype(np.float32, copy=False)
    valid = np.isfinite(array) & (array != -9999.0)
    if not valid.any():
        raise ValueError(f"{values.name} contains no finite prediction values")
    low, high = np.percentile(array[valid], (2.0, 98.0))
    if not np.isfinite(low) or not np.isfinite(high) or np.isclose(low, high):
        centre = float(np.nanmean(array[valid]))
        padding = max(abs(centre) * 0.05, 1.0)
        return centre - padding, centre + padding
    return float(low), float(high)


def render_target(
    values: xr.DataArray,
    dates: np.ndarray,
    output: Path,
    fps: int,
    dpi: int,
) -> None:
    vmin, vmax = colour_limits(values)
    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    figure, axis = plt.subplots(figsize=(7.0, 6.5), layout="constrained")
    first = values.isel(time=0).values.astype(np.float32, copy=False)
    first = np.where((first == -9999.0) | ~np.isfinite(first), np.nan, first)
    image = axis.imshow(first, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    axis.set_axis_off()
    title = axis.set_title("", fontsize=15, fontweight="bold", pad=12)
    colourbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    colourbar.set_label(DISPLAY_NAMES.get(str(values.name), str(values.name)), fontsize=11)

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.stem}.", suffix=".tmp.gif", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        writer = animation.PillowWriter(fps=fps)
        with writer.saving(figure, temporary, dpi=dpi):
            for index, timestamp in enumerate(dates):
                frame = values.isel(time=index).values.astype(np.float32, copy=False)
                frame = np.where((frame == -9999.0) | ~np.isfinite(frame), np.nan, frame)
                image.set_data(frame)
                label = np.datetime_as_string(timestamp, unit="D")
                title.set_text(f"{DISPLAY_NAMES.get(str(values.name), values.name)} — {label}")
                writer.grab_frame(facecolor="white")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Prediction NetCDF to visualize.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.fps < 1 or args.dpi < 50:
        parser.error("fps must be positive and dpi must be at least 50")

    with xr.open_dataset(args.input) as dataset:
        if "time" not in dataset.coords:
            raise ValueError(f"{args.input} has no time coordinate")
        available = [name for name in TARGET_CHANNELS if name in dataset.data_vars]
        missing = [name for name in TARGET_CHANNELS if name not in dataset.data_vars]
        if missing:
            raise ValueError(f"{args.input} is missing target variables: {missing}")
        for name in available:
            output = args.output_dir / f"{args.input.stem}_{name}.gif"
            if output.exists() and not args.overwrite:
                print({"target": name, "status": "skipped_existing", "path": str(output)}, flush=True)
                continue
            render_target(dataset[name], dataset.time.values, output, args.fps, args.dpi)
            print({"target": name, "status": "written", "path": str(output)}, flush=True)


if __name__ == "__main__":
    main()
