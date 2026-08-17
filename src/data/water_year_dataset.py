"""One calendar-forcing replay per site for stateful recurrent training.

Despite the historical filename, this dataset is not Oct--Sep replay: target
arrays are water-year indexed but calendar-year forcing begins on January 1.
It therefore replays the exact Jan--Sep date overlap only.
"""
from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import rasterio
import torch
import xarray as xr
from torch.utils.data import Dataset

from src.data.phase2_qc import DYNAMIC_CHANNELS, TARGET_CHANNELS
from src.data.static_contract import Grid, load_selected_static_stack
from src.data.target_qc import mask_implausible
from src.data.temporal_contract import TEMPORAL_CONTRACT, forcing_index_for_target_date, target_index_for_date


class WaterYearDataset(Dataset):
    """Replay Jan--Sep in chronological order using explicit target/forcing indices."""

    def __init__(self, manifest: Path | str, split: str) -> None:
        with Path(manifest).open(newline="") as handle:
            self.rows = [row for row in csv.DictReader(handle) if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No {split} rows")
        if {row.get("temporal_contract") for row in self.rows} != {TEMPORAL_CONTRACT}:
            raise ValueError("Water-year manifest uses the retired shared-band-index contract; rebuild it from v2 daily screens.")
        self.cache: dict[str, tuple[list[np.ndarray], list[np.ndarray], list[dict[str, str]]]] = {}
        self.static: dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        row = self.rows[i]
        site = row["site_id"]
        water_year = int(row["water_year_end"])
        if site not in self.cache:
            with Path(row["daily_screen_source"]).open(newline="") as handle:
                all_daily = list(csv.DictReader(handle))
            daily = [day for day in all_daily if day.get("forcing_available") == "True"]
            if not daily:
                raise ValueError(f"{site}: no Jan--Sep target/forcing overlap in daily screen")
            dates = [date.fromisoformat(day["date"]) for day in daily]
            if dates[0] != date(water_year, 1, 1) or dates[-1] != date(water_year, 9, 30):
                raise ValueError(f"{site}: expected Jan-01 through Sep-30 overlap, got {dates[0]} through {dates[-1]}")
            if any(next_day - current != timedelta(days=1) for current, next_day in zip(dates, dates[1:])):
                raise ValueError(f"{site}: target/forcing overlap contains a date gap")
            forcing_indices = [int(day["forcing_time_index"]) for day in daily]
            target_indices = [int(day["target_time_index"]) for day in daily]
            expected_forcing = [forcing_index_for_target_date(day, water_year, 366) for day in dates]
            expected_target = [target_index_for_date(day, water_year, 366) for day in dates]
            if forcing_indices != expected_forcing or target_indices != expected_target:
                raise ValueError(f"{site}: daily screen source indices do not match its ISO dates")
            root = Path(row["dynamic_sequence_source"])
            dynamic = []
            for name in DYNAMIC_CHANNELS:
                with rasterio.open(root / f"{name}_{water_year}.tif") as dataset:
                    dynamic.append(dataset.read(indexes=[index + 1 for index in forcing_indices], out_dtype="float32")[:, 1:-1, 1:-1])
            with xr.open_dataset(row["target_source"], engine="h5netcdf", decode_cf=False, mask_and_scale=False) as dataset:
                targets = [np.asarray(dataset[name].isel(time=target_indices).values, dtype=np.float32) for name in TARGET_CHANNELS]
            self.cache[site] = (dynamic, targets, daily)

        dynamic, targets, daily = self.cache[site]
        if site not in self.static:
            with rasterio.open(f"NETCDF:{row['target_source']}:soilmoisture") as dataset:
                grid = Grid(dataset.width, dataset.height, dataset.transform, dataset.crs)
            self.static[site] = torch.from_numpy(load_selected_static_stack(Path(row["static_source"]), grid))
        masked = [mask_implausible(values, name) for name, values in zip(TARGET_CHANNELS, targets)]
        target = np.stack(masked, axis=1)  # [T, C_target, H, W]
        valid = np.isfinite(target)
        loss_days = np.array([day["target_qa_valid"] == "True" and int(day["date"][5:7]) in (6, 7, 8, 9) for day in daily])
        return {
            "site_id": site,
            "dates": [day["date"] for day in daily],
            "source": row["target_source"],
            "dynamic": torch.from_numpy(np.stack(dynamic, axis=1)),
            "static": self.static[site],
            "target": torch.from_numpy(target),
            "valid": torch.from_numpy(valid),
            "loss_days": torch.from_numpy(loss_days),
        }
