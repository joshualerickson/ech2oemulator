"""Canonical sequence dataset for the ECH2O recurrent-model development set."""
from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
import xarray as xr
from torch.utils.data import Dataset

from src.data.phase2_qc import DYNAMIC_CHANNELS, TARGET_CHANNELS
from src.data.static_contract import Grid, STATIC_CHANNELS, load_selected_static_stack
from src.data.target_qc import mask_implausible
from src.data.temporal_contract import (
    TEMPORAL_CONTRACT,
    forcing_index_for_target_date,
    target_index_for_date,
)


class SequenceDataset(Dataset[dict[str, Any]]):
    """Load manifest-defined contiguous forcing sequences and aligned targets."""

    def __init__(self, manifest_path: Path | str, split: str, cache_site_arrays: bool = False) -> None:
        with Path(manifest_path).open(newline="", encoding="utf-8") as handle:
            self.rows = [row for row in csv.DictReader(handle) if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No rows for split {split!r} in {manifest_path}")
        self.split = split
        contracts = {row.get("temporal_contract") for row in self.rows}
        if contracts != {TEMPORAL_CONTRACT}:
            raise ValueError(
                "Manifest does not declare the calendar-forcing/target-water-year v2 contract. "
                "Rebuild daily screens and manifests; historical v1 manifests are intentionally rejected."
            )
        self._static_cache: dict[str, torch.Tensor] = {}
        self._grid_cache: dict[str, Grid] = {}
        self.cache_site_arrays=cache_site_arrays; self._site_arrays: dict[str, tuple[list[np.ndarray],list[np.ndarray]]]={}

    def _arrays(self,row:dict[str,str],water_year:int)->tuple[list[np.ndarray],list[np.ndarray]]:
        site=row['site_id']
        if site not in self._site_arrays:
            directory=Path(row['dynamic_sequence_source']); dynamic=[]
            for channel in DYNAMIC_CHANNELS:
                with rasterio.open(directory/f'{channel}_{water_year}.tif') as ds: dynamic.append(ds.read(out_dtype='float32')[:,1:-1,1:-1])
            with xr.open_dataset(row['target_source'],engine='h5netcdf',decode_cf=False,mask_and_scale=False) as ds: targets=[np.asarray(ds[c].values,dtype=np.float32) for c in TARGET_CHANNELS]
            self._site_arrays[site]=(dynamic,targets)
        return self._site_arrays[site]

    def __len__(self) -> int:
        return len(self.rows)

    def _grid(self, row: dict[str, str]) -> Grid:
        site_id = row["site_id"]
        if site_id not in self._grid_cache:
            with rasterio.open(f"NETCDF:{row['target_source']}:soilmoisture") as dataset:
                self._grid_cache[site_id] = Grid(dataset.width, dataset.height, dataset.transform, dataset.crs)
        return self._grid_cache[site_id]

    def _static(self, row: dict[str, str]) -> torch.Tensor:
        site_id = row["site_id"]
        if site_id not in self._static_cache:
            stack = load_selected_static_stack(Path(row["static_source"]), self._grid(row))
            self._static_cache[site_id] = torch.from_numpy(stack)
        return self._static_cache[site_id]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        sequence_length = int(row["sequence_length"])
        water_year = int(row["water_year_end"])
        target_date = date.fromisoformat(row["target_date"])
        with rasterio.open(Path(row["dynamic_sequence_source"]) / f"{DYNAMIC_CHANNELS[0]}_{water_year}.tif") as reference:
            forcing_count = reference.count
        with xr.open_dataset(row["target_source"], engine="h5netcdf", decode_cf=False, mask_and_scale=False) as reference:
            target_count = int(reference.sizes["time"])
        target_index = target_index_for_date(target_date, water_year, target_count)
        forcing_end_index = forcing_index_for_target_date(target_date, water_year, forcing_count)
        if target_index is None or forcing_end_index is None:
            raise AssertionError(f"Manifest target date is outside the Jan--Sep forcing/target overlap: {row}")
        forcing_start_index = forcing_end_index - sequence_length + 1
        if forcing_start_index < 0:
            raise AssertionError(f"Manifest sequence starts before calendar-year forcing source: {row}")
        persisted = (int(row["target_time_index"]), int(row["forcing_start_index"]), int(row["forcing_end_index"]))
        derived = (target_index, forcing_start_index, forcing_end_index)
        if persisted != derived:
            raise AssertionError(f"Manifest source indices do not match its ISO dates: persisted={persisted}, derived={derived}, row={row}")
        dynamic: list[np.ndarray] = []
        site_dir = Path(row["dynamic_sequence_source"])
        indices = list(range(forcing_start_index + 1, forcing_end_index + 2))
        if self.cache_site_arrays:
            cached_dynamic,cached_targets=self._arrays(row,water_year); dynamic=[values[forcing_start_index:forcing_end_index+1] for values in cached_dynamic]
        else:
            cached_targets=[]
            for channel in DYNAMIC_CHANNELS:
                with rasterio.open(site_dir / f"{channel}_{water_year}.tif") as dataset: dynamic.append(dataset.read(indexes=indices,out_dtype='float32')[:,1:-1,1:-1])
        dynamic_tensor = torch.from_numpy(np.stack(dynamic, axis=1))  # [T, C, H, W]

        targets: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        if not self.cache_site_arrays:
            with xr.open_dataset(row["target_source"],engine="h5netcdf",decode_cf=False,mask_and_scale=False) as dataset: cached_targets=[np.asarray(dataset[c].values,dtype=np.float32) for c in TARGET_CHANNELS]
        for channel,all_values in zip(TARGET_CHANNELS,cached_targets):
                values=all_values[target_index]
                masked = mask_implausible(values, channel)
                targets.append(masked)
                masks.append(np.isfinite(masked))
        return {
            "site_id": row["site_id"], "bbox_id": row["bbox_id"], "target_date": row["target_date"],
            "dynamic_sequence": dynamic_tensor,
            "current_forcing": dynamic_tensor[-1],
            "static_stack": self._static(row),
            "target_stack": torch.from_numpy(np.stack(targets)),
            "valid_mask": torch.from_numpy(np.stack(masks)),
        }


def pad_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable bbox tensors with NaN/False while retaining explicit masks."""
    height = max(sample["target_stack"].shape[-2] for sample in samples)
    width = max(sample["target_stack"].shape[-1] for sample in samples)
    def pad(tensor: torch.Tensor, value: float) -> torch.Tensor:
        result = torch.full((*tensor.shape[:-2], height, width), value, dtype=tensor.dtype)
        result[..., : tensor.shape[-2], : tensor.shape[-1]] = tensor
        return result
    return {
        "site_id": [sample["site_id"] for sample in samples],
        "bbox_id": [sample["bbox_id"] for sample in samples],
        "target_date": [sample["target_date"] for sample in samples],
        "dynamic_sequence": torch.stack([pad(sample["dynamic_sequence"], float("nan")) for sample in samples]),
        "current_forcing": torch.stack([pad(sample["current_forcing"], float("nan")) for sample in samples]),
        "static_stack": torch.stack([pad(sample["static_stack"], float("nan")) for sample in samples]),
        "target_stack": torch.stack([pad(sample["target_stack"], float("nan")) for sample in samples]),
        "valid_mask": torch.stack([pad(sample["valid_mask"], False) for sample in samples]),
    }
