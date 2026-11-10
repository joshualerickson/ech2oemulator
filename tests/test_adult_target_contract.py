"""Regression checks for terra Z-band adult targets and water-year indexing."""

from __future__ import annotations

import tempfile
import unittest
import csv
from pathlib import Path

import numpy as np
import xarray as xr

from scripts.build_adult_target_manifest import screen_site
from scripts.audit_manifest_target_artifacts import parse_manifest
from src.data.phase2_qc import TARGET_CHANNELS
from src.data.target_source import read_target, target_time_count


class AdultTargetContractTest(unittest.TestCase):
    def test_per_variable_z_axes_and_selected_dates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            variables = {}
            for number, name in enumerate(TARGET_CHANNELS, start=1):
                values = np.full((365, 2, 3), 0.2 if name == "soilmoisture" else 20, dtype=np.float32)
                if name == "tskin_pm":
                    values[244, 0, 0] = 90  # One impossible June 2 pixel.
                variables[name] = ((f"Z{number}", "northing", "easting"), values)
            path = Path(directory) / "adult.nc"
            xr.Dataset(variables).to_netcdf(path, engine="h5netcdf")
            with xr.open_dataset(path, engine="h5netcdf", decode_cf=False, mask_and_scale=False) as dataset:
                self.assertEqual(target_time_count(dataset, "soilmoisture"), 365)
                self.assertEqual(read_target(dataset, "tskin_pm", [243, 244]).shape, (2, 2, 3))
            rows = [
                {"site_id": "TEST", "water_year_end": "2017", "target_date": day, "target_time_index": str(index)}
                for day, index in (("2017-06-01", 243), ("2017-06-02", 244))
            ]
            good, rejected = screen_site(rows, path)
            self.assertEqual(good, {243})
            self.assertEqual(rejected, {244: "tskin_pm_out_of_range"})

    def test_time_axis_remains_supported(self) -> None:
        dataset = xr.Dataset({"soilmoisture": (("time", "northing", "easting"), np.ones((3, 2, 2), dtype=np.float32))})
        self.assertEqual(target_time_count(dataset, "soilmoisture"), 3)
        self.assertEqual(read_target(dataset, "soilmoisture", [1]).shape, (1, 2, 2))

    def test_final_fit_rows_are_visible_to_artifact_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("site_id", "split", "water_year_end", "target_source"))
                writer.writeheader()
                writer.writerow({"site_id": "TEST", "split": "final_fit_train", "water_year_end": "2017", "target_source": "adult.nc"})
            self.assertEqual(parse_manifest(path)["TEST"]["target_source"], "adult.nc")


if __name__ == "__main__":
    unittest.main()
