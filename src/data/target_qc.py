"""Transparent target plausibility masking for the canonical data contract."""

from __future__ import annotations

import numpy as np


TARGET_PLAUSIBILITY_RANGES: dict[str, tuple[float, float]] = {
    "soilmoisture": (0.0, 1.0),
    "tskin_am": (-10.0, 70.0),
    "tskin_pm": (-10.0, 70.0),
    "plc_am": (0.0, 100.0),
    "plc_pm": (0.0, 100.0),
}


def plausibility_mask(values: np.ndarray, target_name: str) -> np.ndarray:
    """Return a valid-data mask without altering physically valid target values."""
    lower, upper = TARGET_PLAUSIBILITY_RANGES[target_name]
    return np.isfinite(values) & (values >= lower) & (values <= upper)


def mask_implausible(values: np.ndarray, target_name: str) -> np.ndarray:
    """Convert only non-finite or out-of-range values to NaN; never clip them."""
    result = values.astype(np.float32, copy=True)
    result[~plausibility_mask(result, target_name)] = np.nan
    return result
