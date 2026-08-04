"""Portable resolution of the raw ECH2O input root."""
from __future__ import annotations

import os
from pathlib import Path


def configured_data_root() -> Path | None:
    """Return ECH2O_DATA_ROOT when configured, otherwise None.

    CLI callers keep ``--data-root`` as the explicit, higher-priority choice.
    """
    value = os.environ.get("ECH2O_DATA_ROOT")
    return Path(value).expanduser() if value else None
