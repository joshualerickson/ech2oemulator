"""Explicit calendar join rules for ECH2O targets and daily forcings.

Targets are stored by water-year end label: target band zero is October 1 of
``water_year_end - 1``.  The forcing TIFFs with that same label are instead
calendar-year files: forcing band zero is January 1 of ``water_year_end``.

The two sources must therefore be joined by ISO calendar date, never shared
band number.  Their usable overlap is January 1 through September 30 of the
water-year end year.
"""

from __future__ import annotations

from datetime import date, timedelta


TEMPORAL_CONTRACT = "target_water_year_forcing_calendar_year_v2"


def target_dates(water_year_end: int, count: int) -> list[date]:
    """Dates represented by target NetCDF bands (Oct 1 through Sep 30)."""
    start = date(water_year_end - 1, 10, 1)
    return [start + timedelta(days=index) for index in range(count)]


def forcing_dates(water_year_end: int, count: int) -> list[date]:
    """Dates represented by forcing TIFF bands (Jan 1 through Dec 31)."""
    start = date(water_year_end, 1, 1)
    return [start + timedelta(days=index) for index in range(count)]


def forcing_index_for_target_date(target_date: date, water_year_end: int, forcing_count: int) -> int | None:
    """Return the calendar-year forcing band index for a target date, if present."""
    start = date(water_year_end, 1, 1)
    index = (target_date - start).days
    return index if 0 <= index < forcing_count else None


def target_index_for_date(target_date: date, water_year_end: int, target_count: int) -> int | None:
    """Return the target NetCDF band index for an ISO date, if present."""
    index = (target_date - date(water_year_end - 1, 10, 1)).days
    return index if 0 <= index < target_count else None


def is_training_overlap(target_date: date, water_year_end: int) -> bool:
    """Whether target date has a corresponding forcing date in this contract."""
    return date(water_year_end, 1, 1) <= target_date <= date(water_year_end, 9, 30)
