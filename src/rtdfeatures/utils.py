"""Shared time-grid and lag utilities for current contracts."""

from __future__ import annotations

import re
from datetime import timedelta
from typing import Final, cast

import polars as pl

_DURATION_PATTERN: Final[re.Pattern[str]] = re.compile(r"^(?P<value>\d+)(?P<unit>[smhd])$")
_UNIT_SECONDS: Final[dict[str, int]] = {
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86400,
}


def validate_or_sort_time(
    df: pl.DataFrame, *, time_col: str, order_by_time: bool = False
) -> pl.DataFrame:
    """Validate time ordering or sort when explicitly requested."""
    if time_col not in df.columns:
        raise ValueError(f"time_col '{time_col}' is not present in the input DataFrame.")
    if order_by_time:
        return df.sort(time_col)
    if df.height <= 1:
        return df
    if not df.get_column(time_col).is_sorted():
        raise ValueError(
            f"Input is not sorted by '{time_col}'. "
            "Pass order_by_time=True to opt into sorting."
        )
    return df


def infer_regular_dt(df: pl.DataFrame, *, time_col: str) -> timedelta:
    """Infer dt from a regular time grid, raising on irregular grids."""
    if time_col not in df.columns:
        raise ValueError(f"time_col '{time_col}' is not present in the input DataFrame.")
    if df.height < 2:
        raise ValueError("At least two rows are required to infer dt from the time grid.")

    delta_ns = _infer_regular_dt_ns(df.get_column(time_col), time_col=time_col)
    return timedelta(microseconds=delta_ns / 1_000)


def resolve_and_validate_dt(
    df: pl.DataFrame,
    *,
    time_col: str,
    dt: str | timedelta | None,
) -> timedelta:
    """Resolve dt from data or validate user-supplied dt against the observed grid."""
    observed_dt_ns = _infer_regular_dt_ns(df.get_column(time_col), time_col=time_col)
    if dt is None:
        return timedelta(microseconds=observed_dt_ns / 1_000)

    expected_dt_ns = _duration_like_to_ns(dt, param_name="dt")
    if expected_dt_ns != observed_dt_ns:
        raise ValueError(
            f"Supplied dt ({_ns_to_duration_string(expected_dt_ns)}) does not match observed grid "
            f"({_ns_to_duration_string(observed_dt_ns)})."
        )
    return timedelta(microseconds=expected_dt_ns / 1_000)


def lag_to_steps(value: int | str | timedelta, *, dt: str | timedelta, param_name: str) -> int:
    """Convert a lag-like value into integer lag steps."""
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{param_name} must be non-negative.")
        return value

    lag_ns = _duration_like_to_ns(value, param_name=param_name)
    dt_ns = _duration_like_to_ns(dt, param_name="dt")
    if lag_ns < 0:
        raise ValueError(f"{param_name} must be non-negative.")
    if lag_ns % dt_ns != 0:
        raise ValueError(
            f"{param_name}={value!r} is not aligned to dt={dt!r}; "
            f"expected an exact multiple of dt."
        )
    return lag_ns // dt_ns


def _infer_regular_dt_ns(time_series: pl.Series, *, time_col: str) -> int:
    if time_series.len() < 2:
        raise ValueError("At least two rows are required to infer dt from the time grid.")

    if time_series.null_count() > 0:
        raise ValueError(
            f"time_col '{time_col}' contains nulls; a complete regular grid is required."
        )

    dtype = time_series.dtype
    if dtype not in (pl.Date, pl.Datetime):
        raise ValueError(
            f"time_col '{time_col}' must be Date or Datetime dtype, got {dtype!s}."
        )

    if dtype == pl.Date:
        time_ns = time_series.cast(pl.Datetime("ns"))
    else:
        time_ns = time_series.dt.cast_time_unit("ns")

    step_ns = time_ns.diff().drop_nulls().cast(pl.Int64)
    if step_ns.len() == 0:
        raise ValueError("At least two rows are required to infer dt from the time grid.")

    min_step_raw = step_ns.min()
    max_step_raw = step_ns.max()
    if min_step_raw is None or max_step_raw is None:
        raise ValueError("At least two rows are required to infer dt from the time grid.")
    min_step = int(cast(int, min_step_raw))
    max_step = int(cast(int, max_step_raw))
    if min_step <= 0:
        raise ValueError(
            f"time_col '{time_col}' must be strictly increasing with no duplicates."
        )
    if min_step != max_step:
        raise ValueError(
            f"time_col '{time_col}' is irregular; observed varying steps "
            f"from {_ns_to_duration_string(min_step)} to {_ns_to_duration_string(max_step)}."
        )
    return min_step


def _duration_like_to_ns(value: str | timedelta, *, param_name: str) -> int:
    if isinstance(value, timedelta):
        seconds = value.total_seconds()
        if seconds <= 0:
            raise ValueError(f"{param_name} must be positive.")
        return int(seconds * 1_000_000_000)

    if isinstance(value, str):
        match = _DURATION_PATTERN.fullmatch(value.strip())
        if not match:
            raise ValueError(
                f"{param_name} must be a duration-like string such as '5m', '30m', '2h', or '1d'."
            )
        magnitude = int(match.group("value"))
        unit = match.group("unit")
        seconds = magnitude * _UNIT_SECONDS[unit]
        return seconds * 1_000_000_000

    raise TypeError(f"{param_name} must be a str or datetime.timedelta.")


def _ns_to_duration_string(ns: int) -> str:
    total_seconds = ns // 1_000_000_000
    if total_seconds % 86400 == 0:
        return f"{total_seconds // 86400}d"
    if total_seconds % 3600 == 0:
        return f"{total_seconds // 3600}h"
    if total_seconds % 60 == 0:
        return f"{total_seconds // 60}m"
    return f"{total_seconds}s"
