from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl
import pytest

from rtdfeatures.utils import (
    infer_regular_dt,
    lag_to_steps,
    resolve_and_validate_dt,
    validate_or_sort_time,
)


def _frame(times: list[datetime]) -> pl.DataFrame:
    return pl.DataFrame({"ts": times, "x": list(range(len(times)))})


def test_sorted_input_passes() -> None:
    df = _frame(
        [
            datetime(2026, 1, 1, 0, 0),
            datetime(2026, 1, 1, 0, 5),
            datetime(2026, 1, 1, 0, 10),
        ]
    )
    out = validate_or_sort_time(df, time_col="ts")
    assert out.equals(df)


def test_unsorted_input_raises_by_default() -> None:
    df = _frame(
        [
            datetime(2026, 1, 1, 0, 0),
            datetime(2026, 1, 1, 0, 10),
            datetime(2026, 1, 1, 0, 5),
        ]
    )
    with pytest.raises(ValueError, match="not sorted"):
        validate_or_sort_time(df, time_col="ts")


def test_unsorted_input_sorts_when_opted_in() -> None:
    df = _frame(
        [
            datetime(2026, 1, 1, 0, 0),
            datetime(2026, 1, 1, 0, 10),
            datetime(2026, 1, 1, 0, 5),
        ]
    )
    out = validate_or_sort_time(df, time_col="ts", order_by_time=True)
    assert out["ts"].to_list() == [
        datetime(2026, 1, 1, 0, 0),
        datetime(2026, 1, 1, 0, 5),
        datetime(2026, 1, 1, 0, 10),
    ]


def test_regular_grid_dt_is_inferred() -> None:
    df = _frame(
        [
            datetime(2026, 1, 1, 0, 0),
            datetime(2026, 1, 1, 0, 30),
            datetime(2026, 1, 1, 1, 0),
        ]
    )
    dt = infer_regular_dt(df, time_col="ts")
    assert dt == timedelta(minutes=30)


def test_supplied_dt_mismatch_raises() -> None:
    df = _frame(
        [
            datetime(2026, 1, 1, 0, 0),
            datetime(2026, 1, 1, 0, 30),
            datetime(2026, 1, 1, 1, 0),
        ]
    )
    with pytest.raises(ValueError, match="does not match observed grid"):
        resolve_and_validate_dt(df, time_col="ts", dt="5m")


def test_irregular_grid_raises() -> None:
    df = _frame(
        [
            datetime(2026, 1, 1, 0, 0),
            datetime(2026, 1, 1, 0, 5),
            datetime(2026, 1, 1, 0, 15),
        ]
    )
    with pytest.raises(ValueError, match="irregular"):
        infer_regular_dt(df, time_col="ts")


def test_lag_step_conversion_is_deterministic() -> None:
    assert lag_to_steps("2h", dt="30m", param_name="max_lag") == 4
    assert lag_to_steps("30m", dt=timedelta(minutes=5), param_name="min_lag") == 6
    assert lag_to_steps(7, dt="5m", param_name="max_lag") == 7


def test_lag_step_conversion_rejects_inconsistent_duration() -> None:
    with pytest.raises(ValueError, match="not aligned"):
        lag_to_steps("7m", dt="5m", param_name="max_lag")


def test_validate_or_sort_time_missing_column() -> None:
    df = pl.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError, match="not present"):
        validate_or_sort_time(df, time_col="ts")


def test_validate_or_sort_time_single_row_passes() -> None:
    df = pl.DataFrame({"ts": [datetime(2026, 1, 1, 0, 0)], "x": [1]})
    out = validate_or_sort_time(df, time_col="ts")
    assert out.equals(df)


def test_infer_regular_dt_missing_column() -> None:
    df = pl.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError, match="not present"):
        infer_regular_dt(df, time_col="ts")


def test_infer_regular_dt_single_row() -> None:
    df = pl.DataFrame({"ts": [datetime(2026, 1, 1, 0, 0)]})
    with pytest.raises(ValueError, match="At least two rows"):
        infer_regular_dt(df, time_col="ts")


def test_infer_regular_dt_null_values() -> None:
    null_ts = [datetime(2026, 1, 1, 0, 0), None, datetime(2026, 1, 1, 0, 10)]
    df = pl.DataFrame({"ts": null_ts, "x": [1, 2, 3]}, schema={"ts": pl.Datetime, "x": pl.Int64})
    with pytest.raises(ValueError, match="nulls"):
        infer_regular_dt(df, time_col="ts")


def test_infer_regular_dt_wrong_dtype() -> None:
    df = pl.DataFrame({"ts": [1, 2, 3], "x": [1, 2, 3]})
    with pytest.raises(ValueError, match="must be Date or Datetime"):
        infer_regular_dt(df, time_col="ts")


def test_infer_regular_dt_date_type() -> None:
    df = pl.DataFrame(
        {
            "ts": [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3)],
            "x": [1, 2, 3],
        }
    )
    dt = infer_regular_dt(df, time_col="ts")
    assert dt == timedelta(days=1)


def test_infer_regular_dt_non_positive_step() -> None:
    df = pl.DataFrame(
        {
            "ts": [
                datetime(2026, 1, 1, 0, 0),
                datetime(2026, 1, 1, 0, 0),
                datetime(2026, 1, 1, 0, 10),
            ],
            "x": [1, 2, 3],
        }
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        infer_regular_dt(df, time_col="ts")


def test_lag_to_steps_negative_int() -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        lag_to_steps(-1, dt="5m", param_name="max_lag")


def test_lag_to_steps_negative_duration() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        lag_to_steps(timedelta(hours=-1), dt="30m", param_name="max_lag")


def test_lag_to_steps_bad_string_format() -> None:
    with pytest.raises(ValueError, match="duration-like"):
        lag_to_steps("xyz", dt="5m", param_name="max_lag")


def test_lag_to_steps_wrong_type() -> None:
    with pytest.raises(TypeError, match="must be a str or datetime.timedelta"):
        lag_to_steps(3.14, dt="5m", param_name="max_lag")  # type: ignore[arg-type]


def test_duration_like_to_ns_negative_timedelta() -> None:
    from rtdfeatures.utils import _duration_like_to_ns

    with pytest.raises(ValueError, match="must be positive"):
        _duration_like_to_ns(timedelta(seconds=-10), param_name="dt")


def test_ns_to_duration_string_variants() -> None:
    from rtdfeatures.utils import _ns_to_duration_string

    assert _ns_to_duration_string(86400 * 1_000_000_000) == "1d"
    assert _ns_to_duration_string(3600 * 1_000_000_000) == "1h"
    assert _ns_to_duration_string(60 * 1_000_000_000) == "1m"
    assert _ns_to_duration_string(1_000_000_000) == "1s"
