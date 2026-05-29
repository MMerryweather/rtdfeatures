"""Tests for richer transform reporting with stable feature-table output."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl

from rtdfeatures.features import KernelFeatureBuilder
from rtdfeatures.kernels import UniformKernel


def _make_time(n_rows: int) -> list[datetime]:
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [t0 + timedelta(minutes=i) for i in range(n_rows)]


def test_transform_schema_unchanged_with_richer_reporting() -> None:
    df = pl.DataFrame(
        {
            "timestamp": _make_time(5),
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    kernel = UniformKernel(max_lag_steps=1, min_lag_steps=0, dt=60.0, name="k")
    builder = KernelFeatureBuilder(
        kernels={"k": kernel},
        time_col="timestamp",
        numeric_cols=["x"],
    )

    out = builder.transform(df)
    report = builder.diagnose_transform(df)

    assert out.columns == [
        "timestamp",
        "k_num_x_wmean",
        "k_num_x_wstd",
        "k_num_x_wsum",
        "k_age_mean",
        "k_age_p50",
        "k_age_p90",
        "k_age_tail_gt_threshold",
    ]
    assert set(report.feature_names) == set(out.columns[1:])
    assert builder.last_transform_report == report


def test_report_counts_are_deterministic_and_include_kernel_level_summaries() -> None:
    df = pl.DataFrame(
        {
            "timestamp": _make_time(6),
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    k1 = UniformKernel(max_lag_steps=1, min_lag_steps=0, dt=60.0, name="k1")
    k2 = UniformKernel(max_lag_steps=2, min_lag_steps=0, dt=60.0, name="k2")
    builder = KernelFeatureBuilder(
        kernels={"k1": k1, "k2": k2},
        time_col="timestamp",
        numeric_cols=["x"],
    )

    report = builder.diagnose_transform(df)

    assert report.warmup_rows == 2
    assert report.missing_rows_by_feature["k1_num_x_wmean"] == 1
    assert report.missing_rows_by_feature["k2_num_x_wmean"] == 2
    assert report.zero_denominator_rows_by_feature["k1_num_x_wmean"] == 0
    assert report.zero_denominator_rows_by_feature["k2_num_x_wmean"] == 0
    assert report.missing_rows_by_kernel["k1"] == 3
    assert report.missing_rows_by_kernel["k2"] == 6
    assert report.warmup_unusable_summary == {
        "input_rows": 6,
        "warmup_rows": 2,
        "rows_after_warmup": 4,
        "rows_all_features_usable": 4,
        "rows_with_any_unusable_feature": 2,
    }
    assert report.collision_naming_summary is not None
    assert report.collision_naming_summary["kernel_names"] == ("k1", "k2")


def test_missing_and_zero_denominator_rows_are_distinct() -> None:
    df = pl.DataFrame(
        {
            "timestamp": _make_time(5),
            "x": [10.0, 11.0, 12.0, 13.0, 14.0],
            "w": [0.0, 0.0, 0.0, 1.0, 1.0],
        }
    )
    kernel = UniformKernel(max_lag_steps=1, min_lag_steps=0, dt=60.0, name="k")
    builder = KernelFeatureBuilder(
        kernels={"k": kernel},
        time_col="timestamp",
        numeric_cols=["x"],
        weight_col="w",
    )

    report = builder.diagnose_transform(df)

    # One warmup row + two zero-denominator rows for each numeric feature.
    assert report.missing_rows_by_feature["k_num_x_wmean"] == 3
    assert report.zero_denominator_rows_by_feature["k_num_x_wmean"] == 2
    assert (
        report.missing_rows_by_feature["k_num_x_wmean"]
        > report.zero_denominator_rows_by_feature["k_num_x_wmean"]
    )
    assert report.zero_denominator_rows_by_kernel["k"] == 6
