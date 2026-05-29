"""Snapshot tests: generated feature DataFrames must match expected schema."""

from __future__ import annotations

import polars as pl

from rtdfeatures import FixedDelayKernel, KernelFeatureBuilder


def _schema_signature(df: pl.DataFrame) -> dict[str, pl.DataType]:
    return {name: dtype for name, dtype in zip(df.columns, df.dtypes)}


NUMERIC_COLS = {
    "t": pl.Int64,
    "k_num_x_wmean": pl.Float64,
    "k_num_x_wstd": pl.Float64,
    "k_num_x_wsum": pl.Float64,
    "k_age_mean": pl.Float64,
    "k_age_p50": pl.Float64,
    "k_age_p90": pl.Float64,
    "k_age_tail_gt_threshold": pl.Float64,
}

CAT_KEYS = {
    "t",
    "k_num_x_wmean",
    "k_num_x_wstd",
    "k_num_x_wsum",
    "k_cat_cat_A_frac",
    "k_cat_cat_B_frac",
    "k_cat_cat_entropy",
    "k_age_mean",
    "k_age_p50",
    "k_age_p90",
    "k_age_tail_gt_threshold",
}


def test_numeric_feature_schema_stability() -> None:
    df = pl.DataFrame({"t": range(10), "x": [1.0] * 10})
    builder = KernelFeatureBuilder(
        kernels={"k": FixedDelayKernel(delay_steps=1, max_lag_steps=3, dt=1.0)},
        time_col="t",
        numeric_cols=["x"],
    )
    result = builder.transform_result(df)
    seen = _schema_signature(result.features)
    assert seen == NUMERIC_COLS, f"Schema mismatch: {seen}"


def test_categorical_feature_schema_stability() -> None:
    df = pl.DataFrame({
        "t": range(10),
        "x": [1.0] * 10,
        "cat": ["A", "B"] * 5,
    })
    builder = KernelFeatureBuilder(
        kernels={"k": FixedDelayKernel(delay_steps=1, max_lag_steps=3, dt=1.0)},
        time_col="t",
        numeric_cols=["x"],
        category_cols=["cat"],
    )
    result = builder.transform_result(df)
    seen = _schema_signature(result.features)
    missing = CAT_KEYS - seen.keys()
    assert not missing, f"Missing expected columns: {missing}"
    extra = set(seen.keys()) - CAT_KEYS
    assert not extra, f"Unexpected columns: {extra}"
