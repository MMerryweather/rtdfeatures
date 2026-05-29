"""Shared learned-kernel integration tests for feature generation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from rtdfeatures.features import KernelFeatureBuilder
from rtdfeatures.kernels import UniformKernel
from rtdfeatures.learners import SharedSimplexKernelLearner, SimplexKernelLearner


def _make_time(n_rows: int) -> list[datetime]:
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [t0 + timedelta(minutes=i) for i in range(n_rows)]


def _make_shared_df(n_rows: int = 360) -> pl.DataFrame:
    rng = np.random.default_rng(1234)
    input_a = rng.normal(0.0, 1.0, size=n_rows)
    input_b = rng.normal(0.0, 1.0, size=n_rows)
    target_a = np.zeros(n_rows, dtype=np.float64)
    target_b = np.zeros(n_rows, dtype=np.float64)
    for idx in range(5, n_rows):
        target_a[idx] = 0.7 * input_a[idx - 3] + 0.3 * input_a[idx - 1]
        target_b[idx] = 0.6 * input_b[idx - 2] + 0.4 * input_b[idx - 4]
    target_a += rng.normal(0.0, 0.03, size=n_rows)
    target_b += rng.normal(0.0, 0.03, size=n_rows)
    return pl.DataFrame(
        {
            "timestamp": _make_time(n_rows),
            "input_a": input_a,
            "input_b": input_b,
            "target_a": target_a,
            "target_b": target_b,
        }
    )


def test_shared_fit_kernels_flow_into_builder_transform_contract() -> None:
    df = _make_shared_df()
    fit = SharedSimplexKernelLearner(max_lag=6, min_lag=0, seed=7, loss="mse").fit(
        df,
        input_cols=["input_a", "input_b"],
        target_cols=["target_a", "target_b"],
        time_col="timestamp",
    )
    builder = KernelFeatureBuilder(
        kernels=fit.to_kernels(),
        time_col="timestamp",
        numeric_cols=["input_a", "input_b"],
    )

    transformed = builder.transform(df)

    assert transformed.columns[0] == "timestamp"
    assert transformed.height == df.height
    assert "input_a" not in transformed.columns
    assert "input_b" not in transformed.columns
    assert "target_a" not in transformed.columns
    assert "target_b" not in transformed.columns
    assert "input_a->target_a_num_input_a_wmean" in transformed.columns
    assert "input_b->target_b_num_input_b_wmean" in transformed.columns

    report = builder.diagnose_transform(df)
    assert set(report.feature_names) == set(transformed.columns[1:])
    assert report.output_row_count == transformed.height


def test_shared_fit_to_kernels_rejects_name_collisions() -> None:
    df = _make_shared_df()
    fit = SharedSimplexKernelLearner(max_lag=5, min_lag=0, seed=9).fit(
        df,
        input_cols=["input_a", "input_b"],
        target_cols=["target_a", "target_b"],
        time_col="timestamp",
    )
    with pytest.raises(ValueError, match="collision"):
        fit.to_kernels(
            names={
                "input_a->target_a": "shared_kernel",
                "input_b->target_b": "shared_kernel",
            }
        )


def test_shared_fit_emits_weak_identifiability_warning_for_noisy_pair() -> None:
    n_rows = 420
    rng = np.random.default_rng(777)
    input_good = rng.normal(0.0, 1.0, size=n_rows)
    input_weak = rng.normal(0.0, 1.0, size=n_rows)
    target_good = np.zeros(n_rows, dtype=np.float64)
    for idx in range(4, n_rows):
        target_good[idx] = 0.65 * input_good[idx - 2] + 0.35 * input_good[idx - 4]
    target_good += rng.normal(0.0, 0.03, size=n_rows)
    target_weak = rng.normal(0.0, 1.4, size=n_rows)
    df = pl.DataFrame(
        {
            "timestamp": _make_time(n_rows),
            "input_good": input_good,
            "input_weak": input_weak,
            "target_good": target_good,
            "target_weak": target_weak,
        }
    )

    shared = SharedSimplexKernelLearner(max_lag=7, min_lag=0, seed=31, loss="mse").fit(
        df,
        input_cols=["input_good", "input_weak"],
        target_cols=["target_good", "target_weak"],
        time_col="timestamp",
    )
    weak_warnings = (
        shared.get_pair_result("input_weak->target_weak")
        .identifiability_report.warnings
    )
    assert (
        "Target signal appears noisy or weakly explained." in weak_warnings
        or "Kernel is too diffuse to interpret confidently." in weak_warnings
    )


def test_v01_public_workflow_regression_still_holds() -> None:
    n_rows = 380
    rng = np.random.default_rng(2201)
    x = rng.normal(0.0, 1.0, size=n_rows)
    y = np.zeros(n_rows, dtype=np.float64)
    for idx in range(4, n_rows):
        y[idx] = 0.8 * x[idx - 3] + 0.2 * x[idx - 1]
    y += rng.normal(0.0, 0.03, size=n_rows)
    df = pl.DataFrame(
        {
            "timestamp": _make_time(n_rows),
            "input_signal": x,
            "target_signal": y,
        }
    )
    fit = SimplexKernelLearner(max_lag=6, min_lag=0, seed=44, loss="huber").fit(
        df,
        input_col="input_signal",
        target_col="target_signal",
        time_col="timestamp",
    )

    builder = KernelFeatureBuilder(
        kernels={"learned": fit.kernel},
        time_col="timestamp",
        numeric_cols=["input_signal"],
    )
    features = builder.transform(df)
    report = builder.diagnose_transform(df)

    assert features.columns[0] == "timestamp"
    assert features.height == df.height
    assert "learned_num_input_signal_wmean" in features.columns
    assert report.output_row_count == features.height


def test_v01_feature_families_include_numeric_and_categorical_outputs() -> None:
    df = pl.DataFrame(
        {
            "timestamp": _make_time(5),
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "mode": ["A", "B", "A", "B", "A"],
        }
    )
    kernel = UniformKernel(max_lag_steps=1, min_lag_steps=0, dt=60.0, name="k")
    builder = KernelFeatureBuilder(
        kernels={"k": kernel},
        time_col="timestamp",
        numeric_cols=["x"],
        category_cols=["mode"],
    )

    out = builder.transform(df)

    expected_columns = {
        "k_num_x_wmean",
        "k_num_x_wstd",
        "k_num_x_wsum",
        "k_cat_mode_A_frac",
        "k_cat_mode_B_frac",
        "k_cat_mode_entropy",
        "k_age_mean",
        "k_age_p50",
        "k_age_p90",
        "k_age_tail_gt_threshold",
    }
    assert expected_columns.issubset(set(out.columns))

    row = out.row(2, named=True)
    assert row["k_num_x_wmean"] == pytest.approx(2.5)
    assert row["k_num_x_wsum"] == pytest.approx(2.5)
    assert row["k_num_x_wstd"] == pytest.approx(0.5)
    assert row["k_cat_mode_A_frac"] == pytest.approx(0.5)
    assert row["k_cat_mode_B_frac"] == pytest.approx(0.5)
    assert row["k_cat_mode_entropy"] == pytest.approx(np.log(2.0))
