"""Tests for synthetic parametric-family validation helpers."""

from __future__ import annotations

import polars as pl

from rtdfeatures.features import KernelFeatureBuilder
from rtdfeatures.kernels import Kernel
from rtdfeatures.learners import (
    ExponentialKernelLearner,
    GammaKernelLearner,
    SimplexKernelLearner,
)
from rtdfeatures.reporting import (
    learner_diagnostic_comparison_table,
    learner_diagnostic_warning_table,
)
from rtdfeatures.synthetic import (
    make_exponential_kernel_dataset,
    make_gamma_kernel_dataset,
    make_misspecified_parametric_dataset,
    make_weak_parametric_identifiability_dataset,
)


def _assert_regular_grid(df: pl.DataFrame, *, time_col: str = "time") -> None:
    diffs = df.get_column(time_col).cast(pl.Int64).diff().drop_nulls().to_list()
    assert diffs
    assert len({round(float(d), 12) for d in diffs}) == 1


def _build_features_from_fit(df: pl.DataFrame, fit_kernel: Kernel) -> pl.DataFrame:
    builder = KernelFeatureBuilder(
        kernels={"learned": fit_kernel},
        time_col="time",
        numeric_cols=["input_signal"],
    )
    return pl.DataFrame(builder.transform(df))


def test_exponential_synthetic_helper_end_to_end_and_metadata_contract() -> None:
    synthetic = make_exponential_kernel_dataset(seed=101, n_rows=320, dt=60.0, noise_std=0.02)
    meta = synthetic.true_kernels["input_signal->target_signal"]
    assert meta["parametric_family"] == "exponential"
    assert meta["parametric_parameters"]["rate_lambda"] > 0.0
    _assert_regular_grid(synthetic.data)

    fit = ExponentialKernelLearner(
        max_lag=meta["max_lag"],
        min_lag=meta["min_lag"],
        dt="60s",
        seed=101,
        max_epochs=300,
    ).fit(
        synthetic.data,
        input_col="input_signal",
        target_col="target_signal",
        time_col="time",
    )
    fit.kernel.validate()
    features = _build_features_from_fit(synthetic.data, fit.kernel)
    assert "time" in features.columns
    assert any(col.startswith("learned_num_input_signal_") for col in features.columns)


def test_gamma_synthetic_helper_end_to_end_and_metadata_contract() -> None:
    synthetic = make_gamma_kernel_dataset(seed=103, n_rows=360, dt=60.0, noise_std=0.02)
    meta = synthetic.true_kernels["input_signal->target_signal"]
    assert meta["parametric_family"] == "gamma"
    assert meta["parametric_parameters"]["shape_alpha"] > 0.0
    assert meta["parametric_parameters"]["rate_beta"] > 0.0
    _assert_regular_grid(synthetic.data)

    fit = GammaKernelLearner(
        max_lag=meta["max_lag"],
        min_lag=meta["min_lag"],
        dt="60s",
        seed=103,
        max_epochs=320,
    ).fit(
        synthetic.data,
        input_col="input_signal",
        target_col="target_signal",
        time_col="time",
    )
    fit.kernel.validate()
    features = _build_features_from_fit(synthetic.data, fit.kernel)
    assert "time" in features.columns
    assert any(col.startswith("learned_num_input_signal_") for col in features.columns)


def test_deterministic_seed_handling_for_parametric_helpers() -> None:
    a = make_gamma_kernel_dataset(seed=131, n_rows=260, dt=60.0)
    b = make_gamma_kernel_dataset(seed=131, n_rows=260, dt=60.0)
    c = make_gamma_kernel_dataset(seed=132, n_rows=260, dt=60.0)

    assert a.data.equals(b.data)
    assert a.true_kernels == b.true_kernels
    assert a.scenario == b.scenario
    assert not a.data.equals(c.data)


def test_parametric_vs_simplex_comparison_table_contains_baselines() -> None:
    synthetic = make_exponential_kernel_dataset(seed=151, n_rows=320, dt=60.0, noise_std=0.03)
    max_lag = synthetic.true_kernels["input_signal->target_signal"]["max_lag"]
    min_lag = synthetic.true_kernels["input_signal->target_signal"]["min_lag"]

    fits = {
        "simplex": SimplexKernelLearner(max_lag=max_lag, min_lag=min_lag, dt="60s", seed=151).fit(
            synthetic.data,
            input_col="input_signal",
            target_col="target_signal",
            time_col="time",
        ),
        "exponential": ExponentialKernelLearner(
            max_lag=max_lag,
            min_lag=min_lag,
            dt="60s",
            seed=151,
        ).fit(
            synthetic.data,
            input_col="input_signal",
            target_col="target_signal",
            time_col="time",
        ),
    }
    comparison = learner_diagnostic_comparison_table(fits)
    assert set(comparison["learner_family"].to_list()) == {"simplex", "exponential"}
    assert set(comparison.filter(pl.col("row_type") == "baseline")["candidate"].to_list()) >= {
        "no_lag",
        "best_single_lag",
    }


def test_misspecified_and_weak_parametric_fixtures_emit_clear_warnings() -> None:
    misspecified = make_misspecified_parametric_dataset(
        seed=171, n_rows=420, dt=60.0, noise_std=0.01
    )
    weak = make_weak_parametric_identifiability_dataset(
        seed=181, n_rows=360, dt=60.0, noise_std=0.9
    )

    misspecified_meta = misspecified.true_kernels["input_signal->target_signal"]
    assert misspecified_meta["parametric_family"] == "gamma"
    assert misspecified.scenario["params"]["expected_misspecified_family"] == "exponential"

    max_lag = misspecified_meta["max_lag"]
    min_lag = misspecified_meta["min_lag"]
    fits = {
        "simplex": SimplexKernelLearner(max_lag=max_lag, min_lag=min_lag, dt="60s", seed=171).fit(
            misspecified.data,
            input_col="input_signal",
            target_col="target_signal",
            time_col="time",
        ),
        "gamma": GammaKernelLearner(max_lag=max_lag, min_lag=min_lag, dt="60s", seed=171).fit(
            misspecified.data,
            input_col="input_signal",
            target_col="target_signal",
            time_col="time",
        ),
        "exponential": ExponentialKernelLearner(
            max_lag=max_lag,
            min_lag=min_lag,
            dt="60s",
            seed=171,
        ).fit(
            misspecified.data,
            input_col="input_signal",
            target_col="target_signal",
            time_col="time",
        ),
    }
    warning_table = learner_diagnostic_warning_table(fits)
    assert warning_table.height >= 1
    assert "exponential" in set(warning_table["learner_family"].to_list())

    weak_fit = GammaKernelLearner(
        max_lag=weak.true_kernels["input_signal->target_signal"]["max_lag"],
        min_lag=weak.true_kernels["input_signal->target_signal"]["min_lag"],
        dt="60s",
        seed=181,
        max_epochs=220,
    ).fit(
        weak.data,
        input_col="input_signal",
        target_col="target_signal",
        time_col="time",
    )
    assert not weak_fit.identifiability_report.is_reliable
    assert len(weak_fit.identifiability_report.warning_codes) >= 1
