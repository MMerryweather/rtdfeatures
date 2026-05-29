"""Tests for learner baseline evaluation and fit-result assembly."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from rtdfeatures.diagnostics import BaselineComparison, KernelFitResult
from rtdfeatures.kernels import LearnedKernel
from rtdfeatures.learners import (
    ExponentialKernelLearner,
    GammaKernelLearner,
    SimplexKernelLearner,
)
from rtdfeatures.learners._assembly import (
    BaselineLosses,
    FitAssemblyInput,
    assemble_kernel_fit_result,
    evaluate_baselines,
)
from rtdfeatures.learners._base import (
    LearnerConfig,
    PreparedFitData,
    _numpy_loss_on_finite_pairs,
    best_single_lag_validation_loss,
    exponential_validation_loss,
    prepare_fit_data,
    uniform_validation_loss,
)


def _make_time(n_rows: int) -> list[datetime]:
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [t0 + timedelta(seconds=i) for i in range(n_rows)]


def _dataset(n_rows: int = 260, seed: int = 123) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=n_rows)
    y = np.zeros(n_rows, dtype=np.float64)
    for idx in range(5, n_rows):
        y[idx] = 0.5 * x[idx - 2] + 0.3 * x[idx - 3] + 0.2 * x[idx - 5]
    y += rng.normal(0.0, 0.02, size=n_rows)
    return pl.DataFrame(
        {
            "timestamp": _make_time(n_rows),
            "input_signal": x,
            "target_signal": y,
        }
    )


def _prepared(df: pl.DataFrame) -> PreparedFitData:
    return prepare_fit_data(
        df,
        input_col="input_signal",
        target_col="target_signal",
        time_col="timestamp",
        order_by_time=False,
        config=LearnerConfig(
            max_lag=6,
            min_lag=1,
            dt=None,
            loss="huber",
            validation_fraction=0.2,
            huber_delta=1.0,
        ),
    )


def test_evaluate_baselines_returns_all_baselines() -> None:
    prepared = _prepared(_dataset())
    baselines = evaluate_baselines(prepared=prepared, loss="huber", huber_delta=1.0)

    assert np.isfinite(baselines.no_lag)
    assert np.isfinite(baselines.best_single_lag)
    assert baselines.uniform is not None
    assert baselines.exponential is not None
    assert np.isfinite(baselines.uniform)
    assert np.isfinite(baselines.exponential)
    assert np.isfinite(baselines.best_exponential_alpha)


def test_evaluate_baselines_matches_existing_helper_outputs() -> None:
    prepared = _prepared(_dataset(seed=124))
    baselines = evaluate_baselines(prepared=prepared, loss="huber", huber_delta=1.0)

    expected_no_lag = _numpy_loss_on_finite_pairs(
        prediction=prepared.no_lag_valid_scaled,
        target=prepared.y_valid_scaled,
        loss="huber",
        huber_delta=1.0,
    )
    expected_best_single = best_single_lag_validation_loss(
        x_valid_scaled=prepared.x_valid_scaled,
        y_valid_scaled=prepared.y_valid_scaled,
        loss="huber",
        huber_delta=1.0,
    )
    expected_uniform = uniform_validation_loss(
        x_valid_scaled=prepared.x_valid_scaled,
        y_valid_scaled=prepared.y_valid_scaled,
        loss="huber",
        huber_delta=1.0,
    )
    expected_exp, expected_alpha = exponential_validation_loss(
        x_valid_scaled=prepared.x_valid_scaled,
        y_valid_scaled=prepared.y_valid_scaled,
        loss="huber",
        huber_delta=1.0,
    )

    assert baselines.no_lag == expected_no_lag
    assert baselines.best_single_lag == expected_best_single
    assert baselines.uniform == expected_uniform
    assert baselines.exponential == expected_exp
    assert baselines.best_exponential_alpha == expected_alpha


def test_assemble_kernel_fit_result_returns_kernel_fit_result() -> None:
    prepared = _prepared(_dataset(seed=125))
    baselines = evaluate_baselines(prepared=prepared, loss="huber", huber_delta=1.0)
    lag_steps = tuple(range(prepared.min_lag_steps, prepared.max_lag_steps + 1))
    learned_weights = np.full(len(lag_steps), 1.0 / len(lag_steps), dtype=np.float64)
    kernel = LearnedKernel(
        weights=tuple(float(v) for v in learned_weights),
        lag_steps=lag_steps,
        dt=prepared.dt_seconds,
        min_lag_steps=prepared.min_lag_steps,
        max_lag_steps=prepared.max_lag_steps,
        name="input_signal->target_signal",
    )

    fit = assemble_kernel_fit_result(
        assembly=FitAssemblyInput(
            prepared=prepared,
            kernel=kernel,
            train_loss=0.1,
            validation_loss=0.2,
            learned_weights=learned_weights,
            seed=7,
            loss="huber",
            huber_delta=1.0,
            validation_fraction=0.2,
            smoothness_penalty=0.0,
            flat_floor=1e-8,
            baseline_improvement_margin=0.05,
            baseline_losses=baselines,
            extra_provenance={"parametric_family": "none"},
        )
    )

    assert isinstance(fit, KernelFitResult)


def test_assembled_result_contains_baseline_comparison() -> None:
    prepared = _prepared(_dataset(seed=126))
    baselines = evaluate_baselines(prepared=prepared, loss="huber", huber_delta=1.0)
    lag_steps = tuple(range(prepared.min_lag_steps, prepared.max_lag_steps + 1))
    learned_weights = np.full(len(lag_steps), 1.0 / len(lag_steps), dtype=np.float64)
    kernel = LearnedKernel(
        weights=tuple(float(v) for v in learned_weights),
        lag_steps=lag_steps,
        dt=prepared.dt_seconds,
        min_lag_steps=prepared.min_lag_steps,
        max_lag_steps=prepared.max_lag_steps,
        name="input_signal->target_signal",
    )

    fit = assemble_kernel_fit_result(
        assembly=FitAssemblyInput(
            prepared=prepared,
            kernel=kernel,
            train_loss=0.1,
            validation_loss=0.2,
            learned_weights=learned_weights,
            seed=7,
            loss="huber",
            huber_delta=1.0,
            validation_fraction=0.2,
            smoothness_penalty=0.0,
            flat_floor=1e-8,
            baseline_improvement_margin=0.05,
            baseline_losses=baselines,
        )
    )

    assert isinstance(fit.baseline_comparison, BaselineComparison)
    assert fit.baseline_comparison.uniform_validation_loss is not None
    assert fit.baseline_comparison.exponential_validation_loss is not None


def test_assembled_result_contains_fit_provenance() -> None:
    prepared = _prepared(_dataset(seed=127))
    baselines = evaluate_baselines(prepared=prepared, loss="huber", huber_delta=1.0)
    lag_steps = tuple(range(prepared.min_lag_steps, prepared.max_lag_steps + 1))
    learned_weights = np.full(len(lag_steps), 1.0 / len(lag_steps), dtype=np.float64)
    kernel = LearnedKernel(
        weights=tuple(float(v) for v in learned_weights),
        lag_steps=lag_steps,
        dt=prepared.dt_seconds,
        min_lag_steps=prepared.min_lag_steps,
        max_lag_steps=prepared.max_lag_steps,
        name="input_signal->target_signal",
    )

    fit = assemble_kernel_fit_result(
        assembly=FitAssemblyInput(
            prepared=prepared,
            kernel=kernel,
            train_loss=0.1,
            validation_loss=0.2,
            learned_weights=learned_weights,
            seed=7,
            loss="huber",
            huber_delta=1.0,
            validation_fraction=0.2,
            smoothness_penalty=0.0,
            flat_floor=1e-8,
            baseline_improvement_margin=0.05,
            baseline_losses=baselines,
            extra_provenance={"extra_flag": True},
        )
    )

    assert fit.fit_provenance is not None
    assert fit.fit_provenance["seed"] == 7
    assert fit.fit_provenance["extra_flag"] is True


def test_simplex_gamma_exponential_fit_result_schema_unchanged() -> None:
    df = _dataset(seed=128)
    fits = [
        SimplexKernelLearner(max_lag=6, min_lag=1, seed=21, max_epochs=80).fit(
            df,
            input_col="input_signal",
            target_col="target_signal",
            time_col="timestamp",
        ),
        GammaKernelLearner(max_lag=6, min_lag=1, seed=22, max_epochs=80).fit(
            df,
            input_col="input_signal",
            target_col="target_signal",
            time_col="timestamp",
        ),
        ExponentialKernelLearner(max_lag=6, min_lag=1, seed=23, max_epochs=80).fit(
            df,
            input_col="input_signal",
            target_col="target_signal",
            time_col="timestamp",
        ),
    ]
    required_provenance_keys = {
        "seed",
        "loss",
        "validation_fraction",
        "smoothness_penalty",
        "dt_seconds",
        "train_rows",
        "validation_rows",
        "total_valid_windows",
        "baseline_losses_use_configured_loss",
        "exponential_baseline_alpha_grid",
        "exponential_baseline_best_alpha",
    }
    for fit in fits:
        assert isinstance(fit, KernelFitResult)
        assert fit.fit_provenance is not None
        assert required_provenance_keys.issubset(fit.fit_provenance.keys())
        assert fit.fit_provenance["baseline_losses_use_configured_loss"] is True
        assert fit.fit_provenance["validation_rows"] > 0
        assert fit.fit_provenance["train_rows"] > fit.fit_provenance["validation_rows"]

        baseline = fit.baseline_comparison
        assert baseline.primary_ranking_metric == "validation_loss"
        assert baseline.uniform_validation_loss is not None
        assert baseline.exponential_validation_loss is not None
        assert "no_lag" in baseline.summary_by_baseline
        assert "best_single_lag" in baseline.summary_by_baseline
        assert (
            baseline.summary_by_baseline["no_lag"]["learned_validation_loss"]
            == baseline.learned_validation_loss
        )
        assert isinstance(
            baseline.summary_by_baseline["best_single_lag"]["beats_learned_by_margin"],
            bool,
        )


def test_evaluate_baselines_mse_loss() -> None:
    prepared = _prepared(_dataset())
    baselines = evaluate_baselines(prepared=prepared, loss="mse", huber_delta=1.0)
    assert np.isfinite(baselines.no_lag)
    assert np.isfinite(baselines.best_single_lag)
    assert baselines.uniform is not None
    assert baselines.exponential is not None
    assert np.isfinite(baselines.uniform)
    assert np.isfinite(baselines.exponential)
    assert np.isfinite(baselines.best_exponential_alpha)


def test_assemble_kernel_fit_result_with_none_baselines() -> None:
    prepared = _prepared(_dataset(seed=130))
    baselines = evaluate_baselines(prepared=prepared, loss="huber", huber_delta=1.0)
    lag_steps = tuple(range(prepared.min_lag_steps, prepared.max_lag_steps + 1))
    learned_weights = np.full(len(lag_steps), 1.0 / len(lag_steps), dtype=np.float64)
    kernel = LearnedKernel(
        weights=tuple(float(v) for v in learned_weights),
        lag_steps=lag_steps,
        dt=prepared.dt_seconds,
        min_lag_steps=prepared.min_lag_steps,
        max_lag_steps=prepared.max_lag_steps,
        name="input_signal->target_signal",
    )
    none_baselines = BaselineLosses(
        no_lag=baselines.no_lag,
        best_single_lag=baselines.best_single_lag,
        uniform=None,
        exponential=None,
        best_exponential_alpha=baselines.best_exponential_alpha,
    )
    fit = assemble_kernel_fit_result(
        assembly=FitAssemblyInput(
            prepared=prepared,
            kernel=kernel,
            train_loss=0.1,
            validation_loss=0.2,
            learned_weights=learned_weights,
            seed=7,
            loss="huber",
            huber_delta=1.0,
            validation_fraction=0.2,
            smoothness_penalty=0.0,
            flat_floor=1e-8,
            baseline_improvement_margin=0.05,
            baseline_losses=none_baselines,
            extra_provenance={"parametric_family": "none"},
        )
    )
    assert fit.baseline_comparison.uniform_validation_loss is None
    assert fit.baseline_comparison.exponential_validation_loss is None


def test_assemble_kernel_fit_result_without_extra_provenance() -> None:
    prepared = _prepared(_dataset(seed=131))
    baselines = evaluate_baselines(prepared=prepared, loss="huber", huber_delta=1.0)
    lag_steps = tuple(range(prepared.min_lag_steps, prepared.max_lag_steps + 1))
    learned_weights = np.full(len(lag_steps), 1.0 / len(lag_steps), dtype=np.float64)
    kernel = LearnedKernel(
        weights=tuple(float(v) for v in learned_weights),
        lag_steps=lag_steps,
        dt=prepared.dt_seconds,
        min_lag_steps=prepared.min_lag_steps,
        max_lag_steps=prepared.max_lag_steps,
        name="input_signal->target_signal",
    )
    fit = assemble_kernel_fit_result(
        assembly=FitAssemblyInput(
            prepared=prepared,
            kernel=kernel,
            train_loss=0.1,
            validation_loss=0.2,
            learned_weights=learned_weights,
            seed=7,
            loss="huber",
            huber_delta=1.0,
            validation_fraction=0.2,
            smoothness_penalty=0.0,
            flat_floor=1e-8,
            baseline_improvement_margin=0.05,
            baseline_losses=baselines,
        )
    )
    assert isinstance(fit, KernelFitResult)
