"""legacy milestone tests for exponential parametric learner behavior."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

import rtdfeatures
from rtdfeatures.learners import ExponentialKernelLearner, SimplexKernelLearner


def _make_time(n_rows: int) -> list[datetime]:
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [t0 + timedelta(minutes=i) for i in range(n_rows)]


def _exponential_dataset(
    *,
    rate_lambda: float,
    min_lag: int,
    max_lag: int,
    n_rows: int = 480,
    noise: float = 0.02,
    seed: int = 9,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=n_rows)
    lag_steps = np.arange(min_lag, max_lag + 1, dtype=np.int64)
    weights = np.exp(-rate_lambda * lag_steps.astype(np.float64))
    weights = weights / np.sum(weights)
    y = np.zeros(n_rows, dtype=np.float64)
    for idx in range(max_lag, n_rows):
        y[idx] = float(np.dot(weights, x[idx - lag_steps]))
    y += rng.normal(0.0, noise, size=n_rows)
    return pl.DataFrame(
        {
            "timestamp": _make_time(n_rows),
            "input_signal": x,
            "target_signal": y,
        }
    )


def test_exponential_fit_recovers_plausible_shape_and_parametric_provenance() -> None:
    df = _exponential_dataset(rate_lambda=0.9, min_lag=1, max_lag=6, noise=0.015, seed=42)
    fit = ExponentialKernelLearner(max_lag=6, min_lag=1, seed=17, loss="mse").fit(
        df,
        input_col="input_signal",
        target_col="target_signal",
        time_col="timestamp",
    )
    fit.kernel.validate()
    weights = np.asarray(fit.kernel.weights, dtype=np.float64)
    assert np.all(weights[:-1] >= weights[1:])
    assert fit.fit_provenance is not None
    assert fit.fit_provenance["parametric_family"] == "exponential"
    assert fit.fit_provenance["parametric_conversion_status"] == "ok"
    params = fit.fit_provenance["parametric_parameters"]
    assert params["rate_lambda"] > 0.0
    expected_rate_per_second = 0.9 / 60.0
    assert params["rate_lambda"] == pytest.approx(expected_rate_per_second, rel=0.6)
    assert fit.fit_provenance["parametric_lag_time_grid_seconds"] == pytest.approx(
        tuple(float(step) * 60.0 for step in fit.kernel.lag_steps),
        abs=1e-12,
    )
    assert fit.fit_diagnostics.validation_loss >= 0.0


def test_weak_case_emits_identifiability_warning_instead_of_false_confidence() -> None:
    df = _exponential_dataset(rate_lambda=0.8, min_lag=1, max_lag=6, noise=1.0, seed=43)
    fit = ExponentialKernelLearner(max_lag=6, min_lag=1, seed=18, max_epochs=200).fit(
        df,
        input_col="input_signal",
        target_col="target_signal",
        time_col="timestamp",
    )
    assert not fit.identifiability_report.is_reliable
    assert len(fit.identifiability_report.warning_codes) >= 1
    assert fit.fit_provenance is not None
    assert fit.fit_provenance["parametric_family"] == "exponential"


def test_invalid_parametric_settings_raise_clear_errors() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        ExponentialKernelLearner(max_lag=5, min_lag=1, init_rate_lambda=0.0)

    df = _exponential_dataset(
        rate_lambda=0.7,
        min_lag=0,
        max_lag=0,
        n_rows=120,
        noise=0.01,
        seed=44,
    )
    learner = ExponentialKernelLearner(max_lag=0, min_lag=0, init_rate_lambda=None)
    with pytest.raises(ValueError, match="Provide init_rate_lambda explicitly"):
        learner.fit(
            df,
            input_col="input_signal",
            target_col="target_signal",
            time_col="timestamp",
        )


def test_fit_is_deterministic_given_seed() -> None:
    df = _exponential_dataset(rate_lambda=0.6, min_lag=1, max_lag=5, noise=0.03, seed=45)
    fit_a = ExponentialKernelLearner(max_lag=5, min_lag=1, seed=99).fit(
        df,
        input_col="input_signal",
        target_col="target_signal",
        time_col="timestamp",
    )
    fit_b = ExponentialKernelLearner(max_lag=5, min_lag=1, seed=99).fit(
        df,
        input_col="input_signal",
        target_col="target_signal",
        time_col="timestamp",
    )
    assert fit_a.kernel.weights == pytest.approx(fit_b.kernel.weights, abs=1e-8)
    assert fit_a.fit_diagnostics.validation_loss == pytest.approx(
        fit_b.fit_diagnostics.validation_loss,
        abs=1e-12,
    )
    assert fit_a.fit_provenance is not None
    assert fit_b.fit_provenance is not None
    assert fit_a.fit_provenance["parametric_parameters"] == pytest.approx(
        fit_b.fit_provenance["parametric_parameters"],
        abs=1e-12,
    )


def test_baseline_fields_exist_and_are_finite_and_public_import_works() -> None:
    assert "ExponentialKernelLearner" in rtdfeatures.__all__
    assert rtdfeatures.ExponentialKernelLearner is ExponentialKernelLearner

    df = _exponential_dataset(rate_lambda=0.5, min_lag=1, max_lag=5, noise=0.04, seed=46)
    fit = ExponentialKernelLearner(max_lag=5, min_lag=1, seed=77, loss="huber").fit(
        df,
        input_col="input_signal",
        target_col="target_signal",
        time_col="timestamp",
    )
    baseline = fit.baseline_comparison
    assert np.isfinite(baseline.no_lag_validation_loss)
    assert np.isfinite(baseline.best_single_lag_validation_loss)
    assert np.isfinite(baseline.learned_validation_loss)
    assert baseline.uniform_validation_loss is not None
    assert baseline.exponential_validation_loss is not None
    assert np.isfinite(baseline.uniform_validation_loss)
    assert np.isfinite(baseline.exponential_validation_loss)

    simplex_fit = SimplexKernelLearner(max_lag=5, min_lag=1, seed=77, loss="huber").fit(
        df,
        input_col="input_signal",
        target_col="target_signal",
        time_col="timestamp",
    )
    assert simplex_fit.baseline_comparison.primary_ranking_metric == "validation_loss"
    assert baseline.primary_ranking_metric == "validation_loss"
