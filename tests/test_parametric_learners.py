"""legacy milestone tests for gamma parametric learner behavior."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

import rtdfeatures
from rtdfeatures.kernels.parametric import discrete_gamma_weights
from rtdfeatures.learners import GammaKernelLearner, SimplexKernelLearner


def _make_time(n_rows: int) -> list[datetime]:
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [t0 + timedelta(minutes=i) for i in range(n_rows)]


def _gamma_dataset(
    *,
    shape_alpha: float,
    rate_beta: float,
    min_lag: int,
    max_lag: int,
    n_rows: int = 520,
    noise: float = 0.02,
    seed: int = 21,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=n_rows)
    lag_steps = tuple(range(min_lag, max_lag + 1))
    weights = np.asarray(
        discrete_gamma_weights(
            shape_alpha=shape_alpha,
            rate_beta=rate_beta / 60.0,
            lag_steps=lag_steps,
            dt=60.0,
        ),
        dtype=np.float64,
    )
    lag_arr = np.asarray(lag_steps, dtype=np.int64)
    y = np.zeros(n_rows, dtype=np.float64)
    for idx in range(max_lag, n_rows):
        y[idx] = float(np.dot(weights, x[idx - lag_arr]))
    y += rng.normal(0.0, noise, size=n_rows)
    return pl.DataFrame(
        {
            "timestamp": _make_time(n_rows),
            "input_signal": x,
            "target_signal": y,
        }
    )


def test_gamma_fit_recovers_plausible_unimodal_shape_and_provenance() -> None:
    df = _gamma_dataset(shape_alpha=3.2, rate_beta=0.85, min_lag=1, max_lag=8, noise=0.015, seed=30)
    fit = GammaKernelLearner(max_lag=8, min_lag=1, seed=101, loss="mse").fit(
        df,
        input_col="input_signal",
        target_col="target_signal",
        time_col="timestamp",
    )
    fit.kernel.validate()
    weights = np.asarray(fit.kernel.weights, dtype=np.float64)
    peak_idx = int(np.argmax(weights))
    assert peak_idx > 0
    assert peak_idx < (weights.size - 1)
    assert fit.fit_diagnostics.mean_lag is not None
    assert fit.fit_diagnostics.mean_lag == pytest.approx((3.2 / 0.85) * 60.0, rel=0.6)
    assert fit.fit_provenance is not None
    assert fit.fit_provenance["parametric_family"] == "gamma"
    assert fit.fit_provenance["parametric_conversion_status"] == "ok"
    params = fit.fit_provenance["parametric_parameters"]
    assert params["shape_alpha"] > 0.0
    assert params["rate_beta"] > 0.0
    assert fit.fit_provenance["parametric_lag_time_grid_seconds"] == pytest.approx(
        tuple(float(step) * 60.0 for step in fit.kernel.lag_steps),
        abs=1e-12,
    )


def test_gamma_boundary_or_noisy_case_emits_identifiability_warning() -> None:
    df = _gamma_dataset(shape_alpha=2.8, rate_beta=0.7, min_lag=1, max_lag=8, noise=1.1, seed=31)
    fit = GammaKernelLearner(max_lag=8, min_lag=1, seed=102, max_epochs=220).fit(
        df,
        input_col="input_signal",
        target_col="target_signal",
        time_col="timestamp",
    )
    assert not fit.identifiability_report.is_reliable
    assert len(fit.identifiability_report.warning_codes) >= 1
    assert fit.fit_provenance is not None
    assert fit.fit_provenance["parametric_family"] == "gamma"


def test_gamma_invalid_init_settings_raise_clear_errors() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        GammaKernelLearner(max_lag=6, min_lag=1, init_shape_alpha=0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        GammaKernelLearner(max_lag=6, min_lag=1, init_rate_beta=0.0)

    df = _gamma_dataset(
        shape_alpha=1.2,
        rate_beta=0.5,
        min_lag=0,
        max_lag=2,
        n_rows=120,
        noise=0.01,
        seed=32,
    )
    learner = GammaKernelLearner(max_lag=2, min_lag=0, init_shape_alpha=1.0, init_rate_beta=1.0)
    with pytest.raises(
        ValueError, match="must be strictly greater than 1.0 when lag grid includes zero lag"
    ):
        learner.fit(
            df,
            input_col="input_signal",
            target_col="target_signal",
            time_col="timestamp",
        )


def test_gamma_zero_only_lag_grid_raises_clear_error() -> None:
    rng = np.random.default_rng(35)
    n_rows = 120
    df = pl.DataFrame(
        {
            "timestamp": _make_time(n_rows),
            "input_signal": rng.normal(0.0, 1.0, size=n_rows),
            "target_signal": rng.normal(0.0, 1.0, size=n_rows),
        }
    )
    learner = GammaKernelLearner(max_lag=0, min_lag=0)
    with pytest.raises(
        ValueError,
        match=(
            "requires at least one strictly positive lag step; "
            "min_lag=0 and max_lag=0 is not supported"
        ),
    ):
        learner.fit(
            df,
            input_col="input_signal",
            target_col="target_signal",
            time_col="timestamp",
        )


def test_gamma_fit_is_deterministic_given_seed() -> None:
    df = _gamma_dataset(shape_alpha=2.6, rate_beta=0.8, min_lag=1, max_lag=7, noise=0.03, seed=33)
    fit_a = GammaKernelLearner(max_lag=7, min_lag=1, seed=103).fit(
        df,
        input_col="input_signal",
        target_col="target_signal",
        time_col="timestamp",
    )
    fit_b = GammaKernelLearner(max_lag=7, min_lag=1, seed=103).fit(
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


def test_gamma_baseline_fields_are_finite_and_public_export_works() -> None:
    assert "GammaKernelLearner" in rtdfeatures.__all__
    assert rtdfeatures.GammaKernelLearner is GammaKernelLearner

    df = _gamma_dataset(shape_alpha=2.7, rate_beta=0.75, min_lag=1, max_lag=7, noise=0.04, seed=34)
    fit = GammaKernelLearner(max_lag=7, min_lag=1, seed=104, loss="huber").fit(
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

    simplex_fit = SimplexKernelLearner(max_lag=7, min_lag=1, seed=104, loss="huber").fit(
        df,
        input_col="input_signal",
        target_col="target_signal",
        time_col="timestamp",
    )
    assert simplex_fit.baseline_comparison.primary_ranking_metric == "validation_loss"
    assert baseline.primary_ranking_metric == "validation_loss"
