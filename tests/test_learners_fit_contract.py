"""legacy milestone tests for simplex learner fit behavior."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from rtdfeatures.learners import SimplexKernelLearner


def _make_time(n_rows: int) -> list[datetime]:
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [t0 + timedelta(minutes=i) for i in range(n_rows)]


def _synth_df(
    weights: tuple[float, ...], lag_steps: tuple[int, ...], n_rows: int = 400
) -> pl.DataFrame:
    rng = np.random.default_rng(11)
    x = rng.normal(0.0, 1.0, size=n_rows)
    y = np.zeros(n_rows, dtype=np.float64)
    for idx in range(max(lag_steps), n_rows):
        y[idx] = sum(w * x[idx - lag] for w, lag in zip(weights, lag_steps))
    y += rng.normal(0.0, 0.03, size=n_rows)
    return pl.DataFrame(
        {
            "timestamp": _make_time(n_rows),
            "input_signal": x,
            "target_signal": y,
        }
    )


def test_fit_recovers_plausible_single_delay_kernel() -> None:
    df = _synth_df(weights=(1.0,), lag_steps=(3,))
    learner = SimplexKernelLearner(max_lag=5, min_lag=0, seed=7, loss="huber")
    fit = learner.fit(
        df,
        input_col="input_signal",
        target_col="target_signal",
        time_col="timestamp",
    )
    fit.kernel.validate()
    dominant_idx = int(np.argmax(np.asarray(fit.kernel.weights)))
    assert fit.kernel.lag_steps[dominant_idx] == 3
    assert fit.kernel.weights[dominant_idx] > 0.70


def test_fit_recovers_plausible_spread_kernel() -> None:
    df = _synth_df(weights=(0.2, 0.5, 0.3), lag_steps=(1, 2, 3))
    learner = SimplexKernelLearner(max_lag=5, min_lag=0, seed=42, loss="mse")
    fit = learner.fit(
        df,
        input_col="input_signal",
        target_col="target_signal",
        time_col="timestamp",
    )
    lag_to_weight = dict(zip(fit.kernel.lag_steps, fit.kernel.weights))
    assert lag_to_weight[2] > lag_to_weight[1]
    assert lag_to_weight[2] > lag_to_weight[3]
    assert (lag_to_weight[1] + lag_to_weight[2] + lag_to_weight[3]) > 0.70


def test_fit_is_deterministic_given_seed() -> None:
    df = _synth_df(weights=(0.15, 0.7, 0.15), lag_steps=(2, 3, 4))
    learner_a = SimplexKernelLearner(max_lag=6, min_lag=0, seed=99)
    learner_b = SimplexKernelLearner(max_lag=6, min_lag=0, seed=99)
    fit_a = learner_a.fit(
        df, input_col="input_signal", target_col="target_signal", time_col="timestamp"
    )
    fit_b = learner_b.fit(
        df, input_col="input_signal", target_col="target_signal", time_col="timestamp"
    )
    assert fit_a.kernel.weights == pytest.approx(fit_b.kernel.weights, abs=1e-7)


def test_missing_windows_are_dropped() -> None:
    df = _synth_df(weights=(0.6, 0.4), lag_steps=(1, 2))
    df_missing = df.with_columns(
        pl.when(pl.int_range(0, pl.len()) % 19 == 0)
        .then(None)
        .otherwise(pl.col("input_signal"))
        .alias("input_signal")
    )
    learner = SimplexKernelLearner(max_lag=4, min_lag=0, seed=3)
    fit = learner.fit(
        df_missing,
        input_col="input_signal",
        target_col="target_signal",
        time_col="timestamp",
    )
    total_possible_windows = df.height - 4
    assert fit.fit_provenance is not None
    assert fit.fit_provenance["total_valid_windows"] < total_possible_windows


def test_lag_bounds_match_configured_min_and_max() -> None:
    df = _synth_df(weights=(0.5, 0.5), lag_steps=(4, 5))
    learner = SimplexKernelLearner(max_lag=6, min_lag=2, seed=21)
    fit = learner.fit(
        df,
        input_col="input_signal",
        target_col="target_signal",
        time_col="timestamp",
    )
    assert fit.kernel.min_lag_steps == 2
    assert fit.kernel.max_lag_steps == 6
    assert fit.kernel.lag_steps[0] == 2
    assert fit.kernel.lag_steps[-1] == 6


def test_smoothness_penalty_affects_optimization() -> None:
    df = _synth_df(weights=(1.0,), lag_steps=(3,))
    fit_no_penalty = SimplexKernelLearner(max_lag=7, min_lag=0, seed=5, smoothness_penalty=0.0).fit(
        df,
        input_col="input_signal",
        target_col="target_signal",
        time_col="timestamp",
    )
    fit_with_penalty = SimplexKernelLearner(
        max_lag=7, min_lag=0, seed=5, smoothness_penalty=5.0
    ).fit(
        df,
        input_col="input_signal",
        target_col="target_signal",
        time_col="timestamp",
    )
    w0 = np.asarray(fit_no_penalty.kernel.weights)
    w1 = np.asarray(fit_with_penalty.kernel.weights)
    rough0 = float(np.mean((w0[1:] - w0[:-1]) ** 2))
    rough1 = float(np.mean((w1[1:] - w1[:-1]) ** 2))
    assert rough1 < rough0
    fit_with_penalty.kernel.validate()
