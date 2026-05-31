"""Tests for delayed-exponential and log-normal learners."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from rtdfeatures.learners import DelayedExponentialKernelLearner, LogNormalKernelLearner
from rtdfeatures.synthetic import (
    make_delayed_exponential_kernel_dataset,
    make_lognormal_kernel_dataset,
)


def test_delayed_exponential_fit_recovers_plausible_delay_and_provenance() -> None:
    synthetic = make_delayed_exponential_kernel_dataset(
        seed=211,
        n_rows=360,
        dt=60.0,
        min_lag_steps=0,
        max_lag_steps=10,
        delay=180.0,
        rate_lambda=0.03,
        noise_std=0.02,
    )
    meta = synthetic.true_kernels["input_signal->target_signal"]

    fit = DelayedExponentialKernelLearner(
        max_lag=meta["max_lag"],
        min_lag=meta["min_lag"],
        dt="60s",
        seed=211,
        max_epochs=320,
        loss="mse",
    ).fit(
        synthetic.data,
        input_col="input_signal",
        target_col="target_signal",
        time_col="time",
    )

    fit.kernel.validate()
    assert fit.fit_provenance is not None
    assert fit.fit_provenance["parametric_family"] == "delayed_exponential"
    assert fit.fit_provenance["parametric_conversion_status"] == "ok"
    params = fit.fit_provenance["parametric_parameters"]
    assert params["delay"] >= 0.0
    assert params["rate_lambda"] > 0.0
    assert fit.fit_provenance["parametric_lag_time_grid_seconds"] == pytest.approx(
        tuple(float(step) * 60.0 for step in fit.kernel.lag_steps),
        abs=1e-12,
    )
    delay_steps = int(round(float(params["delay"]) / 60.0))
    assert delay_steps >= meta["min_lag"]
    assert delay_steps <= meta["max_lag"]


def test_lognormal_fit_recovers_plausible_right_skew_and_provenance() -> None:
    synthetic = make_lognormal_kernel_dataset(
        seed=212,
        n_rows=420,
        dt=60.0,
        min_lag_steps=1,
        max_lag_steps=12,
        log_mu=5.0,
        log_sigma=0.5,
        noise_std=0.02,
    )
    meta = synthetic.true_kernels["input_signal->target_signal"]

    fit = LogNormalKernelLearner(
        max_lag=meta["max_lag"],
        min_lag=meta["min_lag"],
        dt="60s",
        seed=212,
        max_epochs=340,
        loss="mse",
    ).fit(
        synthetic.data,
        input_col="input_signal",
        target_col="target_signal",
        time_col="time",
    )

    fit.kernel.validate()
    assert fit.fit_provenance is not None
    assert fit.fit_provenance["parametric_family"] == "lognormal"
    assert fit.fit_provenance["parametric_conversion_status"] == "ok"
    params = fit.fit_provenance["parametric_parameters"]
    assert np.isfinite(params["log_mu"])
    assert params["log_sigma"] > 0.0
    assert fit.fit_diagnostics.mean_lag is not None
    assert fit.fit_diagnostics.p50_lag is not None
    assert fit.fit_diagnostics.mean_lag > fit.fit_diagnostics.p50_lag


def test_invalid_init_parameters_raise_clear_errors() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        DelayedExponentialKernelLearner(max_lag=8, min_lag=0, init_delay=-1.0)
    with pytest.raises(ValueError, match="strictly positive"):
        DelayedExponentialKernelLearner(max_lag=8, min_lag=0, init_rate_lambda=0.0)
    with pytest.raises(ValueError, match="finite when provided"):
        LogNormalKernelLearner(max_lag=8, min_lag=1, init_log_mu=float("inf"))
    with pytest.raises(ValueError, match="strictly positive"):
        LogNormalKernelLearner(max_lag=8, min_lag=1, init_log_sigma=0.0)


def test_lognormal_default_init_log_sigma_is_contract_value() -> None:
    learner = LogNormalKernelLearner(max_lag=8, min_lag=1)
    assert learner.init_log_sigma == pytest.approx(0.5, abs=1e-12)


def test_zero_only_lag_grid_behavior_is_explicit() -> None:
    delayed_data = make_delayed_exponential_kernel_dataset(
        n_rows=140,
        dt=60.0,
        seed=213,
        min_lag_steps=0,
        max_lag_steps=0,
        delay=0.0,
        rate_lambda=0.02,
        noise_std=0.01,
    ).data
    delayed_fit = DelayedExponentialKernelLearner(
        max_lag=0,
        min_lag=0,
        dt="60s",
        seed=213,
        init_rate_lambda=0.02,
    ).fit(
        delayed_data,
        input_col="input_signal",
        target_col="target_signal",
        time_col="time",
    )
    delayed_fit.kernel.validate()
    assert delayed_fit.kernel.lag_steps == (0,)
    assert delayed_fit.kernel.weights == pytest.approx((1.0,), abs=1e-12)
    assert delayed_fit.fit_provenance is not None
    delayed_params = delayed_fit.fit_provenance["parametric_parameters"]
    assert delayed_params["delay"] == pytest.approx(0.0, abs=1e-12)

    rng = np.random.default_rng(214)
    n_rows = 140
    lognormal_data = pl.DataFrame(
        {
            "time": delayed_data.get_column("time"),
            "input_signal": rng.normal(0.0, 1.0, size=n_rows),
            "target_signal": rng.normal(0.0, 1.0, size=n_rows),
        }
    )
    with pytest.raises(
        ValueError,
        match=(
            "requires at least one strictly positive lag step; "
            "min_lag=0 and max_lag=0 is not supported"
        ),
    ):
        LogNormalKernelLearner(max_lag=0, min_lag=0, dt="60s", seed=214).fit(
            lognormal_data,
            input_col="input_signal",
            target_col="target_signal",
            time_col="time",
        )


def test_new_learners_are_available_from_learners_submodule() -> None:
    assert DelayedExponentialKernelLearner is not None
    assert LogNormalKernelLearner is not None


def test_delayed_exponential_out_of_window_init_delay_is_handled_safely() -> None:
    synthetic = make_delayed_exponential_kernel_dataset(
        seed=415,
        n_rows=320,
        dt=60.0,
        min_lag_steps=2,
        max_lag_steps=10,
        delay=300.0,
        rate_lambda=0.02,
        noise_std=0.01,
    )
    fit = DelayedExponentialKernelLearner(
        max_lag=10,
        min_lag=2,
        dt="60s",
        seed=415,
        max_epochs=260,
        loss="mse",
        init_delay=1.0e6,
    ).fit(
        synthetic.data,
        input_col="input_signal",
        target_col="target_signal",
        time_col="time",
    )

    fit.kernel.validate()
    assert fit.fit_provenance is not None
    assert fit.fit_provenance["parametric_initial_parameters"]["delay"] == pytest.approx(1.0e6)
    params = fit.fit_provenance["parametric_parameters"]
    min_lag_time = 2.0 * 60.0
    max_lag_time = 10.0 * 60.0
    assert min_lag_time <= params["delay"] <= max_lag_time
