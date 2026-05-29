"""Behavior tests for fixed-delay and uniform learner families."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

import rtdfeatures
from rtdfeatures.learners import FixedDelayKernelLearner, UniformKernelLearner
from rtdfeatures.synthetic import make_diffuse_kernel_dataset, make_single_delay_dataset


def test_fixed_delay_learner_recovers_true_delay_on_single_delay_fixture() -> None:
    synthetic = make_single_delay_dataset(
        n_rows=420,
        dt=1.0,
        seed=501,
        delay_steps=6,
        noise_std=0.01,
    )
    fit = FixedDelayKernelLearner(max_lag=10, min_lag=0, seed=13, loss="mse").fit(
        synthetic.data,
        input_col="input_signal",
        target_col="target_signal",
        time_col="time",
    )
    dominant_idx = int(np.argmax(np.asarray(fit.kernel.weights, dtype=np.float64)))
    assert fit.kernel.lag_steps[dominant_idx] == 6
    assert fit.kernel.weights[dominant_idx] == pytest.approx(1.0, abs=1e-12)


def test_uniform_learner_returns_uniform_kernel_and_finite_diagnostics() -> None:
    synthetic = make_diffuse_kernel_dataset(n_rows=360, dt=1.0, seed=502, noise_std=0.03)
    fit = UniformKernelLearner(max_lag=10, min_lag=0, seed=17, loss="huber").fit(
        synthetic.data,
        input_col="input_signal",
        target_col="target_signal",
        time_col="time",
    )
    weights = np.asarray(fit.kernel.weights, dtype=np.float64)
    assert np.allclose(weights, np.full(weights.shape[0], 1.0 / weights.shape[0]))
    assert np.isfinite(fit.fit_diagnostics.train_loss)
    assert np.isfinite(fit.fit_diagnostics.validation_loss)


def test_fixed_and_uniform_provide_baselines_and_provenance() -> None:
    synthetic = make_single_delay_dataset(n_rows=320, dt=1.0, seed=503, delay_steps=4)
    for learner in (
        FixedDelayKernelLearner(max_lag=8, min_lag=0, seed=11),
        UniformKernelLearner(max_lag=8, min_lag=0, seed=11),
    ):
        fit = learner.fit(
            synthetic.data,
            input_col="input_signal",
            target_col="target_signal",
            time_col="time",
        )
        baseline = fit.baseline_comparison
        assert np.isfinite(baseline.no_lag_validation_loss)
        assert np.isfinite(baseline.best_single_lag_validation_loss)
        assert baseline.uniform_validation_loss is not None
        assert baseline.exponential_validation_loss is not None
        assert np.isfinite(baseline.uniform_validation_loss)
        assert np.isfinite(baseline.exponential_validation_loss)
        assert fit.fit_provenance is not None
        assert fit.fit_provenance["baseline_losses_use_configured_loss"] is True
        assert fit.fit_provenance["learner_family"] in {"fixed_delay", "uniform"}
        if fit.fit_provenance["learner_family"] == "fixed_delay":
            assert fit.fit_provenance["fixed_delay_selected_step"] in range(0, 9)
            assert fit.fit_provenance["fixed_delay_candidate_steps"] == list(range(0, 9))
            assert (
                fit.fit_provenance["fixed_delay_tie_break"]
                == "validation_loss_then_train_loss_then_lower_delay"
            )
        else:
            assert fit.fit_provenance["uniform_window_min_lag_steps"] == 0
            assert fit.fit_provenance["uniform_window_max_lag_steps"] == 8


def test_fixed_and_uniform_are_deterministic_given_seed() -> None:
    synthetic = make_single_delay_dataset(n_rows=350, dt=1.0, seed=504, delay_steps=5)
    learners = (
        FixedDelayKernelLearner(max_lag=9, min_lag=0, seed=99, loss="mse"),
        UniformKernelLearner(max_lag=9, min_lag=0, seed=99, loss="mse"),
    )
    for learner in learners:
        fit_a = learner.fit(
            synthetic.data,
            input_col="input_signal",
            target_col="target_signal",
            time_col="time",
        )
        fit_b = learner.fit(
            synthetic.data,
            input_col="input_signal",
            target_col="target_signal",
            time_col="time",
        )
        assert fit_a.kernel.weights == pytest.approx(fit_b.kernel.weights, abs=1e-12)
        assert fit_a.fit_diagnostics.validation_loss == pytest.approx(
            fit_b.fit_diagnostics.validation_loss, abs=1e-12
        )


def test_fixed_and_uniform_unsorted_handling_uses_shared_preparation_path() -> None:
    synthetic = make_single_delay_dataset(n_rows=260, dt=1.0, seed=505, delay_steps=3)
    unsorted = synthetic.data.sample(fraction=1.0, shuffle=True, seed=12)
    learners = (
        FixedDelayKernelLearner(max_lag=7, min_lag=0, seed=12),
        UniformKernelLearner(max_lag=7, min_lag=0, seed=12),
    )
    for learner in learners:
        with pytest.raises(ValueError, match="sorted"):
            learner.fit(
                unsorted,
                input_col="input_signal",
                target_col="target_signal",
                time_col="time",
                order_by_time=False,
            )
        fit = learner.fit(
            unsorted,
            input_col="input_signal",
            target_col="target_signal",
            time_col="time",
            order_by_time=True,
        )
        assert fit.kernel.min_lag_steps == 0
        assert fit.kernel.max_lag_steps == 7


def test_fixed_and_uniform_root_exports_are_available() -> None:
    assert "FixedDelayKernelLearner" in rtdfeatures.__all__
    assert "UniformKernelLearner" in rtdfeatures.__all__
    assert rtdfeatures.FixedDelayKernelLearner is FixedDelayKernelLearner
    assert rtdfeatures.UniformKernelLearner is UniformKernelLearner


def test_fixed_and_uniform_constructor_validation_is_strict() -> None:
    with pytest.raises(ValueError, match="either 'huber' or 'mse'"):
        FixedDelayKernelLearner(max_lag=5, loss="mae")
    with pytest.raises(ValueError, match="non-negative"):
        UniformKernelLearner(max_lag=5, smoothness_penalty=-0.1)
    with pytest.raises(ValueError, match="strictly positive"):
        FixedDelayKernelLearner(max_lag=5, learning_rate=0.0)


def test_fixed_delay_learner_uses_earliest_lag_on_exact_tie() -> None:
    n_rows = 240
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    df = pl.DataFrame(
        {
            "time": [t0 + timedelta(seconds=i) for i in range(n_rows)],
            "input_signal": np.ones(n_rows, dtype=np.float64),
            "target_signal": np.ones(n_rows, dtype=np.float64),
        }
    )
    fit = FixedDelayKernelLearner(max_lag=6, min_lag=0, seed=1).fit(
        df,
        input_col="input_signal",
        target_col="target_signal",
        time_col="time",
    )
    dominant_idx = int(np.argmax(np.asarray(fit.kernel.weights, dtype=np.float64)))
    assert fit.kernel.lag_steps[dominant_idx] == 0


def test_fixed_delay_selection_tiebreak_prefers_lower_train_loss_before_lower_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synthetic = make_single_delay_dataset(n_rows=300, dt=1.0, seed=508, delay_steps=2)
    rank_by_delay = {
        0: (1.0, 3.0),
        1: (1.0, 1.0),
        2: (0.5, 0.5),
        3: (0.5, 0.5),
        4: (0.5, 0.5),
    }

    def fake_eval(*, prepared, weights, loss, huber_delta):  # type: ignore[no-untyped-def]
        delay_idx = int(np.argmax(np.asarray(weights, dtype=np.float64)))
        delay_steps = prepared.min_lag_steps + delay_idx
        return rank_by_delay[delay_steps]

    monkeypatch.setattr(
        "rtdfeatures.learners.fixed.evaluate_weight_vector_losses",
        fake_eval,
    )
    fit = FixedDelayKernelLearner(max_lag=4, min_lag=0, seed=31, loss="mse").fit(
        synthetic.data,
        input_col="input_signal",
        target_col="target_signal",
        time_col="time",
    )
    dominant_idx = int(np.argmax(np.asarray(fit.kernel.weights, dtype=np.float64)))
    assert fit.kernel.lag_steps[dominant_idx] == 2
