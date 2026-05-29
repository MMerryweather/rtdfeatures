"""Tests for Erlang learner behavior."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

import rtdfeatures
from rtdfeatures.learners import ErlangKernelLearner
from rtdfeatures.synthetic import make_erlang_kernel_dataset


def test_erlang_fit_selects_plausible_shape_and_provenance() -> None:
    synthetic = make_erlang_kernel_dataset(
        seed=311,
        n_rows=420,
        dt=60.0,
        min_lag_steps=1,
        max_lag_steps=12,
        shape_k=4,
        rate_beta=0.05,
        noise_std=0.02,
    )
    meta = synthetic.true_kernels["input_signal->target_signal"]

    fit = ErlangKernelLearner(
        max_lag=meta["max_lag"],
        min_lag=meta["min_lag"],
        dt="60s",
        seed=311,
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
    assert fit.fit_provenance["parametric_family"] == "erlang"
    assert fit.fit_provenance["parametric_conversion_status"] == "ok"
    params = fit.fit_provenance["parametric_parameters"]
    assert int(params["shape_k"]) in {1, 2, 3, 4, 5, 6, 7, 8}
    assert params["shape_k"] == int(params["shape_k"])
    assert params["rate_beta"] > 0.0
    assert fit.fit_provenance["shape_k_candidates"] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert (
        fit.fit_provenance["shape_k_selection_tie_break"]
        == "validation_loss_then_train_loss_then_lower_shape_k"
    )


@pytest.mark.parametrize(
    ("candidates", "message"),
    [
        ((0, 2), "positive integers"),
        ((1, True), "positive integers"),
        ((1, 2, 2), "duplicate"),
        ((), "non-empty tuple"),
    ],
)
def test_erlang_shape_k_candidates_validation_is_strict(
    candidates: tuple[object, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ErlangKernelLearner(
            max_lag=8,
            min_lag=1,
            shape_k_candidates=candidates,  # type: ignore[arg-type]
        )


def test_erlang_shape_k_candidates_none_uses_default_tuple() -> None:
    learner = ErlangKernelLearner(max_lag=8, min_lag=1, shape_k_candidates=None)
    assert learner.shape_k_candidates == (1, 2, 3, 4, 5, 6, 7, 8)


def test_erlang_fit_is_deterministic_across_candidate_ordering_given_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synthetic = make_erlang_kernel_dataset(
        seed=312,
        n_rows=360,
        dt=60.0,
        min_lag_steps=1,
        max_lag_steps=10,
        shape_k=3,
        rate_beta=0.04,
        noise_std=0.03,
    )

    rank_by_shape = {
        1: (0.9, 0.8),
        2: (0.7, 0.6),
        3: (0.7, 0.4),
        4: (0.7, 0.4),
    }
    def fake_optimize(*args, **kwargs):  # type: ignore[no-untyped-def]
        forward = kwargs["forward"]
        freevars = dict(
            zip(forward.__code__.co_freevars, (cell.cell_contents for cell in forward.__closure__))
        )
        shape = int(freevars["shape_k"])
        validation, train = rank_by_shape[shape]

        class _BestLoss:
            train_loss = train
            validation_loss = validation

        return _BestLoss(), {"rate_beta": 0.02 * shape}, np.asarray([1.0], dtype=np.float64)

    monkeypatch.setattr("rtdfeatures.learners.erlang.optimize_parametric_weights", fake_optimize)

    fit_ordered = ErlangKernelLearner(
        max_lag=10,
        min_lag=1,
        dt="60s",
        seed=313,
        shape_k_candidates=(1, 2, 3, 4),
        max_epochs=8,
    ).fit(
        synthetic.data,
        input_col="input_signal",
        target_col="target_signal",
        time_col="time",
    )
    fit_reversed = ErlangKernelLearner(
        max_lag=10,
        min_lag=1,
        dt="60s",
        seed=313,
        shape_k_candidates=(4, 3, 2, 1),
        max_epochs=8,
    ).fit(
        synthetic.data,
        input_col="input_signal",
        target_col="target_signal",
        time_col="time",
    )

    assert fit_ordered.fit_provenance is not None
    assert fit_reversed.fit_provenance is not None
    assert fit_ordered.fit_provenance["parametric_parameters"]["shape_k"] == 3
    assert fit_reversed.fit_provenance["parametric_parameters"]["shape_k"] == 3


def test_public_exports_include_erlang_learner() -> None:
    assert "ErlangKernelLearner" in rtdfeatures.__all__
    assert rtdfeatures.ErlangKernelLearner is ErlangKernelLearner


def test_zero_only_lag_grid_behavior_is_explicit_for_erlang() -> None:
    n_rows = 140
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rng = np.random.default_rng(314)
    df = pl.DataFrame(
        {
            "time": [t0 + timedelta(seconds=60 * i) for i in range(n_rows)],
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
        ErlangKernelLearner(max_lag=0, min_lag=0, dt="60s", seed=314).fit(
            df,
            input_col="input_signal",
            target_col="target_signal",
            time_col="time",
        )
