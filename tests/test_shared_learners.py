"""legacy milestone tests for shared simplex learner behavior."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from rtdfeatures.learners import SharedSimplexKernelLearner


def _make_time(n_rows: int) -> list[datetime]:
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [t0 + timedelta(minutes=i) for i in range(n_rows)]


def _make_shared_df(n_rows: int = 420) -> pl.DataFrame:
    rng = np.random.default_rng(123)
    x1 = rng.normal(0.0, 1.0, size=n_rows)
    x2 = rng.normal(0.0, 1.0, size=n_rows)

    y1 = np.zeros(n_rows, dtype=np.float64)
    y2 = np.zeros(n_rows, dtype=np.float64)
    for idx in range(5, n_rows):
        y1[idx] = 0.2 * x1[idx - 2] + 0.8 * x1[idx - 3]
        y2[idx] = 0.65 * x2[idx - 1] + 0.35 * x2[idx - 5]

    y1 += rng.normal(0.0, 0.03, size=n_rows)
    y2 += rng.normal(0.0, 0.03, size=n_rows)

    return pl.DataFrame(
        {
            "timestamp": _make_time(n_rows),
            "input_a": x1,
            "input_b": x2,
            "target_a": y1,
            "target_b": y2,
        }
    )


def test_shared_fit_recovers_pair_specific_kernels() -> None:
    df = _make_shared_df()
    learner = SharedSimplexKernelLearner(max_lag=6, min_lag=0, seed=7, loss="mse")
    shared = learner.fit(
        df,
        input_cols=["input_a", "input_b"],
        target_cols=["target_a", "target_b"],
        time_col="timestamp",
    )

    assert shared.pair_ids() == ("input_a->target_a", "input_b->target_b")
    fit_a = shared.get_pair_result("input_a->target_a")
    fit_b = shared.get_pair_result("input_b->target_b")

    lag_to_weight_a = dict(zip(fit_a.kernel.lag_steps, fit_a.kernel.weights))
    lag_to_weight_b = dict(zip(fit_b.kernel.lag_steps, fit_b.kernel.weights))

    assert lag_to_weight_a[3] > lag_to_weight_a[2]
    assert lag_to_weight_a[3] > 0.5
    assert lag_to_weight_b[1] > lag_to_weight_b[5]
    assert (lag_to_weight_b[1] + lag_to_weight_b[5]) > 0.6


def test_shared_fit_rejects_unequal_pair_lengths() -> None:
    df = _make_shared_df()
    learner = SharedSimplexKernelLearner(max_lag=5, min_lag=0, seed=1)
    with pytest.raises(ValueError, match="same length"):
        learner.fit(
            df,
            input_cols=["input_a", "input_b"],
            target_cols=["target_a"],
            time_col="timestamp",
        )


def test_shared_fit_is_deterministic_given_seed() -> None:
    df = _make_shared_df()
    learner_a = SharedSimplexKernelLearner(max_lag=6, min_lag=0, seed=99)
    learner_b = SharedSimplexKernelLearner(max_lag=6, min_lag=0, seed=99)

    shared_a = learner_a.fit(
        df,
        input_cols=["input_a", "input_b"],
        target_cols=["target_a", "target_b"],
        time_col="timestamp",
    )
    shared_b = learner_b.fit(
        df,
        input_cols=["input_a", "input_b"],
        target_cols=["target_a", "target_b"],
        time_col="timestamp",
    )

    for pair_id in shared_a.pair_ids():
        weights_a = shared_a.get_pair_result(pair_id).kernel.weights
        weights_b = shared_b.get_pair_result(pair_id).kernel.weights
        assert weights_a == pytest.approx(weights_b, abs=1e-7)


def test_shared_fit_unsorted_input_obeys_order_by_time_contract() -> None:
    df = _make_shared_df().reverse()
    learner = SharedSimplexKernelLearner(max_lag=5, min_lag=0, seed=3)

    with pytest.raises(ValueError, match="not sorted"):
        learner.fit(
            df,
            input_cols=["input_a", "input_b"],
            target_cols=["target_a", "target_b"],
            time_col="timestamp",
            order_by_time=False,
        )

    sorted_ok = learner.fit(
        df,
        input_cols=["input_a", "input_b"],
        target_cols=["target_a", "target_b"],
        time_col="timestamp",
        order_by_time=True,
    )
    assert all(pair.succeeded for pair in sorted_ok.pairs)


def test_shared_fit_missing_data_is_pair_local() -> None:
    df = _make_shared_df().with_columns(
        pl.when(pl.int_range(0, pl.len()) % 2 == 0)
        .then(None)
        .otherwise(pl.col("input_a"))
        .alias("input_a")
    )
    learner = SharedSimplexKernelLearner(max_lag=4, min_lag=0, seed=5)
    shared = learner.fit(
        df,
        input_cols=["input_a", "input_b"],
        target_cols=["target_a", "target_b"],
        time_col="timestamp",
    )

    pair_a = shared.get_pair("input_a->target_a")
    pair_b = shared.get_pair("input_b->target_b")

    assert pair_a.succeeded is False
    assert pair_a.error is not None
    assert (
        "No valid lag windows remain" in pair_a.error
        or "Not enough valid lag windows" in pair_a.error
    )
    assert pair_b.succeeded is True
    assert pair_b.fit_result is not None


def test_shared_fit_accepts_explicit_pair_names() -> None:
    df = _make_shared_df()
    learner = SharedSimplexKernelLearner(max_lag=5, min_lag=0, seed=11)
    shared = learner.fit(
        df,
        input_cols=["input_a", "input_b"],
        target_cols=["target_a", "target_b"],
        pair_names=["Pair A", "Pair B"],
        time_col="timestamp",
    )

    assert shared.pair_ids() == ("Pair A", "Pair B")
    assert shared.get_pair("Pair A").input_col == "input_a"
    assert shared.get_pair("Pair B").target_col == "target_b"


def test_shared_fit_rejects_pair_names_length_mismatch() -> None:
    df = _make_shared_df()
    learner = SharedSimplexKernelLearner(max_lag=5, min_lag=0, seed=12)
    with pytest.raises(ValueError, match="pair_names must have the same length"):
        learner.fit(
            df,
            input_cols=["input_a", "input_b"],
            target_cols=["target_a", "target_b"],
            pair_names=["only_one_name"],
            time_col="timestamp",
        )


def test_shared_fit_rejects_duplicate_pair_ids() -> None:
    df = _make_shared_df()
    learner = SharedSimplexKernelLearner(max_lag=5, min_lag=0, seed=13)
    with pytest.raises(ValueError, match="Duplicate pair_id detected"):
        learner.fit(
            df,
            input_cols=["input_a", "input_b"],
            target_cols=["target_a", "target_b"],
            pair_names=["dup", "dup"],
            time_col="timestamp",
        )
