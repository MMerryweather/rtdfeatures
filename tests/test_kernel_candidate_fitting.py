"""Tests for v0.7 kernel candidate fitting engine."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from rtdfeatures.candidates import _WindowedKernelEvaluator, fit_kernel_candidates
from rtdfeatures.diagnostics import KernelCandidate, KernelCandidateSet


def _make_df(n_rows: int = 64) -> pl.DataFrame:
    ts = pl.datetime_range(
        start=pl.datetime(2024, 1, 1, 0, 0, 0),
        end=pl.datetime(2024, 1, 1, 0, 0, 0) + pl.duration(minutes=n_rows - 1),
        interval="1m",
        eager=True,
    )
    x = [float(i) for i in range(n_rows)]
    y = [
        0.7 * x[i - 2] + 0.2 * x[i - 1] + 0.1 * x[i] if i >= 2 else float(i)
        for i in range(n_rows)
    ]
    return pl.DataFrame({"ts": ts, "x": x, "y": y})


def test_fit_kernel_candidates_mixed_success_failure() -> None:
    df = _make_df()
    candidate_set = KernelCandidateSet(
        candidate_set_id="mixed-1",
        input_col="x",
        target_col="y",
        time_col="ts",
        candidates=(
            KernelCandidate(
                candidate_id="ok-simplex",
                family="simplex",
                candidate_type="empirical_learner",
                min_lag="0m",
                max_lag="3m",
                learner_parameters={"max_epochs": 10, "seed": 1},
            ),
            KernelCandidate(
                candidate_id="bad-family",
                family="does-not-exist",
                candidate_type="parametric_learner",
                min_lag="0m",
                max_lag="3m",
                learner_parameters={"max_epochs": 10, "seed": 1},
            ),
        ),
    )
    result = fit_kernel_candidates(df, candidate_set)
    assert len(result.family_results) == 2
    assert result.family_results[0].succeeded is True
    assert result.family_results[0].fit_result is not None
    assert result.family_results[1].succeeded is False
    assert result.family_results[1].fit_result is None
    assert "Unsupported learner family" in (result.family_results[1].error or "")
    assert result.comparison_table.height == 2
    assert len(result.warnings) == 1


def test_fit_kernel_candidates_fixed_kernel_candidate() -> None:
    df = _make_df()
    candidate_set = KernelCandidateSet(
        candidate_set_id="fixed-1",
        input_col="x",
        target_col="y",
        time_col="ts",
        candidates=(
            KernelCandidate(
                candidate_id="fixed-delay",
                family="fixed_delay",
                candidate_type="fixed_kernel",
                min_lag="0m",
                max_lag="3m",
                fixed_parameters={"delay_steps": 2},
            ),
        ),
    )
    result = fit_kernel_candidates(df, candidate_set)
    fit = result.family_results[0]
    assert fit.succeeded is True
    assert fit.fit_result is None
    assert fit.is_baseline is False
    assert fit.error is None
    assert fit.evaluated_fixed_kernel is not None
    assert fit.evaluated_fixed_kernel.name == "fixed-delay"


def test_fixed_delay_candidate_typoed_delay_key_fails_explicitly() -> None:
    df = _make_df()
    candidate_set = KernelCandidateSet(
        candidate_set_id="fixed-typo-key",
        input_col="x",
        target_col="y",
        time_col="ts",
        candidates=(
            KernelCandidate(
                candidate_id="fixed-delay-typo",
                family="fixed_delay",
                candidate_type="fixed_kernel",
                min_lag="0m",
                max_lag="3m",
                fixed_parameters={"delay_step": 2},
            ),
        ),
    )
    result = fit_kernel_candidates(df, candidate_set)
    fit = result.family_results[0]
    assert fit.succeeded is False
    assert fit.fit_result is None
    assert "delay_steps" in (fit.error or "")


def test_fixed_kernel_candidate_unknown_fixed_parameter_key_fails_closed() -> None:
    df = _make_df()
    candidate_set = KernelCandidateSet(
        candidate_set_id="fixed-unknown-key",
        input_col="x",
        target_col="y",
        time_col="ts",
        candidates=(
            KernelCandidate(
                candidate_id="fixed-delay-unknown-key",
                family="fixed_delay",
                candidate_type="fixed_kernel",
                min_lag="0m",
                max_lag="3m",
                fixed_parameters={"delay_steps": 2, "delay_step_typo": 2},
            ),
        ),
    )
    result = fit_kernel_candidates(df, candidate_set)
    fit = result.family_results[0]
    assert fit.succeeded is False
    assert fit.fit_result is None
    assert "Unsupported fixed_parameters keys" in (fit.error or "")
    assert "delay_step_typo" in (fit.error or "")


def test_fit_kernel_candidates_simplex_candidate() -> None:
    df = _make_df()
    candidate_set = KernelCandidateSet(
        candidate_set_id="simplex-1",
        input_col="x",
        target_col="y",
        time_col="ts",
        candidates=(
            KernelCandidate(
                candidate_id="simplex",
                family="simplex",
                candidate_type="empirical_learner",
                min_lag="0m",
                max_lag="3m",
                learner_parameters={"max_epochs": 10, "seed": 2},
            ),
        ),
    )
    result = fit_kernel_candidates(df, candidate_set)
    fit = result.family_results[0]
    assert fit.succeeded is True
    assert fit.fit_result is not None
    assert fit.fit_result.fit_diagnostics.validation_loss >= 0.0


def test_fit_kernel_candidates_parametric_candidate() -> None:
    df = _make_df()
    candidate_set = KernelCandidateSet(
        candidate_set_id="param-1",
        input_col="x",
        target_col="y",
        time_col="ts",
        candidates=(
            KernelCandidate(
                candidate_id="exp",
                family="exponential",
                candidate_type="parametric_learner",
                min_lag="0m",
                max_lag="3m",
                learner_parameters={"max_epochs": 10, "seed": 3},
            ),
        ),
    )
    result = fit_kernel_candidates(df, candidate_set)
    fit = result.family_results[0]
    assert fit.succeeded is True
    assert fit.fit_result is not None
    assert fit.is_parametric is True


def test_fit_kernel_candidates_baseline_candidate() -> None:
    df = _make_df()
    candidate_set = KernelCandidateSet(
        candidate_set_id="baseline-1",
        input_col="x",
        target_col="y",
        time_col="ts",
        candidates=(
            KernelCandidate(
                candidate_id="baseline-no-lag",
                family="no_lag",
                candidate_type="baseline",
                min_lag="0m",
                max_lag="3m",
            ),
        ),
    )
    result = fit_kernel_candidates(df, candidate_set)
    fit = result.family_results[0]
    assert fit.succeeded is True
    assert fit.fit_result is None
    assert fit.is_baseline is True
    assert fit.error is None


def test_fit_kernel_candidates_invalid_candidate_failure() -> None:
    df = _make_df()
    candidate_set = KernelCandidateSet(
        candidate_set_id="invalid-1",
        input_col="x",
        target_col="y",
        time_col="ts",
        candidates=(
            KernelCandidate(
                candidate_id="bad-baseline",
                family="not-a-baseline",
                candidate_type="baseline",
                min_lag="0m",
                max_lag="3m",
            ),
        ),
    )
    result = fit_kernel_candidates(df, candidate_set)
    fit = result.family_results[0]
    assert fit.succeeded is False
    assert fit.fit_result is None
    assert "Unsupported baseline family" in (fit.error or "")


def test_fit_kernel_candidates_unsorted_time_raises_by_default() -> None:
    df = _make_df().sort("ts", descending=True)
    candidate_set = KernelCandidateSet(
        candidate_set_id="unsorted-raise",
        input_col="x",
        target_col="y",
        time_col="ts",
        candidates=(
            KernelCandidate(
                candidate_id="simplex",
                family="simplex",
                candidate_type="empirical_learner",
                min_lag="0m",
                max_lag="3m",
                learner_parameters={"max_epochs": 10, "seed": 11},
            ),
        ),
    )
    with pytest.raises(ValueError, match="sorted"):
        fit_kernel_candidates(df, candidate_set)


def test_fit_kernel_candidates_unsorted_time_sorts_when_opted_in() -> None:
    df = _make_df().sort("ts", descending=True)
    candidate_set = KernelCandidateSet(
        candidate_set_id="unsorted-sort",
        input_col="x",
        target_col="y",
        time_col="ts",
        candidates=(
            KernelCandidate(
                candidate_id="simplex",
                family="simplex",
                candidate_type="empirical_learner",
                min_lag="0m",
                max_lag="3m",
                learner_parameters={"max_epochs": 10, "seed": 12},
            ),
        ),
    )
    result = fit_kernel_candidates(df, candidate_set, order_by_time=True)
    fit = result.family_results[0]
    assert fit.succeeded is True
    assert fit.fit_result is not None


def test_fit_kernel_candidates_irregular_grid_records_candidate_failure() -> None:
    df = _make_df().with_columns(
        pl.when(pl.arange(0, pl.len()) >= 20)
        .then(pl.col("ts") + pl.duration(minutes=1))
        .otherwise(pl.col("ts"))
        .alias("ts")
    )
    candidate_set = KernelCandidateSet(
        candidate_set_id="irregular-grid-api-path",
        input_col="x",
        target_col="y",
        time_col="ts",
        candidates=(
            KernelCandidate(
                candidate_id="fixed-delay",
                family="fixed_delay",
                candidate_type="fixed_kernel",
                min_lag="0m",
                max_lag="3m",
                fixed_parameters={"delay_steps": 2},
            ),
        ),
    )

    result = fit_kernel_candidates(df, candidate_set)

    fit = result.family_results[0]
    assert fit.succeeded is False
    assert fit.fit_result is None
    assert "irregular" in (fit.error or "")
    assert result.comparison_table.height == 1
    assert result.comparison_table["succeeded"][0] is False


def test_fixed_kernel_no_lag_baseline_uses_aligned_1d_series() -> None:
    df = _make_df()
    candidate_set = KernelCandidateSet(
        candidate_set_id="fixed-no-lag-1d",
        input_col="x",
        target_col="y",
        time_col="ts",
        candidates=(
            KernelCandidate(
                candidate_id="fixed-delay",
                family="fixed_delay",
                candidate_type="fixed_kernel",
                min_lag="0m",
                max_lag="3m",
                fixed_parameters={"delay_steps": 2, "loss": "mse"},
            ),
        ),
    )

    result = fit_kernel_candidates(df, candidate_set)

    fit = result.family_results[0]
    assert fit.succeeded is True
    assert fit.fixed_baseline_comparison is not None

    evaluator = _WindowedKernelEvaluator(loss="mse", huber_delta=1.0, validation_fraction=0.2)
    windows = evaluator._scaled_validation_windows(
        df=df,
        time_col="ts",
        input_col="x",
        target_col="y",
        min_lag_steps=0,
        max_lag_steps=3,
    )

    assert windows.no_lag_valid_scaled.ndim == 1
    expected_no_lag_loss = float(
        np.mean((windows.no_lag_valid_scaled - windows.y_valid_scaled) ** 2)
    )
    observed_no_lag_loss = fit.fixed_baseline_comparison.no_lag_validation_loss

    assert fit.validation_loss is not None
    assert observed_no_lag_loss is not None
    assert observed_no_lag_loss == pytest.approx(expected_no_lag_loss)
    assert result.comparison_table["beats_no_lag"][0] == (
        fit.validation_loss < observed_no_lag_loss
    )
