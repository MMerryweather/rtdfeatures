"""Tests for single-learner out-of-fold feature generation."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from rtdfeatures.diagnostics import OutOfFoldKernelFeatureResult
from rtdfeatures.features import KernelFeatureBuilder
from rtdfeatures.kernels import UniformKernel
from rtdfeatures.learners import SimplexKernelLearner
from rtdfeatures.oof import ForwardChainingSplitConfig, fit_transform_oof


def _make_df(n_rows: int = 30) -> pl.DataFrame:
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return pl.DataFrame(
        {
            "timestamp": [t0 + timedelta(minutes=i) for i in range(n_rows)],
            "x": [float(i) for i in range(n_rows)],
            "y": [0.5 * float(i) + 1.0 for i in range(n_rows)],
        }
    )


def _run_oof(df: pl.DataFrame) -> OutOfFoldKernelFeatureResult:
    learner = SimplexKernelLearner(
        max_lag=2,
        min_lag=0,
        seed=0,
        max_epochs=60,
        validation_fraction=0.2,
    )
    split_config = ForwardChainingSplitConfig(
        n_folds=2,
        min_train_size=12,
        validation_size=4,
        gap=0,
    )
    return fit_transform_oof(
        df=df,
        learner=learner,
        split_config=split_config,
        input_col="x",
        target_col="y",
        time_col="timestamp",
        numeric_cols=["x"],
    )


def _generated_cols(result_features: pl.DataFrame) -> list[str]:
    return [c for c in result_features.columns if c != "timestamp"]


def _numeric_generated_cols(result_features: pl.DataFrame) -> list[str]:
    return [c for c in _generated_cols(result_features) if "_num_" in c]


def test_no_validation_leakage() -> None:
    base_df = _make_df()
    changed_df = _make_df().with_columns(
        pl.when(pl.int_range(pl.len()) >= 16)
        .then(pl.col("y") * 10.0)
        .otherwise(pl.col("y"))
        .alias("y")
    )

    base_result = _run_oof(base_df)
    changed_result = _run_oof(changed_df)

    assert base_result.features.equals(changed_result.features)


def test_output_row_count_and_time_col_match_input() -> None:
    df = _make_df()
    result = _run_oof(df)

    assert result.features.height == df.height
    assert result.features.get_column("timestamp").to_list() == df.get_column("timestamp").to_list()


def test_row_order_is_preserved() -> None:
    df = _make_df()
    result = _run_oof(df)

    assert result.features.get_column("timestamp").to_list() == df.get_column("timestamp").to_list()


def test_uncovered_rows_are_nan_for_generated_features() -> None:
    result = _run_oof(_make_df())

    generated_cols = _generated_cols(result.features)
    uncovered_indices = list(range(0, 12)) + list(range(20, 30))
    for col in generated_cols:
        values = result.features.get_column(col).to_list()
        for idx in uncovered_indices:
            assert values[idx] is None


def test_warmup_rows_inside_each_fold_are_null() -> None:
    result = _run_oof(_make_df())

    generated_cols = _numeric_generated_cols(result.features)
    warmup_rows = [12, 13, 16, 17]
    usable_rows = [14, 15, 18, 19]
    for col in generated_cols:
        values = result.features.get_column(col).cast(pl.Float64).to_list()
        for idx in warmup_rows:
            assert values[idx] is None or math.isnan(values[idx])
        for idx in usable_rows:
            assert isinstance(values[idx], float)
            assert not math.isnan(values[idx])


def test_fold_report_aggregation_contract() -> None:
    result = _run_oof(_make_df())

    assert len(result.fold_results) == 2
    assert len(result.fold_reports) == 2
    assert result.combined_transform_report.output_row_count == result.features.height
    assert set(result.combined_transform_report.feature_names) == set(
        _generated_cols(result.features)
    )
    assert result.split_summary.n_folds == 2
    assert result.combined_transform_report.warmup_rows == sum(
        report.warmup_rows for report in result.fold_reports
    )
    assert (
        result.combined_transform_report.warmup_unusable_summary["warmup_rows"]
        == result.combined_transform_report.warmup_rows
    )


def test_default_transform_path_is_unchanged() -> None:
    df = _make_df(n_rows=8)
    kernel = UniformKernel(max_lag_steps=1, min_lag_steps=0, dt=60.0, name="k")
    builder = KernelFeatureBuilder(kernels={"k": kernel}, time_col="timestamp", numeric_cols=["x"])

    out = builder.transform(df)

    assert out.height == df.height
    assert out.columns == [
        "timestamp",
        "k_num_x_wmean",
        "k_num_x_wstd",
        "k_num_x_wsum",
        "k_age_mean",
        "k_age_p50",
        "k_age_p90",
        "k_age_tail_gt_threshold",
    ]


def test_single_learner_path_has_no_candidate_comparison_provenance() -> None:
    result = _run_oof(_make_df())
    assert result.feature_evidence_report is not None
    evidence_items = result.feature_evidence_report.feature_evidence
    assert evidence_items
    assert all(item.candidate_id is None for item in evidence_items)
    fold_evidence = evidence_items[0].metadata.get("fold_evidence")
    assert isinstance(fold_evidence, list)
    assert all(row["candidate_id"] is None for row in fold_evidence)


def test_unsorted_input_raises_by_default() -> None:
    df = _make_df().reverse()
    with pytest.raises(ValueError, match="not sorted"):
        _run_oof(df)


def test_unsorted_input_order_by_time_opt_in() -> None:
    learner = SimplexKernelLearner(
        max_lag=2,
        min_lag=0,
        seed=0,
        max_epochs=60,
        validation_fraction=0.2,
    )
    split_config = ForwardChainingSplitConfig(
        n_folds=2,
        min_train_size=12,
        validation_size=4,
        gap=0,
    )
    df = _make_df().reverse()
    result = fit_transform_oof(
        df=df,
        learner=learner,
        split_config=split_config,
        input_col="x",
        target_col="y",
        time_col="timestamp",
        numeric_cols=["x"],
        order_by_time=True,
    )
    assert result.features.get_column("timestamp").to_list() == sorted(
        df.get_column("timestamp").to_list()
    )


def test_irregular_grid_raises() -> None:
    irregular_df = _make_df().with_columns(
        pl.when(pl.int_range(pl.len()) >= 8)
        .then(pl.col("timestamp") + timedelta(minutes=1))
        .otherwise(pl.col("timestamp"))
        .alias("timestamp")
    )
    with pytest.raises(ValueError, match="irregular"):
        _run_oof(irregular_df)
