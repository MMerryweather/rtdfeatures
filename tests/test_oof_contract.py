"""Contract tests for v0.95 out-of-fold result objects."""

from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from rtdfeatures.diagnostics import (
    FeatureEvidenceReport,
    OutOfFoldKernelFeatureResult,
    OutOfFoldSplitSummary,
    TransformReport,
)
from rtdfeatures.learners import SimplexKernelLearner
from rtdfeatures.oof import ForwardChainingSplitConfig, fit_transform_oof


def _minimal_transform_report() -> TransformReport:
    return TransformReport(
        row_count=10,
        output_row_count=10,
        warmup_rows=2,
        feature_names=("k_num_feed_wmean",),
        missing_rows_by_feature={"k_num_feed_wmean": 2},
        zero_denominator_rows_by_feature={"k_num_feed_wmean": 0},
    )


def test_oof_dataclass_fields_are_stable() -> None:
    assert [field.name for field in fields(OutOfFoldSplitSummary)] == [
        "n_folds",
        "split_strategy",
        "fold_boundaries",
        "min_train_rows",
        "validation_rows_total",
        "rows_with_features",
        "rows_without_features",
        "warnings",
    ]
    assert [field.name for field in fields(OutOfFoldKernelFeatureResult)] == [
        "features",
        "fold_results",
        "fold_reports",
        "combined_transform_report",
        "feature_evidence_report",
        "split_summary",
        "warnings",
    ]


def test_oof_summary_validation_rejects_invalid_boundaries() -> None:
    with_boundary_gap = (
        {
            "fold_id": 0,
            "train_start": 0,
            "train_end": 5,
            "validation_start": 6,
            "validation_end": 7,
            "gap": 0,
        },
    )

    OutOfFoldSplitSummary(
        n_folds=1,
        split_strategy="forward_chaining",
        fold_boundaries=with_boundary_gap,
        min_train_rows=6,
        validation_rows_total=2,
        rows_with_features=2,
        rows_without_features=8,
    )

    with_overlap = (
        {
            "fold_id": 0,
            "train_start": 0,
            "train_end": 5,
            "validation_start": 5,
            "validation_end": 7,
            "gap": 0,
        },
    )
    try:
        OutOfFoldSplitSummary(
            n_folds=1,
            split_strategy="forward_chaining",
            fold_boundaries=with_overlap,
            min_train_rows=6,
            validation_rows_total=2,
            rows_with_features=2,
            rows_without_features=8,
        )
    except ValueError as exc:
        assert "strictly after" in str(exc)
    else:
        raise AssertionError("Expected ValueError for overlapping train/validation boundary")


def test_oof_result_validates_fold_result_report_lengths() -> None:
    summary = OutOfFoldSplitSummary(
        n_folds=1,
        split_strategy="forward_chaining",
        fold_boundaries=(
            {
                "fold_id": 0,
                "train_start": 0,
                "train_end": 6,
                "validation_start": 8,
                "validation_end": 9,
                "gap": 1,
            },
        ),
        min_train_rows=7,
        validation_rows_total=2,
        rows_with_features=2,
        rows_without_features=8,
    )

    report = _minimal_transform_report()
    features = pl.DataFrame({"timestamp": [1, 2], "k_num_feed_wmean": [0.1, 0.2]})

    OutOfFoldKernelFeatureResult(
        features=features,
        fold_results=({"fold_id": 0, "status": "ok"},),
        fold_reports=(report,),
        combined_transform_report=report,
        feature_evidence_report=None,
        split_summary=summary,
        warnings=(),
    )

    try:
        OutOfFoldKernelFeatureResult(
            features=features,
            fold_results=({"fold_id": 0, "status": "ok"},),
            fold_reports=(),
            combined_transform_report=report,
            feature_evidence_report=FeatureEvidenceReport(
                feature_evidence=(),
                feature_count=0,
                kernel_count=0,
                source_columns=(),
                warning_summary={},
                evidence_summary_by_kernel={},
                evidence_summary_by_feature_family={},
            ),
            split_summary=summary,
            warnings=(),
        )
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("Expected ValueError for fold_results/fold_reports length mismatch")


def test_oof_objects_are_exported() -> None:
    expected = {
        "OutOfFoldKernelFeatureResult",
        "OutOfFoldSplitSummary",
    }
    import rtdfeatures.diagnostics as _diag
    assert expected.issubset(set(dir(_diag)))


def test_order_guardrails_match_public_contract() -> None:
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    sorted_df = pl.DataFrame(
        {
            "timestamp": [t0 + timedelta(minutes=i) for i in range(24)],
            "x": [float(i) for i in range(24)],
            "y": [0.5 * float(i) for i in range(24)],
        }
    )
    unsorted_df = sorted_df.reverse()
    learner = SimplexKernelLearner(max_lag=2, min_lag=0, seed=0, max_epochs=40)
    split = ForwardChainingSplitConfig(n_folds=1, min_train_size=12, validation_size=4, gap=0)

    with pytest.raises(ValueError, match="not sorted"):
        fit_transform_oof(
            df=unsorted_df,
            learner=learner,
            split_config=split,
            input_col="x",
            target_col="y",
            time_col="timestamp",
            numeric_cols=["x"],
        )

    out = fit_transform_oof(
        df=unsorted_df,
        learner=learner,
        split_config=split,
        input_col="x",
        target_col="y",
        time_col="timestamp",
        numeric_cols=["x"],
        order_by_time=True,
    )
    assert out.features.get_column("timestamp").to_list() == sorted_df.get_column(
        "timestamp"
    ).to_list()
