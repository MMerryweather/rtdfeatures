from __future__ import annotations

import pytest

from rtdfeatures.diagnostics.transform import OutOfFoldKernelFeatureResult, OutOfFoldSplitSummary

_VALID_BOUNDARIES = (
    {
        "fold_id": 0,
        "train_start": 0,
        "train_end": 10,
        "validation_start": 12,
        "validation_end": 15,
        "gap": 1,
    },
)


def test_oot_split_summary_n_folds_positive() -> None:
    with pytest.raises(ValueError, match="n_folds must be a positive integer"):
        OutOfFoldSplitSummary(
            n_folds=0, split_strategy="loo",
            fold_boundaries=(),
            min_train_rows=1, validation_rows_total=5,
            rows_with_features=5, rows_without_features=0,
        )


def test_oot_split_summary_fold_boundaries_length_mismatch() -> None:
    with pytest.raises(ValueError, match="fold_boundaries length must match"):
        OutOfFoldSplitSummary(
            n_folds=2, split_strategy="loo",
            fold_boundaries=_VALID_BOUNDARIES,
            min_train_rows=1, validation_rows_total=5,
            rows_with_features=5, rows_without_features=0,
        )


def test_oot_split_summary_bad_boundary_keys() -> None:
    bad_boundaries = (
        {"fold_id": 0, "train_start": 0, "train_end": 10},
    )
    with pytest.raises(ValueError, match="Each fold boundary must include exactly"):
        OutOfFoldSplitSummary(
            n_folds=1, split_strategy="loo",
            fold_boundaries=bad_boundaries,
            min_train_rows=1, validation_rows_total=5,
            rows_with_features=5, rows_without_features=0,
        )


def test_oot_split_summary_validation_not_after_train() -> None:
    bad_boundaries = (
        {
            "fold_id": 0,
            "train_start": 0,
            "train_end": 10,
            "validation_start": 8,
            "validation_end": 15,
            "gap": 0,
        },
    )
    with pytest.raises(ValueError, match="must be strictly after"):
        OutOfFoldSplitSummary(
            n_folds=1, split_strategy="loo",
            fold_boundaries=bad_boundaries,
            min_train_rows=1, validation_rows_total=5,
            rows_with_features=5, rows_without_features=0,
        )


def test_oot_split_summary_negative_gap() -> None:
    bad_boundaries = (
        {
            "fold_id": 0,
            "train_start": 0,
            "train_end": 10,
            "validation_start": 12,
            "validation_end": 15,
            "gap": -1,
        },
    )
    with pytest.raises(ValueError, match="gap must be >= 0"):
        OutOfFoldSplitSummary(
            n_folds=1, split_strategy="loo",
            fold_boundaries=bad_boundaries,
            min_train_rows=1, validation_rows_total=5,
            rows_with_features=5, rows_without_features=0,
        )


def test_oot_split_summary_negative_counts() -> None:
    with pytest.raises(ValueError, match="min_train_rows must be a positive"):
        OutOfFoldSplitSummary(
            n_folds=1, split_strategy="loo",
            fold_boundaries=_VALID_BOUNDARIES,
            min_train_rows=0, validation_rows_total=5,
            rows_with_features=5, rows_without_features=0,
        )

    with pytest.raises(ValueError, match="validation_rows_total must be >= 0"):
        OutOfFoldSplitSummary(
            n_folds=1, split_strategy="loo",
            fold_boundaries=_VALID_BOUNDARIES,
            min_train_rows=1, validation_rows_total=-1,
            rows_with_features=5, rows_without_features=0,
        )

    with pytest.raises(ValueError, match="rows_with_features must be >= 0"):
        OutOfFoldSplitSummary(
            n_folds=1, split_strategy="loo",
            fold_boundaries=_VALID_BOUNDARIES,
            min_train_rows=1, validation_rows_total=5,
            rows_with_features=-1, rows_without_features=0,
        )

    with pytest.raises(ValueError, match="rows_without_features must be >= 0"):
        OutOfFoldSplitSummary(
            n_folds=1, split_strategy="loo",
            fold_boundaries=_VALID_BOUNDARIES,
            min_train_rows=1, validation_rows_total=5,
            rows_with_features=5, rows_without_features=-1,
        )


def test_oot_split_summary_empty_split_strategy_raises() -> None:
    with pytest.raises(ValueError, match="split_strategy"):
        OutOfFoldSplitSummary(
            n_folds=1, split_strategy="",
            fold_boundaries=_VALID_BOUNDARIES,
            min_train_rows=1, validation_rows_total=5,
            rows_with_features=5, rows_without_features=0,
        )


def test_out_of_fold_kernel_feature_result_fold_mismatch() -> None:
    summary = OutOfFoldSplitSummary(
        n_folds=1, split_strategy="loo",
        fold_boundaries=_VALID_BOUNDARIES,
        min_train_rows=1, validation_rows_total=5,
        rows_with_features=5, rows_without_features=0,
    )
    with pytest.raises(ValueError, match="must have the same length"):
        OutOfFoldKernelFeatureResult(
            features=[],
            fold_results=({},),
            fold_reports=(),
            combined_transform_report=None,  # type: ignore[arg-type]
            feature_evidence_report=None,
            split_summary=summary,
            warnings=(),
        )
