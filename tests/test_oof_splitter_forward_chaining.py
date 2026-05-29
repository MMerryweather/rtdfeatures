"""Tests for deterministic forward-chaining OOF split generation."""

from __future__ import annotations

import pytest

from rtdfeatures.oof import ForwardChainingSplitConfig, generate_forward_chaining_splits


def test_forward_chaining_boundaries_are_deterministic() -> None:
    config = ForwardChainingSplitConfig(
        n_folds=3,
        min_train_size=5,
        validation_size=2,
        gap=1,
    )
    first = generate_forward_chaining_splits(n_rows=20, config=config)
    second = generate_forward_chaining_splits(n_rows=20, config=config)

    assert first == second
    boundaries = [
        (fold.train_start, fold.train_end, fold.validation_start, fold.validation_end)
        for fold in first
    ]
    assert boundaries == [
        (0, 4, 6, 7),
        (0, 6, 8, 9),
        (0, 8, 10, 11),
    ]


def test_validation_is_after_training_no_self_leakage() -> None:
    config = ForwardChainingSplitConfig(
        n_folds=2,
        min_train_size=4,
        validation_size=3,
        gap=0,
    )
    splits = generate_forward_chaining_splits(n_rows=20, config=config)

    for fold in splits:
        assert max(fold.train_indices) < min(fold.validation_indices)
        assert set(fold.train_indices).isdisjoint(set(fold.validation_indices))


def test_gap_excludes_rows_between_train_and_validation() -> None:
    config = ForwardChainingSplitConfig(
        n_folds=1,
        min_train_size=5,
        validation_size=2,
        gap=3,
    )
    split = generate_forward_chaining_splits(n_rows=20, config=config)[0]

    assert split.train_indices == (0, 1, 2, 3, 4)
    assert split.validation_indices == (8, 9)


def test_max_train_size_caps_training_window() -> None:
    config = ForwardChainingSplitConfig(
        n_folds=2,
        min_train_size=4,
        validation_size=2,
        gap=1,
        max_train_size=5,
    )
    splits = generate_forward_chaining_splits(n_rows=20, config=config)

    assert splits[0].train_indices == (0, 1, 2, 3)
    assert splits[1].train_indices == (1, 2, 3, 4, 5)


def test_invalid_config_errors_are_clear() -> None:
    with pytest.raises(ValueError, match="n_folds must be a positive integer"):
        ForwardChainingSplitConfig(n_folds=0, min_train_size=1, validation_size=1)

    with pytest.raises(ValueError, match="min_train_size must be a positive integer"):
        ForwardChainingSplitConfig(n_folds=1, min_train_size=0, validation_size=1)

    with pytest.raises(ValueError, match="validation_size must be a positive integer"):
        ForwardChainingSplitConfig(n_folds=1, min_train_size=1, validation_size=0)

    with pytest.raises(ValueError, match="gap must be a non-negative integer"):
        ForwardChainingSplitConfig(n_folds=1, min_train_size=1, validation_size=1, gap=-1)

    with pytest.raises(ValueError, match="max_train_size must be a positive integer"):
        ForwardChainingSplitConfig(
            n_folds=1,
            min_train_size=1,
            validation_size=1,
            max_train_size=0,
        )


def test_invalid_row_budget_is_rejected() -> None:
    config = ForwardChainingSplitConfig(
        n_folds=3,
        min_train_size=5,
        validation_size=2,
        gap=1,
    )
    with pytest.raises(ValueError, match="need at least"):
        generate_forward_chaining_splits(n_rows=11, config=config)


def test_no_future_leakage_across_earlier_folds() -> None:
    config = ForwardChainingSplitConfig(
        n_folds=3,
        min_train_size=3,
        validation_size=2,
        gap=1,
    )
    splits = generate_forward_chaining_splits(n_rows=20, config=config)

    for left, right in zip(splits, splits[1:]):
        assert left.validation_end < right.validation_start
        assert max(left.train_indices) < right.validation_start
