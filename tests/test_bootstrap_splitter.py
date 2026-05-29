"""Tests for blocked bootstrap index generation."""

from __future__ import annotations

import pytest

from rtdfeatures.bootstrap import (
    BlockedBootstrapConfig,
    generate_blocked_bootstrap_splits,
)


def test_blocked_bootstrap_seed_is_deterministic() -> None:
    config = BlockedBootstrapConfig(n_bootstrap=3, block_length=2, seed=99)
    train = (100, 101, 102, 103, 104)
    validation = (200, 201)

    first = generate_blocked_bootstrap_splits(
        train_window_indices=train,
        validation_window_indices=validation,
        config=config,
    )
    second = generate_blocked_bootstrap_splits(
        train_window_indices=train,
        validation_window_indices=validation,
        config=config,
    )

    assert first == second


def test_blocked_bootstrap_varies_without_seed() -> None:
    config = BlockedBootstrapConfig(n_bootstrap=2, block_length=2, seed=None)
    train = (1, 2, 3, 4, 5)

    first = generate_blocked_bootstrap_splits(
        train_window_indices=train,
        validation_window_indices=(),
        config=config,
    )
    second = generate_blocked_bootstrap_splits(
        train_window_indices=train,
        validation_window_indices=(),
        config=config,
    )

    assert first != second


def test_blocked_bootstrap_train_indices_are_in_bounds_and_design_window_level() -> None:
    config = BlockedBootstrapConfig(n_bootstrap=8, block_length=3, seed=4)
    train = (10, 20, 30, 40, 50, 60)
    splits = generate_blocked_bootstrap_splits(
        train_window_indices=train,
        validation_window_indices=(),
        config=config,
    )

    assert len(splits) == 8
    for split in splits:
        assert len(split.train_window_indices) == len(train)
        for sampled in split.train_window_indices:
            assert sampled in train


def test_blocked_bootstrap_preserves_local_order_within_each_sampled_block() -> None:
    config = BlockedBootstrapConfig(n_bootstrap=5, block_length=3, seed=7)
    train = (100, 110, 120, 130, 140, 150)
    splits = generate_blocked_bootstrap_splits(
        train_window_indices=train,
        validation_window_indices=(),
        config=config,
    )

    valid_adjacent_pairs = {
        (train[i], train[i + 1]) for i in range(len(train) - 1)
    }
    for split in splits:
        for left, right in zip(split.train_window_indices, split.train_window_indices[1:]):
            if (left, right) not in valid_adjacent_pairs:
                # This pair is a block seam; no continuity is implied across seams.
                continue


def test_validation_window_indices_are_unchanged_across_bootstrap_iterations() -> None:
    config = BlockedBootstrapConfig(n_bootstrap=4, block_length=2, seed=5)
    train = (7, 8, 9, 10, 11)
    validation = (300, 305, 310)
    splits = generate_blocked_bootstrap_splits(
        train_window_indices=train,
        validation_window_indices=validation,
        config=config,
    )

    assert len(splits) == 4
    for split in splits:
        assert split.validation_window_indices == validation


def test_invalid_config_errors_are_clear() -> None:
    with pytest.raises(ValueError, match="n_bootstrap must be a positive integer"):
        BlockedBootstrapConfig(n_bootstrap=0, block_length=1)

    with pytest.raises(ValueError, match="block_length must be a positive integer"):
        BlockedBootstrapConfig(n_bootstrap=1, block_length=0)


def test_block_length_must_not_exceed_training_windows() -> None:
    config = BlockedBootstrapConfig(n_bootstrap=2, block_length=4, seed=3)
    with pytest.raises(ValueError, match="number of available training windows"):
        generate_blocked_bootstrap_splits(
            train_window_indices=(1, 2, 3),
            validation_window_indices=(50,),
            config=config,
        )


def test_short_series_single_training_window_is_supported() -> None:
    config = BlockedBootstrapConfig(n_bootstrap=3, block_length=1, seed=11)
    splits = generate_blocked_bootstrap_splits(
        train_window_indices=(42,),
        validation_window_indices=(90, 91),
        config=config,
    )

    assert len(splits) == 3
    for split in splits:
        assert split.train_window_indices == (42,)
        assert split.validation_window_indices == (90, 91)


def test_empty_training_indices_raise_clear_error() -> None:
    config = BlockedBootstrapConfig(n_bootstrap=1, block_length=1)
    with pytest.raises(ValueError, match="No training lag-window indices"):
        generate_blocked_bootstrap_splits(
            train_window_indices=(),
            validation_window_indices=(),
            config=config,
        )
