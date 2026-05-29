"""Forward-chaining split configuration and generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ForwardChainingSplitConfig:
    """Configuration for deterministic forward-chaining fold boundaries."""

    n_folds: int
    min_train_size: int
    validation_size: int
    gap: int = 0
    max_train_size: int | None = None

    def __post_init__(self) -> None:
        if self.n_folds <= 0:
            raise ValueError("n_folds must be a positive integer.")
        if self.min_train_size <= 0:
            raise ValueError("min_train_size must be a positive integer.")
        if self.validation_size <= 0:
            raise ValueError("validation_size must be a positive integer.")
        if self.gap < 0:
            raise ValueError("gap must be a non-negative integer.")
        if self.max_train_size is not None and self.max_train_size <= 0:
            raise ValueError("max_train_size must be a positive integer when provided.")


@dataclass(frozen=True)
class ForwardChainingFoldSplit:
    """One fold split with explicit row-index boundaries."""

    fold_id: int
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int
    gap: int


def generate_forward_chaining_splits(
    *,
    n_rows: int,
    config: ForwardChainingSplitConfig,
) -> tuple[ForwardChainingFoldSplit, ...]:
    """Generate deterministic leakage-safe forward-chaining fold splits."""
    if n_rows <= 0:
        raise ValueError("n_rows must be a positive integer.")

    required_rows = config.min_train_size + config.gap + config.validation_size * config.n_folds
    if n_rows < required_rows:
        raise ValueError(
            "Invalid split configuration: need at least "
            f"{required_rows} rows but received {n_rows}."
        )

    folds: list[ForwardChainingFoldSplit] = []
    for fold_id in range(config.n_folds):
        validation_start = config.min_train_size + config.gap + fold_id * config.validation_size
        validation_end = validation_start + config.validation_size
        train_end_exclusive = validation_start - config.gap
        train_start = 0
        if config.max_train_size is not None:
            train_start = max(0, train_end_exclusive - config.max_train_size)
        train_end = train_end_exclusive - 1

        train_indices = tuple(range(train_start, train_end_exclusive))
        validation_indices = tuple(range(validation_start, validation_end))
        if not train_indices:
            raise ValueError(
                f"Fold {fold_id} has no training rows; increase min_train_size or reduce gap."
            )
        if len(train_indices) < config.min_train_size:
            raise ValueError(
                f"Fold {fold_id} has {len(train_indices)} training rows, "
                f"below min_train_size={config.min_train_size}."
            )
        if train_indices[-1] >= validation_indices[0]:
            raise ValueError(
                f"Fold {fold_id} leakage detected: training overlaps validation rows."
            )

        folds.append(
            ForwardChainingFoldSplit(
                fold_id=fold_id,
                train_indices=train_indices,
                validation_indices=validation_indices,
                train_start=train_start,
                train_end=train_end,
                validation_start=validation_start,
                validation_end=validation_end - 1,
                gap=config.gap,
            )
        )

    return tuple(folds)
