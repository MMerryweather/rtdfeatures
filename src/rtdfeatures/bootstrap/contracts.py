"""Bootstrap data-class contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BlockedBootstrapConfig:
    """Configuration for blocked bootstrap index generation."""

    n_bootstrap: int
    block_length: int
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.n_bootstrap <= 0:
            raise ValueError("n_bootstrap must be a positive integer.")
        if self.block_length <= 0:
            raise ValueError("block_length must be a positive integer.")


@dataclass(frozen=True)
class BootstrapIndexSplit:
    """One bootstrap split over design-window training indices."""

    bootstrap_id: int
    train_window_indices: tuple[int, ...]
    validation_window_indices: tuple[int, ...]


@dataclass(frozen=True)
class _BootstrapContext:
    x_train_scaled: Any  # np.ndarray
    y_train_scaled: Any
    x_valid_scaled: Any
    y_valid_scaled: Any
    lag_steps: tuple[int, ...]
    dt_seconds: float
    train_size: int
    total_windows: int
    validation_window_indices: tuple[int, ...]
    validation_fraction: float
    loss: str
    huber_delta: float
    learner_parameters: dict[str, Any]
