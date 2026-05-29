"""Deterministic seed helpers."""

from __future__ import annotations

import numpy as np


def validate_seed(seed: int) -> int:
    if not isinstance(seed, int):
        raise TypeError("seed must be an int")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    return seed


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(validate_seed(seed))
