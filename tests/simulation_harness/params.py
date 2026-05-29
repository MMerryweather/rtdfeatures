"""Validation helpers for common simulation harness parameters."""

from __future__ import annotations


def validate_positive_timestep(dt: float) -> float:
    if dt <= 0:
        raise ValueError("dt must be positive")
    return float(dt)


def validate_non_negative_mass(feed_mass: float) -> float:
    if feed_mass < 0:
        raise ValueError("feed_mass must be non-negative")
    return float(feed_mass)


def validate_lag_bounds(min_lag: int, max_lag: int) -> tuple[int, int]:
    if min_lag < 0:
        raise ValueError("min_lag must be non-negative")
    if max_lag < 0:
        raise ValueError("max_lag must be non-negative")
    if max_lag < min_lag:
        raise ValueError("max_lag must be >= min_lag")
    return min_lag, max_lag
