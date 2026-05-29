"""Centralised feature name construction — all generated feature names come from this module."""

from __future__ import annotations

_NUMERIC_METRICS = frozenset({"wmean", "wstd", "wsum"})
_AGE_METRICS = frozenset({"mean", "p50", "p90", "tail_gt_threshold"})


def numeric_feature_name(kernel_name: str, source_col: str, metric: str) -> str:
    if not kernel_name:
        raise ValueError("kernel_name must be non-empty.")
    if not source_col:
        raise ValueError("source_col must be non-empty.")
    if metric not in _NUMERIC_METRICS:
        raise ValueError(
            f"Unknown numeric metric '{metric}'. "
            f"Allowed: {sorted(_NUMERIC_METRICS)}"
        )
    return f"{kernel_name}_num_{source_col}_{metric}"


def categorical_fraction_feature_name(kernel_name: str, source_col: str, level: str) -> str:
    if not kernel_name:
        raise ValueError("kernel_name must be non-empty.")
    if not source_col:
        raise ValueError("source_col must be non-empty.")
    return f"{kernel_name}_cat_{source_col}_{level}_frac"


def categorical_entropy_feature_name(kernel_name: str, source_col: str) -> str:
    if not kernel_name:
        raise ValueError("kernel_name must be non-empty.")
    if not source_col:
        raise ValueError("source_col must be non-empty.")
    return f"{kernel_name}_cat_{source_col}_entropy"


def age_feature_name(kernel_name: str, metric: str) -> str:
    if not kernel_name:
        raise ValueError("kernel_name must be non-empty.")
    if metric not in _AGE_METRICS:
        raise ValueError(
            f"Unknown age metric '{metric}'. "
            f"Allowed: {sorted(_AGE_METRICS)}"
        )
    return f"{kernel_name}_age_{metric}"
