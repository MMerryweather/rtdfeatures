"""Categorical feature computation helpers."""

from __future__ import annotations

import math

import numpy as np


def categorical_fraction_and_entropy_series(
    *,
    category_values: np.ndarray,
    levels: list[str],
    lag_steps: np.ndarray,
    lag_weights: np.ndarray,
    max_lag_steps: int,
    weight_values: np.ndarray | None,
) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    level_arrays = {
        level: np.full(category_values.shape[0], np.nan, dtype=np.float64)
        for level in levels
    }
    entropy_array = np.full(category_values.shape[0], np.nan, dtype=np.float64)
    lag_steps_arr = np.asarray(lag_steps, dtype=np.int64)
    lag_weights_arr = np.asarray(lag_weights, dtype=np.float64)
    zero_denominator_count = 0

    for idx in range(max_lag_steps, category_values.shape[0]):
        window_idx = idx - lag_steps_arr
        cat_window = category_values[window_idx]
        if any(level is None for level in cat_window):
            continue

        if weight_values is None:
            effective_weights = lag_weights_arr
        else:
            w = weight_values[window_idx]
            if not np.isfinite(w).all():
                continue
            effective_weights = lag_weights_arr * w

        denominator = float(np.sum(effective_weights))
        if math.isclose(denominator, 0.0, abs_tol=1e-12):
            zero_denominator_count += 1
            continue

        fractions: list[float] = []
        cat_window_str = [str(level) for level in cat_window]
        for level in levels:
            mask = np.asarray(
                [window_level == level for window_level in cat_window_str],
                dtype=np.float64,
            )
            frac = float(np.sum(effective_weights * mask) / denominator)
            level_arrays[level][idx] = frac
            fractions.append(frac)

        positive = [frac for frac in fractions if frac > 0.0]
        entropy_array[idx] = float(-sum(frac * math.log(frac) for frac in positive))

    return level_arrays, entropy_array, zero_denominator_count
