"""Numeric feature computation helpers."""

from __future__ import annotations

import math

import numpy as np


def weighted_numeric_series(
    *,
    values: np.ndarray,
    lag_steps: np.ndarray,
    lag_weights: np.ndarray,
    max_lag_steps: int,
    weight_values: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    mean_out = np.full(values.shape[0], np.nan, dtype=np.float64)
    std_out = np.full(values.shape[0], np.nan, dtype=np.float64)
    sum_out = np.full(values.shape[0], np.nan, dtype=np.float64)
    lag_steps_arr = np.asarray(lag_steps, dtype=np.int64)
    lag_weights_arr = np.asarray(lag_weights, dtype=np.float64)
    zero_denominator_count = 0

    for idx in range(max_lag_steps, values.shape[0]):
        window_idx = idx - lag_steps_arr
        x = values[window_idx]
        if not np.isfinite(x).all():
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
        numerator = float(np.sum(effective_weights * x))
        mean = numerator / denominator
        var = float(np.sum(effective_weights * np.square(x - mean)) / denominator)
        mean_out[idx] = mean
        std_out[idx] = math.sqrt(max(var, 0.0))
        sum_out[idx] = numerator

    return mean_out, std_out, sum_out, zero_denominator_count
