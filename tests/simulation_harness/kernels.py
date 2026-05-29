"""Discrete-kernel utilities for harness fixtures and assertions."""

from __future__ import annotations

import math

import numpy as np

from .contracts import KernelMetadata


def build_lag_grid(min_lag: int, max_lag: int) -> list[int]:
    if min_lag < 0 or max_lag < 0:
        raise ValueError("lag bounds must be non-negative")
    if max_lag < min_lag:
        raise ValueError("max_lag must be >= min_lag")
    return list(range(min_lag, max_lag + 1))


def normalize_weights(weights: list[float], *, tol: float = 1e-12) -> list[float]:
    if len(weights) == 0:
        raise ValueError("weights must be non-empty")
    if any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative")
    total = float(sum(weights))
    if total <= tol:
        raise ValueError("weights sum must be positive")
    return [float(w / total) for w in weights]


def summarize_kernel(lag_steps: list[int], weights: list[float], *, dt: float) -> KernelMetadata:
    if len(lag_steps) != len(weights):
        raise ValueError("lag_steps and weights must have equal length")
    if len(lag_steps) == 0:
        raise ValueError("lag_steps must be non-empty")
    normalized = normalize_weights(weights)
    lags = np.asarray(lag_steps, dtype=np.float64)
    ws = np.asarray(normalized, dtype=np.float64)
    mean_lag = float(np.dot(lags, ws))

    cdf = np.cumsum(ws)
    p50_lag = float(lags[int(np.searchsorted(cdf, 0.5, side="left"))])
    p90_lag = float(lags[int(np.searchsorted(cdf, 0.9, side="left"))])

    return {
        "lag_steps": [int(v) for v in lag_steps],
        "weights": normalized,
        "dt": float(dt),
        "min_lag": int(min(lag_steps)),
        "max_lag": int(max(lag_steps)),
        "mean_lag": mean_lag,
        "p50_lag": p50_lag,
        "p90_lag": p90_lag,
    }


def convolve_discrete_kernels(
    lag_steps_a: list[int],
    weights_a: list[float],
    lag_steps_b: list[int],
    weights_b: list[float],
) -> tuple[list[int], list[float]]:
    if len(lag_steps_a) != len(weights_a) or len(lag_steps_b) != len(weights_b):
        raise ValueError("each lag list must match its weight list length")
    wa = normalize_weights(weights_a)
    wb = normalize_weights(weights_b)

    accum: dict[int, float] = {}
    for lag_a, w_a in zip(lag_steps_a, wa):
        for lag_b, w_b in zip(lag_steps_b, wb):
            out_lag = lag_a + lag_b
            accum[out_lag] = accum.get(out_lag, 0.0) + (w_a * w_b)

    lags = sorted(accum)
    weights = normalize_weights([accum[lag] for lag in lags])
    return lags, weights


def make_plug_flow_kernel(*, lag: int = 6, dt: float = 1.0) -> KernelMetadata:
    if lag < 0:
        raise ValueError("lag must be non-negative")
    return summarize_kernel([lag], [1.0], dt=dt)


def make_narrow_spread_kernel(*, dt: float = 1.0) -> KernelMetadata:
    return summarize_kernel([5, 6, 7], [0.2, 0.6, 0.2], dt=dt)


def make_tank_kernel(*, rho: float, max_lag: int, dt: float = 1.0) -> KernelMetadata:
    if not (0 <= rho < 1):
        raise ValueError("rho must satisfy 0 <= rho < 1")
    if max_lag < 0:
        raise ValueError("max_lag must be non-negative")
    lag_steps = list(range(max_lag + 1))
    weights = [(1.0 - rho) * (rho**k) for k in lag_steps]
    return summarize_kernel(lag_steps, weights, dt=dt)


def make_recycle_pass_mixture(
    pass_delays: list[int], *, product_split: float, recycle_split: float, dt: float = 1.0
) -> KernelMetadata:
    if len(pass_delays) == 0:
        raise ValueError("pass_delays must be non-empty")
    if any(delay < 0 for delay in pass_delays):
        raise ValueError("pass_delays must be non-negative")
    if product_split <= 0:
        raise ValueError("product_split must be positive")
    if recycle_split < 0:
        raise ValueError("recycle_split must be non-negative")

    weights = [product_split * (recycle_split**idx) for idx, _ in enumerate(pass_delays)]
    return summarize_kernel(pass_delays, weights, dt=dt)


def feed_grade_signal(time_index: int, *, n_rows: int) -> float:
    step = 0.5 if time_index >= (n_rows // 3) else 0.0
    sinusoid = 0.1 * math.sin(2.0 * math.pi * time_index / 32.0)
    return 1.0 + step + sinusoid
