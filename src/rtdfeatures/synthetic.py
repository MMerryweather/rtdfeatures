"""Deterministic synthetic helpers for kernel learning and feature generation tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, TypedDict, cast

import numpy as np
import polars as pl

from rtdfeatures.kernels.parametric import (
    discrete_delayed_exponential_weights,
    discrete_erlang_weights,
    discrete_exponential_weights,
    discrete_gamma_weights,
    discrete_lognormal_weights,
    parametric_lag_steps,
)


class _RequiredKernelMetadata(TypedDict):
    lag_steps: list[int]
    weights: list[float]
    dt: float
    min_lag: int
    max_lag: int
    mean_lag: float
    p50_lag: float
    p90_lag: float


class KernelMetadata(_RequiredKernelMetadata, total=False):
    parametric_family: str
    parametric_parameters: dict[str, float | int]


class SyntheticScenario(TypedDict, total=False):
    name: str
    seed: int
    n_rows: int
    dt: float
    params: dict[str, Any]


@dataclass(frozen=True)
class SyntheticDataset:
    data: pl.DataFrame
    true_kernels: dict[str, KernelMetadata]
    scenario: SyntheticScenario


def _regular_time(n_rows: int, dt: float) -> list[datetime]:
    if n_rows <= 0:
        raise ValueError("n_rows must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be strictly positive")
    start = datetime(2020, 1, 1)
    return [start + timedelta(seconds=(i * dt)) for i in range(n_rows)]


def _normalize_weights(weights: list[float]) -> list[float]:
    total = float(sum(weights))
    if total <= 0.0:
        raise ValueError("kernel weights must sum to a positive value")
    return [float(w / total) for w in weights]


def _validate_erlang_shape_k(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"shape_k must be a positive integer for Erlang fixtures; got {value!r}.")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(
                f"shape_k must be a positive integer for Erlang fixtures; got {value!r}."
            )
        return value
    if isinstance(value, float):
        as_int = int(value)
        if as_int <= 0 or float(as_int) != value:
            raise ValueError(
                f"shape_k must be a positive integer for Erlang fixtures; got {value!r}."
            )
        return as_int
    raise ValueError(f"shape_k must be a positive integer for Erlang fixtures; got {value!r}.")


def _kernel_metadata(lag_steps: list[int], weights: list[float], *, dt: float) -> KernelMetadata:
    if len(lag_steps) != len(weights) or not lag_steps:
        raise ValueError("lag_steps and weights must have equal non-zero length")
    if sorted(lag_steps) != lag_steps:
        raise ValueError("lag_steps must be sorted")
    w = _normalize_weights(weights)
    cdf = 0.0

    def _quantile(q: float) -> float:
        nonlocal cdf
        cdf = 0.0
        for step, wt in zip(lag_steps, w):
            cdf += wt
            if cdf >= q:
                return float(step * dt)
        return float(lag_steps[-1] * dt)

    mean_lag = float(sum(step * wt for step, wt in zip(lag_steps, w)) * dt)
    return {
        "lag_steps": [int(step) for step in lag_steps],
        "weights": [float(wi) for wi in w],
        "dt": float(dt),
        "min_lag": int(lag_steps[0]),
        "max_lag": int(lag_steps[-1]),
        "mean_lag": mean_lag,
        "p50_lag": _quantile(0.5),
        "p90_lag": _quantile(0.9),
    }


def _apply_kernel(signal: np.ndarray, lag_steps: list[int], weights: list[float]) -> np.ndarray:
    out = np.zeros(signal.shape[0], dtype=np.float64)
    for t in range(signal.shape[0]):
        value = 0.0
        for lag, weight in zip(lag_steps, weights):
            src = t - lag
            if src >= 0:
                value += weight * float(signal[src])
        out[t] = value
    return out


def _build_single_pair_dataset(
    *,
    n_rows: int,
    dt: float,
    seed: int,
    lag_steps: list[int],
    weights: list[float],
    noise_std: float,
    name: str,
) -> SyntheticDataset:
    time = _regular_time(n_rows, dt)
    metadata = _kernel_metadata(lag_steps, weights, dt=dt)
    kernel_weights = metadata["weights"]

    rng = np.random.default_rng(seed)
    x_base = rng.normal(loc=0.0, scale=1.0, size=n_rows)
    trend = np.sin(np.linspace(0.0, 6.0 * np.pi, n_rows, dtype=np.float64))
    x = (0.65 * x_base) + (0.35 * trend)

    y = _apply_kernel(x, metadata["lag_steps"], kernel_weights)
    if noise_std > 0.0:
        y = y + rng.normal(loc=0.0, scale=noise_std, size=n_rows)

    data = pl.DataFrame(
        {
            "time": time,
            "input_signal": x.astype(np.float64),
            "target_signal": y.astype(np.float64),
        }
    )
    return SyntheticDataset(
        data=data,
        true_kernels={"input_signal->target_signal": metadata},
        scenario={
            "name": name,
            "seed": seed,
            "n_rows": n_rows,
            "dt": dt,
            "params": {"noise_std": noise_std},
        },
    )


def _build_parametric_single_pair_dataset(
    *,
    n_rows: int,
    dt: float,
    seed: int,
    family: str,
    min_lag_steps: int,
    max_lag_steps: int,
    parameters: dict[str, float | int],
    noise_std: float,
    name: str,
    scenario_params: dict[str, Any] | None = None,
) -> SyntheticDataset:
    lag_steps = parametric_lag_steps(
        min_lag_steps=min_lag_steps,
        max_lag_steps=max_lag_steps,
    )
    if family == "exponential":
        weights = list(
            discrete_exponential_weights(
                rate_lambda=parameters["rate_lambda"],
                lag_steps=lag_steps,
                dt=dt,
            )
        )
    elif family == "delayed_exponential":
        weights = list(
            discrete_delayed_exponential_weights(
                delay=parameters["delay"],
                rate_lambda=parameters["rate_lambda"],
                lag_steps=lag_steps,
                dt=dt,
            )
        )
    elif family == "gamma":
        weights = list(
            discrete_gamma_weights(
                shape_alpha=parameters["shape_alpha"],
                rate_beta=parameters["rate_beta"],
                lag_steps=lag_steps,
                dt=dt,
            )
        )
    elif family == "lognormal":
        weights = list(
            discrete_lognormal_weights(
                log_mu=parameters["log_mu"],
                log_sigma=parameters["log_sigma"],
                lag_steps=lag_steps,
                dt=dt,
            )
        )
    elif family == "erlang":
        shape_k = _validate_erlang_shape_k(parameters["shape_k"])
        weights = list(
            discrete_erlang_weights(
                shape_k=shape_k,
                rate_beta=parameters["rate_beta"],
                lag_steps=lag_steps,
                dt=dt,
            )
        )
    else:
        raise ValueError(
            "family must be one of 'exponential', 'delayed_exponential', 'gamma', "
            "'lognormal', or 'erlang'."
        )

    out = _build_single_pair_dataset(
        n_rows=n_rows,
        dt=dt,
        seed=seed,
        lag_steps=list(lag_steps),
        weights=weights,
        noise_std=noise_std,
        name=name,
    )
    source_metadata = out.true_kernels["input_signal->target_signal"]
    metadata: KernelMetadata = {
        "lag_steps": list(source_metadata["lag_steps"]),
        "weights": list(source_metadata["weights"]),
        "dt": float(source_metadata["dt"]),
        "min_lag": int(source_metadata["min_lag"]),
        "max_lag": int(source_metadata["max_lag"]),
        "mean_lag": float(source_metadata["mean_lag"]),
        "p50_lag": float(source_metadata["p50_lag"]),
        "p90_lag": float(source_metadata["p90_lag"]),
    }
    metadata["parametric_family"] = family
    metadata["parametric_parameters"] = dict(parameters)
    scenario = cast(SyntheticScenario, dict(out.scenario))
    params = {
        "noise_std": noise_std,
        "family": family,
        "family_parameters": dict(metadata["parametric_parameters"]),
    }
    if scenario_params:
        params.update(scenario_params)
    scenario["params"] = params
    return SyntheticDataset(
        data=out.data,
        true_kernels={"input_signal->target_signal": cast(KernelMetadata, metadata)},
        scenario=scenario,
    )


def make_single_delay_dataset(
    *,
    n_rows: int = 240,
    dt: float = 1.0,
    seed: int = 7,
    delay_steps: int = 6,
    noise_std: float = 0.02,
) -> SyntheticDataset:
    delay = int(delay_steps)
    if delay < 0:
        raise ValueError(f"delay_steps must be non-negative; got {delay}.")
    return _build_single_pair_dataset(
        n_rows=n_rows,
        dt=dt,
        seed=seed,
        lag_steps=[delay],
        weights=[1.0],
        noise_std=noise_std,
        name="single_delay",
    )


def make_spread_delay_dataset(
    *,
    n_rows: int = 240,
    dt: float = 1.0,
    seed: int = 11,
    noise_std: float = 0.03,
) -> SyntheticDataset:
    return _build_single_pair_dataset(
        n_rows=n_rows,
        dt=dt,
        seed=seed,
        lag_steps=[4, 5, 6, 7],
        weights=[0.15, 0.35, 0.35, 0.15],
        noise_std=noise_std,
        name="spread_delay",
    )


def make_noisy_identifiable_dataset(
    *,
    n_rows: int = 320,
    dt: float = 1.0,
    seed: int = 19,
    noise_std: float = 0.08,
) -> SyntheticDataset:
    return _build_single_pair_dataset(
        n_rows=n_rows,
        dt=dt,
        seed=seed,
        lag_steps=[2, 3, 4],
        weights=[0.25, 0.55, 0.20],
        noise_std=noise_std,
        name="noisy_identifiable",
    )


def make_weak_identifiability_dataset(
    *,
    n_rows: int = 320,
    dt: float = 1.0,
    seed: int = 23,
    noise_std: float = 0.5,
) -> SyntheticDataset:
    return _build_single_pair_dataset(
        n_rows=n_rows,
        dt=dt,
        seed=seed,
        lag_steps=[2, 3, 4],
        weights=[0.30, 0.40, 0.30],
        noise_std=noise_std,
        name="weak_identifiability",
    )


def make_multi_pair_dataset(
    *,
    n_rows: int = 280,
    dt: float = 1.0,
    seed: int = 29,
    noise_std: float = 0.04,
) -> SyntheticDataset:
    time = _regular_time(n_rows, dt)
    rng = np.random.default_rng(seed)

    pairs = {
        "a": _kernel_metadata([2, 3], [0.65, 0.35], dt=dt),
        "b": _kernel_metadata([4, 5, 6], [0.20, 0.60, 0.20], dt=dt),
        "c": _kernel_metadata([1], [1.0], dt=dt),
    }

    columns: dict[str, Any] = {"time": time}
    true_kernels: dict[str, KernelMetadata] = {}
    for pair_id, kernel in pairs.items():
        input_name = f"input_signal_{pair_id}"
        target_name = f"target_signal_{pair_id}"
        x = rng.normal(loc=0.0, scale=1.0, size=n_rows)
        y = _apply_kernel(x, kernel["lag_steps"], kernel["weights"]) + rng.normal(
            loc=0.0, scale=noise_std, size=n_rows
        )
        columns[input_name] = x.astype(np.float64)
        columns[target_name] = y.astype(np.float64)
        true_kernels[f"{input_name}->{target_name}"] = kernel

    return SyntheticDataset(
        data=pl.DataFrame(columns),
        true_kernels=true_kernels,
        scenario={
            "name": "multi_pair",
            "seed": seed,
            "n_rows": n_rows,
            "dt": dt,
            "params": {"pair_count": len(pairs), "noise_std": noise_std},
        },
    )


def make_missing_window_dataset(
    *,
    n_rows: int = 260,
    dt: float = 1.0,
    seed: int = 31,
    missing_window_start: int = 90,
    missing_window_len: int = 14,
    noise_std: float = 0.03,
) -> SyntheticDataset:
    out = _build_single_pair_dataset(
        n_rows=n_rows,
        dt=dt,
        seed=seed,
        lag_steps=[3, 4, 5],
        weights=[0.25, 0.50, 0.25],
        noise_std=noise_std,
        name="missing_window",
    )

    start = int(max(0, missing_window_start))
    length = int(max(0, missing_window_len))
    end = min(n_rows, start + length)
    data = out.data
    if end > start:
        data = data.with_columns(
            pl.when((pl.arange(0, n_rows) >= start) & (pl.arange(0, n_rows) < end))
            .then(None)
            .otherwise(pl.col("input_signal"))
            .alias("input_signal"),
            pl.when((pl.arange(0, n_rows) >= start) & (pl.arange(0, n_rows) < end))
            .then(None)
            .otherwise(pl.col("target_signal"))
            .alias("target_signal"),
        )
    scenario = cast(SyntheticScenario, dict(out.scenario))
    scenario["params"] = {
        "noise_std": noise_std,
        "missing_window_start": start,
        "missing_window_len": max(0, end - start),
    }
    return SyntheticDataset(data=data, true_kernels=out.true_kernels, scenario=scenario)


def make_boundary_kernel_dataset(
    *,
    n_rows: int = 280,
    dt: float = 1.0,
    seed: int = 37,
    noise_std: float = 0.02,
) -> SyntheticDataset:
    return _build_single_pair_dataset(
        n_rows=n_rows,
        dt=dt,
        seed=seed,
        lag_steps=[0, 1, 2, 3, 4, 5, 6],
        weights=[0.72, 0.10, 0.06, 0.04, 0.03, 0.03, 0.02],
        noise_std=noise_std,
        name="boundary_kernel",
    )


def make_diffuse_kernel_dataset(
    *,
    n_rows: int = 300,
    dt: float = 1.0,
    seed: int = 41,
    noise_std: float = 0.03,
) -> SyntheticDataset:
    lags = list(range(0, 11))
    weights = [1.0 for _ in lags]
    return _build_single_pair_dataset(
        n_rows=n_rows,
        dt=dt,
        seed=seed,
        lag_steps=lags,
        weights=weights,
        noise_std=noise_std,
        name="diffuse_kernel",
    )


def make_baseline_challenge_dataset(
    *,
    n_rows: int = 360,
    dt: float = 1.0,
    seed: int = 43,
    noise_std: float = 0.04,
) -> SyntheticDataset:
    return _build_single_pair_dataset(
        n_rows=n_rows,
        dt=dt,
        seed=seed,
        lag_steps=[5, 6, 7, 8],
        weights=[0.10, 0.40, 0.35, 0.15],
        noise_std=noise_std,
        name="baseline_challenge",
    )


def make_exponential_kernel_dataset(
    *,
    n_rows: int = 360,
    dt: float = 60.0,
    seed: int = 47,
    min_lag_steps: int = 1,
    max_lag_steps: int = 8,
    rate_lambda: float = 0.02,
    noise_std: float = 0.03,
) -> SyntheticDataset:
    return _build_parametric_single_pair_dataset(
        n_rows=n_rows,
        dt=dt,
        seed=seed,
        family="exponential",
        min_lag_steps=min_lag_steps,
        max_lag_steps=max_lag_steps,
        parameters={"rate_lambda": rate_lambda},
        noise_std=noise_std,
        name="parametric_exponential",
    )


def make_gamma_kernel_dataset(
    *,
    n_rows: int = 420,
    dt: float = 60.0,
    seed: int = 53,
    min_lag_steps: int = 1,
    max_lag_steps: int = 10,
    shape_alpha: float = 3.0,
    rate_beta: float = 0.06,
    noise_std: float = 0.03,
) -> SyntheticDataset:
    return _build_parametric_single_pair_dataset(
        n_rows=n_rows,
        dt=dt,
        seed=seed,
        family="gamma",
        min_lag_steps=min_lag_steps,
        max_lag_steps=max_lag_steps,
        parameters={"shape_alpha": shape_alpha, "rate_beta": rate_beta},
        noise_std=noise_std,
        name="parametric_gamma",
    )


def make_delayed_exponential_kernel_dataset(
    *,
    n_rows: int = 360,
    dt: float = 60.0,
    seed: int = 67,
    min_lag_steps: int = 0,
    max_lag_steps: int = 10,
    delay: float = 180.0,
    rate_lambda: float = 0.03,
    noise_std: float = 0.03,
) -> SyntheticDataset:
    return _build_parametric_single_pair_dataset(
        n_rows=n_rows,
        dt=dt,
        seed=seed,
        family="delayed_exponential",
        min_lag_steps=min_lag_steps,
        max_lag_steps=max_lag_steps,
        parameters={"delay": delay, "rate_lambda": rate_lambda},
        noise_std=noise_std,
        name="parametric_delayed_exponential",
    )


def make_lognormal_kernel_dataset(
    *,
    n_rows: int = 420,
    dt: float = 60.0,
    seed: int = 71,
    min_lag_steps: int = 1,
    max_lag_steps: int = 12,
    log_mu: float = 5.0,
    log_sigma: float = 0.5,
    noise_std: float = 0.03,
) -> SyntheticDataset:
    return _build_parametric_single_pair_dataset(
        n_rows=n_rows,
        dt=dt,
        seed=seed,
        family="lognormal",
        min_lag_steps=min_lag_steps,
        max_lag_steps=max_lag_steps,
        parameters={"log_mu": log_mu, "log_sigma": log_sigma},
        noise_std=noise_std,
        name="parametric_lognormal",
    )


def make_erlang_kernel_dataset(
    *,
    n_rows: int = 420,
    dt: float = 60.0,
    seed: int = 73,
    min_lag_steps: int = 0,
    max_lag_steps: int = 12,
    shape_k: int = 3,
    rate_beta: float = 0.05,
    noise_std: float = 0.03,
) -> SyntheticDataset:
    validated_shape_k = _validate_erlang_shape_k(shape_k)
    return _build_parametric_single_pair_dataset(
        n_rows=n_rows,
        dt=dt,
        seed=seed,
        family="erlang",
        min_lag_steps=min_lag_steps,
        max_lag_steps=max_lag_steps,
        parameters={"shape_k": validated_shape_k, "rate_beta": rate_beta},
        noise_std=noise_std,
        name="parametric_erlang",
    )


def make_misspecified_parametric_dataset(
    *,
    n_rows: int = 420,
    dt: float = 60.0,
    seed: int = 59,
    min_lag_steps: int = 1,
    max_lag_steps: int = 10,
    true_shape_alpha: float = 6.0,
    true_rate_beta: float = 0.08,
    noise_std: float = 0.02,
) -> SyntheticDataset:
    return _build_parametric_single_pair_dataset(
        n_rows=n_rows,
        dt=dt,
        seed=seed,
        family="gamma",
        min_lag_steps=min_lag_steps,
        max_lag_steps=max_lag_steps,
        parameters={"shape_alpha": true_shape_alpha, "rate_beta": true_rate_beta},
        noise_std=noise_std,
        name="parametric_misspecified_family",
        scenario_params={
            "expected_misspecified_family": "exponential",
            "true_family": "gamma",
        },
    )


def make_weak_parametric_identifiability_dataset(
    *,
    n_rows: int = 360,
    dt: float = 60.0,
    seed: int = 61,
    min_lag_steps: int = 1,
    max_lag_steps: int = 10,
    shape_alpha: float = 2.5,
    rate_beta: float = 0.05,
    noise_std: float = 0.8,
) -> SyntheticDataset:
    return _build_parametric_single_pair_dataset(
        n_rows=n_rows,
        dt=dt,
        seed=seed,
        family="gamma",
        min_lag_steps=min_lag_steps,
        max_lag_steps=max_lag_steps,
        parameters={"shape_alpha": shape_alpha, "rate_beta": rate_beta},
        noise_std=noise_std,
        name="parametric_weak_identifiability",
        scenario_params={"weak_identifiability": True},
    )


__all__ = [
    "KernelMetadata",
    "SyntheticScenario",
    "SyntheticDataset",
    "make_single_delay_dataset",
    "make_spread_delay_dataset",
    "make_noisy_identifiable_dataset",
    "make_weak_identifiability_dataset",
    "make_multi_pair_dataset",
    "make_missing_window_dataset",
    "make_boundary_kernel_dataset",
    "make_diffuse_kernel_dataset",
    "make_baseline_challenge_dataset",
    "make_exponential_kernel_dataset",
    "make_gamma_kernel_dataset",
    "make_delayed_exponential_kernel_dataset",
    "make_lognormal_kernel_dataset",
    "make_erlang_kernel_dataset",
    "make_misspecified_parametric_dataset",
    "make_weak_parametric_identifiability_dataset",
]
