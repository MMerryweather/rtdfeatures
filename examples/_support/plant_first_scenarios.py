"""Private deterministic scenario fixtures for plant-first example generation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from rtdfeatures.synthetic import (
    KernelMetadata,
    SyntheticDataset,
    make_delayed_exponential_kernel_dataset,
    make_diffuse_kernel_dataset,
    make_erlang_kernel_dataset,
    make_exponential_kernel_dataset,
    make_lognormal_kernel_dataset,
    make_single_delay_dataset,
)


@dataclass(frozen=True)
class ScenarioFixture:
    name: str
    positive_kernels: tuple[str, ...]
    comparison_kernels: tuple[str, ...]
    dataset_factory: Callable[[], SyntheticDataset]


def _regular_time(n_rows: int, dt: float) -> list[datetime]:
    start = datetime(2020, 1, 1)
    return [start + timedelta(seconds=(idx * dt)) for idx in range(n_rows)]


def _normalise(weights: list[float]) -> list[float]:
    total = float(sum(weights))
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    return [float(weight / total) for weight in weights]


def _kernel_metadata(lag_steps: list[int], weights: list[float], *, dt: float) -> KernelMetadata:
    if len(lag_steps) != len(weights) or not lag_steps:
        raise ValueError("lag_steps and weights must have equal non-zero length")
    normalised = _normalise(weights)
    cdf = 0.0

    def _quantile(q: float) -> float:
        nonlocal cdf
        cdf = 0.0
        for lag, weight in zip(lag_steps, normalised):
            cdf += weight
            if cdf >= q:
                return float(lag * dt)
        return float(lag_steps[-1] * dt)

    return {
        "lag_steps": [int(lag) for lag in lag_steps],
        "weights": [float(weight) for weight in normalised],
        "dt": float(dt),
        "min_lag": int(lag_steps[0]),
        "max_lag": int(lag_steps[-1]),
        "mean_lag": float(sum(lag * weight for lag, weight in zip(lag_steps, normalised)) * dt),
        "p50_lag": _quantile(0.5),
        "p90_lag": _quantile(0.9),
    }


def _apply_kernel(signal: np.ndarray, lag_steps: list[int], weights: list[float]) -> np.ndarray:
    output = np.zeros(signal.shape[0], dtype=np.float64)
    for row_idx in range(signal.shape[0]):
        value = 0.0
        for lag, weight in zip(lag_steps, weights):
            source_idx = row_idx - lag
            if source_idx >= 0:
                value += weight * float(signal[source_idx])
        output[row_idx] = value
    return output


def _convolve_discrete(
    lag_a: list[int], weight_a: list[float], lag_b: list[int], weight_b: list[float]
) -> tuple[list[int], list[float]]:
    accum: dict[int, float] = {}
    for lag1, w1 in zip(lag_a, weight_a):
        for lag2, w2 in zip(lag_b, weight_b):
            lag = int(lag1 + lag2)
            accum[lag] = float(accum.get(lag, 0.0) + (w1 * w2))
    lags = sorted(accum)
    weights = _normalise([float(accum[lag]) for lag in lags])
    return lags, weights


def _recycle_product_kernel(
    *,
    pass_lags: list[int],
    pass_weights: list[float],
    product_split: float,
    recycle_split: float,
    max_passes: int,
) -> tuple[list[int], list[float]]:
    pass_support = list(pass_lags)
    pass_support_weights = _normalise(list(pass_weights))
    product_accum: dict[int, float] = {}
    for pass_idx in range(max_passes):
        recycle_weight = recycle_split**pass_idx
        product_weight = product_split * recycle_weight
        for lag, weight in zip(pass_support, pass_support_weights):
            product_accum[lag] = float(product_accum.get(lag, 0.0) + (product_weight * weight))
        if pass_idx < (max_passes - 1):
            pass_support, pass_support_weights = _convolve_discrete(
                pass_support, pass_support_weights, pass_lags, pass_weights
            )
    support = sorted(product_accum)
    return support, _normalise([float(product_accum[lag]) for lag in support])


def core_scenario_fixtures() -> tuple[ScenarioFixture, ...]:
    return (
        ScenarioFixture(
            name="conveyor",
            positive_kernels=("FixedDelayKernelLearner",),
            comparison_kernels=(
                "SimplexKernelLearner",
                "GammaKernelLearner",
                "ExponentialKernelLearner",
            ),
            dataset_factory=lambda: make_single_delay_dataset(
                n_rows=420, dt=60.0, seed=101, delay_steps=6, noise_std=0.015
            ),
        ),
        ScenarioFixture(
            name="cstr",
            positive_kernels=("ExponentialKernelLearner",),
            comparison_kernels=("SimplexKernelLearner", "GammaKernelLearner"),
            dataset_factory=lambda: make_exponential_kernel_dataset(
                n_rows=480,
                dt=60.0,
                seed=102,
                min_lag_steps=0,
                max_lag_steps=10,
                rate_lambda=0.025,
                noise_std=0.02,
            ),
        ),
        ScenarioFixture(
            name="tanks_in_series",
            positive_kernels=("GammaKernelLearner", "ErlangKernelLearner"),
            comparison_kernels=("SimplexKernelLearner", "ExponentialKernelLearner"),
            dataset_factory=lambda: make_erlang_kernel_dataset(
                n_rows=520,
                dt=60.0,
                seed=103,
                min_lag_steps=0,
                max_lag_steps=12,
                shape_k=3,
                rate_beta=0.045,
                noise_std=0.02,
            ),
        ),
        ScenarioFixture(
            name="flotation_banks",
            positive_kernels=("LogNormalKernelLearner",),
            comparison_kernels=("SimplexKernelLearner", "GammaKernelLearner"),
            dataset_factory=lambda: make_lognormal_kernel_dataset(
                n_rows=520,
                dt=60.0,
                seed=104,
                min_lag_steps=1,
                max_lag_steps=14,
                log_mu=5.6,
                log_sigma=0.45,
                noise_std=0.025,
            ),
        ),
        ScenarioFixture(
            name="closed_loop_crushing",
            positive_kernels=("DelayedExponentialKernelLearner",),
            comparison_kernels=(
                "SimplexKernelLearner",
                "GammaKernelLearner",
                "ExponentialKernelLearner",
            ),
            dataset_factory=lambda: make_delayed_exponential_kernel_dataset(
                n_rows=520,
                dt=60.0,
                seed=105,
                min_lag_steps=0,
                max_lag_steps=14,
                delay=240.0,
                rate_lambda=0.025,
                noise_std=0.025,
            ),
        ),
        ScenarioFixture(
            name="bounded_hold_up_tank",
            positive_kernels=("UniformKernelLearner",),
            comparison_kernels=(
                "SimplexKernelLearner",
                "FixedDelayKernelLearner",
                "ExponentialKernelLearner",
            ),
            dataset_factory=lambda: make_diffuse_kernel_dataset(
                n_rows=520, dt=60.0, seed=106, noise_std=0.02
            ),
        ),
    )


def make_mini_flowsheet_dataset(
    *,
    n_rows: int = 720,
    dt: float = 60.0,
    seed: int = 107,
) -> SyntheticDataset:
    if n_rows <= 0:
        raise ValueError("n_rows must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    rng = np.random.default_rng(seed)
    time = _regular_time(n_rows, dt)

    row_idx = np.arange(n_rows, dtype=np.float64)
    transition_row = int(round(0.45 * n_rows))
    ore_type = np.where(row_idx >= transition_row, "A", "B")
    ore_a_flag = np.where(ore_type == "A", 1.0, 0.0)

    feed_mass = (
        1000.0
        + 40.0 * np.sin(np.linspace(0.0, 4.0 * np.pi, n_rows, dtype=np.float64))
        + 18.0 * np.cos(np.linspace(0.0, 1.6 * np.pi, n_rows, dtype=np.float64))
    )
    feed_mass = np.maximum(feed_mass, 700.0)

    feed_grade_trend = 0.96 + 0.10 * np.sin(np.linspace(0.0, 2.4 * np.pi, n_rows, dtype=np.float64))
    feed_grade_step = np.zeros(n_rows, dtype=np.float64)
    feed_grade_step[int(0.20 * n_rows) :] += 0.045
    feed_grade_step[int(0.58 * n_rows) :] -= 0.030
    feed_grade_step[int(0.78 * n_rows) :] += 0.018
    deterministic_noise = rng.normal(0.0, 0.006, size=n_rows)
    feed_copper_grade = np.clip(
        feed_grade_trend + feed_grade_step + deterministic_noise, 0.70, 1.30
    )

    crusher_lags = [1, 2, 3]
    crusher_weights = [0.20, 0.60, 0.20]
    ball_mill_lags, ball_mill_weights = _recycle_product_kernel(
        pass_lags=[3, 4, 5],
        pass_weights=[0.25, 0.50, 0.25],
        product_split=0.40,
        recycle_split=0.60,
        max_passes=16,
    )
    cyclone_lags = [0, 1]
    cyclone_weights = [0.30, 0.70]
    flotation_cell_lags = list(range(0, 17))
    flotation_cell_weights = _normalise([(0.65**lag) for lag in flotation_cell_lags])
    cleaner_lags = list(range(0, 13))
    cleaner_weights = _normalise([(0.50**lag) for lag in cleaner_lags])

    flotation_bank_lags, flotation_bank_weights = _convolve_discrete(
        flotation_cell_lags, flotation_cell_weights, flotation_cell_lags, flotation_cell_weights
    )
    flotation_bank_lags, flotation_bank_weights = _convolve_discrete(
        flotation_bank_lags, flotation_bank_weights, flotation_cell_lags, flotation_cell_weights
    )

    feed_to_cleaner_lags, feed_to_cleaner_weights = _convolve_discrete(
        crusher_lags, _normalise(crusher_weights), ball_mill_lags, ball_mill_weights
    )
    feed_to_cleaner_lags, feed_to_cleaner_weights = _convolve_discrete(
        feed_to_cleaner_lags, feed_to_cleaner_weights, cyclone_lags, _normalise(cyclone_weights)
    )
    feed_to_cleaner_lags, feed_to_cleaner_weights = _convolve_discrete(
        feed_to_cleaner_lags, feed_to_cleaner_weights, flotation_bank_lags, flotation_bank_weights
    )
    feed_to_cleaner_lags, feed_to_cleaner_weights = _convolve_discrete(
        feed_to_cleaner_lags, feed_to_cleaner_weights, cleaner_lags, cleaner_weights
    )

    crusher_output_mass = _apply_kernel(feed_mass, crusher_lags, _normalise(crusher_weights))
    ball_mill_product_mass = _apply_kernel(crusher_output_mass, ball_mill_lags, ball_mill_weights)
    cyclone_delayed_mass = _apply_kernel(
        ball_mill_product_mass,
        cyclone_lags,
        _normalise(cyclone_weights),
    )
    cyclone_overflow_mass = 0.70 * cyclone_delayed_mass
    cyclone_underflow_recycle_mass = 0.30 * cyclone_delayed_mass
    flotation_bank_1_mass = _apply_kernel(
        cyclone_overflow_mass,
        flotation_cell_lags,
        flotation_cell_weights,
    )
    flotation_bank_2_mass = _apply_kernel(
        flotation_bank_1_mass,
        flotation_cell_lags,
        flotation_cell_weights,
    )
    flotation_bank_3_mass = _apply_kernel(
        flotation_bank_2_mass,
        flotation_cell_lags,
        flotation_cell_weights,
    )
    cleaner_product_mass = _apply_kernel(flotation_bank_3_mass, cleaner_lags, cleaner_weights)

    delayed_feed_grade = _apply_kernel(
        feed_copper_grade,
        feed_to_cleaner_lags,
        feed_to_cleaner_weights,
    )
    delayed_ore_a = _apply_kernel(ore_a_flag, feed_to_cleaner_lags, feed_to_cleaner_weights)
    ore_recovery_factor = 1.0 + (0.05 * delayed_ore_a)
    cleaner_copper_grade_true = delayed_feed_grade * ore_recovery_factor
    cleaner_copper_grade = cleaner_copper_grade_true + rng.normal(0.0, 0.004, size=n_rows)
    cleaner_recovered_copper_mass = cleaner_product_mass * cleaner_copper_grade_true

    data = pl.DataFrame(
        {
            "time": time,
            "feed_mass": feed_mass.astype(np.float64),
            "feed_copper_grade": feed_copper_grade.astype(np.float64),
            "ore_type": ore_type.tolist(),
            "crusher_output_mass": crusher_output_mass.astype(np.float64),
            "ball_mill_product_mass": ball_mill_product_mass.astype(np.float64),
            "cyclone_overflow_mass": cyclone_overflow_mass.astype(np.float64),
            "cyclone_underflow_recycle_mass": cyclone_underflow_recycle_mass.astype(np.float64),
            "flotation_bank_1_mass": flotation_bank_1_mass.astype(np.float64),
            "flotation_bank_2_mass": flotation_bank_2_mass.astype(np.float64),
            "flotation_bank_3_mass": flotation_bank_3_mass.astype(np.float64),
            "cleaner_product_mass": cleaner_product_mass.astype(np.float64),
            "cleaner_copper_grade": cleaner_copper_grade.astype(np.float64),
            "cleaner_recovered_copper_mass": cleaner_recovered_copper_mass.astype(np.float64),
        }
    )

    true_kernels = {
        "feed_copper_grade->cleaner_copper_grade": _kernel_metadata(
            feed_to_cleaner_lags, feed_to_cleaner_weights, dt=dt
        ),
        "synthetic_reference_only_feed_to_cleaner": _kernel_metadata(
            feed_to_cleaner_lags, feed_to_cleaner_weights, dt=dt
        ),
    }
    return SyntheticDataset(
        data=data,
        true_kernels=true_kernels,
        scenario={
            "name": "mini_flowsheet",
            "seed": seed,
            "n_rows": n_rows,
            "dt": dt,
            "params": {
                "ore_transition_row": transition_row,
                "ore_transition_time": time[transition_row],
                "ore_recovery_factor_a_vs_b": 1.05,
                "cleaner_grade_noise_std": 0.004,
            },
        },
    )
