"""Deterministic single-unit scenario fixtures for simulation-harness tests."""

from __future__ import annotations

from typing import cast

import polars as pl

from .contracts import GeneratorOutput, ScenarioMetadata
from .defaults import (
    DEFAULT_COMPLEX_ROWS,
    DEFAULT_DT,
    DEFAULT_FEED_MASS,
    DEFAULT_SEED,
    DEFAULT_SINGLE_UNIT_ROWS,
)
from .genealogy import empty_genealogy
from .kernels import (
    build_lag_grid,
    convolve_discrete_kernels,
    feed_grade_signal,
    make_narrow_spread_kernel,
    make_plug_flow_kernel,
    make_tank_kernel,
    normalize_weights,
    summarize_kernel,
)
from .params import validate_non_negative_mass, validate_positive_timestep


def _coerce_int_kwarg(kwargs: dict[str, object], key: str, default: int) -> int:
    value = kwargs.pop(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be an integer-like number")
    return int(value)


def _coerce_float_kwarg(kwargs: dict[str, object], key: str, default: float) -> float:
    value = kwargs.pop(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be a float-like number")
    return float(value)


def _regular_time_index(n_rows: int, *, dt: float) -> list[float]:
    return [float(i * dt) for i in range(n_rows)]


def _target_from_kernel(
    feed_grade: list[float], lag_steps: list[int], weights: list[float]
) -> list[float]:
    n_rows = len(feed_grade)
    target: list[float] = []
    for t in range(n_rows):
        value = 0.0
        for lag, weight in zip(lag_steps, weights):
            src = t - lag
            if src >= 0:
                value += weight * feed_grade[src]
        target.append(float(value))
    return target


def _genealogy_from_kernel(
    *,
    n_rows: int,
    feed_mass: float,
    lag_steps: list[int],
    weights: list[float],
    unit: str,
    path: str,
) -> pl.DataFrame:
    max_lag = int(max(lag_steps))
    rows: list[dict[str, int | float | bool | str]] = []
    for output_time in range(n_rows):
        for lag, weight in zip(lag_steps, weights):
            source_time = output_time - lag
            if source_time < 0:
                continue
            rows.append(
                {
                    "output_time": output_time,
                    "source_time": source_time,
                    "unit": unit,
                    "path": path,
                    "source_mass": float(feed_mass),
                    "contribution_mass": float(feed_mass * weight),
                    "contribution_fraction": float(weight),
                    "is_warmup": output_time < max_lag,
                }
            )
    if not rows:
        return empty_genealogy()
    return pl.DataFrame(rows).select(
        "output_time",
        "source_time",
        "unit",
        "path",
        "source_mass",
        "contribution_mass",
        "contribution_fraction",
        "is_warmup",
    )


def _single_unit_dataset(
    *,
    scenario_name: str,
    kernel_name: str,
    lag_steps: list[int],
    weights: list[float],
    n_rows: int,
    dt: float,
    seed: int,
    feed_mass: float,
) -> GeneratorOutput:
    time = _regular_time_index(n_rows, dt=dt)
    feed_mass_col = [float(feed_mass)] * n_rows
    feed_grade = [feed_grade_signal(t, n_rows=n_rows) for t in range(n_rows)]
    target_grade = _target_from_kernel(feed_grade, lag_steps, weights)
    data = pl.DataFrame(
        {
            "time": time,
            "feed_mass": feed_mass_col,
            "feed_grade": feed_grade,
            "target_grade": target_grade,
        }
    )
    true_kernel = summarize_kernel(lag_steps, weights, dt=dt)
    genealogy = _genealogy_from_kernel(
        n_rows=n_rows,
        feed_mass=feed_mass,
        lag_steps=lag_steps,
        weights=weights,
        unit=kernel_name,
        path=kernel_name,
    )
    return GeneratorOutput(
        data=data,
        true_kernels={kernel_name: true_kernel},
        genealogy=genealogy,
        scenario={
            "name": scenario_name,
            "seed": seed,
            "n_rows": n_rows,
            "dt": dt,
            "params": {"feed_mass": feed_mass},
        },
    )


def make_tank_dataset(
    *,
    n_rows: int = DEFAULT_SINGLE_UNIT_ROWS,
    dt: float = DEFAULT_DT,
    seed: int = DEFAULT_SEED,
    feed_mass: float = DEFAULT_FEED_MASS,
) -> GeneratorOutput:
    dt = validate_positive_timestep(dt)
    feed_mass = validate_non_negative_mass(feed_mass)
    kernel = make_tank_kernel(rho=0.80, max_lag=32, dt=dt)
    return _single_unit_dataset(
        scenario_name="tank",
        kernel_name="tank",
        lag_steps=kernel["lag_steps"],
        weights=kernel["weights"],
        n_rows=n_rows,
        dt=dt,
        seed=seed,
        feed_mass=feed_mass,
    )


def make_plug_flow_dataset(
    *,
    n_rows: int = DEFAULT_SINGLE_UNIT_ROWS,
    dt: float = DEFAULT_DT,
    seed: int = DEFAULT_SEED,
    feed_mass: float = DEFAULT_FEED_MASS,
) -> GeneratorOutput:
    dt = validate_positive_timestep(dt)
    feed_mass = validate_non_negative_mass(feed_mass)
    kernel = make_plug_flow_kernel(lag=6, dt=dt)
    return _single_unit_dataset(
        scenario_name="plug_flow",
        kernel_name="plug_flow",
        lag_steps=kernel["lag_steps"],
        weights=kernel["weights"],
        n_rows=n_rows,
        dt=dt,
        seed=seed,
        feed_mass=feed_mass,
    )


def make_plug_flow_spread_dataset(
    *,
    n_rows: int = DEFAULT_SINGLE_UNIT_ROWS,
    dt: float = DEFAULT_DT,
    seed: int = DEFAULT_SEED,
    feed_mass: float = DEFAULT_FEED_MASS,
) -> GeneratorOutput:
    dt = validate_positive_timestep(dt)
    feed_mass = validate_non_negative_mass(feed_mass)
    kernel = make_narrow_spread_kernel(dt=dt)
    return _single_unit_dataset(
        scenario_name="plug_flow_spread",
        kernel_name="plug_flow_spread",
        lag_steps=kernel["lag_steps"],
        weights=kernel["weights"],
        n_rows=n_rows,
        dt=dt,
        seed=seed,
        feed_mass=feed_mass,
    )


def make_flotation_bank_dataset(
    *,
    n_rows: int = DEFAULT_SINGLE_UNIT_ROWS,
    dt: float = DEFAULT_DT,
    seed: int = DEFAULT_SEED,
    feed_mass: float = DEFAULT_FEED_MASS,
    n_cells: int = 3,
) -> GeneratorOutput:
    dt = validate_positive_timestep(dt)
    feed_mass = validate_non_negative_mass(feed_mass)
    if n_cells <= 0:
        raise ValueError("n_cells must be positive")

    cell = make_tank_kernel(rho=0.65, max_lag=16, dt=dt)
    bank_lags = cell["lag_steps"]
    bank_weights = cell["weights"]
    for _ in range(1, n_cells):
        bank_lags, bank_weights = convolve_discrete_kernels(
            bank_lags, bank_weights, cell["lag_steps"], cell["weights"]
        )

    out = _single_unit_dataset(
        scenario_name="flotation_bank",
        kernel_name="flotation_bank",
        lag_steps=bank_lags,
        weights=bank_weights,
        n_rows=n_rows,
        dt=dt,
        seed=seed,
        feed_mass=feed_mass,
    )
    scenario = cast(ScenarioMetadata, dict(out.scenario))
    scenario["params"] = {
        "feed_mass": feed_mass,
        "n_cells": n_cells,
        "cell_rho": 0.65,
        "cell_max_lag": 16,
    }
    return GeneratorOutput(
        data=out.data, true_kernels=out.true_kernels, genealogy=out.genealogy, scenario=scenario
    )


def make_closed_loop_crushing_dataset(*args: object, **kwargs: object) -> GeneratorOutput:
    if args:
        raise TypeError("make_closed_loop_crushing_dataset only accepts keyword arguments")

    kw: dict[str, object] = dict(kwargs)
    n_rows = _coerce_int_kwarg(kw, "n_rows", DEFAULT_COMPLEX_ROWS)
    dt = _coerce_float_kwarg(kw, "dt", DEFAULT_DT)
    seed = _coerce_int_kwarg(kw, "seed", DEFAULT_SEED)
    feed_mass = _coerce_float_kwarg(kw, "feed_mass", DEFAULT_FEED_MASS)
    if kw:
        unknown = ", ".join(sorted(kw))
        raise TypeError(f"unexpected keyword arguments: {unknown}")

    dt = validate_positive_timestep(dt)
    feed_mass = validate_non_negative_mass(feed_mass)

    crusher_pass_lags = [1, 2, 3]
    crusher_pass_weights = [0.2, 0.6, 0.2]
    product_split = 0.35
    recycle_split = 0.65
    max_passes = 12
    max_lag = 64

    product_accum: dict[int, float] = {}
    discharge_accum: dict[int, float] = {}
    pass_lags = crusher_pass_lags
    pass_weights = crusher_pass_weights
    for pass_index in range(max_passes):
        discharge_weight = recycle_split**pass_index
        product_weight = product_split * discharge_weight
        for lag, weight in zip(pass_lags, pass_weights):
            discharge_accum[lag] = discharge_accum.get(lag, 0.0) + (discharge_weight * weight)
            product_accum[lag] = product_accum.get(lag, 0.0) + (product_weight * weight)
        if pass_index < (max_passes - 1):
            pass_lags, pass_weights = convolve_discrete_kernels(
                pass_lags,
                pass_weights,
                crusher_pass_lags,
                crusher_pass_weights,
            )

    support = build_lag_grid(0, max_lag)
    product_raw = [float(product_accum.get(lag, 0.0)) for lag in support]
    discharge_weights = [float(discharge_accum.get(lag, 0.0)) for lag in support]
    product_weights = normalize_weights(product_raw)

    def _series_from_kernel(
        signal: list[float], lag_steps: list[int], weights: list[float]
    ) -> list[float]:
        series: list[float] = []
        for t in range(len(signal)):
            value = 0.0
            for lag, weight in zip(lag_steps, weights):
                source_idx = t - lag
                if source_idx >= 0:
                    value += weight * signal[source_idx]
            series.append(float(value))
        return series

    time = _regular_time_index(n_rows, dt=dt)
    feed_mass_col = [float(feed_mass)] * n_rows
    feed_grade = [feed_grade_signal(t, n_rows=n_rows) for t in range(n_rows)]
    target_grade = _target_from_kernel(feed_grade, support, product_weights)
    crusher_screen_discharge_mass = _series_from_kernel(feed_mass_col, support, discharge_weights)
    product_mass = [float(product_split * mass) for mass in crusher_screen_discharge_mass]
    recycle_mass = [float(recycle_split * mass) for mass in crusher_screen_discharge_mass]

    data = pl.DataFrame(
        {
            "time": time,
            "feed_mass": feed_mass_col,
            "feed_grade": feed_grade,
            "target_grade": target_grade,
            "crusher_screen_discharge_mass": crusher_screen_discharge_mass,
            "product_mass": product_mass,
            "recycle_mass": recycle_mass,
        }
    )

    product_kernel = summarize_kernel(support, product_weights, dt=dt)
    crusher_pass_kernel = summarize_kernel(crusher_pass_lags, crusher_pass_weights, dt=dt)
    genealogy = _genealogy_from_kernel(
        n_rows=n_rows,
        feed_mass=feed_mass,
        lag_steps=support,
        weights=product_weights,
        unit="closed_loop_crushing",
        path="closed_loop_crushing/product",
    )
    return GeneratorOutput(
        data=data,
        true_kernels={
            "closed_loop_crushing_product_effective": product_kernel,
            "closed_loop_crushing_crusher_pass": crusher_pass_kernel,
        },
        genealogy=genealogy,
        scenario={
            "name": "closed_loop_crushing",
            "seed": seed,
            "n_rows": n_rows,
            "dt": dt,
            "params": {
                "feed_mass": feed_mass,
                "crusher_pass_lags": crusher_pass_lags,
                "crusher_pass_weights": crusher_pass_weights,
                "product_split": product_split,
                "recycle_split": recycle_split,
                "max_passes": max_passes,
                "max_lag": max_lag,
            },
        },
    )


def make_toy_full_plant_dataset(*args: object, **kwargs: object) -> GeneratorOutput:
    if args:
        raise TypeError("make_toy_full_plant_dataset only accepts keyword arguments")

    kw: dict[str, object] = dict(kwargs)
    n_rows = _coerce_int_kwarg(kw, "n_rows", DEFAULT_COMPLEX_ROWS)
    dt = _coerce_float_kwarg(kw, "dt", DEFAULT_DT)
    seed = _coerce_int_kwarg(kw, "seed", DEFAULT_SEED)
    feed_mass = _coerce_float_kwarg(kw, "feed_mass", DEFAULT_FEED_MASS)
    if kw:
        unknown = ", ".join(sorted(kw))
        raise TypeError(f"unexpected keyword arguments: {unknown}")

    dt = validate_positive_timestep(dt)
    feed_mass = validate_non_negative_mass(feed_mass)

    def _series_from_kernel(
        signal: list[float], lag_steps: list[int], weights: list[float]
    ) -> list[float]:
        series: list[float] = []
        for t in range(len(signal)):
            value = 0.0
            for lag, weight in zip(lag_steps, weights):
                source_idx = t - lag
                if source_idx >= 0:
                    value += weight * signal[source_idx]
            series.append(float(value))
        return series

    def _build_recycle_product_kernel(
        *,
        pass_lags: list[int],
        pass_weights: list[float],
        product_split: float,
        recycle_split: float,
        max_passes: int,
        max_lag: int,
    ) -> tuple[list[int], list[float]]:
        pass_support = pass_lags
        pass_support_weights = pass_weights
        product_accum: dict[int, float] = {}
        for pass_index in range(max_passes):
            pass_weight = recycle_split**pass_index
            product_weight = product_split * pass_weight
            for lag, weight in zip(pass_support, pass_support_weights):
                product_accum[lag] = product_accum.get(lag, 0.0) + (product_weight * weight)
            if pass_index < (max_passes - 1):
                pass_support, pass_support_weights = convolve_discrete_kernels(
                    pass_support,
                    pass_support_weights,
                    pass_lags,
                    pass_weights,
                )

        support = build_lag_grid(0, max_lag)
        kernel_raw = [float(product_accum.get(lag, 0.0)) for lag in support]
        return support, normalize_weights(kernel_raw)

    crusher_lags = [1, 2, 3]
    crusher_weights = [0.2, 0.6, 0.2]

    ball_mill_pass_lags = [3, 4, 5]
    ball_mill_pass_weights = [0.25, 0.50, 0.25]
    ball_mill_product_split = 0.40
    ball_mill_recycle_split = 0.60
    ball_mill_max_passes = 16
    ball_mill_max_lag = 96
    ball_mill_effective_lags, ball_mill_effective_weights = _build_recycle_product_kernel(
        pass_lags=ball_mill_pass_lags,
        pass_weights=ball_mill_pass_weights,
        product_split=ball_mill_product_split,
        recycle_split=ball_mill_recycle_split,
        max_passes=ball_mill_max_passes,
        max_lag=ball_mill_max_lag,
    )

    cyclone_delay_lags = [0, 1]
    cyclone_delay_weights = [0.3, 0.7]
    cyclone_overflow_split = 0.70
    cyclone_underflow_recycle_split = 0.30

    flotation_cell = make_tank_kernel(rho=0.65, max_lag=16, dt=dt)
    flotation_bank_lags = flotation_cell["lag_steps"]
    flotation_bank_weights = flotation_cell["weights"]
    for _ in range(2):
        flotation_bank_lags, flotation_bank_weights = convolve_discrete_kernels(
            flotation_bank_lags,
            flotation_bank_weights,
            flotation_cell["lag_steps"],
            flotation_cell["weights"],
        )

    cleaner_kernel = make_tank_kernel(rho=0.50, max_lag=12, dt=dt)
    cleaner_lags = cleaner_kernel["lag_steps"]
    cleaner_weights = cleaner_kernel["weights"]

    stage_kernels_for_path: list[tuple[list[int], list[float]]] = [
        (crusher_lags, crusher_weights),
        (ball_mill_effective_lags, ball_mill_effective_weights),
        (cyclone_delay_lags, cyclone_delay_weights),
        (flotation_bank_lags, flotation_bank_weights),
        (flotation_bank_lags, flotation_bank_weights),
        (flotation_bank_lags, flotation_bank_weights),
        (cleaner_lags, cleaner_weights),
    ]
    final_lags = crusher_lags
    final_weights = normalize_weights(crusher_weights)
    for stage_lags, stage_weights in stage_kernels_for_path[1:]:
        final_lags, final_weights = convolve_discrete_kernels(
            final_lags, final_weights, stage_lags, stage_weights
        )
    final_weights = normalize_weights(final_weights)

    time = _regular_time_index(n_rows, dt=dt)
    feed_mass_col = [float(feed_mass)] * n_rows
    feed_grade = [feed_grade_signal(t, n_rows=n_rows) for t in range(n_rows)]
    target_grade = _target_from_kernel(feed_grade, final_lags, final_weights)

    crusher_output_mass = _series_from_kernel(
        feed_mass_col, crusher_lags, normalize_weights(crusher_weights)
    )
    ball_mill_product_mass = _series_from_kernel(
        crusher_output_mass, ball_mill_effective_lags, ball_mill_effective_weights
    )
    cyclone_delayed_mass = _series_from_kernel(
        ball_mill_product_mass, cyclone_delay_lags, normalize_weights(cyclone_delay_weights)
    )
    cyclone_overflow_mass = [float(cyclone_overflow_split * mass) for mass in cyclone_delayed_mass]
    cyclone_underflow_recycle_mass = [
        float(cyclone_underflow_recycle_split * mass) for mass in cyclone_delayed_mass
    ]
    flotation_bank_1_mass = _series_from_kernel(
        cyclone_overflow_mass, flotation_bank_lags, flotation_bank_weights
    )
    flotation_bank_2_mass = _series_from_kernel(
        flotation_bank_1_mass, flotation_bank_lags, flotation_bank_weights
    )
    flotation_bank_3_mass = _series_from_kernel(
        flotation_bank_2_mass, flotation_bank_lags, flotation_bank_weights
    )
    cleaner_product_mass = _series_from_kernel(flotation_bank_3_mass, cleaner_lags, cleaner_weights)

    data = pl.DataFrame(
        {
            "time": time,
            "feed_mass": feed_mass_col,
            "feed_grade": feed_grade,
            "target_grade": target_grade,
            "crusher_output_mass": crusher_output_mass,
            "ball_mill_product_mass": ball_mill_product_mass,
            "cyclone_overflow_mass": cyclone_overflow_mass,
            "cyclone_underflow_recycle_mass": cyclone_underflow_recycle_mass,
            "flotation_bank_1_mass": flotation_bank_1_mass,
            "flotation_bank_2_mass": flotation_bank_2_mass,
            "flotation_bank_3_mass": flotation_bank_3_mass,
            "cleaner_product_mass": cleaner_product_mass,
        }
    )

    final_effective_kernel = summarize_kernel(final_lags, final_weights, dt=dt)
    genealogy = _genealogy_from_kernel(
        n_rows=n_rows,
        feed_mass=feed_mass,
        lag_steps=final_lags,
        weights=final_weights,
        unit="toy_full_plant",
        path="feed_to_cleaner_product",
    )
    return GeneratorOutput(
        data=data,
        true_kernels={
            "toy_full_plant_open_loop_crusher": summarize_kernel(
                crusher_lags, crusher_weights, dt=dt
            ),
            "toy_full_plant_closed_loop_ball_mill_pass": summarize_kernel(
                ball_mill_pass_lags, ball_mill_pass_weights, dt=dt
            ),
            "toy_full_plant_closed_loop_ball_mill_product_effective": summarize_kernel(
                ball_mill_effective_lags, ball_mill_effective_weights, dt=dt
            ),
            "toy_full_plant_cyclone_delay": summarize_kernel(
                cyclone_delay_lags, cyclone_delay_weights, dt=dt
            ),
            "toy_full_plant_flotation_bank_1": summarize_kernel(
                flotation_bank_lags, flotation_bank_weights, dt=dt
            ),
            "toy_full_plant_flotation_bank_2": summarize_kernel(
                flotation_bank_lags, flotation_bank_weights, dt=dt
            ),
            "toy_full_plant_flotation_bank_3": summarize_kernel(
                flotation_bank_lags, flotation_bank_weights, dt=dt
            ),
            "toy_full_plant_cleaner": summarize_kernel(cleaner_lags, cleaner_weights, dt=dt),
            "toy_full_plant_final_effective": final_effective_kernel,
        },
        genealogy=genealogy,
        scenario={
            "name": "toy_full_plant",
            "seed": seed,
            "n_rows": n_rows,
            "dt": dt,
            "params": {
                "feed_mass": feed_mass,
                "open_loop_crusher": {"lags": crusher_lags, "weights": crusher_weights},
                "closed_loop_ball_mill": {
                    "pass_lags": ball_mill_pass_lags,
                    "pass_weights": ball_mill_pass_weights,
                    "product_split": ball_mill_product_split,
                    "recycle_split": ball_mill_recycle_split,
                    "max_passes": ball_mill_max_passes,
                    "max_lag": ball_mill_max_lag,
                },
                "cyclone": {
                    "overflow_split": cyclone_overflow_split,
                    "underflow_recycle_split": cyclone_underflow_recycle_split,
                    "delay_lags": cyclone_delay_lags,
                    "delay_weights": cyclone_delay_weights,
                },
                "flotation_banks": {
                    "bank_count": 3,
                    "cells_per_bank": 3,
                    "cell_rho": 0.65,
                    "cell_max_lag": 16,
                },
                "cleaner": {"rho": 0.50, "max_lag": 12},
            },
        },
    )
