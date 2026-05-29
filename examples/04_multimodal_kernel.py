#!/usr/bin/env python3
"""Multimodal empirical kernel: synthetic data with two delay modes (fast + recycle)
and a SimplexKernelLearner fit that shows the bimodal pattern."""

from __future__ import annotations

import os
from datetime import datetime, timedelta

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import polars as pl

from rtdfeatures import GammaKernelLearner, SimplexKernelLearner
from rtdfeatures.reporting import (
    fit_result_warning_summary_table,
    learner_diagnostic_comparison_table,
)
from rtdfeatures.synthetic import SyntheticDataset, _normalize_weights


def _regular_time(n_rows: int, dt: float) -> list[datetime]:
    start = datetime(2020, 1, 1)
    return [start + timedelta(seconds=(i * dt)) for i in range(n_rows)]


def _apply_kernel(signal: np.ndarray, lag_steps: list[int], weights: list[float]) -> np.ndarray:
    out = np.zeros(signal.shape[0], dtype=np.float64)
    for t in range(signal.shape[0]):
        v = 0.0
        for lag, w in zip(lag_steps, weights):
            src = t - lag
            if src >= 0:
                v += w * float(signal[src])
        out[t] = v
    return out


def make_multimodal_dataset() -> SyntheticDataset:
    n_rows = 500
    dt = 60.0
    seed = 81
    rng = np.random.default_rng(seed)
    time = _regular_time(n_rows, dt)

    x_base = rng.normal(0.0, 1.0, size=n_rows)
    x = 0.6 * x_base + 0.4 * np.sin(np.linspace(0, 8 * np.pi, n_rows))

    lag_steps = [2, 3, 9, 10, 11]
    weights = _normalize_weights([0.24, 0.26, 0.16, 0.24, 0.10])
    y = _apply_kernel(x, lag_steps, weights) + rng.normal(0.0, 0.025, size=n_rows)

    data = pl.DataFrame({
        "time": time,
        "feed": x.astype(np.float64),
        "product": y.astype(np.float64),
    })
    return SyntheticDataset(
        data=data,
        true_kernels={"feed->product": {
            "lag_steps": lag_steps, "weights": weights, "dt": dt,
            "min_lag": lag_steps[0], "max_lag": lag_steps[-1],
            "mean_lag": float(sum(s * w for s, w in zip(lag_steps, weights)) * dt),
            "p50_lag": float(3 * dt), "p90_lag": float(11 * dt),
        }},
        scenario={"name": "multimodal", "seed": seed, "n_rows": n_rows, "dt": dt, "params": {}},
    )


def main() -> None:
    ds = make_multimodal_dataset()
    df = ds.data
    true_meta = ds.true_kernels["feed->product"]
    print(
        f"True kernel: lag_steps={true_meta['lag_steps']}, "
        f"weights={[round(w, 3) for w in true_meta['weights']]}"
    )
    print("  This is a bimodal distribution — fast path (lags 2-3) and slow/recycle (lags 9-11)")
    print(f"Data: {df.shape[0]} rows\n")

    min_lag, max_lag = 1, 13

    simplex_learner = SimplexKernelLearner(
        min_lag=min_lag, max_lag=max_lag, seed=101, max_epochs=400
    )
    simplex_result = simplex_learner.fit(
        df, input_col="feed", target_col="product", time_col="time",
    )
    gamma_learner = GammaKernelLearner(
        min_lag=min_lag, max_lag=max_lag, seed=102, max_epochs=400
    )
    gamma_result = gamma_learner.fit(
        df, input_col="feed", target_col="product", time_col="time",
    )

    print("Fitted kernels:")
    for name, r in [("Empirical (simplex)", simplex_result), ("Parametric (gamma)", gamma_result)]:
        k = r.kernel
        print(f"  {name}:")
        print(f"    weights: {[round(w, 4) for w in k.weights]}")
        print(f"    loss: {r.fit_diagnostics.validation_loss:.5f}")

    print("\nIMPORTANT: The simplex kernel should show weight at BOTH lag groups (2-3 and 9-11).")
    print("The gamma kernel (unimodal by construction) cannot capture the second mode.")
    print("If your process has bypass or recycle paths, use an empirical learner.")

    warnings = fit_result_warning_summary_table(simplex_result)
    print(f"\nSimplex warning summary:\n{warnings}")

    comparison = learner_diagnostic_comparison_table(
        fit_results_by_family={
            "simplex": simplex_result,
            "gamma": gamma_result,
        }
    )
    print(f"\nLearner comparison:\n{comparison}")

    print("\nDone — 04_multimodal_kernel.py completed.")


if __name__ == "__main__":
    main()
