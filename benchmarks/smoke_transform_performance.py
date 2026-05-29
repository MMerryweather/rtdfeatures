#!/usr/bin/env python3
"""
Smoke benchmark for transform performance.

Not a rigorous benchmark — just documents order-of-growth.
Generates synthetic datasets at varying sizes, fits a simplex kernel,
builds features with several numeric columns, and times transform().

Run from the repository root:

    python benchmarks/smoke_transform_performance.py
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import polars as pl

from rtdfeatures import KernelFeatureBuilder, SimplexKernelLearner


def _regular_time(n_rows: int, dt: float) -> list[datetime]:
    start = datetime(2020, 1, 1)
    return [start + timedelta(seconds=(i * dt)) for i in range(n_rows)]


def _synthetic_dataset(
    n_rows: int, dt: float, seed: int, n_numeric_cols: int,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    cols: dict[str, pl.Series | list[datetime]] = {
        "time": _regular_time(n_rows, dt),
    }
    for i in range(n_numeric_cols):
        signal = rng.normal(0.0, 1.0, size=n_rows).astype(np.float64)
        wave = np.sin(np.linspace(0.0, 8.0 * np.pi, n_rows, dtype=np.float64))
        cols[f"input_{i}"] = 0.7 * signal + 0.3 * wave
    target = rng.normal(0.0, 1.0, size=n_rows).astype(np.float64)
    cols["target"] = target + 0.3 * cols["input_0"]
    return pl.DataFrame(cols)


def main() -> None:
    row_counts = [100, 1000, 5000]
    n_numeric_cols = 5
    dt = 60.0

    print("Smoke transform performance benchmark")
    print(f"{'Rows':>8}  {'Transform (s)':>14}  {'Features':>9}  {'Cols':>6}")
    print("-" * 44)

    for n_rows in row_counts:
        df = _synthetic_dataset(
            n_rows=n_rows, dt=dt, seed=42, n_numeric_cols=n_numeric_cols,
        )
        target_col = "target"

        learner = SimplexKernelLearner(min_lag=1, max_lag=10, seed=42, max_epochs=200)
        fit_result = learner.fit(
            df, input_col="input_0", target_col=target_col, time_col="time",
        )
        kernel = fit_result.kernel

        numeric_cols = [f"input_{i}" for i in range(n_numeric_cols)]
        builder = KernelFeatureBuilder(
            kernels={"k": kernel},
            time_col="time",
            numeric_cols=numeric_cols,
        )

        start = time.perf_counter()
        result = builder.transform_result(df)
        elapsed = time.perf_counter() - start

        n_features = len(result.feature_registry)
        n_cols = len(result.features.columns)

        print(f"{n_rows:>8}  {elapsed:>12.4f}s  {n_features:>9}  {n_cols:>6}")

    print()
    print("Note: results are indicative only. Actual performance")
    print("depends on hardware, Polars version, and kernel complexity.")


if __name__ == "__main__":
    main()
