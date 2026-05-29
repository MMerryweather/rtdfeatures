#!/usr/bin/env python3
"""Categorical genealogy: generate synthetic data with a switching feed source,
fit a kernel, and build categorical fraction + entropy features."""

from __future__ import annotations

import os
from datetime import datetime, timedelta

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import polars as pl

from rtdfeatures import KernelFeatureBuilder, SimplexKernelLearner
from rtdfeatures.features import feature_evidence_table
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


def make_genealogy_dataset() -> SyntheticDataset:
    n_rows = 400
    dt = 60.0
    seed = 42
    rng = np.random.default_rng(seed)
    time = _regular_time(n_rows, dt)

    feed_pattern = (
        ["Pit A"] * 60 + ["Pit B"] * 60 + ["Pit A"] * 40
        + ["Pit C"] * 70 + ["Pit A"] * 50 + ["Pit B"] * 50 + ["Pit C"] * 70
    )
    feed_source = (feed_pattern * 3)[:n_rows]

    x_base = rng.normal(0.0, 1.0, size=n_rows)
    x = 0.6 * x_base + 0.4 * np.sin(np.linspace(0, 8 * np.pi, n_rows))

    lag_steps = [2, 3, 4, 5]
    weights = _normalize_weights([0.15, 0.35, 0.35, 0.15])
    y = _apply_kernel(x, lag_steps, weights) + rng.normal(0.0, 0.03, size=n_rows)

    data = pl.DataFrame({
        "time": time,
        "feed_source": feed_source,
        "concentration": x.astype(np.float64),
        "product_quality": y.astype(np.float64),
    })

    return SyntheticDataset(
        data=data,
        true_kernels={"concentration->product_quality": {
            "lag_steps": lag_steps,
            "weights": weights,
            "dt": dt,
            "min_lag": lag_steps[0],
            "max_lag": lag_steps[-1],
            "mean_lag": float(sum(s * w for s, w in zip(lag_steps, weights)) * dt),
            "p50_lag": float(3 * dt),
            "p90_lag": float(5 * dt),
        }},
        scenario={
            "name": "categorical_genealogy", "seed": seed,
            "n_rows": n_rows, "dt": dt, "params": {},
        },
    )


def main() -> None:
    ds = make_genealogy_dataset()
    df = ds.data
    print(f"Data: {df.shape[0]} rows, {len(df.columns)} columns")
    print(f"Feed source levels: {sorted(df['feed_source'].unique().to_list())}")

    learner = SimplexKernelLearner(min_lag=1, max_lag=7, seed=42, max_epochs=400)
    fit_result = learner.fit(
        df, input_col="concentration", target_col="product_quality", time_col="time"
    )
    kernel = fit_result.kernel
    print(f"\nLearned kernel weights: {[round(w, 3) for w in kernel.weights]}")

    builder = KernelFeatureBuilder(
        kernels={"genealogy": kernel},
        time_col="time",
        numeric_cols=["concentration"],
        category_cols=["feed_source"],
    )
    result = builder.transform_result(df)
    features = result.features
    report = result.report
    print(f"\nGenerated features ({len(features.columns) - 1}):")
    for col in features.columns:
        if col == "time":
            continue
        vals = features[col].drop_nulls()
        print(f"  {col}: {len(vals)} non-null, mean={vals.mean():.4f}")

    print(f"\nFeature table head:\n{features.head(8)}")

    evidence = builder.diagnose_feature_evidence(
        feature_registry=result.feature_registry,
        fit_result_by_kernel={"genealogy": fit_result},
    )
    evidence_table = feature_evidence_table(evidence)
    print(f"\nFeature evidence:\n{evidence_table}")

    if report:
        print(f"\nTransform report: {len(report.feature_names)} features, "
              f"warmup={report.warmup_rows} rows")

    print("\nDone — 03_categorical_genealogy.py completed.")


if __name__ == "__main__":
    main()
