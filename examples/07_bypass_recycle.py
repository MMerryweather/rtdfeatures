#!/usr/bin/env python3
"""Bypass + recycle: two-mode kernel (short bypass + spread recycle) learned from data.

Shows how the learner resolves a bimodal kernel with a fast bypass path and a
slow recirculation path — common in chemical reactors, re-circulating flows,
and systems with parallel fast/slow pathways.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from rtdfeatures import (
    DelayedExponentialKernel,
    GammaKernel,
    KernelFeatureBuilder,
    SimplexKernelLearner,
)
from rtdfeatures.features import feature_evidence_table
from rtdfeatures.reporting import fit_result_warning_summary_table


def main() -> None:
    n_rows = 600
    dt = 1.0
    rng = np.random.default_rng(42)

    # Ground-truth kernel:
    #   bypass:  30 % mass at lag 2  (fast, concentrated)
    #   recycle: 70 % spread across lags 5-9  (slow, diffuse)
    true_weights = [0.00, 0.00, 0.30, 0.00, 0.00, 0.25, 0.20, 0.12, 0.08, 0.05]
    true_lag_steps = list(range(10))

    time = [
        datetime(2020, 1, 1) + timedelta(seconds=i * dt)
        for i in range(n_rows)
    ]
    x_base = rng.normal(0.0, 1.0, n_rows)
    trend = np.sin(np.linspace(0.0, 6.0 * np.pi, n_rows))
    x = (0.65 * x_base) + (0.35 * trend)

    y = np.zeros(n_rows)
    for t in range(n_rows):
        total = 0.0
        for step, w in zip(true_lag_steps, true_weights):
            if t - step >= 0:
                total += w * x[t - step]
        y[t] = total
    y += rng.normal(0.0, 0.02, n_rows)

    df = pl.DataFrame(
        {"time": time, "input_signal": x, "target_signal": y}
    )
    print(f"Data: {df.shape[0]} rows\n")

    # Learn the kernel
    learner = SimplexKernelLearner(
        min_lag=0, max_lag=10, seed=42, max_epochs=600,
    )
    fit_result = learner.fit(
        df,
        input_col="input_signal",
        target_col="target_signal",
        time_col="time",
    )
    kernel = fit_result.kernel

    print("True kernel (bypass + recycle):")
    print(
        f"  weights: {[round(w, 4) for w in true_weights]}"
    )
    print(f"  lag_steps: {true_lag_steps}\n")

    print("Learned kernel:")
    print(
        f"  weights: {[round(w, 4) for w in kernel.weights]}"
    )
    print(f"  lag_steps: {list(kernel.lag_steps)}")
    print(f"  summary: {kernel.summary()}\n")

    diag = fit_result.fit_diagnostics
    print("Fit diagnostics:")
    print(f"  validation_loss: {diag.validation_loss:.6f}")
    print(f"  mean_lag: {diag.mean_lag:.2f}s")
    print(f"  p50_lag:  {diag.p50_lag:.2f}s")
    print(f"  p90_lag:  {diag.p90_lag:.2f}s\n")

    ident = fit_result.identifiability_report
    print(f"Identifiability warning codes: {ident.warning_codes}\n")

    # Build features from learned kernel
    builder = KernelFeatureBuilder(
        kernels={"bypass_recycle": kernel},
        time_col="time",
        numeric_cols=["input_signal"],
    )
    features = builder.transform(df)
    print(
        f"Features: {features.shape[0]} rows, "
        f"columns={features.columns}"
    )
    print(features.head(6), "\n")

    # Evidence
    evidence = builder.diagnose_feature_evidence(
        fit_result_by_kernel={"bypass_recycle": fit_result},
    )
    evidence_table = feature_evidence_table(evidence)
    print(f"Feature evidence:\n{evidence_table}\n")

    # Warning summary
    warnings_df = fit_result_warning_summary_table(fit_result)
    print(f"Warning summary:\n{warnings_df}\n")

    print("Interpretation:")
    print(
        "  - The bypass mode (short lag) appears as a sharp peak at"
        " low lags"
    )
    print(
        "  - The recycle mode appears as a broad spread over longer"
        " lags"
    )
    print(
        "  - If the learner cannot resolve both modes, identifiability"
        " warnings may appear"
    )

    # Other parametric families are available as direct constructors,
    # though they cannot capture this bimodal shape well.
    fixed_gamma = GammaKernel(
        shape_alpha=2.0, rate_beta=0.5,
        min_lag_steps=0, max_lag_steps=10, dt=1.0,
        name="gamma_fixed",
    )
    fixed_delayed = DelayedExponentialKernel(
        delay=1.0, rate_lambda=0.8,
        min_lag_steps=0, max_lag_steps=10, dt=1.0,
        name="delayed_exp_fixed",
    )
    print("\nFixed parametric kernels (not learned, for comparison):")
    print(f"  {fixed_gamma.name}: mean_lag={fixed_gamma.mean_lag():.2f}s")
    print(f"  {fixed_delayed.name}: mean_lag={fixed_delayed.mean_lag():.2f}s")

    print("\nDone — 07_bypass_recycle.py completed.")


if __name__ == "__main__":
    main()
