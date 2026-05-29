#!/usr/bin/env python3
"""Weak identifiability case: generate data where multiple kernels explain the data
similarly, and show diagnostics that warn about weak evidence."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


from rtdfeatures import ExponentialKernelLearner, GammaKernelLearner, SimplexKernelLearner
from rtdfeatures.reporting import (
    baseline_comparison_table,
    fit_result_warning_summary_table,
    learner_diagnostic_comparison_table,
)
from rtdfeatures.synthetic import make_weak_identifiability_dataset


def main() -> None:
    ds = make_weak_identifiability_dataset(n_rows=400, dt=60.0, seed=23, noise_std=1.0)
    df = ds.data
    print(f"Data: {df.shape[0]} rows")
    print("NOTE: High noise (noise_std=1.0) makes the kernel hard to identify.\n")

    min_lag, max_lag = 1, 6

    results = {}
    learners = [
        ("Empirical (simplex)", SimplexKernelLearner(
            min_lag=min_lag, max_lag=max_lag, seed=101, max_epochs=400,
        )),
        ("Parametric (gamma)", GammaKernelLearner(
            min_lag=min_lag, max_lag=max_lag, seed=102, max_epochs=400,
        )),
        ("Parametric (exponential)", ExponentialKernelLearner(
            min_lag=min_lag, max_lag=max_lag, seed=103, max_epochs=400,
        )),
    ]
    for name, learner in learners:
        r = learner.fit(df, input_col="input_signal", target_col="target_signal", time_col="time")
        results[name] = r
        k = r.kernel
        d = r.fit_diagnostics
        ident = r.identifiability_report
        print(f"  {name}:")
        print(f"    loss={d.validation_loss:.5f}, mean_lag={d.mean_lag:.1f}s")
        print(f"    weights: {[round(w, 4) for w in k.weights]}")
        print(f"    warnings: {ident.warning_codes}")

    print("\nBaseline comparisons:")
    for name, r in results.items():
        bc = r.baseline_comparison
        if bc:
            tbl = baseline_comparison_table(bc)
            print(f"  {name}:\n{tbl}")

    warnings = fit_result_warning_summary_table(results["Empirical (simplex)"])
    print(f"\nSimplex warning summary:\n{warnings}")

    comparison = learner_diagnostic_comparison_table(
        fit_results_by_family={
            "simplex": results["Empirical (simplex)"],
            "gamma": results["Parametric (gamma)"],
            "exponential": results["Parametric (exponential)"],
        }
    )
    print(f"\nAll-learner comparison:\n{comparison}")

    print("\nCONCLUSION: All fits show similar loss and diffuse weights.")
    print("This is weak identifiability — no kernel can be trusted for feature generation.")

    print("\nDone — 05_weak_identifiability.py completed.")


if __name__ == "__main__":
    main()
