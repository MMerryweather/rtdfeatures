#!/usr/bin/env python3
"""Minimal end-to-end: generate synthetic data, fit a SimplexKernelLearner,
build features, show diagnostics."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


from rtdfeatures import KernelFeatureBuilder, SimplexKernelLearner
from rtdfeatures.features import feature_evidence_table
from rtdfeatures.reporting import baseline_comparison_table, fit_result_warning_summary_table
from rtdfeatures.synthetic import make_single_delay_dataset


def main() -> None:
    ds = make_single_delay_dataset(n_rows=300, dt=60.0, seed=7, delay_steps=6, noise_std=0.02)
    df = ds.data
    print(f"Data: {df.shape[0]} rows, columns={df.columns}")

    learner = SimplexKernelLearner(min_lag=1, max_lag=10, seed=42, max_epochs=400)
    fit_result = learner.fit(
        df, input_col="input_signal", target_col="target_signal", time_col="time"
    )
    kernel = fit_result.kernel
    print(f"\nLearned kernel: {kernel.summary()}")
    print(f"  weights ({len(kernel.weights)}): {[round(w, 4) for w in kernel.weights]}")
    print(f"  lag_steps: {list(kernel.lag_steps)}")

    diag = fit_result.fit_diagnostics
    print("\nFit diagnostics:")
    print(f"  validation_loss: {diag.validation_loss:.6f}")
    print(f"  mean_lag: {diag.mean_lag:.2f}s")
    print(f"  p50_lag: {diag.p50_lag:.2f}s")
    print(f"  p90_lag: {diag.p90_lag:.2f}s")

    ident = fit_result.identifiability_report
    print(f"  warning_codes: {ident.warning_codes}")

    baseline = fit_result.baseline_comparison
    if baseline:
        print("\nBaseline comparison:")
        print(baseline_comparison_table(baseline))

    warnings_df = fit_result_warning_summary_table(fit_result)
    print(f"\nWarning summary:\n{warnings_df}")

    builder = KernelFeatureBuilder(
        kernels={"simplex": kernel},
        time_col="time",
        numeric_cols=["input_signal"],
    )
    result = builder.transform_result(df)
    features = result.features
    report = result.report
    print(f"\nFeatures: {features.shape[0]} rows, columns={features.columns}")
    print(features.head(6))

    if report:
        print(f"\nTransform report: {report.row_count} rows, {len(report.feature_names)} features")

    evidence = builder.diagnose_feature_evidence(
        feature_registry=result.feature_registry,
        fit_result_by_kernel={"simplex": fit_result},
    )
    evidence_table = feature_evidence_table(evidence)
    print(f"\nFeature evidence:\n{evidence_table}")

    print("\nDone — 01_quickstart_simplex.py completed.")


if __name__ == "__main__":
    main()
