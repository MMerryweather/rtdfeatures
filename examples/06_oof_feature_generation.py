#!/usr/bin/env python3
"""Leakage-safe OOF feature generation with forward-chaining splits."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import polars as pl

from rtdfeatures import KernelFeatureBuilder, SimplexKernelLearner
from rtdfeatures.oof import ForwardChainingSplitConfig, fit_transform_oof
from rtdfeatures.synthetic import make_single_delay_dataset


def main() -> None:
    ds = make_single_delay_dataset(n_rows=600, dt=60.0, seed=7, delay_steps=6, noise_std=0.02)
    df = ds.data
    print(f"Data: {df.shape[0]} rows\n")

    split_config = ForwardChainingSplitConfig(
        n_folds=4,
        min_train_size=120,
        validation_size=60,
        gap=0,
    )

    learner = SimplexKernelLearner(min_lag=1, max_lag=10, seed=42, max_epochs=400)

    oof_result = fit_transform_oof(
        df=df,
        learner=learner,
        split_config=split_config,
        input_col="input_signal",
        target_col="target_signal",
        time_col="time",
        numeric_cols=["input_signal"],
    )

    oof_features = oof_result.features
    print(f"OOF features: {oof_features.shape[0]} rows, columns={oof_features.columns}")
    print(f"\nOOF feature table head:\n{oof_features.head(10)}")

    print("\nFold results:")
    for fr in oof_result.fold_results:
        print(f"  Fold {fr['fold_id']}: {fr['status']}, "
              f"train={fr['train_rows']}, validation={fr['validation_rows']}")

    split_summary = oof_result.split_summary
    print(f"\nSplit summary: {split_summary.n_folds} folds, "
          f"validation_rows={split_summary.validation_rows_total}, "
          f"warnings={len(split_summary.warnings)}")

    combined_report = oof_result.combined_transform_report
    print(f"\nCombined transform report: {len(combined_report.feature_names)} features, "
          f"warmup={combined_report.warmup_rows} rows")

    for feat_name, missing_count in combined_report.missing_rows_by_feature.items():
        missing_pct = 100.0 * missing_count / max(oof_features.shape[0], 1)
        if missing_pct > 1.0:
            print(f"  WARNING: {feat_name}: {missing_count} missing ({missing_pct:.1f}%)")

    evidence = oof_result.feature_evidence_report
    if evidence:
        print(f"\nFeature evidence report: {len(evidence.feature_evidence)} evidence rows")

    ####################################################################
    # Naive in-sample transform (leakage demo)
    ####################################################################
    naive_kernel = SimplexKernelLearner(min_lag=1, max_lag=10, seed=42, max_epochs=400).fit(
        df, input_col="input_signal", target_col="target_signal", time_col="time",
    ).kernel

    builder = KernelFeatureBuilder(
        kernels={"naive": naive_kernel},
        time_col="time",
        numeric_cols=["input_signal"],
    )
    in_sample_features = builder.transform(df)

    oof_count = oof_features.select(pl.col("^.*_wmean$")).null_count().sum_horizontal().to_list()[0]
    insample_count = (
        in_sample_features.select(pl.col("^.*_wmean$"))
        .null_count().sum_horizontal().to_list()[0]
    )

    print("\n--- Leakage risk comparison ---")
    print(f"OOF null count: {oof_count} (expected: warmup rows only)")
    print(f"In-sample null count: {insample_count}")
    print("WARNING: In-sample features use the full dataset to fit AND transform.")
    print("         Use fit_transform_oof for leakage-safe feature generation.")

    print("\nDone — 06_oof_feature_generation.py completed.")


if __name__ == "__main__":
    main()
