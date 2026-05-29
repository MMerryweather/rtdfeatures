"""Out-of-fold feature generation (fit-transform per fold + stitching)."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from rtdfeatures.diagnostics import (
    KernelCandidateSet,
    KernelComparisonResult,
    KernelSelectionResult,
    OutOfFoldKernelFeatureResult,
    OutOfFoldSplitSummary,
    TransformReport,
)
from rtdfeatures.features import KernelFeatureBuilder
from rtdfeatures.kernels import Kernel
from rtdfeatures.learners import SimplexKernelLearner
from rtdfeatures.oof.reports import (
    RecoverableFoldError,
    _deterministic_fallback_candidate_id,
    _failed_fold_report,
    _resolve_selected_kernel_from_comparison,
)
from rtdfeatures.oof.splits import ForwardChainingSplitConfig, generate_forward_chaining_splits
from rtdfeatures.utils import validate_or_sort_time


def fit_transform_oof(
    *,
    df: pl.DataFrame,
    learner: SimplexKernelLearner | None = None,
    candidate_set: KernelCandidateSet | None = None,
    select_candidate_per_fold: bool = True,
    selection_loss_tolerance_fraction: float = 0.02,
    split_config: ForwardChainingSplitConfig,
    input_col: str,
    target_col: str,
    time_col: str,
    numeric_cols: list[str] | tuple[str, ...],
    category_cols: list[str] | tuple[str, ...] | None = None,
    weight_col: str | None = None,
    age_tail_threshold: float | None = None,
    order_by_time: bool = False,
) -> OutOfFoldKernelFeatureResult:
    """Fit one learner per fold and produce stitched OOF features."""
    from rtdfeatures import oof as _oof_module
    ordered = validate_or_sort_time(df, time_col=time_col, order_by_time=order_by_time)
    indexed = ordered.with_row_index("_row_idx")
    folds = generate_forward_chaining_splits(n_rows=ordered.height, config=split_config)

    if learner is not None and candidate_set is not None:
        raise ValueError("Provide exactly one of learner or candidate_set, not both.")
    if learner is None and candidate_set is None:
        raise ValueError("Provide learner or candidate_set for OOF generation.")
    if candidate_set is not None:
        if candidate_set.input_col != input_col:
            raise ValueError("candidate_set.input_col must match input_col.")
        if candidate_set.target_col != target_col:
            raise ValueError("candidate_set.target_col must match target_col.")
        if candidate_set.time_col != time_col:
            raise ValueError("candidate_set.time_col must match time_col.")

    fold_results: list[dict[str, Any]] = []
    fold_reports: list[TransformReport] = []
    fold_feature_frames: list[tuple[pl.DataFrame, tuple[str, ...]]] = []
    feature_names_seen: list[str] = []
    warnings: list[str] = []
    successful_kernel_by_name: dict[str, Kernel] = {}
    successful_candidate_by_kernel: dict[str, str] = {}
    fold_kernel_summaries_by_name: dict[str, list[dict[str, Any]]] = {}
    fold_evidence_records: list[dict[str, Any]] = []

    for fold in folds:
        train_df = indexed.filter(pl.col("_row_idx").is_in(fold.train_indices)).drop("_row_idx")
        validation_df = (
            indexed.filter(pl.col("_row_idx").is_in(fold.validation_indices))
            .drop("_row_idx")
        )

        selected_kernel: Kernel | None = None
        selected_kernel_name: str | None = None
        comparison_result: KernelComparisonResult | None = None
        selection_result: KernelSelectionResult | None = None
        fold_validation_loss: float | None = None
        selected_candidate_id: str | None = None

        try:
            if candidate_set is None:
                if learner is None:
                    raise RuntimeError("learner is required when candidate_set is not provided.")
                fit_result = learner.fit(
                    train_df, input_col=input_col, target_col=target_col,
                    time_col=time_col, order_by_time=False,
                )
                selected_kernel = fit_result.kernel
                selected_kernel_name = fit_result.kernel.name or f"fold_{fold.fold_id}_kernel"
                fold_validation_loss = fit_result.fit_diagnostics.validation_loss
            else:
                comparison_result = _oof_module.fit_kernel_candidates(
                    train_df, candidate_set, order_by_time=False
                )
                if select_candidate_per_fold:
                    selection_result = _oof_module.select_kernel_candidate(
                        comparison_result,
                        loss_tolerance_fraction=selection_loss_tolerance_fraction,
                    )
                    selected_candidate_id = selection_result.selected_candidate_id
                    selected_kernel = selection_result.selected_kernel
                    selected_kernel_name = selection_result.selected_candidate_id
                else:
                    selected_candidate_id = _deterministic_fallback_candidate_id(comparison_result)
                    if selected_candidate_id is not None:
                        selected_kernel = _resolve_selected_kernel_from_comparison(
                            comparison_result, selected_candidate_id
                        )
                        selected_kernel_name = selected_candidate_id
                    selection_result = None
                if selected_kernel is None:
                    raise RecoverableFoldError(
                        "No fold kernel selected from candidate comparison. "
                        "Check that at least one candidate fits the training data "
                        "or relax candidate constraints."
                    )
                selected_row = comparison_result.comparison_table.filter(
                    pl.col("candidate_id") == selected_kernel_name
                )
                if selected_row.height > 0 and "validation_loss" in selected_row.columns:
                    selected_loss = selected_row.get_column("validation_loss").to_list()[0]
                    if selected_loss is not None:
                        fold_validation_loss = float(selected_loss)

            kernel_name = selected_kernel_name or f"fold_{fold.fold_id}_kernel"
            if selected_kernel is None:
                raise RuntimeError("selected_kernel must be available before transform.")
            builder = KernelFeatureBuilder(
                kernels={kernel_name: selected_kernel},
                time_col=time_col,
                numeric_cols=numeric_cols,
                category_cols=category_cols,
                weight_col=weight_col,
                age_tail_threshold=age_tail_threshold,
            )
            fold_features = builder.augment_cols(validation_df, order_by_time=False)
            fold_report = builder.last_transform_report
            if fold_report is None:
                raise RuntimeError("Transform report missing after fold transform.")
            fold_reports.append(fold_report)
            generated_cols = tuple(
                name for name in fold_report.feature_names if name in fold_features.columns
            )
            for name in generated_cols:
                if name not in feature_names_seen:
                    feature_names_seen.append(name)
            fold_feature_frames.append((
                pl.DataFrame({
                    "_row_idx": list(fold.validation_indices),
                    **{name: fold_features.get_column(name) for name in generated_cols},
                }),
                generated_cols,
            ))
        except RecoverableFoldError as exc:
            reason = f"{type(exc).__name__}: {exc}"
            warnings.append(f"Fold {fold.fold_id}: {reason}; validation rows keep null features.")
            fold_reports.append(_failed_fold_report(validation_df.height))
            fold_results.append({
                "fold_id": fold.fold_id, "train_rows": len(fold.train_indices),
                "validation_rows": len(fold.validation_indices), "status": "failed",
                "failure_reason": reason, "comparison_result": comparison_result,
                "selection_result": selection_result,
            })
            fold_evidence_records.append({
                "evidence_scope": "oof", "fold_id": fold.fold_id,
                "status": "failed", "failure_reason": reason,
                "kernel_name": None, "candidate_id": None,
            })
            continue

        kernel_name = selected_kernel_name or f"fold_{fold.fold_id}_kernel"
        fold_results.append({
            "fold_id": fold.fold_id, "train_rows": len(fold.train_indices),
            "validation_rows": len(fold.validation_indices), "kernel_name": kernel_name,
            "validation_loss": fold_validation_loss, "status": "succeeded",
            "comparison_result": comparison_result, "selection_result": selection_result,
        })
        successful_kernel_by_name.setdefault(kernel_name, selected_kernel)
        if selected_candidate_id is not None and kernel_name not in successful_candidate_by_kernel:
            successful_candidate_by_kernel[kernel_name] = selected_candidate_id
        fold_kernel_summaries_by_name.setdefault(kernel_name, []).append({
            "fold_id": fold.fold_id, "status": "succeeded",
            "candidate_id": selected_candidate_id,
            "kernel_summary": selected_kernel.summary(),
        })
        fold_evidence_records.append({
            "evidence_scope": "oof", "fold_id": fold.fold_id,
            "status": "succeeded", "failure_reason": None,
            "kernel_name": kernel_name, "candidate_id": selected_candidate_id,
        })

    feature_names = sorted(feature_names_seen)
    out = indexed.select(["_row_idx", time_col])
    for name in feature_names:
        out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias(name))

    for frame, frame_feature_names in fold_feature_frames:
        out = out.join(frame, on="_row_idx", how="left", suffix="_fold")
        for name in frame_feature_names:
            out = out.with_columns(
                pl.coalesce(pl.col(name), pl.col(f"{name}_fold")).alias(name)
            ).drop(f"{name}_fold")

    out = out.sort("_row_idx").drop("_row_idx")

    missing_rows_by_feature: dict[str, int] = {}
    missing_fraction_by_feature: dict[str, float] = {}
    for name in feature_names:
        values = out.get_column(name).cast(pl.Float64).to_numpy()
        missing_count = int(np.sum(~np.isfinite(values)))
        missing_rows_by_feature[name] = missing_count
        missing_fraction_by_feature[name] = (
            float(missing_count) / float(out.height) if out.height > 0 else 0.0
        )

    zero_denominator_rows_by_feature: dict[str, int] = {}
    missing_rows_by_kernel: dict[str, int] = {}
    missing_fraction_by_kernel: dict[str, float] = {}
    zero_denominator_rows_by_kernel: dict[str, int] = {}
    for report in fold_reports:
        for key, value in report.zero_denominator_rows_by_feature.items():
            zero_denominator_rows_by_feature[key] = (
                zero_denominator_rows_by_feature.get(key, 0) + value
            )
        for key, value in report.missing_rows_by_kernel.items():
            missing_rows_by_kernel[key] = missing_rows_by_kernel.get(key, 0) + value
        for key, value in report.zero_denominator_rows_by_kernel.items():
            zero_denominator_rows_by_kernel[key] = (
                zero_denominator_rows_by_kernel.get(key, 0) + value
            )

    for kernel_name, kernel_missing_rows in missing_rows_by_kernel.items():
        kernel_feature_count = max(
            len([name for name in feature_names if name.startswith(f"{kernel_name}_")]), 1,
        )
        missing_fraction_by_kernel[kernel_name] = float(kernel_missing_rows) / float(
            out.height * kernel_feature_count
        )

    finite_feature_rows_all = 0
    finite_feature_rows_any = 0
    if feature_names:
        finite_masks = [
            np.isfinite(out.get_column(name).cast(pl.Float64).to_numpy()) for name in feature_names
        ]
        finite_feature_rows_all = int(np.sum(np.logical_and.reduce(finite_masks)))
        finite_feature_rows_any = int(np.sum(np.logical_or.reduce(finite_masks)))

    total_warmup_rows = sum(report.warmup_rows for report in fold_reports)

    combined_report = TransformReport(
        row_count=ordered.height, output_row_count=out.height,
        warmup_rows=total_warmup_rows, feature_names=tuple(feature_names),
        missing_rows_by_feature=missing_rows_by_feature,
        zero_denominator_rows_by_feature=zero_denominator_rows_by_feature,
        missing_fraction_by_feature=missing_fraction_by_feature,
        missing_rows_by_kernel=missing_rows_by_kernel,
        missing_fraction_by_kernel=missing_fraction_by_kernel,
        zero_denominator_rows_by_kernel=zero_denominator_rows_by_kernel,
        warmup_unusable_summary={
            "input_rows": out.height, "warmup_rows": total_warmup_rows,
            "rows_after_warmup": max(out.height - total_warmup_rows, 0),
            "rows_all_features_usable": finite_feature_rows_all,
            "rows_with_any_unusable_feature": out.height - finite_feature_rows_all,
        },
        collision_naming_summary={
            "kernel_names": tuple(sorted(missing_rows_by_kernel.keys())),
            "feature_count_by_kernel": {
                kernel_name: len(
                    [name for name in feature_names if name.startswith(f"{kernel_name}_")]
                )
                for kernel_name in missing_rows_by_kernel
            },
            "total_feature_count": len(feature_names),
            "has_name_collision": False,
        },
    )

    split_summary = OutOfFoldSplitSummary(
        n_folds=len(folds), split_strategy="forward_chaining",
        fold_boundaries=tuple({
            "fold_id": fold.fold_id, "train_start": fold.train_start,
            "train_end": fold.train_end, "validation_start": fold.validation_start,
            "validation_end": fold.validation_end, "gap": fold.gap,
        } for fold in folds),
        min_train_rows=min(len(fold.train_indices) for fold in folds),
        validation_rows_total=sum(len(fold.validation_indices) for fold in folds),
        rows_with_features=finite_feature_rows_any,
        rows_without_features=out.height - finite_feature_rows_any,
        warnings=tuple(warnings),
    )

    feature_evidence_report = None
    if feature_names and successful_kernel_by_name:
        evidence_builder = KernelFeatureBuilder(
            kernels=successful_kernel_by_name, time_col=time_col,
            numeric_cols=numeric_cols, category_cols=category_cols,
            weight_col=weight_col, age_tail_threshold=age_tail_threshold,
        )
        metadata_by_feature = {
            feature_name: {
                "evidence_scope": "oof", "full_data_evidence_available": False,
                "fold_evidence": fold_evidence_records,
                "fold_kernel_summaries": fold_kernel_summaries_by_name,
            }
            for feature_name in feature_names
        }
        registry = evidence_builder._build_feature_registry(feature_names=tuple(feature_names))
        feature_evidence_report = _oof_module.build_feature_evidence(
            builder=evidence_builder, feature_registry=registry,
            candidate_id_by_kernel=successful_candidate_by_kernel,
            metadata_by_feature=metadata_by_feature,
        )

    return OutOfFoldKernelFeatureResult(
        features=out, fold_results=tuple(fold_results),
        fold_reports=tuple(fold_reports),
        combined_transform_report=combined_report,
        feature_evidence_report=feature_evidence_report,
        split_summary=split_summary, warnings=tuple(warnings),
    )
