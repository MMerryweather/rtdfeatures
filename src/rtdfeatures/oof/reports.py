"""Out-of-fold report helpers and fold error types."""

from __future__ import annotations

import polars as pl

from rtdfeatures.diagnostics import KernelComparisonResult, TransformReport
from rtdfeatures.kernels import Kernel


class RecoverableFoldError(RuntimeError):
    """Raised when a single fold can be marked failed while OOF continues."""


def _failed_fold_report(validation_rows: int) -> TransformReport:
    return TransformReport(
        row_count=validation_rows,
        output_row_count=validation_rows,
        warmup_rows=0,
        feature_names=tuple(),
        missing_rows_by_feature={},
        zero_denominator_rows_by_feature={},
        missing_fraction_by_feature={},
        missing_rows_by_kernel={},
        missing_fraction_by_kernel={},
        zero_denominator_rows_by_kernel={},
        warmup_unusable_summary={
            "input_rows": validation_rows,
            "warmup_rows": 0,
            "rows_after_warmup": validation_rows,
            "rows_all_features_usable": 0,
            "rows_with_any_unusable_feature": validation_rows,
        },
        collision_naming_summary={
            "kernel_names": tuple(),
            "feature_count_by_kernel": {},
            "total_feature_count": 0,
            "has_name_collision": False,
        },
    )


def _resolve_selected_kernel_from_comparison(
    comparison_result: KernelComparisonResult,
    candidate_id: str,
) -> Kernel | None:
    for family_result in comparison_result.family_results:
        if family_result.candidate.candidate_id != candidate_id:
            continue
        if family_result.fit_result is not None:
            return family_result.fit_result.kernel
        if family_result.evaluated_fixed_kernel is not None:
            return family_result.evaluated_fixed_kernel
    return None


def _deterministic_fallback_candidate_id(
    comparison_result: KernelComparisonResult,
) -> str | None:
    table = comparison_result.comparison_table
    if table.height == 0:
        return None
    if "succeeded" in table.columns:
        table = table.filter(pl.col("succeeded"))
    if table.height == 0:
        return None
    if "validation_loss" in table.columns:
        table = table.with_columns(
            pl.when(pl.col("validation_loss").is_null())
            .then(pl.lit(float("inf")))
            .otherwise(pl.col("validation_loss"))
            .alias("_selection_loss")
        ).sort(by=["_selection_loss", "candidate_id"])
    else:
        table = table.sort(by="candidate_id")
    candidate_id = table.get_column("candidate_id").to_list()[0]
    return str(candidate_id) if candidate_id is not None else None
