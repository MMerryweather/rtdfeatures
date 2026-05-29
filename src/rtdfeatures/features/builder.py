"""Kernel feature builder — generate kernel-weighted features from learned or fixed kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import polars as pl

from rtdfeatures.diagnostics import (
    FeatureEvidenceReport,
    FeatureInterpretationLabel,
    KernelFitResult,
    TransformReport,
)
from rtdfeatures.features.accumulator import FeatureAccumulator
from rtdfeatures.features.age import age_feature_values, resolve_age_tail_threshold
from rtdfeatures.features.categorical import categorical_fraction_and_entropy_series
from rtdfeatures.features.evidence import build_feature_evidence
from rtdfeatures.features.names import (
    age_feature_name,
    categorical_entropy_feature_name,
    categorical_fraction_feature_name,
    numeric_feature_name,
)
from rtdfeatures.features.numeric import weighted_numeric_series
from rtdfeatures.features.registry import FeatureRegistry, FeatureSpec, TransformResult
from rtdfeatures.kernels import Kernel
from rtdfeatures.utils import validate_or_sort_time


@dataclass(frozen=True)
class _ExecutionResult:
    features: pl.DataFrame
    report: TransformReport
    feature_registry: FeatureRegistry


@dataclass(frozen=True)
class _FeatureComputation:
    feature_arrays: dict[str, np.ndarray]
    feature_specs: tuple[FeatureSpec, ...]
    missing_rows_by_feature: dict[str, int]
    missing_fraction_by_feature: dict[str, float]
    zero_denominator_rows_by_feature: dict[str, int]
    missing_rows_by_kernel: dict[str, int]
    missing_fraction_by_kernel: dict[str, float]
    zero_denominator_rows_by_kernel: dict[str, int]
    kernel_feature_names: dict[str, list[str]]
    max_warmup: int


class KernelFeatureBuilder:
    """Generate kernel-based features from one or more learned/fixed kernels."""

    def __init__(
        self,
        *,
        kernels: dict[str, Kernel],
        time_col: str,
        numeric_cols: list[str] | tuple[str, ...] | None = None,
        category_cols: list[str] | tuple[str, ...] | None = None,
        weight_col: str | None = None,
        age_tail_threshold: float | None = None,
    ) -> None:
        if not kernels:
            raise ValueError("kernels must include at least one named kernel.")
        self.kernels = dict(kernels)
        self.time_col = time_col
        self.numeric_cols = tuple(numeric_cols or ())
        self.category_cols = tuple(category_cols or ())
        self.category_levels_by_col: dict[str, tuple[str, ...]] = {}
        self.weight_col = weight_col
        self.age_tail_threshold = age_tail_threshold
        self.last_transform_report: TransformReport | None = None
        self.last_feature_registry: FeatureRegistry | None = None

    def transform(self, df: pl.DataFrame, *, order_by_time: bool = False) -> pl.DataFrame:
        execution = self._execute(df, order_by_time=order_by_time)
        self.last_transform_report = execution.report
        self.last_feature_registry = execution.feature_registry
        return execution.features

    def transform_with_report(
        self, df: pl.DataFrame, *, order_by_time: bool = False
    ) -> tuple[pl.DataFrame, TransformReport]:
        execution = self._execute(df, order_by_time=order_by_time)
        self.last_transform_report = execution.report
        self.last_feature_registry = execution.feature_registry
        return execution.features, execution.report

    def transform_result(
        self, df: pl.DataFrame, *, order_by_time: bool = False
    ) -> TransformResult:
        execution = self._execute(df, order_by_time=order_by_time)
        self.last_transform_report = execution.report
        self.last_feature_registry = execution.feature_registry
        return TransformResult(
            features=execution.features,
            report=execution.report,
            feature_registry=execution.feature_registry,
        )

    def augment_cols(self, df: pl.DataFrame, *, order_by_time: bool = False) -> pl.DataFrame:
        ordered = validate_or_sort_time(df, time_col=self.time_col, order_by_time=order_by_time)
        execution = self._execute(ordered, order_by_time=False)
        self.last_transform_report = execution.report
        self.last_feature_registry = execution.feature_registry
        return cast(
            pl.DataFrame,
            ordered.hstack(execution.features.drop(self.time_col).get_columns()),
        )

    def diagnose_transform(
        self, df: pl.DataFrame, *, order_by_time: bool = False
    ) -> TransformReport:
        execution = self._execute(df, order_by_time=order_by_time)
        self.last_transform_report = execution.report
        self.last_feature_registry = execution.feature_registry
        return execution.report

    def diagnose_feature_evidence(
        self,
        *,
        feature_registry: FeatureRegistry | None = None,
        feature_names: tuple[str, ...] | list[str] | None = None,
        fit_result_by_kernel: dict[str, KernelFitResult] | None = None,
        interpretation_by_kernel: dict[str, FeatureInterpretationLabel] | None = None,
        interpretation_by_feature: dict[str, FeatureInterpretationLabel] | None = None,
        candidate_id_by_kernel: dict[str, str] | None = None,
        baseline_summary_by_kernel: dict[str, dict[str, Any]] | None = None,
        bootstrap_summary_by_kernel: dict[str, dict[str, Any]] | None = None,
        metadata_by_kernel: dict[str, dict[str, Any]] | None = None,
        metadata_by_feature: dict[str, dict[str, Any]] | None = None,
    ) -> FeatureEvidenceReport:
        if feature_registry is None:
            registry = self._build_feature_registry_from_specs(feature_names=feature_names)
        else:
            registry = feature_registry
        return build_feature_evidence(
            builder=self,
            feature_registry=registry,
            fit_result_by_kernel=fit_result_by_kernel,
            interpretation_by_kernel=interpretation_by_kernel,
            interpretation_by_feature=interpretation_by_feature,
            candidate_id_by_kernel=candidate_id_by_kernel,
            baseline_summary_by_kernel=baseline_summary_by_kernel,
            bootstrap_summary_by_kernel=bootstrap_summary_by_kernel,
            metadata_by_kernel=metadata_by_kernel,
            metadata_by_feature=metadata_by_feature,
        )

    # ------------------------------------------------------------------
    # Internal orchestration
    # ------------------------------------------------------------------

    def _execute(self, df: pl.DataFrame, *, order_by_time: bool) -> _ExecutionResult:
        ordered = self._prepare_input(df, order_by_time=order_by_time)
        missing = sorted(self._required_columns() - set(ordered.columns))
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. "
                "Ensure these columns exist in the DataFrame "
                f"(available: {sorted(ordered.columns)})."
            )
        comp = self._compute_all_features(ordered)
        out_df = self._build_output_dataframe(ordered, comp.feature_arrays)
        report = self._build_transform_report(
            n_input_rows=df.height,
            out_df=out_df,
            max_warmup=comp.max_warmup,
            feature_arrays=comp.feature_arrays,
            missing_rows_by_feature=comp.missing_rows_by_feature,
            missing_fraction_by_feature=comp.missing_fraction_by_feature,
            zero_denominator_rows_by_feature=comp.zero_denominator_rows_by_feature,
            missing_rows_by_kernel=comp.missing_rows_by_kernel,
            missing_fraction_by_kernel=comp.missing_fraction_by_kernel,
            zero_denominator_rows_by_kernel=comp.zero_denominator_rows_by_kernel,
            kernel_feature_names=comp.kernel_feature_names,
            kernel_names=tuple(self.kernels.keys()),
        )
        return _ExecutionResult(
            features=out_df,
            report=report,
            feature_registry=FeatureRegistry(specs=comp.feature_specs),
        )

    # ------------------------------------------------------------------
    # Step helpers
    # ------------------------------------------------------------------

    def _prepare_input(self, df: pl.DataFrame, *, order_by_time: bool) -> pl.DataFrame:
        return validate_or_sort_time(df, time_col=self.time_col, order_by_time=order_by_time)

    def _required_columns(self) -> set[str]:
        required = {self.time_col, *self.numeric_cols, *self.category_cols}
        if self.weight_col is not None:
            required.add(self.weight_col)
        return required

    def _compute_all_features(self, ordered: pl.DataFrame) -> _FeatureComputation:
        accumulator = FeatureAccumulator(n_rows=ordered.height)
        missing_rows_by_kernel: dict[str, int] = {}
        missing_fraction_by_kernel: dict[str, float] = {}
        zero_denominator_rows_by_kernel: dict[str, int] = {}
        max_warmup = 0

        for kernel_name, kernel in self.kernels.items():
            kernel.validate()
            if not kernel_name:
                raise ValueError("kernel_name must be a non-empty string.")
            accumulator.ensure_kernel(kernel_name)
            max_warmup = max(max_warmup, kernel.max_lag_steps)
            kernel_zero_denominator_total = 0
            kernel_zero_denominator_total += self._add_numeric_features(
                ordered=ordered,
                kernel_name=kernel_name,
                kernel=kernel,
                accumulator=accumulator,
            )
            kernel_zero_denominator_total += self._add_categorical_features(
                ordered=ordered,
                kernel_name=kernel_name,
                kernel=kernel,
                accumulator=accumulator,
            )
            self._add_age_features(
                n_rows=ordered.height,
                kernel_name=kernel_name,
                kernel=kernel,
                accumulator=accumulator,
            )

            kernel_missing_rows = sum(
                accumulator.missing_rows_by_feature[name]
                for name in accumulator.kernel_feature_names[kernel_name]
            )
            missing_rows_by_kernel[kernel_name] = int(kernel_missing_rows)
            missing_fraction_by_kernel[kernel_name] = (
                float(kernel_missing_rows)
                / float(ordered.height * len(accumulator.kernel_feature_names[kernel_name]))
                if ordered.height > 0 and accumulator.kernel_feature_names[kernel_name]
                else 0.0
            )
            zero_denominator_rows_by_kernel[kernel_name] = int(kernel_zero_denominator_total)

        accumulator.finalize_missing_fractions()

        return _FeatureComputation(
            feature_arrays=accumulator.arrays,
            feature_specs=tuple(accumulator.specs),
            missing_rows_by_feature=accumulator.missing_rows_by_feature,
            missing_fraction_by_feature=accumulator.missing_fraction_by_feature,
            zero_denominator_rows_by_feature=accumulator.zero_denominator_rows_by_feature,
            missing_rows_by_kernel=missing_rows_by_kernel,
            missing_fraction_by_kernel=missing_fraction_by_kernel,
            zero_denominator_rows_by_kernel=zero_denominator_rows_by_kernel,
            kernel_feature_names=accumulator.kernel_feature_names,
            max_warmup=max_warmup,
        )

    def _add_numeric_features(
        self,
        *,
        ordered: pl.DataFrame,
        kernel_name: str,
        kernel: Kernel,
        accumulator: FeatureAccumulator,
    ) -> int:
        kernel_summary = kernel.summary()
        lag_steps = tuple(kernel.lag_steps)
        zero_denominator_total = 0
        for numeric_col in self.numeric_cols:
            mean_feature_name = numeric_feature_name(kernel_name, numeric_col, "wmean")
            std_feature_name = numeric_feature_name(kernel_name, numeric_col, "wstd")
            sum_feature_name = numeric_feature_name(kernel_name, numeric_col, "wsum")
            values = ordered.get_column(numeric_col).cast(pl.Float64).to_numpy()
            weight_values = (
                ordered.get_column(self.weight_col).cast(pl.Float64).to_numpy()
                if self.weight_col is not None
                else None
            )
            mean_arr, std_arr, sum_arr, zero_count = weighted_numeric_series(
                values=values,
                lag_steps=np.asarray(kernel.lag_steps, dtype=np.int64),
                lag_weights=np.asarray(kernel.weights, dtype=np.float64),
                max_lag_steps=kernel.max_lag_steps,
                weight_values=weight_values,
            )
            for feature_name, metric, arr in (
                (mean_feature_name, "wmean", mean_arr),
                (std_feature_name, "wstd", std_arr),
                (sum_feature_name, "wsum", sum_arr),
            ):
                accumulator.add(
                    name=feature_name,
                    values=arr,
                    spec=FeatureSpec(
                        name=feature_name,
                        kernel_name=kernel_name,
                        source_col=numeric_col,
                        family="numeric",
                        metric=metric,
                        category_level=None,
                        lag_steps=lag_steps,
                        kernel_summary=kernel_summary,
                    ),
                    zero_denominator_rows=int(zero_count),
                )
            zero_denominator_total += int(zero_count) * 3
        return zero_denominator_total

    def _add_categorical_features(
        self,
        *,
        ordered: pl.DataFrame,
        kernel_name: str,
        kernel: Kernel,
        accumulator: FeatureAccumulator,
    ) -> int:
        kernel_summary = kernel.summary()
        lag_steps = tuple(kernel.lag_steps)
        zero_denominator_total = 0
        for category_col in self.category_cols:
            category_values = ordered.get_column(category_col).to_numpy()
            levels = sorted({str(level) for level in category_values if level is not None})
            if not levels:
                continue
            (
                fraction_arrays,
                entropy_array,
                zero_count,
            ) = categorical_fraction_and_entropy_series(
                category_values=category_values,
                levels=levels,
                lag_steps=np.asarray(kernel.lag_steps, dtype=np.int64),
                lag_weights=np.asarray(kernel.weights, dtype=np.float64),
                max_lag_steps=kernel.max_lag_steps,
                weight_values=(
                    ordered.get_column(self.weight_col).cast(pl.Float64).to_numpy()
                    if self.weight_col is not None
                    else None
                ),
            )
            for level in levels:
                feature_name = categorical_fraction_feature_name(
                    kernel_name,
                    category_col,
                    level,
                )
                accumulator.add(
                    name=feature_name,
                    values=fraction_arrays[level],
                    spec=FeatureSpec(
                        name=feature_name,
                        kernel_name=kernel_name,
                        source_col=category_col,
                        family="categorical",
                        metric="frac",
                        category_level=level,
                        lag_steps=lag_steps,
                        kernel_summary=kernel_summary,
                    ),
                    zero_denominator_rows=int(zero_count),
                )

            entropy_feature_name = categorical_entropy_feature_name(kernel_name, category_col)
            accumulator.add(
                name=entropy_feature_name,
                values=entropy_array,
                spec=FeatureSpec(
                    name=entropy_feature_name,
                    kernel_name=kernel_name,
                    source_col=category_col,
                    family="categorical",
                    metric="entropy",
                    category_level=None,
                    lag_steps=lag_steps,
                    kernel_summary=kernel_summary,
                ),
                zero_denominator_rows=int(zero_count),
            )
            zero_denominator_total += int(zero_count) * (len(levels) + 1)
        return zero_denominator_total

    def _add_age_features(
        self,
        *,
        n_rows: int,
        kernel_name: str,
        kernel: Kernel,
        accumulator: FeatureAccumulator,
    ) -> None:
        kernel_summary = kernel.summary()
        lag_steps = tuple(kernel.lag_steps)
        age_threshold = resolve_age_tail_threshold(
            min_lag_steps=kernel.min_lag_steps,
            max_lag_steps=kernel.max_lag_steps,
            dt=kernel.dt,
            configured_threshold=self.age_tail_threshold,
        )
        raw_age_values = age_feature_values(kernel=kernel, threshold=age_threshold)
        for metric, value in raw_age_values.items():
            feature_name = age_feature_name(kernel_name, metric)
            accumulator.add(
                name=feature_name,
                values=np.full(n_rows, float(value), dtype=np.float64),
                spec=FeatureSpec(
                    name=feature_name,
                    kernel_name=kernel_name,
                    source_col="__kernel__",
                    family="age",
                    metric=metric,
                    category_level=None,
                    lag_steps=lag_steps,
                    kernel_summary=kernel_summary,
                ),
            )

    def _build_output_dataframe(
        self,
        ordered: pl.DataFrame,
        feature_arrays: dict[str, np.ndarray],
    ) -> pl.DataFrame:
        out_data: dict[str, pl.Series] = {self.time_col: ordered.get_column(self.time_col)}
        for feature_name, arr in feature_arrays.items():
            out_data[feature_name] = pl.Series(feature_name, arr)
        return pl.DataFrame(out_data)

    @staticmethod
    def _build_transform_report(
        n_input_rows: int,
        out_df: pl.DataFrame,
        max_warmup: int,
        feature_arrays: dict[str, np.ndarray],
        missing_rows_by_feature: dict[str, int],
        missing_fraction_by_feature: dict[str, float],
        zero_denominator_rows_by_feature: dict[str, int],
        missing_rows_by_kernel: dict[str, int],
        missing_fraction_by_kernel: dict[str, float],
        zero_denominator_rows_by_kernel: dict[str, int],
        kernel_feature_names: dict[str, list[str]],
        kernel_names: tuple[str, ...],
    ) -> TransformReport:
        n_rows = out_df.height
        finite_feature_rows = 0
        if feature_arrays:
            finite_masks = [np.isfinite(arr) for arr in feature_arrays.values()]
            all_finite = np.logical_and.reduce(finite_masks)
            finite_feature_rows = int(np.sum(all_finite))
        warmup_region_rows = min(max_warmup, n_rows)
        rows_after_warmup = max(n_rows - warmup_region_rows, 0)
        report_summary = {
            "input_rows": n_rows,
            "warmup_rows": warmup_region_rows,
            "rows_after_warmup": rows_after_warmup,
            "rows_all_features_usable": finite_feature_rows,
            "rows_with_any_unusable_feature": n_rows - finite_feature_rows,
        }
        naming_summary = {
            "kernel_names": kernel_names,
            "feature_count_by_kernel": {
                kernel_name: len(names) for kernel_name, names in kernel_feature_names.items()
            },
            "total_feature_count": len(feature_arrays),
            "has_name_collision": False,
        }
        return TransformReport(
            row_count=n_input_rows,
            output_row_count=out_df.height,
            warmup_rows=max_warmup,
            feature_names=tuple(feature_arrays.keys()),
            missing_rows_by_feature=missing_rows_by_feature,
            zero_denominator_rows_by_feature=zero_denominator_rows_by_feature,
            missing_fraction_by_feature=missing_fraction_by_feature,
            missing_rows_by_kernel=missing_rows_by_kernel,
            missing_fraction_by_kernel=missing_fraction_by_kernel,
            zero_denominator_rows_by_kernel=zero_denominator_rows_by_kernel,
            warmup_unusable_summary=report_summary,
            collision_naming_summary=naming_summary,
        )

    # ------------------------------------------------------------------
    # Feature registry construction
    # ------------------------------------------------------------------

    def _build_feature_registry(
        self,
        feature_names: tuple[str, ...] | list[str] | None = None,
    ) -> FeatureRegistry:
        return self._build_feature_registry_from_specs(feature_names=feature_names)

    def _build_feature_registry_from_specs(
        self,
        feature_names: tuple[str, ...] | list[str] | None = None,
    ) -> FeatureRegistry:
        if self.last_feature_registry is None:
            if feature_names is not None:
                selected_order = tuple(dict.fromkeys(feature_names))
                selected = set(selected_order)
                specs = self._build_feature_specs_from_configuration(
                    selected_names=selected_order
                )
                return FeatureRegistry(
                    specs=tuple(spec for spec in specs if spec.name in selected)
                )
            raise ValueError(
                "Call transform() or transform_with_report() before building a registry."
            )
        specs = self.last_feature_registry.specs
        if feature_names is None:
            return FeatureRegistry(specs=specs)
        selected = set(feature_names)
        return FeatureRegistry(specs=tuple(spec for spec in specs if spec.name in selected))

    def _build_feature_specs_from_configuration(
        self,
        *,
        selected_names: tuple[str, ...] | None = None,
    ) -> tuple[FeatureSpec, ...]:
        specs: list[FeatureSpec] = []
        selected = selected_names or ()
        for kernel_name, kernel in self.kernels.items():
            kernel.validate()
            kernel_summary = kernel.summary()
            lag_steps = tuple(kernel.lag_steps)
            for numeric_col in self.numeric_cols:
                for metric in ("wmean", "wstd", "wsum"):
                    feature_name = numeric_feature_name(kernel_name, numeric_col, metric)
                    specs.append(
                        FeatureSpec(
                            name=feature_name,
                            kernel_name=kernel_name,
                            source_col=numeric_col,
                            family="numeric",
                            metric=metric,
                            category_level=None,
                            lag_steps=lag_steps,
                            kernel_summary=kernel_summary,
                        )
                    )
            for category_col in self.category_cols:
                for level in self.category_levels_by_col.get(category_col, ()):
                    feature_name = categorical_fraction_feature_name(
                        kernel_name,
                        category_col,
                        level,
                    )
                    specs.append(
                        FeatureSpec(
                            name=feature_name,
                            kernel_name=kernel_name,
                            source_col=category_col,
                            family="categorical",
                            metric="frac",
                            category_level=level,
                            lag_steps=lag_steps,
                            kernel_summary=kernel_summary,
                        )
                    )
                # Fresh-builder fallback can be asked for exact categorical fraction
                # names before transform() populated a concrete registry.
                configured_levels = set(self.category_levels_by_col.get(category_col, ()))
                prefix = f"{kernel_name}_cat_{category_col}_"
                for requested_name in selected:
                    if (
                        not requested_name.startswith(prefix)
                        or not requested_name.endswith("_frac")
                    ):
                        continue
                    level = requested_name[len(prefix) : -len("_frac")]
                    if not level or level in configured_levels:
                        continue
                    specs.append(
                        FeatureSpec(
                            name=requested_name,
                            kernel_name=kernel_name,
                            source_col=category_col,
                            family="categorical",
                            metric="frac",
                            category_level=level,
                            lag_steps=lag_steps,
                            kernel_summary=kernel_summary,
                        )
                    )
                feature_name = categorical_entropy_feature_name(kernel_name, category_col)
                specs.append(
                    FeatureSpec(
                        name=feature_name,
                        kernel_name=kernel_name,
                        source_col=category_col,
                        family="categorical",
                        metric="entropy",
                        category_level=None,
                        lag_steps=lag_steps,
                        kernel_summary=kernel_summary,
                    )
                )
            for metric in ("mean", "p50", "p90", "tail_gt_threshold"):
                feature_name = age_feature_name(kernel_name, metric)
                specs.append(
                    FeatureSpec(
                        name=feature_name,
                        kernel_name=kernel_name,
                        source_col="__kernel__",
                        family="age",
                        metric=metric,
                        category_level=None,
                        lag_steps=lag_steps,
                        kernel_summary=kernel_summary,
                    )
                )
        return tuple(specs)
