"""Transform-time diagnostic data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rtdfeatures.diagnostics._helpers import _require_non_empty_name
from rtdfeatures.diagnostics.warnings import FeatureEvidenceReport


@dataclass(frozen=True)
class TransformReport:
    """Diagnostics for feature transformation output integrity."""

    row_count: int
    output_row_count: int
    warmup_rows: int
    feature_names: tuple[str, ...]
    missing_rows_by_feature: dict[str, int]
    zero_denominator_rows_by_feature: dict[str, int]
    missing_fraction_by_feature: dict[str, float] = field(default_factory=dict)
    missing_rows_by_kernel: dict[str, int] = field(default_factory=dict)
    missing_fraction_by_kernel: dict[str, float] = field(default_factory=dict)
    zero_denominator_rows_by_kernel: dict[str, int] = field(default_factory=dict)
    warmup_unusable_summary: dict[str, int] = field(default_factory=dict)
    collision_naming_summary: dict[str, object] | None = None


@dataclass(frozen=True)
class OutOfFoldSplitSummary:
    """Split metadata summary for leakage-safe out-of-fold generation."""

    n_folds: int
    split_strategy: str
    fold_boundaries: tuple[dict[str, int], ...]
    min_train_rows: int
    validation_rows_total: int
    rows_with_features: int
    rows_without_features: int
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.n_folds <= 0:
            raise ValueError("n_folds must be a positive integer.")
        object.__setattr__(
            self,
            "split_strategy",
            _require_non_empty_name("split_strategy", self.split_strategy),
        )
        if len(self.fold_boundaries) != self.n_folds:
            raise ValueError("fold_boundaries length must match n_folds.")
        required_keys = {
            "fold_id",
            "train_start",
            "train_end",
            "validation_start",
            "validation_end",
            "gap",
        }
        for boundary in self.fold_boundaries:
            if set(boundary.keys()) != required_keys:
                raise ValueError(
                    "Each fold boundary must include exactly: "
                    "fold_id, train_start, train_end, validation_start, validation_end, gap."
                )
            if boundary["validation_start"] <= boundary["train_end"]:
                raise ValueError("validation_start must be strictly after train_end.")
            if boundary["gap"] < 0:
                raise ValueError("gap must be >= 0 for every fold boundary.")
        if self.min_train_rows <= 0:
            raise ValueError("min_train_rows must be a positive integer.")
        if self.validation_rows_total < 0:
            raise ValueError("validation_rows_total must be >= 0.")
        if self.rows_with_features < 0:
            raise ValueError("rows_with_features must be >= 0.")
        if self.rows_without_features < 0:
            raise ValueError("rows_without_features must be >= 0.")


@dataclass(frozen=True)
class OutOfFoldKernelFeatureResult:
    """Out-of-fold feature-generation artifacts and diagnostics contract."""

    features: Any  # pl.DataFrame
    fold_results: tuple[dict[str, Any], ...]
    fold_reports: tuple[TransformReport, ...]
    combined_transform_report: TransformReport
    feature_evidence_report: FeatureEvidenceReport | None
    split_summary: OutOfFoldSplitSummary
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.fold_results) != len(self.fold_reports):
            raise ValueError("fold_results and fold_reports must have the same length.")
