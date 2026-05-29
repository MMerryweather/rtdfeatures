"""Warning/identifiability diagnostic data structures and bootstrap schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import polars as pl

from rtdfeatures.diagnostics._helpers import _require_non_empty_name, _validate_json_serializable

BOOTSTRAP_WARNING_CODES: tuple[str, ...] = (
    "BOOTSTRAP_TOO_FEW_SUCCESSES",
    "BOOTSTRAP_WEIGHT_UNSTABLE",
    "BOOTSTRAP_PARAMETER_UNSTABLE",
    "BOOTSTRAP_PARAMETER_PROVENANCE_MISSING",
    "BOOTSTRAP_LAG_SUMMARY_UNSTABLE",
    "BOOTSTRAP_FAMILY_UNSTABLE",
    "BOOTSTRAP_INTERVAL_TOUCHES_BOUNDARY",
    "BOOTSTRAP_VALIDATION_WINDOW_CHANGED",
    "BOOTSTRAP_CONTEXT_MISMATCH",
    "BOOTSTRAP_BLOCK_LENGTH_INVALID",
)
DEFAULT_BOOTSTRAP_INTERVAL_QUANTILES: tuple[float, float] = (0.025, 0.975)
FeatureInterpretationLabel = Literal[
    "material_memory",
    "process_response",
    "statistical_pattern",
    "unknown",
]
FeatureEvidenceCompletenessLabel = Literal[
    "kernel_only",
    "fit_evidence",
    "comparison_evidence",
    "bootstrap_evidence",
    "full_evidence",
]
FEATURE_INTERPRETATION_LABELS: tuple[FeatureInterpretationLabel, ...] = (
    "material_memory",
    "process_response",
    "statistical_pattern",
    "unknown",
)
FEATURE_EVIDENCE_COMPLETENESS_LABELS: tuple[FeatureEvidenceCompletenessLabel, ...] = (
    "kernel_only",
    "fit_evidence",
    "comparison_evidence",
    "bootstrap_evidence",
    "full_evidence",
)


def bootstrap_weight_samples_schema() -> dict[str, type[pl.DataType]]:
    """Return deterministic schema for bootstrap weight sample tables."""
    return {
        "bootstrap_id": pl.Int64,
        "candidate_id": pl.String,
        "lag_step": pl.Int64,
        "lag_time": pl.Float64,
        "weight": pl.Float64,
    }


def bootstrap_parameter_samples_schema() -> dict[str, type[pl.DataType]]:
    """Return deterministic schema for bootstrap parameter sample tables."""
    return {
        "bootstrap_id": pl.Int64,
        "candidate_id": pl.String,
        "parameter_name": pl.String,
        "parameter_value": pl.Float64,
    }


def bootstrap_lag_summary_samples_schema() -> dict[str, type[pl.DataType]]:
    """Return deterministic schema for bootstrap lag-summary sample tables."""
    return {
        "bootstrap_id": pl.Int64,
        "candidate_id": pl.String,
        "mean_lag": pl.Float64,
        "p50_lag": pl.Float64,
        "p90_lag": pl.Float64,
        "tail_mass": pl.Float64,
    }


def parameter_uncertainty_summary_schema() -> dict[str, type[pl.DataType]]:
    """Return deterministic schema for parameter uncertainty summary tables."""
    return {
        "parameter_name": pl.String,
        "estimate": pl.Float64,
        "lower": pl.Float64,
        "upper": pl.Float64,
        "bootstrap_std": pl.Float64,
        "n_samples": pl.Int64,
    }


def weight_uncertainty_summary_schema() -> dict[str, type[pl.DataType]]:
    """Return deterministic schema for lag-weight uncertainty summary tables."""
    return {
        "lag_step": pl.Int64,
        "lag_time": pl.Float64,
        "weight_estimate": pl.Float64,
        "lower": pl.Float64,
        "upper": pl.Float64,
        "bootstrap_std": pl.Float64,
    }


@dataclass(frozen=True)
class BootstrapWeightSample:
    """One lag-weight sample record for a single bootstrap iteration."""

    bootstrap_id: int
    candidate_id: str
    lag_step: int
    lag_time: float
    weight: float


@dataclass(frozen=True)
class BootstrapParameterSample:
    """One fitted-parameter sample record for a single bootstrap iteration."""

    bootstrap_id: int
    candidate_id: str
    parameter_name: str
    parameter_value: float | None


@dataclass(frozen=True)
class BootstrapLagSummarySample:
    """One lag-summary sample record for a single bootstrap iteration."""

    bootstrap_id: int
    candidate_id: str
    mean_lag: float
    p50_lag: float
    p90_lag: float
    tail_mass: float


@dataclass(frozen=True)
class ParameterUncertaintySummary:
    """Bootstrap uncertainty interval summary for one named parameter."""

    parameter_name: str
    estimate: float
    lower: float
    upper: float
    bootstrap_std: float
    n_samples: int


@dataclass(frozen=True)
class WeightUncertaintySummary:
    """Bootstrap uncertainty interval summary for one lag-weight coordinate."""

    lag_step: int
    lag_time: float
    weight_estimate: float
    lower: float
    upper: float
    bootstrap_std: float


@dataclass(frozen=True)
class KernelBootstrapSummary:
    """Aggregate bootstrap uncertainty summary for one kernel candidate."""

    mean_lag_interval: tuple[float, float]
    p50_lag_interval: tuple[float, float]
    p90_lag_interval: tuple[float, float]
    tail_mass_interval: tuple[float, float]
    weight_interval_by_lag: tuple[WeightUncertaintySummary, ...]
    parameter_interval_by_name: tuple[ParameterUncertaintySummary, ...]
    stability_score: float | None


@dataclass(frozen=True)
class FeatureEvidence:
    """Evidence descriptor for one generated feature.

    Optional evidence fields use ``None`` when the corresponding evidence layer
    is unavailable by design or not computed in the current workflow.
    """

    feature_name: str
    source_col: str
    feature_family: str
    kernel_name: str
    kernel_family: str
    kernel_summary: dict[str, Any]
    fit_result_id: str | None
    candidate_id: str | None
    baseline_summary: dict[str, Any] | None
    identifiability_warnings: tuple[str, ...] | None
    bootstrap_summary: dict[str, Any] | None
    interpretation: FeatureInterpretationLabel
    evidence_completeness: FeatureEvidenceCompletenessLabel
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "feature_name", _require_non_empty_name("feature_name", self.feature_name)
        )
        object.__setattr__(
            self, "source_col", _require_non_empty_name("source_col", self.source_col)
        )
        object.__setattr__(
            self, "feature_family", _require_non_empty_name("feature_family", self.feature_family)
        )
        object.__setattr__(
            self, "kernel_name", _require_non_empty_name("kernel_name", self.kernel_name)
        )
        object.__setattr__(
            self, "kernel_family", _require_non_empty_name("kernel_family", self.kernel_family)
        )
        if self.fit_result_id is not None:
            object.__setattr__(
                self, "fit_result_id", _require_non_empty_name("fit_result_id", self.fit_result_id)
            )
        if self.candidate_id is not None:
            object.__setattr__(
                self, "candidate_id", _require_non_empty_name("candidate_id", self.candidate_id)
            )
        if self.interpretation not in FEATURE_INTERPRETATION_LABELS:
            raise ValueError(
                "interpretation must be one of: " + ", ".join(FEATURE_INTERPRETATION_LABELS)
            )
        if self.evidence_completeness not in FEATURE_EVIDENCE_COMPLETENESS_LABELS:
            raise ValueError(
                "evidence_completeness must be one of: "
                + ", ".join(FEATURE_EVIDENCE_COMPLETENESS_LABELS)
            )
        _validate_json_serializable("kernel_summary", self.kernel_summary)
        _validate_json_serializable("baseline_summary", self.baseline_summary)
        _validate_json_serializable("identifiability_warnings", self.identifiability_warnings)
        _validate_json_serializable("bootstrap_summary", self.bootstrap_summary)
        _validate_json_serializable("metadata", self.metadata)


@dataclass(frozen=True)
class FeatureEvidenceReport:
    """Aggregated evidence summary across generated features."""

    feature_evidence: tuple[FeatureEvidence, ...]
    feature_count: int
    kernel_count: int
    source_columns: tuple[str, ...]
    warning_summary: dict[str, int]
    evidence_summary_by_kernel: dict[str, dict[str, int]]
    evidence_summary_by_feature_family: dict[str, dict[str, int]]

    def __post_init__(self) -> None:
        if self.feature_count < 0:
            raise ValueError("feature_count must be >= 0.")
        if self.kernel_count < 0:
            raise ValueError("kernel_count must be >= 0.")
        if self.feature_count != len(self.feature_evidence):
            raise ValueError("feature_count must match len(feature_evidence).")
        if len(set(self.source_columns)) != len(self.source_columns):
            raise ValueError("source_columns must be unique.")
        for source_col in self.source_columns:
            _require_non_empty_name("source_columns", source_col)
        _validate_json_serializable("warning_summary", self.warning_summary)
        _validate_json_serializable(
            "evidence_summary_by_kernel", self.evidence_summary_by_kernel
        )
        _validate_json_serializable(
            "evidence_summary_by_feature_family",
            self.evidence_summary_by_feature_family,
        )


@dataclass(frozen=True)
class BootstrapResult:
    """Bootstrap run artifacts and outcomes for candidate uncertainty reporting."""

    n_bootstrap: int
    n_succeeded: int
    n_failed: int
    failures: tuple[dict[str, Any], ...]
    weight_samples: tuple[BootstrapWeightSample, ...]
    parameter_samples: tuple[BootstrapParameterSample, ...]
    lag_summary_samples: tuple[BootstrapLagSummarySample, ...]
    family_selection_counts: dict[str, int]
    warnings: tuple[str, ...]
    bootstrap_config: dict[str, Any]
