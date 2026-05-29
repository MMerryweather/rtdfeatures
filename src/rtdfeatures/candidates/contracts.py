"""Candidate data-class contracts.

These types were originally defined in ``diagnostics.py`` and are re-exported
from there for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import polars as pl

from rtdfeatures.diagnostics._helpers import _require_non_empty_name, _validate_json_serializable
from rtdfeatures.diagnostics.fit import BaselineComparison, KernelFitResult
from rtdfeatures.kernels import Kernel

KernelCandidateType = Literal[
    "fixed_kernel",
    "empirical_learner",
    "parametric_learner",
    "baseline",
]
_ALLOWED_CANDIDATE_TYPES: tuple[KernelCandidateType, ...] = (
    "fixed_kernel",
    "empirical_learner",
    "parametric_learner",
    "baseline",
)
_ALLOWED_SELECTION_METRICS: tuple[str, ...] = ("validation_loss",)


@dataclass(frozen=True)
class KernelCandidate:
    """Serializable descriptor for one kernel candidate configuration."""

    candidate_id: str
    family: str
    candidate_type: KernelCandidateType
    min_lag: str | int | float
    max_lag: str | int | float
    fixed_parameters: dict[str, Any] = field(default_factory=dict)
    learner_parameters: dict[str, Any] = field(default_factory=dict)
    interpretation_hint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "candidate_id", _require_non_empty_name("candidate_id", self.candidate_id)
        )
        object.__setattr__(self, "family", _require_non_empty_name("family", self.family))
        if self.candidate_type not in _ALLOWED_CANDIDATE_TYPES:
            raise ValueError(
                "candidate_type must be one of: " + ", ".join(_ALLOWED_CANDIDATE_TYPES)
            )
        if self.interpretation_hint is not None:
            object.__setattr__(
                self,
                "interpretation_hint",
                _require_non_empty_name("interpretation_hint", self.interpretation_hint),
            )
        _validate_json_serializable("fixed_parameters", self.fixed_parameters)
        _validate_json_serializable("learner_parameters", self.learner_parameters)
        _validate_json_serializable("metadata", self.metadata)
        _validate_json_serializable("min_lag", self.min_lag)
        _validate_json_serializable("max_lag", self.max_lag)
        if self.candidate_type == "fixed_kernel" and not self.fixed_parameters:
            raise ValueError("fixed_kernel candidates require non-empty fixed_parameters.")
        if (
            self.candidate_type in {"empirical_learner", "parametric_learner"}
            and not self.learner_parameters
        ):
            raise ValueError(
                "empirical_learner and parametric_learner candidates require "
                "non-empty learner_parameters."
            )
        if self.candidate_type == "baseline" and self.fixed_parameters:
            raise ValueError("baseline candidates must not include fixed_parameters.")
        if self.candidate_type == "baseline" and self.learner_parameters:
            raise ValueError("baseline candidates must not include learner_parameters.")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable descriptor payload."""
        return {
            "candidate_id": self.candidate_id,
            "family": self.family,
            "candidate_type": self.candidate_type,
            "min_lag": self.min_lag,
            "max_lag": self.max_lag,
            "fixed_parameters": self.fixed_parameters,
            "learner_parameters": self.learner_parameters,
            "interpretation_hint": self.interpretation_hint,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> KernelCandidate:
        """Build a validated candidate descriptor from dict payload."""
        return cls(
            candidate_id=payload["candidate_id"],
            family=payload["family"],
            candidate_type=payload["candidate_type"],
            min_lag=payload["min_lag"],
            max_lag=payload["max_lag"],
            fixed_parameters=payload.get("fixed_parameters", {}),
            learner_parameters=payload.get("learner_parameters", {}),
            interpretation_hint=payload.get("interpretation_hint"),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True)
class KernelCandidateSet:
    """Validated candidate set for one input/target/time triple."""

    candidate_set_id: str
    input_col: str
    target_col: str
    time_col: str
    candidates: tuple[KernelCandidate, ...]
    baseline_names: tuple[str, ...] = ()
    selection_metric: str = "validation_loss"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "candidate_set_id",
            _require_non_empty_name("candidate_set_id", self.candidate_set_id)
        )
        object.__setattr__(
            self, "input_col", _require_non_empty_name("input_col", self.input_col)
        )
        object.__setattr__(
            self, "target_col", _require_non_empty_name("target_col", self.target_col)
        )
        object.__setattr__(
            self, "time_col", _require_non_empty_name("time_col", self.time_col)
        )
        object.__setattr__(
            self, "selection_metric",
            _require_non_empty_name("selection_metric", self.selection_metric)
        )
        if self.selection_metric not in _ALLOWED_SELECTION_METRICS:
            raise ValueError(
                "selection_metric must be one of: " + ", ".join(_ALLOWED_SELECTION_METRICS)
            )
        if not self.candidates:
            raise ValueError("KernelCandidateSet requires at least one candidate.")
        ids = [candidate.candidate_id for candidate in self.candidates]
        if len(set(ids)) != len(ids):
            duplicates: list[str] = []
            seen: set[str] = set()
            for candidate_id in ids:
                if candidate_id in seen and candidate_id not in duplicates:
                    duplicates.append(candidate_id)
                seen.add(candidate_id)
            raise ValueError(
                "KernelCandidateSet requires unique candidate_id values; duplicates: "
                + ", ".join(duplicates)
            )
        cleaned_baselines: list[str] = []
        for name in self.baseline_names:
            cleaned_baselines.append(_require_non_empty_name("baseline_names", name))
        if len(set(cleaned_baselines)) != len(cleaned_baselines):
            raise ValueError("baseline_names must be unique.")
        object.__setattr__(self, "baseline_names", tuple(cleaned_baselines))
        candidate_baseline_families = {
            candidate.family
            for candidate in self.candidates
            if candidate.candidate_type == "baseline"
        }
        missing_baselines = sorted(
            set(cleaned_baselines).difference(candidate_baseline_families)
        )
        if missing_baselines:
            raise ValueError(
                "baseline_names must reference candidate baseline families; missing: "
                + ", ".join(missing_baselines)
            )
        _validate_json_serializable("metadata", self.metadata)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable payload for candidate-set transport."""
        return {
            "candidate_set_id": self.candidate_set_id,
            "input_col": self.input_col,
            "target_col": self.target_col,
            "time_col": self.time_col,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "baseline_names": list(self.baseline_names),
            "selection_metric": self.selection_metric,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> KernelCandidateSet:
        """Build validated candidate set from serialized payload."""
        return cls(
            candidate_set_id=payload["candidate_set_id"],
            input_col=payload["input_col"],
            target_col=payload["target_col"],
            time_col=payload["time_col"],
            candidates=tuple(
                KernelCandidate.from_dict(item) for item in payload.get("candidates", [])
            ),
            baseline_names=tuple(payload.get("baseline_names", [])),
            selection_metric=payload.get("selection_metric", "validation_loss"),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True)
class KernelComparisonConfig:
    """Shared candidate-comparison context for fair candidate bootstrapping."""

    loss: Literal["huber", "mse"] = "huber"
    huber_delta: float = 1.0
    validation_fraction: float = 0.2
    loss_tolerance_fraction: float = 0.02

    def __post_init__(self) -> None:
        if self.loss not in {"huber", "mse"}:
            raise ValueError("loss must be either 'huber' or 'mse'.")
        if self.huber_delta <= 0.0:
            raise ValueError("huber_delta must be strictly positive.")
        if self.validation_fraction <= 0.0 or self.validation_fraction >= 0.5:
            raise ValueError("validation_fraction must be in (0.0, 0.5).")
        if self.loss_tolerance_fraction < 0.0:
            raise ValueError("loss_tolerance_fraction must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "huber_delta": self.huber_delta,
            "validation_fraction": self.validation_fraction,
            "loss_tolerance_fraction": self.loss_tolerance_fraction,
        }


@dataclass(frozen=True)
class KernelFamilyFitResult:
    """One candidate outcome; keeps fit result optional for non-learner rows."""

    candidate: KernelCandidate
    fit_result: KernelFitResult | None
    succeeded: bool
    error: str | None
    is_parametric: bool
    is_empirical: bool
    is_baseline: bool
    n_parameters: int | None = None
    validation_loss: float | None = None
    train_loss: float | None = None
    warning_codes: tuple[str, ...] = ()
    evaluated_fixed_kernel: Kernel | None = None
    fixed_baseline_comparison: BaselineComparison | None = None
    evaluation_provenance: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.succeeded and self.error is not None:
            raise ValueError("succeeded results must not include an error message.")
        if not self.succeeded and not self.error:
            raise ValueError("failed results must include an explicit error message.")
        if self.fit_result is not None and not self.succeeded:
            raise ValueError("failed results must not include fit_result.")
        if self.n_parameters is not None and self.n_parameters < 0:
            raise ValueError("n_parameters must be >= 0 when provided.")
        if self.validation_loss is not None and not self.succeeded:
            raise ValueError("failed results must not include validation_loss.")
        if self.train_loss is not None and not self.succeeded:
            raise ValueError("failed results must not include train_loss.")
        if self.evaluated_fixed_kernel is not None and self.fit_result is not None:
            raise ValueError("evaluated_fixed_kernel must be None when fit_result is present.")
        if self.fixed_baseline_comparison is not None and self.fit_result is not None:
            raise ValueError(
                "fixed_baseline_comparison must be None when fit_result is present."
            )


@dataclass(frozen=True)
class KernelComparisonResult:
    """Aggregate candidate outcomes and comparison table for one candidate set."""

    candidate_set: KernelCandidateSet
    family_results: tuple[KernelFamilyFitResult, ...]
    comparison_table: pl.DataFrame
    warnings: tuple[str, ...] = ()
    selection_summary: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KernelSelectionResult:
    """Optional selected-kernel summary for explicit kernel-only selection calls."""

    selected_candidate_id: str | None
    selected_kernel: Kernel | None
    selected_fit_result: KernelFitResult | None
    selection_reason: str | None
    selection_warnings: tuple[str, ...]
    all_candidates: KernelComparisonResult

    def __post_init__(self) -> None:
        if self.selected_candidate_id is not None:
            object.__setattr__(
                self,
                "selected_candidate_id",
                _require_non_empty_name("selected_candidate_id", self.selected_candidate_id),
            )
        if self.selected_candidate_id is None and (
            self.selected_kernel is not None or self.selected_fit_result is not None
        ):
            raise ValueError(
                "selected_kernel and selected_fit_result must be None when "
                "selected_candidate_id is None."
            )
