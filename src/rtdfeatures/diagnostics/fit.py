"""Fit-time diagnostic data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rtdfeatures.kernels import Kernel


@dataclass(frozen=True)
class FitDiagnostics:
    """Fit-time diagnostics used for kernel quality interpretation."""

    train_loss: float
    validation_loss: float
    input_variance: float
    target_variance: float
    kernel_weight_sum: float
    mean_lag: float
    p50_lag: float
    p90_lag: float
    tail_mass: float
    boundary_mass_fraction: float


@dataclass(frozen=True)
class IdentifiabilityReport:
    """Warnings and signals for kernel identifiability confidence."""

    warnings: tuple[str, ...]
    is_reliable: bool
    warning_codes: tuple[str, ...] = ()
    warning_severity_by_code: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BaselineComparison:
    """Validation-loss comparison against baseline methods."""

    no_lag_validation_loss: float
    best_single_lag_validation_loss: float
    learned_validation_loss: float
    uniform_validation_loss: float | None = None
    exponential_validation_loss: float | None = None
    primary_ranking_metric: str = "validation_loss"
    summary_by_baseline: dict[str, dict[str, float | bool]] = field(default_factory=dict)


@dataclass(frozen=True)
class KernelShapeSummary:
    """Compact concentration/spread summary for learned kernel weights."""

    normalized_entropy: float
    max_weight: float
    min_weight: float
    concentration_hhi: float
    effective_lag_count: float


@dataclass(frozen=True)
class FitDataCoverageSummary:
    """Coverage summary for fit windows retained after filtering."""

    total_rows: int
    valid_windows: int
    train_windows: int
    validation_windows: int
    retained_row_fraction: float
    retained_window_fraction: float


@dataclass(frozen=True)
class KernelFitResult:
    """Primary learner result with kernel and diagnostic artifacts."""

    kernel: Kernel
    fit_diagnostics: FitDiagnostics
    identifiability_report: IdentifiabilityReport
    baseline_comparison: BaselineComparison
    kernel_shape_summary: KernelShapeSummary | None = None
    fit_data_coverage_summary: FitDataCoverageSummary | None = None
    fit_provenance: dict[str, Any] | None = None


@dataclass(frozen=True)
class SharedPairFitResult:
    """Pair-level shared learner fit outcome."""

    pair_id: str
    input_col: str
    target_col: str
    fit_result: KernelFitResult | None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """Return whether this pair produced a fit result."""
        return self.fit_result is not None and self.error is None


@dataclass(frozen=True)
class SharedKernelFitResult:
    """Aggregate shared-learning fit result ordered by input/target pairing."""

    pairs: tuple[SharedPairFitResult, ...]

    def __post_init__(self) -> None:
        pair_ids = [pair.pair_id for pair in self.pairs]
        if len(set(pair_ids)) != len(pair_ids):
            duplicates: list[str] = []
            seen: set[str] = set()
            for pair_id in pair_ids:
                if pair_id in seen and pair_id not in duplicates:
                    duplicates.append(pair_id)
                seen.add(pair_id)
            raise ValueError(
                "SharedKernelFitResult requires unique pair_id values; duplicates: "
                + ", ".join(duplicates)
            )

    @staticmethod
    def make_pair_id(input_col: str, target_col: str, *, pair_name: str | None = None) -> str:
        """Return deterministic pair id from explicit name or input/target names."""
        if pair_name is not None:
            cleaned = pair_name.strip()
            if not cleaned:
                raise ValueError("pair_name must not be empty when provided.")
            return cleaned
        return f"{input_col}->{target_col}"

    def pair_ids(self) -> tuple[str, ...]:
        """Return ordered pair identifiers."""
        return tuple(pair.pair_id for pair in self.pairs)

    def _pair_index(self) -> dict[str, SharedPairFitResult]:
        """Build pair lookup while rejecting duplicate ids."""
        indexed: dict[str, SharedPairFitResult] = {}
        for pair in self.pairs:
            if pair.pair_id in indexed:
                raise ValueError(
                    f"Duplicate pair_id detected in SharedKernelFitResult: '{pair.pair_id}'."
                )
            indexed[pair.pair_id] = pair
        return indexed

    def get_pair(self, pair_id: str) -> SharedPairFitResult:
        """Return pair outcome by identifier."""
        try:
            return self._pair_index()[pair_id]
        except KeyError as exc:
            raise KeyError(f"Unknown pair_id: {pair_id}.") from exc

    def get_pair_result(self, pair_id: str) -> KernelFitResult:
        """Return successful per-pair fit result, else raise a clear error."""
        pair = self.get_pair(pair_id)
        if not pair.succeeded or pair.fit_result is None:
            detail = pair.error or "fit failed without an explicit error message."
            raise ValueError(f"Pair '{pair_id}' did not produce a fit result: {detail}")
        return pair.fit_result

    def summary(self) -> dict[str, dict[str, Any]]:
        """Return per-pair summary with kernel and diagnostics when available."""
        result: dict[str, dict[str, Any]] = {}
        for pair in self._pair_index().values():
            if pair.succeeded and pair.fit_result is not None:
                result[pair.pair_id] = {
                    "status": "ok",
                    "input_col": pair.input_col,
                    "target_col": pair.target_col,
                    "kernel": pair.fit_result.kernel.summary(),
                    "validation_loss": pair.fit_result.fit_diagnostics.validation_loss,
                    "warnings": pair.fit_result.identifiability_report.warnings,
                }
            else:
                result[pair.pair_id] = {
                    "status": "failed",
                    "input_col": pair.input_col,
                    "target_col": pair.target_col,
                    "error": pair.error or "unknown fit failure",
                }
        return result

    def to_kernels(self, *, names: dict[str, str] | None = None) -> dict[str, Kernel]:
        """Return successful learned kernels keyed by pair id or explicit names.

        By default, each kernel key uses the pair id (`"{input}->{target}"`).
        Callers can override names with a `names` mapping keyed by pair id.
        """
        kernels: dict[str, Kernel] = {}
        for pair in self.pairs:
            fit = pair.fit_result
            if not pair.succeeded or fit is None:
                detail = pair.error or "fit failed without an explicit error message."
                raise ValueError(
                    "Cannot convert shared fit result because pair "
                    f"'{pair.pair_id}' failed: {detail}"
                )
            kernel_name = (
                names.get(pair.pair_id, pair.pair_id)
                if names is not None
                else pair.pair_id
            )
            if not kernel_name:
                raise ValueError(f"Kernel name for pair '{pair.pair_id}' must be non-empty.")
            if kernel_name in kernels:
                raise ValueError(
                    f"Kernel name collision while converting shared fit result: '{kernel_name}'."
                )
            kernels[kernel_name] = fit.kernel
        return kernels
