"""Shared learner identifiability report builder."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rtdfeatures.diagnostics import FitDiagnostics, IdentifiabilityReport


@dataclass(frozen=True)
class IdentifiabilityPolicy:
    """Threshold policy for identifiability warning rules."""

    flat_variance_threshold: float = 1e-8
    validation_gap_ratio: float = 2.0
    baseline_improvement_margin: float = 0.05
    boundary_mass_threshold: float = 0.35
    diffuse_entropy_fraction: float = 0.85
    diffuse_max_weight_threshold: float = 0.20


WARNING_DEFINITIONS = {
    "INPUT_TOO_FLAT": ("Input is too flat.", "high"),
    "TARGET_TOO_FLAT": ("Target signal is too flat.", "high"),
    "WEAK_NO_LAG_IMPROVEMENT": ("Target signal appears noisy or weakly explained.", "medium"),
    "LARGE_VALIDATION_GAP": ("Validation loss is much worse than training loss.", "high"),
    "BOUNDARY_PILED_KERNEL": ("Kernel piles mass at the lag boundary.", "medium"),
    "DIFFUSE_KERNEL": ("Kernel is too diffuse to interpret confidently.", "medium"),
    "BEST_SINGLE_LAG_BEATS_LEARNED": ("best_single_lag beats the learned kernel.", "medium"),
    "UNIFORM_BASELINE_BEATS_LEARNED": ("uniform baseline beats the learned kernel.", "medium"),
    "EXPONENTIAL_BASELINE_BEATS_LEARNED": (
        "exponential baseline beats the learned kernel.",
        "medium",
    ),
}


def build_identifiability_report(
    *,
    fit_diagnostics: FitDiagnostics,
    learned_weights: np.ndarray,
    no_lag_validation_loss: float,
    best_single_lag_validation_loss: float,
    uniform_validation_loss: float | None,
    exponential_validation_loss: float | None,
    policy: IdentifiabilityPolicy = IdentifiabilityPolicy(),
) -> IdentifiabilityReport:
    """Build identifiability warnings from learned fit diagnostics and baselines."""
    warnings: list[str] = []
    warning_codes: list[str] = []
    flat_floor = policy.flat_variance_threshold
    if fit_diagnostics.input_variance < flat_floor:
        _append_warning(warnings, warning_codes, "INPUT_TOO_FLAT")
    if fit_diagnostics.target_variance < flat_floor:
        _append_warning(warnings, warning_codes, "TARGET_TOO_FLAT")

    no_lag_improvement = (
        (no_lag_validation_loss - fit_diagnostics.validation_loss)
        / max(no_lag_validation_loss, flat_floor)
    )
    if no_lag_improvement < policy.baseline_improvement_margin:
        _append_warning(warnings, warning_codes, "WEAK_NO_LAG_IMPROVEMENT")

    gap_ratio = fit_diagnostics.validation_loss / max(fit_diagnostics.train_loss, flat_floor)
    if gap_ratio > policy.validation_gap_ratio:
        _append_warning(warnings, warning_codes, "LARGE_VALIDATION_GAP")

    min_weight = float(learned_weights[0])
    max_weight = float(learned_weights[-1])
    if min_weight >= policy.boundary_mass_threshold or max_weight >= policy.boundary_mass_threshold:
        _append_warning(warnings, warning_codes, "BOUNDARY_PILED_KERNEL")

    if learned_weights.size > 1:
        safe_weights = learned_weights[learned_weights > 0.0]
        entropy = float(-np.sum(safe_weights * np.log(safe_weights)))
        normalized_entropy = entropy / math.log(learned_weights.size)
        max_weight_any = float(np.max(learned_weights))
        if (
            normalized_entropy >= policy.diffuse_entropy_fraction
            and max_weight_any <= policy.diffuse_max_weight_threshold
        ):
            _append_warning(warnings, warning_codes, "DIFFUSE_KERNEL")

    best_single_lag_delta = (
        (fit_diagnostics.validation_loss - best_single_lag_validation_loss)
        / max(fit_diagnostics.validation_loss, flat_floor)
    )
    if best_single_lag_delta >= policy.baseline_improvement_margin:
        _append_warning(warnings, warning_codes, "BEST_SINGLE_LAG_BEATS_LEARNED")
    if _baseline_beats_learned_by_margin(
        baseline_loss=uniform_validation_loss,
        learned_loss=fit_diagnostics.validation_loss,
        flat_floor=flat_floor,
        policy=policy,
    ):
        _append_warning(warnings, warning_codes, "UNIFORM_BASELINE_BEATS_LEARNED")
    if _baseline_beats_learned_by_margin(
        baseline_loss=exponential_validation_loss,
        learned_loss=fit_diagnostics.validation_loss,
        flat_floor=flat_floor,
        policy=policy,
    ):
        _append_warning(warnings, warning_codes, "EXPONENTIAL_BASELINE_BEATS_LEARNED")
    severity_by_code = {code: WARNING_DEFINITIONS[code][1] for code in warning_codes}
    return IdentifiabilityReport(
        warnings=tuple(warnings),
        is_reliable=len(warnings) == 0,
        warning_codes=tuple(warning_codes),
        warning_severity_by_code=severity_by_code,
    )


def _append_warning(warnings: list[str], warning_codes: list[str], code: str) -> None:
    message, _severity = WARNING_DEFINITIONS[code]
    warnings.append(message)
    warning_codes.append(code)


def _baseline_beats_learned_by_margin(
    *,
    baseline_loss: float | None,
    learned_loss: float,
    flat_floor: float,
    policy: IdentifiabilityPolicy,
) -> bool:
    if baseline_loss is None or not np.isfinite(baseline_loss):
        return False
    baseline_delta = (learned_loss - baseline_loss) / max(learned_loss, flat_floor)
    return baseline_delta >= policy.baseline_improvement_margin
