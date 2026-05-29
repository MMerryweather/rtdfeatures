"""Kernel candidate selection logic."""

from __future__ import annotations

from typing import Any

import numpy as np

from rtdfeatures.candidates.contracts import (
    BaselineComparison,
    Kernel,
    KernelComparisonResult,
    KernelFamilyFitResult,
    KernelSelectionResult,
)
from rtdfeatures.candidates.fitting import (
    DEFAULT_SELECTION_TOLERANCE,
)


def select_kernel_candidate(
    comparison_result: KernelComparisonResult,
    *,
    loss_tolerance_fraction: float = DEFAULT_SELECTION_TOLERANCE,
) -> KernelSelectionResult:
    """Optionally select a conservative kernel candidate from comparison results."""
    if loss_tolerance_fraction < 0.0:
        raise ValueError("loss_tolerance_fraction must be non-negative.")

    ordered_results = _ordered_family_results(comparison_result.family_results)
    mismatch_warning = _selection_context_mismatch_warning(ordered_results)
    if mismatch_warning is not None:
        return KernelSelectionResult(
            selected_candidate_id=None,
            selected_kernel=None,
            selected_fit_result=None,
            selection_reason=None,
            selection_warnings=(
                "No strong recommendation; candidate validation contexts are not comparable.",
                mismatch_warning,
            ),
            all_candidates=comparison_result,
        )

    ranked = [_rankable_selection_row(result) for result in ordered_results]
    passed = [row for row in ranked if row["passes_all_gates"]]

    if not passed:
        fixed_ineligible: list[str] = []
        for row in ranked:
            warning = _fixed_selection_ineligible_warning(row["result"])
            if warning is not None:
                fixed_ineligible.append(warning)
        warnings = ["No strong recommendation; all candidates failed conservative gates."]
        warnings.extend(fixed_ineligible)
        return KernelSelectionResult(
            selected_candidate_id=None,
            selected_kernel=None,
            selected_fit_result=None,
            selection_reason=None,
            selection_warnings=tuple(warnings),
            all_candidates=comparison_result,
        )

    best_loss = min(float(row["validation_loss"]) for row in passed)
    tolerance_pool = [
        row
        for row in passed
        if _loss_delta_fraction(best_loss, float(row["validation_loss"])) <= loss_tolerance_fraction
    ]
    best = sorted(
        tolerance_pool,
        key=lambda row: (
            _simplicity_rank(row["result"]),
            float(row["validation_loss"]),
            row["result"].candidate.candidate_id,
        ),
    )[0]
    selected_fit_result = best["result"].fit_result
    selected_kernel = _selected_kernel_for_result(best["result"])
    selected_candidate_id = str(best["result"].candidate.candidate_id)
    near_ties = [row for row in tolerance_pool if row["result"] is not best["result"]]
    selection_warnings: list[str] = []
    if near_ties:
        selection_warnings.append(
            "Multiple candidates are within tolerance; selecting simpler candidate "
            "deterministically."
        )
    reason = (
        "Selected by conservative ranking: passed reliability gates, then simplicity tie-break, "
        "then validation loss."
    )
    return KernelSelectionResult(
        selected_candidate_id=selected_candidate_id,
        selected_kernel=selected_kernel,
        selected_fit_result=selected_fit_result,
        selection_reason=reason,
        selection_warnings=tuple(selection_warnings),
        all_candidates=comparison_result,
    )


def _selected_kernel_for_result(result: KernelFamilyFitResult) -> Kernel | None:
    fit_result = result.fit_result
    if fit_result is not None:
        return fit_result.kernel
    if result.candidate.candidate_type != "fixed_kernel":
        return None
    return result.evaluated_fixed_kernel


def _fixed_selection_ineligible_warning(result: KernelFamilyFitResult) -> str | None:
    if result.candidate.candidate_type != "fixed_kernel":
        return None
    missing: list[str] = []
    if result.evaluated_fixed_kernel is None:
        missing.append("evaluated_fixed_kernel")
    if result.fixed_baseline_comparison is None:
        missing.append("fixed_baseline_comparison")
    if not isinstance(result.evaluation_provenance, dict):
        missing.append("evaluation_provenance")
    if not missing:
        return None
    return (
        f"Fixed-kernel candidate '{result.candidate.candidate_id}' is not selection-eligible; "
        f"missing required evidence: {', '.join(missing)}."
    )


def _ordered_family_results(
    family_results: tuple[KernelFamilyFitResult, ...],
) -> list[KernelFamilyFitResult]:
    return sorted(
        family_results,
        key=lambda result: (
            not result.succeeded,
            not _is_finite_loss(result.validation_loss),
            _loss_or_inf(result.validation_loss),
            _simplicity_rank(result),
            result.candidate.candidate_id,
        ),
    )


def _rankable_selection_row(result: KernelFamilyFitResult) -> dict[str, Any]:
    validation_loss = (
        float(result.validation_loss) if result.validation_loss is not None else float("inf")
    )
    baseline_ok = _baseline_warnings_acceptable(result)
    identifiability_ok = _identifiability_warnings_acceptable(result)
    fixed_evidence_ok = _fixed_selection_evidence_acceptable(result)
    return {
        "result": result,
        "validation_loss": validation_loss,
        "baseline_ok": baseline_ok,
        "identifiability_ok": identifiability_ok,
        "fixed_evidence_ok": fixed_evidence_ok,
        "passes_all_gates": bool(
            result.succeeded
            and _is_finite_loss(result.validation_loss)
            and baseline_ok
            and identifiability_ok
            and fixed_evidence_ok
            and not result.is_baseline
        ),
        "simplicity_rank": _simplicity_rank(result),
    }


def _baseline_warnings_acceptable(result: KernelFamilyFitResult) -> bool:
    baseline: BaselineComparison | None
    if result.fit_result is not None:
        baseline = result.fit_result.baseline_comparison
    elif result.candidate.candidate_type == "fixed_kernel":
        baseline = result.fixed_baseline_comparison
    else:
        return False
    if baseline is None:
        return False
    return (
        result.validation_loss is not None
        and result.validation_loss < float(baseline.no_lag_validation_loss)
        and result.validation_loss < float(baseline.best_single_lag_validation_loss)
    )


def _identifiability_warnings_acceptable(result: KernelFamilyFitResult) -> bool:
    if result.fit_result is None:
        if result.candidate.candidate_type == "fixed_kernel":
            return result.evaluated_fixed_kernel is not None
        return False
    return bool(result.fit_result.identifiability_report.is_reliable)


def _fixed_selection_evidence_acceptable(result: KernelFamilyFitResult) -> bool:
    if result.candidate.candidate_type != "fixed_kernel":
        return True
    return (
        result.evaluated_fixed_kernel is not None
        and result.fixed_baseline_comparison is not None
        and isinstance(result.evaluation_provenance, dict)
    )


def _selection_context_mismatch_warning(
    ordered_results: list[KernelFamilyFitResult],
) -> str | None:
    comparable_ids: list[str] = []
    signatures: dict[str, tuple[Any, ...]] = {}
    for result in ordered_results:
        if (
            not result.succeeded
            or result.is_baseline
            or not _is_finite_loss(result.validation_loss)
        ):
            continue
        comparable_ids.append(result.candidate.candidate_id)
        signature = _selection_context_signature(result)
        if signature is None:
            continue
        signatures[result.candidate.candidate_id] = signature
    if len(comparable_ids) > 1 and 0 < len(signatures) < len(comparable_ids):
        missing = sorted(
            candidate_id for candidate_id in comparable_ids if candidate_id not in signatures
        )
        return (
            "Missing evaluation context/provenance signature for candidate(s): "
            f"{', '.join(missing)}."
        )
    if len(signatures) <= 1:
        return None
    reference_id = sorted(signatures)[0]
    reference_sig = signatures[reference_id]
    mismatched = sorted(
        candidate_id for candidate_id, signature in signatures.items() if signature != reference_sig
    )
    if not mismatched:
        return None
    return (
        "Incompatible evaluation context/provenance across candidates: "
        f"{reference_id} vs {', '.join(mismatched)}."
    )


def _selection_context_signature(result: KernelFamilyFitResult) -> tuple[Any, ...] | None:
    if result.fit_result is not None and isinstance(result.fit_result.fit_provenance, dict):
        provenance = result.fit_result.fit_provenance
    elif isinstance(result.evaluation_provenance, dict):
        provenance = result.evaluation_provenance
    else:
        return None
    loss_name = _first_present_str(provenance, ("loss_name", "loss"))
    dt_seconds = provenance.get("dt_seconds")
    validation_fraction = provenance.get("validation_fraction")
    total_valid_windows = provenance.get("total_valid_windows")
    validation_windows = provenance.get("validation_windows")
    if validation_windows is None:
        validation_windows = provenance.get("validation_rows")
    if (
        dt_seconds is None
        or validation_fraction is None
        or total_valid_windows is None
        or validation_windows is None
        or loss_name is None
    ):
        return None
    huber_delta = provenance.get("huber_delta") if loss_name == "huber" else None
    if loss_name == "huber" and huber_delta is None:
        return None
    return (
        dt_seconds,
        validation_fraction,
        total_valid_windows,
        validation_windows,
        loss_name,
        huber_delta,
    )


def _first_present_str(provenance: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = provenance.get(key)
        if value is None:
            continue
        return str(value)
    return None


def _simplicity_rank(result: KernelFamilyFitResult) -> int:
    if result.is_baseline:
        return 4
    if result.candidate.candidate_type == "fixed_kernel":
        return 1
    if result.candidate.candidate_type == "empirical_learner":
        return 2
    if result.candidate.candidate_type == "parametric_learner":
        return 3
    return 5


def _is_finite_loss(loss: float | None) -> bool:
    return loss is not None and np.isfinite(loss)


def _loss_delta_fraction(loss_a: float, loss_b: float) -> float:
    return abs(loss_b - loss_a) / max(abs(loss_a), 1e-12)


def _loss_or_inf(loss: float | None) -> float:
    if loss is None:
        return float("inf")
    return float(loss)
