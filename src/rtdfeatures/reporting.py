"""Lightweight presentation helpers for fit diagnostics reporting.

These helpers consume public diagnostics/result objects and return compact
string, dictionary, and Polars table summaries. They do not mutate the input
objects and do not recommend downstream model choices.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from rtdfeatures.diagnostics import BaselineComparison, IdentifiabilityReport, KernelFitResult

DEFAULT_PARAMETRIC_DIAGNOSTIC_WARNING_MARGIN = 0.05


def baseline_comparison_table(
    baseline: BaselineComparison,
    *,
    include_optional_baselines: bool = True,
) -> pl.DataFrame:
    """Return compact baseline-comparison rows sorted in a deterministic order."""
    rows: list[dict[str, Any]] = []
    learned_loss = baseline.learned_validation_loss

    ordered_candidates: list[tuple[str, float | None]] = [
        ("learned", learned_loss),
        ("no_lag", baseline.no_lag_validation_loss),
        ("best_single_lag", baseline.best_single_lag_validation_loss),
    ]
    if include_optional_baselines:
        ordered_candidates.extend(
            [
                ("uniform", baseline.uniform_validation_loss),
                ("exponential", baseline.exponential_validation_loss),
            ]
        )

    for name, loss in ordered_candidates:
        if loss is None:
            continue
        # Positive means baseline validates better than learned (lower loss).
        delta_vs_learned = (learned_loss - loss) / max(abs(learned_loss), 1e-12)
        rows.append(
            {
                "baseline": name,
                "validation_loss": float(loss),
                "learned_validation_loss": float(learned_loss),
                "delta_fraction_vs_learned": float(delta_vs_learned),
                "beats_learned": bool(name != "learned" and loss < learned_loss),
                "is_learned": bool(name == "learned"),
            }
        )

    return pl.DataFrame(rows)


def baseline_comparison_compact_dict(baseline: BaselineComparison) -> dict[str, Any]:
    """Return a compact dictionary summary for baseline diagnostics."""
    return {
        "primary_ranking_metric": baseline.primary_ranking_metric,
        "learned_validation_loss": baseline.learned_validation_loss,
        "available_baselines": tuple(baseline_comparison_table(baseline)["baseline"].to_list()),
        "summary_by_baseline": baseline.summary_by_baseline,
    }


def baseline_comparison_compact_text(baseline: BaselineComparison) -> str:
    """Return one-line text summary without selecting a winner."""
    table = baseline_comparison_table(baseline)
    parts = [
        f"{row['baseline']}={row['validation_loss']:.6g}"
        for row in table.select(["baseline", "validation_loss"]).to_dicts()
    ]
    return "validation losses: " + ", ".join(parts)


def warning_summary_table(report: IdentifiabilityReport) -> pl.DataFrame:
    """Return warning rows including code and severity when available."""
    rows: list[dict[str, Any]] = []
    for idx, message in enumerate(report.warnings):
        code = report.warning_codes[idx] if idx < len(report.warning_codes) else ""
        rows.append(
            {
                "warning_index": idx,
                "warning_code": code,
                "warning_severity": report.warning_severity_by_code.get(code, "") if code else "",
                "warning_message": message,
            }
        )

    return pl.DataFrame(
        rows,
        schema={
            "warning_index": pl.Int64,
            "warning_code": pl.String,
            "warning_severity": pl.String,
            "warning_message": pl.String,
        },
    )


def warning_summary_compact_dict(report: IdentifiabilityReport) -> dict[str, Any]:
    """Return compact warning summary dictionary."""
    return {
        "is_reliable": report.is_reliable,
        "warning_count": len(report.warnings),
        "warning_codes": report.warning_codes,
        "warning_severity_by_code": dict(report.warning_severity_by_code),
    }


def warning_summary_compact_text(report: IdentifiabilityReport) -> str:
    """Return one-line warning summary text."""
    if not report.warnings:
        return "warnings: none"
    if report.warning_codes:
        pairs = [
            f"{code}:{report.warning_severity_by_code.get(code, 'unknown')}"
            for code in report.warning_codes
        ]
        return "warnings: " + ", ".join(pairs)
    return f"warnings: {len(report.warnings)}"


def fit_result_warning_summary_table(result: KernelFitResult) -> pl.DataFrame:
    """Return compact warning summary for a full fit result."""
    return warning_summary_table(result.identifiability_report)


def fit_result_baseline_summary_table(result: KernelFitResult) -> pl.DataFrame:
    """Return compact baseline summary for a full fit result."""
    return baseline_comparison_table(result.baseline_comparison)


def learner_diagnostic_comparison_table(
    fit_results_by_family: dict[str, KernelFitResult],
    *,
    include_baselines: bool = True,
) -> pl.DataFrame:
    """Return deterministic diagnostic comparison rows across learner families.

    This helper is presentation-only: it compares provided fit results and does
    not refit, rank, or choose a final learner family.
    """
    _assert_comparable_fit_results(fit_results_by_family)
    rows: list[dict[str, Any]] = []
    ordered_families = _ordered_families(fit_results_by_family)

    for family in ordered_families:
        fit_result = fit_results_by_family[family]
        learned_loss = float(fit_result.fit_diagnostics.validation_loss)
        rows.append(
            {
                "learner_family": family,
                "row_type": "learned",
                "candidate": family,
                "validation_loss": learned_loss,
                "delta_fraction_vs_learned": 0.0,
                "is_parametric_family": bool(family in {"gamma", "exponential"}),
            }
        )
        if not include_baselines:
            continue

        baseline_table = baseline_comparison_table(fit_result.baseline_comparison)
        for baseline_row in baseline_table.to_dicts():
            baseline_name = str(baseline_row["baseline"])
            if baseline_name == "learned":
                continue
            rows.append(
                {
                    "learner_family": family,
                    "row_type": "baseline",
                    "candidate": baseline_name,
                    "validation_loss": float(baseline_row["validation_loss"]),
                    "delta_fraction_vs_learned": float(
                        baseline_row["delta_fraction_vs_learned"]
                    ),
                    "is_parametric_family": bool(family in {"gamma", "exponential"}),
                }
            )

    return pl.DataFrame(
        rows,
        schema={
            "learner_family": pl.String,
            "row_type": pl.String,
            "candidate": pl.String,
            "validation_loss": pl.Float64,
            "delta_fraction_vs_learned": pl.Float64,
            "is_parametric_family": pl.Boolean,
        },
    )


def learner_diagnostic_warning_table(
    fit_results_by_family: dict[str, KernelFitResult],
    *,
    warning_margin: float = DEFAULT_PARAMETRIC_DIAGNOSTIC_WARNING_MARGIN,
) -> pl.DataFrame:
    """Return deterministic warnings for weak parametric-vs-reference diagnostics.

    Lower validation loss remains better. This helper emits warning rows only
    and does not recommend or select a final learner family.
    """
    if warning_margin < 0.0:
        raise ValueError("warning_margin must be non-negative.")
    _assert_comparable_fit_results(fit_results_by_family)

    rows: list[dict[str, Any]] = []
    simplex_result = fit_results_by_family.get("simplex")
    simplex_loss = (
        float(simplex_result.fit_diagnostics.validation_loss)
        if simplex_result is not None
        else None
    )
    for family in _ordered_families(fit_results_by_family):
        if family not in {"gamma", "exponential"}:
            continue
        fit_result = fit_results_by_family[family]
        learned_loss = float(fit_result.fit_diagnostics.validation_loss)

        if simplex_loss is not None:
            delta_vs_simplex = _delta_fraction_worse(
                candidate_loss=learned_loss,
                reference_loss=simplex_loss,
            )
            if delta_vs_simplex >= warning_margin:
                rows.append(
                    {
                        "learner_family": family,
                        "warning_code": "PARAMETRIC_WORSE_THAN_SIMPLEX",
                        "reference": "simplex",
                        "warning_margin": float(warning_margin),
                        "delta_fraction_worse": float(delta_vs_simplex),
                        "candidate_validation_loss": learned_loss,
                        "reference_validation_loss": simplex_loss,
                    }
                )

        baseline = fit_result.baseline_comparison
        simple_baselines = {
            "no_lag": float(baseline.no_lag_validation_loss),
            "best_single_lag": float(baseline.best_single_lag_validation_loss),
        }
        for baseline_name, baseline_loss in simple_baselines.items():
            delta_vs_baseline = _delta_fraction_worse(
                candidate_loss=learned_loss,
                reference_loss=baseline_loss,
            )
            if delta_vs_baseline >= warning_margin:
                rows.append(
                    {
                        "learner_family": family,
                        "warning_code": "PARAMETRIC_WORSE_THAN_BASELINE",
                        "reference": baseline_name,
                        "warning_margin": float(warning_margin),
                        "delta_fraction_worse": float(delta_vs_baseline),
                        "candidate_validation_loss": learned_loss,
                        "reference_validation_loss": baseline_loss,
                    }
                )

    return pl.DataFrame(
        rows,
        schema={
            "learner_family": pl.String,
            "warning_code": pl.String,
            "reference": pl.String,
            "warning_margin": pl.Float64,
            "delta_fraction_worse": pl.Float64,
            "candidate_validation_loss": pl.Float64,
            "reference_validation_loss": pl.Float64,
        },
    )


def _ordered_families(fit_results_by_family: dict[str, KernelFitResult]) -> list[str]:
    preferred = ("simplex", "gamma", "exponential")
    preferred_present = [name for name in preferred if name in fit_results_by_family]
    extras = sorted(name for name in fit_results_by_family if name not in preferred)
    return preferred_present + extras


def _delta_fraction_worse(*, candidate_loss: float, reference_loss: float) -> float:
    return (candidate_loss - reference_loss) / max(abs(reference_loss), 1e-12)


def _assert_comparable_fit_results(fit_results_by_family: dict[str, KernelFitResult]) -> None:
    """Raise when fit provenance indicates incompatible validation contexts."""
    ordered_families = _ordered_families(fit_results_by_family)
    if len(ordered_families) <= 1:
        return

    provenance_keys = ("validation_fraction", "dt_seconds", "total_valid_windows")
    loss_config_keys = ("loss_name", "loss")
    reference_family = ordered_families[0]
    reference_provenance = fit_results_by_family[reference_family].fit_provenance
    if not isinstance(reference_provenance, dict):
        return

    for family in ordered_families[1:]:
        family_provenance = fit_results_by_family[family].fit_provenance
        if not isinstance(family_provenance, dict):
            continue
        for key in provenance_keys:
            if key not in reference_provenance or key not in family_provenance:
                continue
            reference_value = reference_provenance[key]
            candidate_value = family_provenance[key]
            if reference_value != candidate_value:
                raise ValueError(
                    "Incompatible fit results for learner diagnostic comparison: "
                    f"fit_provenance['{key}'] mismatch between '{reference_family}' "
                    f"({reference_value!r}) and '{family}' ({candidate_value!r})."
                )
        reference_loss_name = _first_present_str(reference_provenance, loss_config_keys)
        candidate_loss_name = _first_present_str(family_provenance, loss_config_keys)
        if (
            reference_loss_name is not None
            and candidate_loss_name is not None
            and reference_loss_name != candidate_loss_name
        ):
            raise ValueError(
                "Incompatible fit results for learner diagnostic comparison: "
                f"loss mismatch between '{reference_family}' ({reference_loss_name!r}) "
                f"and '{family}' ({candidate_loss_name!r})."
            )
        if (
            reference_loss_name == "huber"
            and candidate_loss_name == "huber"
            and "huber_delta" in reference_provenance
            and "huber_delta" in family_provenance
            and reference_provenance["huber_delta"] != family_provenance["huber_delta"]
        ):
            raise ValueError(
                "Incompatible fit results for learner diagnostic comparison: "
                f"fit_provenance['huber_delta'] mismatch between '{reference_family}' "
                f"({reference_provenance['huber_delta']!r}) and '{family}' "
                f"({family_provenance['huber_delta']!r})."
            )


def _first_present_str(provenance: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = provenance.get(key)
        if value is None:
            continue
        return str(value)
    return None
