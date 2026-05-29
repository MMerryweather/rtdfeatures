"""Feature evidence generation and reporting."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

from rtdfeatures.diagnostics import (
    FeatureEvidence,
    FeatureEvidenceCompletenessLabel,
    FeatureEvidenceReport,
    FeatureInterpretationLabel,
    KernelFitResult,
)
from rtdfeatures.features.registry import FeatureRegistry, FeatureSpec

if TYPE_CHECKING:
    from rtdfeatures.features.builder import KernelFeatureBuilder


def build_feature_evidence(
    *,
    builder: KernelFeatureBuilder,
    feature_registry: FeatureRegistry,
    fit_result_by_kernel: dict[str, KernelFitResult] | None = None,
    interpretation_by_kernel: dict[str, FeatureInterpretationLabel] | None = None,
    interpretation_by_feature: dict[str, FeatureInterpretationLabel] | None = None,
    candidate_id_by_kernel: dict[str, str] | None = None,
    baseline_summary_by_kernel: dict[str, dict[str, Any]] | None = None,
    bootstrap_summary_by_kernel: dict[str, dict[str, Any]] | None = None,
    metadata_by_kernel: dict[str, dict[str, Any]] | None = None,
    metadata_by_feature: dict[str, dict[str, Any]] | None = None,
) -> FeatureEvidenceReport:
    resolved_feature_names = tuple(spec.name for spec in feature_registry.specs)
    spec_by_name = {spec.name: spec for spec in feature_registry.specs}
    fit_lookup = fit_result_by_kernel or {}
    interpretation_kernel = interpretation_by_kernel or {}
    interpretation_feature = interpretation_by_feature or {}
    candidate_lookup = candidate_id_by_kernel or {}
    baseline_lookup = baseline_summary_by_kernel or {}
    bootstrap_lookup = bootstrap_summary_by_kernel or {}
    kernel_metadata_lookup = metadata_by_kernel or {}
    feature_metadata_lookup = metadata_by_feature or {}

    evidence_items: list[FeatureEvidence] = []
    warning_summary: dict[str, int] = {}
    evidence_summary_by_kernel: dict[str, dict[str, int]] = {}
    evidence_summary_by_feature_family: dict[str, dict[str, int]] = {}

    participating_kernels: set[str] = set()

    for feature_name in resolved_feature_names:
        spec = spec_by_name.get(feature_name)
        if spec is None:
            raise ValueError(
                f"Feature '{feature_name}' is not present in the feature registry."
            )
        kernel_name = spec.kernel_name
        kernel = builder.kernels.get(kernel_name)
        if kernel is None:
            raise ValueError(
                f"Feature '{feature_name}' references unknown kernel '{kernel_name}'."
            )
        participating_kernels.add(kernel_name)
        fit_result = fit_lookup.get(kernel_name)
        warnings = _warning_codes_from_fit(fit_result)
        for code in warnings:
            warning_summary[code] = warning_summary.get(code, 0) + 1

        interpretation = interpretation_feature.get(
            feature_name,
            interpretation_kernel.get(kernel_name, "unknown"),
        )
        candidate_id = candidate_lookup.get(kernel_name)
        baseline_summary = baseline_lookup.get(kernel_name)
        has_comparison_evidence = (
            candidate_id is not None or baseline_summary is not None
        )
        evidence_completeness = _evidence_completeness(
            has_fit_evidence=fit_result is not None,
            has_comparison_evidence=has_comparison_evidence,
            has_bootstrap_evidence=(kernel_name in bootstrap_lookup),
        )
        kernel_summary = kernel.summary()
        kernel_family = str(
            kernel_summary.get(
                "parametric_family",
                kernel.__class__.__name__.removesuffix("Kernel").lower(),
            )
        )
        feature_family = _spec_family_to_feature_family(spec)
        metadata: dict[str, Any] = {}
        metadata.update(kernel_metadata_lookup.get(kernel_name, {}))
        metadata.update(feature_metadata_lookup.get(feature_name, {}))
        evidence = FeatureEvidence(
            feature_name=feature_name,
            source_col=spec.source_col,
            feature_family=feature_family,
            kernel_name=kernel_name,
            kernel_family=kernel_family,
            kernel_summary=kernel_summary,
            fit_result_id=_fit_result_id(fit_result),
            candidate_id=candidate_id,
            baseline_summary=baseline_summary,
            identifiability_warnings=warnings or None,
            bootstrap_summary=bootstrap_lookup.get(kernel_name),
            interpretation=interpretation,
            evidence_completeness=evidence_completeness,
            metadata=metadata,
        )
        evidence_items.append(evidence)

        kernel_summary_counts = evidence_summary_by_kernel.setdefault(kernel_name, {})
        kernel_summary_counts[evidence_completeness] = (
            kernel_summary_counts.get(evidence_completeness, 0) + 1
        )
        family_counts = evidence_summary_by_feature_family.setdefault(
            evidence.feature_family, {}
        )
        family_counts[evidence_completeness] = (
            family_counts.get(evidence_completeness, 0) + 1
        )

    source_columns = sorted({item.source_col for item in evidence_items})
    return FeatureEvidenceReport(
        feature_evidence=tuple(evidence_items),
        feature_count=len(evidence_items),
        kernel_count=len(participating_kernels),
        source_columns=tuple(source_columns),
        warning_summary=warning_summary,
        evidence_summary_by_kernel=evidence_summary_by_kernel,
        evidence_summary_by_feature_family=evidence_summary_by_feature_family,
    )


def feature_evidence_table(report: FeatureEvidenceReport) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for evidence in sorted(report.feature_evidence, key=lambda item: item.feature_name):
        warning_codes = tuple(evidence.identifiability_warnings or ())
        rows.append(
            {
                "feature_name": evidence.feature_name,
                "source_col": evidence.source_col,
                "feature_family": evidence.feature_family,
                "kernel_name": evidence.kernel_name,
                "kernel_family": evidence.kernel_family,
                "interpretation": evidence.interpretation,
                "evidence_completeness": evidence.evidence_completeness,
                "warning_count": len(warning_codes),
                "warning_codes": "|".join(warning_codes),
                "has_fit_evidence": evidence.fit_result_id is not None,
                "has_comparison_evidence": _has_comparison_evidence(evidence),
                "has_bootstrap_evidence": evidence.bootstrap_summary is not None,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "feature_name": pl.String,
            "source_col": pl.String,
            "feature_family": pl.String,
            "kernel_name": pl.String,
            "kernel_family": pl.String,
            "interpretation": pl.String,
            "evidence_completeness": pl.String,
            "warning_count": pl.Int64,
            "warning_codes": pl.String,
            "has_fit_evidence": pl.Boolean,
            "has_comparison_evidence": pl.Boolean,
            "has_bootstrap_evidence": pl.Boolean,
        },
    )


def feature_evidence_compact_dict(report: FeatureEvidenceReport) -> dict[str, Any]:
    by_feature_name: dict[str, dict[str, Any]] = {}
    for evidence in sorted(report.feature_evidence, key=lambda item: item.feature_name):
        by_feature_name[evidence.feature_name] = {
            "kernel_name": evidence.kernel_name,
            "feature_family": evidence.feature_family,
            "source_col": evidence.source_col,
            "interpretation": evidence.interpretation,
            "evidence_completeness": evidence.evidence_completeness,
            "warning_codes": tuple(evidence.identifiability_warnings or ()),
            "has_fit_evidence": evidence.fit_result_id is not None,
            "has_comparison_evidence": _has_comparison_evidence(evidence),
            "has_bootstrap_evidence": evidence.bootstrap_summary is not None,
        }
    return {
        "feature_count": report.feature_count,
        "kernel_count": report.kernel_count,
        "source_columns": report.source_columns,
        "warning_summary": dict(sorted(report.warning_summary.items())),
        "by_feature_name": by_feature_name,
    }


def feature_evidence_compact_text(report: FeatureEvidenceReport) -> str:
    table = feature_evidence_table(report)
    by_completeness = (
        table.group_by("evidence_completeness")
        .len()
        .sort("evidence_completeness")
        .to_dicts()
    )
    counts_text = ", ".join(
        f"{row['evidence_completeness']}={row['len']}" for row in by_completeness
    )
    return (
        f"features={report.feature_count}; kernels={report.kernel_count}; "
        f"source_cols={len(report.source_columns)}; evidence={counts_text}"
    )


def _warning_codes_from_fit(fit_result: KernelFitResult | None) -> tuple[str, ...]:
    if fit_result is None:
        return ()
    return tuple(fit_result.identifiability_report.warning_codes)


def _fit_result_id(fit_result: KernelFitResult | None) -> str | None:
    if fit_result is None or fit_result.fit_provenance is None:
        return None
    fit_id = fit_result.fit_provenance.get("fit_result_id")
    if fit_id is None:
        return None
    return str(fit_id)


def _has_comparison_evidence(evidence: FeatureEvidence) -> bool:
    return (
        evidence.candidate_id is not None
        or evidence.baseline_summary is not None
    )


def _evidence_completeness(
    *,
    has_fit_evidence: bool,
    has_comparison_evidence: bool,
    has_bootstrap_evidence: bool,
) -> FeatureEvidenceCompletenessLabel:
    if has_fit_evidence and has_comparison_evidence and has_bootstrap_evidence:
        return "full_evidence"
    if has_bootstrap_evidence:
        return "bootstrap_evidence"
    if has_comparison_evidence:
        return "comparison_evidence"
    if has_fit_evidence:
        return "fit_evidence"
    return "kernel_only"


def _spec_family_to_feature_family(spec: FeatureSpec) -> str:
    if spec.family == "numeric":
        return f"numeric_{spec.metric}"
    if spec.family == "categorical":
        if spec.metric == "frac":
            return "categorical_fraction"
        return "categorical_entropy"
    if spec.family == "age":
        return f"age_{spec.metric}"
    return f"{spec.family}_{spec.metric}"
