"""legacy milestone contract tests for v0.9 feature evidence objects."""

from __future__ import annotations

from dataclasses import fields

import pytest

from rtdfeatures.diagnostics import (
    FEATURE_EVIDENCE_COMPLETENESS_LABELS,
    FEATURE_INTERPRETATION_LABELS,
    FeatureEvidence,
    FeatureEvidenceReport,
)


def test_feature_evidence_dataclass_fields_are_stable() -> None:
    assert [field.name for field in fields(FeatureEvidence)] == [
        "feature_name",
        "source_col",
        "feature_family",
        "kernel_name",
        "kernel_family",
        "kernel_summary",
        "fit_result_id",
        "candidate_id",
        "baseline_summary",
        "identifiability_warnings",
        "bootstrap_summary",
        "interpretation",
        "evidence_completeness",
        "metadata",
    ]
    assert [field.name for field in fields(FeatureEvidenceReport)] == [
        "feature_evidence",
        "feature_count",
        "kernel_count",
        "source_columns",
        "warning_summary",
        "evidence_summary_by_kernel",
        "evidence_summary_by_feature_family",
    ]


def test_feature_evidence_label_constants_are_stable() -> None:
    assert FEATURE_INTERPRETATION_LABELS == (
        "material_memory",
        "process_response",
        "statistical_pattern",
        "unknown",
    )
    assert FEATURE_EVIDENCE_COMPLETENESS_LABELS == (
        "kernel_only",
        "fit_evidence",
        "comparison_evidence",
        "bootstrap_evidence",
        "full_evidence",
    )


def test_feature_evidence_supports_missing_optional_evidence() -> None:
    evidence = FeatureEvidence(
        feature_name="learned_num_feed_wmean",
        source_col="feed",
        feature_family="numeric_wmean",
        kernel_name="learned",
        kernel_family="simplex",
        kernel_summary={"mean_lag": 2.0},
        fit_result_id=None,
        candidate_id=None,
        baseline_summary=None,
        identifiability_warnings=None,
        bootstrap_summary=None,
        interpretation="unknown",
        evidence_completeness="kernel_only",
        metadata={},
    )
    assert evidence.fit_result_id is None
    assert evidence.bootstrap_summary is None


def test_feature_evidence_objects_are_exported() -> None:
    expected = {
        "FeatureEvidence",
        "FeatureEvidenceReport",
        "FEATURE_INTERPRETATION_LABELS",
        "FEATURE_EVIDENCE_COMPLETENESS_LABELS",
    }
    import rtdfeatures.diagnostics as _diag
    assert expected.issubset(set(dir(_diag)))


def test_feature_evidence_rejects_invalid_labels() -> None:
    with pytest.raises(ValueError, match="interpretation must be one of"):
        FeatureEvidence(
            feature_name="learned_num_feed_wmean",
            source_col="feed",
            feature_family="numeric_wmean",
            kernel_name="learned",
            kernel_family="simplex",
            kernel_summary={"mean_lag": 2.0},
            fit_result_id=None,
            candidate_id=None,
            baseline_summary=None,
            identifiability_warnings=None,
            bootstrap_summary=None,
            interpretation="not_a_label",  # type: ignore[arg-type]
            evidence_completeness="kernel_only",
            metadata={},
        )
    with pytest.raises(ValueError, match="evidence_completeness must be one of"):
        FeatureEvidence(
            feature_name="learned_num_feed_wmean",
            source_col="feed",
            feature_family="numeric_wmean",
            kernel_name="learned",
            kernel_family="simplex",
            kernel_summary={"mean_lag": 2.0},
            fit_result_id=None,
            candidate_id=None,
            baseline_summary=None,
            identifiability_warnings=None,
            bootstrap_summary=None,
            interpretation="unknown",
            evidence_completeness="not_a_label",  # type: ignore[arg-type]
            metadata={},
        )


def test_feature_evidence_report_validates_count_contract() -> None:
    evidence = FeatureEvidence(
        feature_name="learned_num_feed_wmean",
        source_col="feed",
        feature_family="numeric_wmean",
        kernel_name="learned",
        kernel_family="simplex",
        kernel_summary={"mean_lag": 2.0},
        fit_result_id=None,
        candidate_id=None,
        baseline_summary=None,
        identifiability_warnings=None,
        bootstrap_summary=None,
        interpretation="unknown",
        evidence_completeness="kernel_only",
        metadata={},
    )
    with pytest.raises(ValueError, match="feature_count must match"):
        FeatureEvidenceReport(
            feature_evidence=(evidence,),
            feature_count=2,
            kernel_count=1,
            source_columns=("feed",),
            warning_summary={},
            evidence_summary_by_kernel={},
            evidence_summary_by_feature_family={},
        )
