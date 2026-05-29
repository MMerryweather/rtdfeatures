"""Diagnostics package — fit, transform, and warning/identifiability diagnostics.

Re-exports all public symbols from the submodules for backward compatibility.
Candidate contracts re-exported from ``candidates.contracts`` for convenience.
"""

# ruff: noqa: I001 — import order is deliberate to avoid circular deps:
#   candidates.contracts (loaded via fitting → learners) must come
#   AFTER diagnostics submodules so BaselineComparison etc. are
#   defined when learners/exponential.py re-enters diagnostics.

from rtdfeatures.diagnostics.fit import (
    BaselineComparison,
    FitDataCoverageSummary,
    FitDiagnostics,
    IdentifiabilityReport,
    KernelFitResult,
    KernelShapeSummary,
    SharedKernelFitResult,
    SharedPairFitResult,
)
from rtdfeatures.diagnostics.transform import (
    OutOfFoldKernelFeatureResult,
    OutOfFoldSplitSummary,
    TransformReport,
)
from rtdfeatures.diagnostics.warnings import (
    BOOTSTRAP_WARNING_CODES,
    DEFAULT_BOOTSTRAP_INTERVAL_QUANTILES,
    FEATURE_EVIDENCE_COMPLETENESS_LABELS,
    FEATURE_INTERPRETATION_LABELS,
    BootstrapLagSummarySample,
    BootstrapParameterSample,
    BootstrapResult,
    BootstrapWeightSample,
    FeatureEvidence,
    FeatureEvidenceCompletenessLabel,
    FeatureEvidenceReport,
    FeatureInterpretationLabel,
    KernelBootstrapSummary,
    ParameterUncertaintySummary,
    WeightUncertaintySummary,
    bootstrap_lag_summary_samples_schema,
    bootstrap_parameter_samples_schema,
    bootstrap_weight_samples_schema,
    parameter_uncertainty_summary_schema,
    weight_uncertainty_summary_schema,
)
# Candidate contracts re-exported for backward compatibility
from rtdfeatures.candidates.contracts import (
    KernelCandidate,
    KernelCandidateSet,
    KernelComparisonConfig,
    KernelComparisonResult,
    KernelFamilyFitResult,
    KernelSelectionResult,
)

__all__ = [
    "BaselineComparison",
    "BOOTSTRAP_WARNING_CODES",
    "BootstrapLagSummarySample",
    "BootstrapParameterSample",
    "BootstrapResult",
    "BootstrapWeightSample",
    "DEFAULT_BOOTSTRAP_INTERVAL_QUANTILES",
    "FEATURE_EVIDENCE_COMPLETENESS_LABELS",
    "FEATURE_INTERPRETATION_LABELS",
    "FeatureEvidence",
    "FeatureEvidenceCompletenessLabel",
    "FeatureEvidenceReport",
    "FeatureInterpretationLabel",
    "FitDataCoverageSummary",
    "FitDiagnostics",
    "IdentifiabilityReport",
    "KernelBootstrapSummary",
    "KernelCandidate",
    "KernelCandidateSet",
    "KernelComparisonConfig",
    "KernelComparisonResult",
    "KernelFamilyFitResult",
    "KernelFitResult",
    "KernelSelectionResult",
    "KernelShapeSummary",
    "OutOfFoldKernelFeatureResult",
    "OutOfFoldSplitSummary",
    "ParameterUncertaintySummary",
    "SharedKernelFitResult",
    "SharedPairFitResult",
    "TransformReport",
    "WeightUncertaintySummary",
    "bootstrap_lag_summary_samples_schema",
    "bootstrap_parameter_samples_schema",
    "bootstrap_weight_samples_schema",
    "parameter_uncertainty_summary_schema",
    "weight_uncertainty_summary_schema",
]
