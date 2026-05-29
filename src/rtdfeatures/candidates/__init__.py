"""Kernel candidate fitting and selection package."""

from rtdfeatures.candidates.contracts import (
    KernelCandidate,
    KernelCandidateSet,
    KernelComparisonConfig,
    KernelComparisonResult,
    KernelFamilyFitResult,
    KernelSelectionResult,
)
from rtdfeatures.candidates.fitting import (
    DEFAULT_SELECTION_TOLERANCE,
    _WindowedKernelEvaluator,  # noqa: F401 — re-exported for tests
    fit_kernel_candidates,
    kernel_comparison_compact_dict,
    kernel_comparison_compact_text,
    kernel_comparison_table,
)
from rtdfeatures.candidates.selection import select_kernel_candidate

__all__ = [
    "DEFAULT_SELECTION_TOLERANCE",
    "KernelCandidate",
    "KernelCandidateSet",
    "KernelComparisonConfig",
    "KernelComparisonResult",
    "KernelFamilyFitResult",
    "KernelSelectionResult",
    "fit_kernel_candidates",
    "kernel_comparison_compact_dict",
    "kernel_comparison_compact_text",
    "kernel_comparison_table",
    "select_kernel_candidate",
]
