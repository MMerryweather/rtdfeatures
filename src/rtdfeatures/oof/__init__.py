"""Out-of-fold feature generation package."""

from rtdfeatures.candidates import fit_kernel_candidates, select_kernel_candidate
from rtdfeatures.features import build_feature_evidence
from rtdfeatures.oof.generation import fit_transform_oof
from rtdfeatures.oof.reports import RecoverableFoldError
from rtdfeatures.oof.splits import (
    ForwardChainingFoldSplit,
    ForwardChainingSplitConfig,
    generate_forward_chaining_splits,
)

__all__ = [
    "ForwardChainingFoldSplit",
    "ForwardChainingSplitConfig",
    "RecoverableFoldError",
    "build_feature_evidence",
    "fit_kernel_candidates",
    "fit_transform_oof",
    "generate_forward_chaining_splits",
    "select_kernel_candidate",
]
