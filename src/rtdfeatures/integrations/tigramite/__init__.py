"""Optional Tigramite adapter utilities.

This module intentionally avoids importing Tigramite so it remains importable
without optional dependencies installed.
"""

from rtdfeatures.integrations.tigramite.helpers import (
    TIGRAMITE_CONTEMPORANEOUS_LINK_IGNORED,
    TIGRAMITE_GRAPH_MARK_UNSUPPORTED,
    TIGRAMITE_KERNEL_SUPPORT_THRESHOLD_BREACH,
    TIGRAMITE_LAG_RANGE_EMPTY,
    TIGRAMITE_NO_LINKS_FOUND,
    TIGRAMITE_PAYLOAD_SHAPE_INVALID,
    TIGRAMITE_VALUES_NOT_KERNEL_WEIGHTS,
    TIGRAMITE_VARIABLE_NAME_MISSING,
    KernelTigramiteSupportComparison,
    LagCandidateDescriptor,
    LagCandidateExtractionResult,
    TigramiteLagCandidateResult,
    candidate_set_from_tigramite_links,
    compare_kernel_to_tigramite_links,
    lag_candidates_from_pcmci_graph,
    parent_pairs_from_tigramite_graph,
    tigramite_lag_candidate_compact_dict,
    tigramite_lag_candidate_compact_text,
    tigramite_lag_candidate_table,
)
from rtdfeatures.integrations.tigramite.payloads import (
    TigramitePayloadError,
    parse_mock_graph_payload,
    parse_mock_pvalue_matrix,
    parse_mock_value_matrix,
)

__all__ = [
    "TigramitePayloadError",
    "LagCandidateDescriptor",
    "LagCandidateExtractionResult",
    "TigramiteLagCandidateResult",
    "KernelTigramiteSupportComparison",
    "TIGRAMITE_NO_LINKS_FOUND",
    "TIGRAMITE_LAG_RANGE_EMPTY",
    "TIGRAMITE_CONTEMPORANEOUS_LINK_IGNORED",
    "TIGRAMITE_GRAPH_MARK_UNSUPPORTED",
    "TIGRAMITE_KERNEL_SUPPORT_THRESHOLD_BREACH",
    "TIGRAMITE_VALUES_NOT_KERNEL_WEIGHTS",
    "TIGRAMITE_PAYLOAD_SHAPE_INVALID",
    "TIGRAMITE_VARIABLE_NAME_MISSING",
    "lag_candidates_from_pcmci_graph",
    "parent_pairs_from_tigramite_graph",
    "candidate_set_from_tigramite_links",
    "compare_kernel_to_tigramite_links",
    "tigramite_lag_candidate_table",
    "tigramite_lag_candidate_compact_dict",
    "tigramite_lag_candidate_compact_text",
    "parse_mock_graph_payload",
    "parse_mock_pvalue_matrix",
    "parse_mock_value_matrix",
]
