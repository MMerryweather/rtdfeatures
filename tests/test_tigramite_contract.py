"""Phase 0 contract tests for optional Tigramite adapter public API."""

from __future__ import annotations

import pytest

from rtdfeatures.integrations.tigramite import (
    TIGRAMITE_CONTEMPORANEOUS_LINK_IGNORED,
    TIGRAMITE_GRAPH_MARK_UNSUPPORTED,
    TIGRAMITE_KERNEL_SUPPORT_THRESHOLD_BREACH,
    TIGRAMITE_LAG_RANGE_EMPTY,
    TIGRAMITE_NO_LINKS_FOUND,
    TIGRAMITE_PAYLOAD_SHAPE_INVALID,
    TIGRAMITE_VALUES_NOT_KERNEL_WEIGHTS,
    TIGRAMITE_VARIABLE_NAME_MISSING,
    TigramiteLagCandidateResult,
    candidate_set_from_tigramite_links,
    compare_kernel_to_tigramite_links,
    parent_pairs_from_tigramite_graph,
)
from rtdfeatures.integrations.tigramite.payloads import TigramitePayloadError
from rtdfeatures.kernels import Kernel


def test_phase0_public_contract_exports_are_available() -> None:
    assert TIGRAMITE_NO_LINKS_FOUND == "TIGRAMITE_NO_LINKS_FOUND"
    assert TIGRAMITE_LAG_RANGE_EMPTY == "TIGRAMITE_LAG_RANGE_EMPTY"
    assert TIGRAMITE_CONTEMPORANEOUS_LINK_IGNORED == "TIGRAMITE_CONTEMPORANEOUS_LINK_IGNORED"
    assert TIGRAMITE_GRAPH_MARK_UNSUPPORTED == "TIGRAMITE_GRAPH_MARK_UNSUPPORTED"
    assert TIGRAMITE_VALUES_NOT_KERNEL_WEIGHTS == "TIGRAMITE_VALUES_NOT_KERNEL_WEIGHTS"
    assert TIGRAMITE_PAYLOAD_SHAPE_INVALID == "TIGRAMITE_PAYLOAD_SHAPE_INVALID"
    assert TIGRAMITE_VARIABLE_NAME_MISSING == "TIGRAMITE_VARIABLE_NAME_MISSING"


def test_phase0_tigramite_lag_candidate_result_fields_and_shape() -> None:
    result = TigramiteLagCandidateResult(
        source_col="feed",
        target_col="product",
        lag_steps=(1, 2),
        min_lag_step=1,
        max_lag_step=2,
        link_values=(0.6, 0.4),
        p_values=(0.01, 0.03),
        graph_marks=("->", "->"),
        source="tigramite_link_adapter",
        warnings=(TIGRAMITE_VALUES_NOT_KERNEL_WEIGHTS,),
        metadata={"kind": "candidate_evidence"},
    )

    payload = result.to_dict()
    assert payload["source_col"] == "feed"
    assert payload["lag_steps"] == [1, 2]
    assert payload["min_lag_step"] == 1
    assert payload["max_lag_step"] == 2
    assert payload["warnings"] == [TIGRAMITE_VALUES_NOT_KERNEL_WEIGHTS]


def test_phase0_tigramite_lag_candidate_result_rejects_mismatched_link_values_length() -> None:
    with pytest.raises(
        ValueError,
        match="link_values length must match lag_steps length when link_values are provided.",
    ):
        TigramiteLagCandidateResult(
            source_col="feed",
            target_col="product",
            lag_steps=(1, 2),
            min_lag_step=1,
            max_lag_step=2,
            link_values=(0.6,),
            p_values=(0.01, 0.03),
            graph_marks=("->", "->"),
            source="tigramite_link_adapter",
            warnings=(),
            metadata={},
        )


def test_phase0_tigramite_lag_candidate_result_rejects_mismatched_p_values_length() -> None:
    with pytest.raises(
        ValueError,
        match="p_values length must match lag_steps length when p_values are provided.",
    ):
        TigramiteLagCandidateResult(
            source_col="feed",
            target_col="product",
            lag_steps=(1, 2),
            min_lag_step=1,
            max_lag_step=2,
            link_values=(0.6, 0.4),
            p_values=(0.01,),
            graph_marks=("->", "->"),
            source="tigramite_link_adapter",
            warnings=(),
            metadata={},
        )


def test_phase0_tigramite_lag_candidate_result_rejects_mismatched_graph_marks_length() -> None:
    with pytest.raises(
        ValueError,
        match="graph_marks length must match lag_steps length when graph_marks are provided.",
    ):
        TigramiteLagCandidateResult(
            source_col="feed",
            target_col="product",
            lag_steps=(1, 2),
            min_lag_step=1,
            max_lag_step=2,
            link_values=(0.6, 0.4),
            p_values=(0.01, 0.03),
            graph_marks=("->",),
            source="tigramite_link_adapter",
            warnings=(),
            metadata={},
        )


def test_phase0_warning_code_prefix_for_graph_handling() -> None:
    graph = [
        [["", ""], ["->", "o-o"]],
        [["", ""], ["", ""]],
    ]

    with pytest.warns(UserWarning) as record:
        pairs = parent_pairs_from_tigramite_graph(graph, var_names=["x", "y"])

    assert pairs == ()
    messages = [str(item.message) for item in record]
    assert any(msg.startswith(TIGRAMITE_CONTEMPORANEOUS_LINK_IGNORED) for msg in messages)
    assert any(msg.startswith(TIGRAMITE_GRAPH_MARK_UNSUPPORTED) for msg in messages)
    assert any(msg.startswith(TIGRAMITE_NO_LINKS_FOUND) for msg in messages)


def test_phase0_warning_code_prefix_for_payload_and_var_name() -> None:
    with pytest.warns(UserWarning, match=rf"^{TIGRAMITE_PAYLOAD_SHAPE_INVALID}:"):
        with pytest.raises(TigramitePayloadError):
            parent_pairs_from_tigramite_graph([["bad"]], var_names=["x"])

    graph = [
        [["", ""], ["", ""]],
        [["", ""], ["", ""]],
    ]
    with pytest.warns(UserWarning, match=rf"^{TIGRAMITE_VARIABLE_NAME_MISSING}:"):
        with pytest.raises(TigramitePayloadError):
            parent_pairs_from_tigramite_graph(graph, var_names={0: "x"})


def test_phase0_warning_code_prefix_for_empty_lag_range() -> None:
    graph = [
        [["", ""], ["", ""]],
        [["", ""], ["", ""]],
    ]

    with pytest.warns(UserWarning, match=rf"^{TIGRAMITE_LAG_RANGE_EMPTY}:"):
        payload = candidate_set_from_tigramite_links(
            graph,
            var_names=["target", "input"],
            input_col="input",
            target_col="target",
        )

    assert len(payload["candidates"]) == 2
    assert {item["family"] for item in payload["candidates"]} == {
        "no_lag",
        "best_single_lag",
    }
    assert all(item["candidate_type"] == "baseline" for item in payload["candidates"])
    assert payload["baseline_names"] == ["no_lag", "best_single_lag"]


def test_phase0_warning_code_prefix_for_support_threshold_breach() -> None:
    kernel = Kernel(
        weights=(0.1, 0.2, 0.3, 0.4),
        lag_steps=(1, 2, 3, 4),
        dt=1.0,
        min_lag_steps=1,
        max_lag_steps=4,
        name="phase0",
    )

    with pytest.warns(
        UserWarning,
        match=rf"^{TIGRAMITE_KERNEL_SUPPORT_THRESHOLD_BREACH}:",
    ):
        comparison = compare_kernel_to_tigramite_links(
            kernel,
            [2, 4],
            outside_support_warning_threshold=0.2,
        )

    assert comparison.outside_support_exceeds_threshold is True


def test_phase0_warning_code_prefix_for_value_matrix_interpretation() -> None:
    graph = [
        [["", ""], ["", "->"]],
        [["", ""], ["", ""]],
    ]
    val_matrix = [
        [[0.0, 0.0], [0.0, 0.8]],
        [[0.0, 0.0], [0.0, 0.0]],
    ]

    with pytest.warns(
        UserWarning,
        match=rf"^{TIGRAMITE_VALUES_NOT_KERNEL_WEIGHTS}:",
    ):
        payload = candidate_set_from_tigramite_links(
            graph,
            var_names=["target", "input"],
            input_col="input",
            target_col="target",
            val_matrix=val_matrix,
        )

    assert payload["metadata"]["graph_metadata"]["val_matrix"] == val_matrix
