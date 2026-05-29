"""Phase 3 tests for Tigramite graph helper extraction and metadata behavior."""

from __future__ import annotations

import json
from typing import Any

import pytest

from rtdfeatures.integrations.tigramite import (
    TigramitePayloadError,
    lag_candidates_from_pcmci_graph,
    parent_pairs_from_tigramite_graph,
    parse_mock_graph_payload,
    parse_mock_pvalue_matrix,
    parse_mock_value_matrix,
)


def test_lagged_graph_extraction_and_sign_convention() -> None:
    graph = [
        [["", "", ""], ["", "->", ""]],
        [["", "", "->"], ["", "", ""]],
    ]

    pairs = parent_pairs_from_tigramite_graph(graph, var_names=["a", "b"])

    assert pairs == (
        ("b", "a", 1, "->"),
        ("a", "b", 2, "->"),
    )


def test_parent_pairs_helper_accepts_direct_dict_graph_payload() -> None:
    graph_payload = {"graph": [[["", ""], ["", "->"]], [["", ""], ["", ""]]]}

    pairs = parent_pairs_from_tigramite_graph(
        graph_payload,
        var_names=["target", "source"],
    )

    assert pairs == (("source", "target", 1, "->"),)


def test_parsed_mock_payloads_work_end_to_end_with_graph_helper() -> None:
    graph = parse_mock_graph_payload({"graph": [[["", ""], ["", "->"]], [["", ""], ["", ""]]]})
    p_matrix = parse_mock_pvalue_matrix(
        {"p_matrix": [[[1.0, 1.0], [1.0, 0.01]], [[1.0, 1.0], [1.0, 1.0]]]}
    )
    val_matrix = parse_mock_value_matrix(
        {"val_matrix": [[[0.0, 0.0], [0.0, 0.8]], [[0.0, 0.0], [0.0, 0.0]]]}
    )

    result = lag_candidates_from_pcmci_graph(
        graph,
        var_names=["target", "source"],
        p_matrix=p_matrix,
        val_matrix=val_matrix,
    )

    assert len(result.candidates) == 1
    assert result.candidates[0].target_col == "target"
    assert result.candidates[0].source_col == "source"
    assert result.candidates[0].lag_steps == 1
    assert result.candidates[0].p_value == 0.01
    assert result.candidates[0].value == 0.8


def test_public_lag_helper_accepts_plain_dict_payload_and_string_index_var_names() -> None:
    graph_payload = {"graph": [[["", ""], ["", "->"]], [["", ""], ["", ""]]]}
    p_payload = {"p_matrix": [[[1.0, 1.0], [1.0, 0.01]], [[1.0, 1.0], [1.0, 1.0]]]}
    val_payload = {"val_matrix": [[[0.0, 0.0], [0.0, 0.8]], [[0.0, 0.0], [0.0, 0.0]]]}

    result = lag_candidates_from_pcmci_graph(
        graph_payload,
        var_names={"0": "target", "1": "source"},
        p_matrix=p_payload,
        val_matrix=val_payload,
    )

    assert len(result.candidates) == 1
    assert result.candidates[0].source_col == "source"
    assert result.candidates[0].target_col == "target"
    assert result.candidates[0].lag_steps == 1
    assert result.candidates[0].p_value == 0.01
    assert result.candidates[0].value == 0.8


def test_metadata_preserves_p_and_value_matrices() -> None:
    graph = [
        [["", ""], ["", "->"], ["", ""]],
        [["", ""], ["", ""], ["", ""]],
        [["", ""], ["", ""], ["", ""]],
    ]
    p_matrix = [
        [[1.0, 1.0], [1.0, 0.01], [1.0, 1.0]],
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
    ]
    val_matrix = [
        [[0.0, 0.0], [0.0, 0.8], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    ]

    result = lag_candidates_from_pcmci_graph(
        graph,
        var_names=["target", "source", "extra"],
        p_matrix=p_matrix,
        val_matrix=val_matrix,
    )
    payload = result.to_dict()

    assert payload["metadata"]["p_matrix"] == p_matrix
    assert payload["metadata"]["val_matrix"] == val_matrix
    assert payload["candidates"][0]["p_value"] == 0.01
    assert payload["candidates"][0]["value"] == 0.8


def test_unsupported_mark_warns() -> None:
    graph = [
        [["", ""], ["", "o-o"]],
        [["", ""], ["", ""]],
    ]

    with pytest.warns(UserWarning, match="Unsupported graph mark"):
        pairs = parent_pairs_from_tigramite_graph(graph, var_names=["x", "y"])

    assert pairs == ()


def test_contemporaneous_link_warns_and_is_ignored() -> None:
    graph = [
        [["", ""], ["->", ""]],
        [["", ""], ["", ""]],
    ]

    with pytest.warns(UserWarning, match="Contemporaneous links"):
        pairs = parent_pairs_from_tigramite_graph(graph, var_names=["x", "y"])

    assert pairs == ()


def test_2d_graph_payload_warns_that_lag_axis_is_missing() -> None:
    message = "2D Tigramite graph payload was provided without a lag axis"
    with pytest.warns(UserWarning, match=message):
        graph = parse_mock_graph_payload({"graph": [["", "->"], ["", ""]]})

    assert graph == [[[""], ["->"]], [[""], [""]]]


def test_2d_graph_payload_warning_is_explicit_and_extraction_is_contract_safe() -> None:
    message = "2D Tigramite graph payload was provided without a lag axis"
    with pytest.warns(UserWarning, match=message):
        graph = parse_mock_graph_payload({"graph": [["", "->"], ["", ""]]})

    with pytest.warns(UserWarning, match="Contemporaneous links"):
        pairs = parent_pairs_from_tigramite_graph(graph, var_names=["target", "source"])

    assert pairs == ()


def test_no_links_warns() -> None:
    graph = [
        [["", ""], ["", ""]],
        [["", ""], ["", ""]],
    ]

    with pytest.warns(UserWarning, match="No lagged directed links"):
        pairs = parent_pairs_from_tigramite_graph(graph, var_names=["x", "y"])

    assert pairs == ()


def test_invalid_payload_shape_warns_then_raises() -> None:
    bad_graph = [["not-a-cube"]]

    with pytest.warns(UserWarning, match="Invalid Tigramite payload shape"):
        with pytest.raises(TigramitePayloadError, match="Malformed graph payload"):
            parent_pairs_from_tigramite_graph(bad_graph, var_names=["x"])


def test_ragged_optional_p_matrix_warns_then_raises() -> None:
    graph = [
        [["", ""], ["", "->"]],
        [["", ""], ["", ""]],
    ]
    ragged_p_matrix = [
        [[1.0, 1.0], [1.0, 0.01]],
        [[1.0, 1.0]],
    ]

    with pytest.warns(UserWarning, match="TIGRAMITE_PAYLOAD_SHAPE_INVALID"):
        with pytest.raises(TigramitePayloadError, match="Malformed payload: 'p_matrix' shape"):
            lag_candidates_from_pcmci_graph(
                graph,
                var_names=["target", "source"],
                p_matrix=ragged_p_matrix,
            )


def test_ragged_optional_val_matrix_warns_then_raises() -> None:
    graph = [
        [["", "", ""], ["", "->", ""]],
        [["", "", ""], ["", "", ""]],
    ]
    ragged_val_matrix = [
        [[0.0, 0.0, 0.0], [0.0, 0.8]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ]

    with pytest.warns(UserWarning, match="TIGRAMITE_PAYLOAD_SHAPE_INVALID"):
        with pytest.raises(TigramitePayloadError, match="Malformed payload: 'val_matrix' shape"):
            lag_candidates_from_pcmci_graph(
                graph,
                var_names=["target", "source"],
                val_matrix=ragged_val_matrix,
            )


def test_missing_variable_name_warns_then_raises() -> None:
    graph = [
        [["", ""], ["", ""]],
        [["", ""], ["", ""]],
    ]

    with pytest.warns(UserWarning, match="Missing or invalid variable name"):
        with pytest.raises(TigramitePayloadError, match="Missing variable-name mapping"):
            parent_pairs_from_tigramite_graph(graph, var_names={0: "x"})


def test_duplicate_var_names_raise_value_error() -> None:
    graph = [
        [["", ""], ["", "->"]],
        [["", ""], ["", ""]],
    ]

    with pytest.raises(ValueError, match="var_names must be unique"):
        parent_pairs_from_tigramite_graph(graph, var_names=["dup", "dup"])


def test_allowed_directed_marks_bare_string_raises_type_error() -> None:
    graph = [
        [["", ""], ["", "->"]],
        [["", ""], ["", ""]],
    ]

    with pytest.raises(TypeError, match="non-string iterable of strings"):
        parent_pairs_from_tigramite_graph(
            graph,
            var_names=["target", "source"],
            allowed_directed_marks="->",
        )


def test_no_kernel_weights_emitted_and_output_is_serializable() -> None:
    graph = [
        [["", ""], ["", "->"], ["", ""]],
        [["", ""], ["", ""], ["", ""]],
        [["", ""], ["", ""], ["", ""]],
    ]

    result = lag_candidates_from_pcmci_graph(graph, var_names=["target", "source", "extra"])
    payload = result.to_dict()

    encoded = json.dumps(payload)
    decoded = json.loads(encoded)

    assert "kernel_weight" not in encoded
    assert decoded["candidates"][0]["lag_steps"] == 1
    assert decoded["candidates"][0]["source_col"] == "source"
    assert decoded["candidates"][0]["target_col"] == "target"


def test_lag_step_range_filter_applies_on_public_helper() -> None:
    graph = [
        [["", "", "", ""], ["", "->", "->", "->"]],
        [["", "", "", ""], ["", "", "", ""]],
    ]

    result = lag_candidates_from_pcmci_graph(
        graph,
        var_names=["target", "source"],
        lag_step_range=(2, 2),
    )

    assert [item.lag_steps for item in result.candidates] == [2]


@pytest.mark.parametrize("bad_lag_step_range", [(2,), (), (1, 2, 3)])
def test_lag_step_range_malformed_shape_raises_value_error(
    bad_lag_step_range: Any,
) -> None:
    graph = [
        [["", "", ""], ["", "->", "->"]],
        [["", "", ""], ["", "", ""]],
    ]

    with pytest.raises(ValueError, match="lag_step_range must be a 2-item sequence"):
        lag_candidates_from_pcmci_graph(
            graph,
            var_names=["target", "source"],
            lag_step_range=bad_lag_step_range,
        )
