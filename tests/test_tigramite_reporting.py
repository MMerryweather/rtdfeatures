"""Phase 5b tests for Tigramite lag-candidate reporting helpers."""

from __future__ import annotations

from rtdfeatures.integrations.tigramite import (
    TIGRAMITE_LAG_RANGE_EMPTY,
    TigramiteLagCandidateResult,
    tigramite_lag_candidate_compact_dict,
    tigramite_lag_candidate_compact_text,
    tigramite_lag_candidate_table,
)


def test_phase5b_table_schema_and_ordering_are_stable() -> None:
    result = TigramiteLagCandidateResult(
        source_col="feed",
        target_col="product",
        lag_steps=(3, 1, 2),
        min_lag_step=1,
        max_lag_step=3,
        link_values=(0.2, 0.8, 0.5),
        p_values=(0.03, 0.01, 0.02),
        graph_marks=("->", "->", "->"),
        source="tigramite_link_adapter",
        warnings=(TIGRAMITE_LAG_RANGE_EMPTY,),
        metadata={},
    )

    table = tigramite_lag_candidate_table(result)

    assert table.columns == [
        "source_col",
        "target_col",
        "lag_step",
        "link_value",
        "p_value",
        "graph_mark",
        "source",
        "warning_count",
        "warning_codes",
    ]
    assert table["lag_step"].to_list() == [3, 1, 2]

    compact = tigramite_lag_candidate_compact_dict(result)
    assert [row["lag_step"] for row in compact["rows"]] == [1, 2, 3]


def test_phase5b_missing_p_values_are_handled() -> None:
    result = TigramiteLagCandidateResult(
        source_col="feed",
        target_col="product",
        lag_steps=(1, 2),
        min_lag_step=1,
        max_lag_step=2,
        link_values=(0.6, 0.4),
        p_values=(),
        graph_marks=("->", "->"),
        source="tigramite_link_adapter",
        warnings=(),
        metadata={},
    )

    table = tigramite_lag_candidate_table(result)
    assert table["p_value"].to_list() == [None, None]


def test_phase5b_missing_value_matrix_is_handled() -> None:
    result = TigramiteLagCandidateResult(
        source_col="feed",
        target_col="product",
        lag_steps=(1, 2),
        min_lag_step=1,
        max_lag_step=2,
        link_values=(),
        p_values=(0.1, 0.2),
        graph_marks=("->", "->"),
        source="tigramite_link_adapter",
        warnings=(),
        metadata={},
    )

    table = tigramite_lag_candidate_table(result)
    assert table["link_value"].to_list() == [None, None]


def test_phase5b_compact_dict_and_text() -> None:
    result = TigramiteLagCandidateResult(
        source_col="feed",
        target_col="product",
        lag_steps=(1, 2),
        min_lag_step=1,
        max_lag_step=2,
        link_values=(0.7, 0.2),
        p_values=(0.01, 0.03),
        graph_marks=("->", "->"),
        source="tigramite_link_adapter",
        warnings=("W1", "W2"),
        metadata={},
    )

    compact = tigramite_lag_candidate_compact_dict(result)
    assert compact["source_col"] == "feed"
    assert compact["target_col"] == "product"
    assert compact["lag_steps"] == [1, 2]
    assert compact["warning_codes"] == ["W1", "W2"]
    assert len(compact["rows"]) == 2

    text = tigramite_lag_candidate_compact_text(result)
    assert text == "feed->product source=tigramite_link_adapter lags=1,2 warnings=W1,W2"


def test_phase5b_compact_text_uses_deterministic_lag_order() -> None:
    result = TigramiteLagCandidateResult(
        source_col="feed",
        target_col="product",
        lag_steps=(4, 1, 3, 2),
        min_lag_step=1,
        max_lag_step=4,
        link_values=(0.4, 0.1, 0.3, 0.2),
        p_values=(0.04, 0.01, 0.03, 0.02),
        graph_marks=("->", "->", "->", "->"),
        source="tigramite_link_adapter",
        warnings=(),
        metadata={},
    )

    compact = tigramite_lag_candidate_compact_dict(result)
    assert compact["lag_steps"] == [1, 2, 3, 4]
    assert [row["lag_step"] for row in compact["rows"]] == [1, 2, 3, 4]

    text = tigramite_lag_candidate_compact_text(result)
    assert text == "feed->product source=tigramite_link_adapter lags=1,2,3,4 warnings=none"
