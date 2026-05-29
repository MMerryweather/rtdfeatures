"""Phase 4 tests for Tigramite candidate-set descriptor integration."""

from __future__ import annotations

import json
from typing import Any

import polars as pl
import pytest

from rtdfeatures.candidates import fit_kernel_candidates
from rtdfeatures.diagnostics import KernelCandidateSet
from rtdfeatures.integrations.tigramite import candidate_set_from_tigramite_links


def _graph_payload(
) -> tuple[
    list[list[list[str]]],
    list[list[list[float]]],
    list[list[list[float]]],
]:
    graph = [
        [["", "", ""], ["", "->", "->"], ["", "", ""]],
        [["", "", ""], ["", "", ""], ["", "", ""]],
        [["", "", ""], ["", "", ""], ["", "", ""]],
    ]
    p_matrix = [
        [[1.0, 1.0, 1.0], [1.0, 0.02, 0.04], [1.0, 1.0, 1.0]],
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    ]
    val_matrix = [
        [[0.0, 0.0, 0.0], [0.0, 0.7, 0.5], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ]
    return graph, p_matrix, val_matrix


def _fit_df(n_rows: int = 64) -> pl.DataFrame:
    ts = pl.datetime_range(
        start=pl.datetime(2024, 1, 1, 0, 0, 0),
        end=pl.datetime(2024, 1, 1, 0, 0, 0) + pl.duration(minutes=n_rows - 1),
        interval="1m",
        eager=True,
    )
    x = [float(i) for i in range(n_rows)]
    y = [
        0.7 * x[i - 2] + 0.2 * x[i - 1] + 0.1 * x[i] if i >= 2 else float(i)
        for i in range(n_rows)
    ]
    return pl.DataFrame({"ts": ts, "input": x, "target": y})


def test_phase4_candidate_set_construction() -> None:
    graph, p_matrix, val_matrix = _graph_payload()

    payload = candidate_set_from_tigramite_links(
        graph,
        var_names=["target", "input", "extra"],
        input_col="input",
        target_col="target",
        time_col="ts",
        p_matrix=p_matrix,
        val_matrix=val_matrix,
    )

    assert payload["input_col"] == "input"
    assert payload["target_col"] == "target"
    assert payload["time_col"] == "ts"
    assert payload["candidates"]
    assert {item["family"] for item in payload["candidates"]} >= {
        "simplex",
        "fixed_delay",
        "gamma",
        "exponential",
    }
    assert payload["baseline_names"] == ["no_lag", "best_single_lag"]


def test_phase4_candidate_payload_is_json_serializable() -> None:
    graph, p_matrix, val_matrix = _graph_payload()
    payload = candidate_set_from_tigramite_links(
        graph,
        var_names=["target", "input", "extra"],
        input_col="input",
        target_col="target",
        p_matrix=p_matrix,
        val_matrix=val_matrix,
    )

    encoded = json.dumps(payload)
    decoded = json.loads(encoded)

    assert decoded["candidate_set_id"] == "tigramite:input->target"
    assert isinstance(decoded["candidates"], list)


def test_phase4_family_defaults_are_configurable() -> None:
    graph, p_matrix, val_matrix = _graph_payload()

    payload = candidate_set_from_tigramite_links(
        graph,
        var_names=["target", "input", "extra"],
        input_col="input",
        target_col="target",
        candidate_families=["simplex", "fixed_delay"],
        family_defaults={
            "simplex": {"max_epochs": 21, "learning_rate": 0.03},
            "fixed_delay": {"delay_steps": 2},
        },
        p_matrix=p_matrix,
        val_matrix=val_matrix,
    )

    by_family = {item["family"]: item for item in payload["candidates"]}
    assert {"simplex", "fixed_delay"} <= set(by_family)
    assert by_family["simplex"]["learner_parameters"]["max_epochs"] == 21
    assert by_family["fixed_delay"]["fixed_parameters"]["delay_steps"] == 2


def test_phase4_metadata_preservation_and_lag_window_derivation() -> None:
    graph, p_matrix, val_matrix = _graph_payload()

    payload = candidate_set_from_tigramite_links(
        graph,
        var_names=["target", "input", "extra"],
        input_col="input",
        target_col="target",
        p_matrix=p_matrix,
        val_matrix=val_matrix,
    )

    summary = payload["metadata"]["lag_evidence_summary"]
    assert summary["lag_steps"] == [1, 2]
    assert summary["min_lag_steps"] == 1
    assert summary["max_lag_steps"] == 2
    assert summary["p_value_min"] == 0.02
    assert summary["value_max"] == 0.7
    assert "->" in summary["graph_marks"]


def test_phase4_empty_lag_evidence_warns() -> None:
    graph = [
        [["", "", ""], ["", "", ""], ["", "", ""]],
        [["", "", ""], ["", "", ""], ["", "", ""]],
        [["", "", ""], ["", "->", ""], ["", "", ""]],
    ]

    with pytest.warns(UserWarning, match="No usable lag evidence"):
        payload = candidate_set_from_tigramite_links(
            graph,
            var_names=["target", "input", "other"],
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


def test_phase4_empty_lag_evidence_payload_round_trips_to_candidate_set() -> None:
    graph = [
        [["", "", ""], ["", "", ""], ["", "", ""]],
        [["", "", ""], ["", "", ""], ["", "", ""]],
        [["", "", ""], ["", "->", ""], ["", "", ""]],
    ]

    with pytest.warns(UserWarning, match="No usable lag evidence"):
        payload = candidate_set_from_tigramite_links(
            graph,
            var_names=["target", "input", "other"],
            input_col="input",
            target_col="target",
        )

    restored = KernelCandidateSet.from_dict(payload)
    assert restored.baseline_names == ("no_lag", "best_single_lag")
    assert len(restored.candidates) == 2
    assert all(candidate.candidate_type == "baseline" for candidate in restored.candidates)
    assert {candidate.family for candidate in restored.candidates} == {
        "no_lag",
        "best_single_lag",
    }


def test_phase4_non_empty_lag_evidence_payload_round_trips_to_candidate_set() -> None:
    graph, p_matrix, val_matrix = _graph_payload()
    payload = candidate_set_from_tigramite_links(
        graph,
        var_names=["target", "input", "extra"],
        input_col="input",
        target_col="target",
        p_matrix=p_matrix,
        val_matrix=val_matrix,
    )

    restored = KernelCandidateSet.from_dict(payload)
    assert restored.candidate_set_id == "tigramite:input->target"
    assert len(restored.candidates) >= 4
    assert all(
        candidate.learner_parameters
        for candidate in restored.candidates
        if candidate.candidate_type in {"empirical_learner", "parametric_learner"}
    )


def test_phase4_parametric_candidates_fit_downstream_with_init_parameter_keys() -> None:
    graph, p_matrix, val_matrix = _graph_payload()
    payload = candidate_set_from_tigramite_links(
        graph,
        var_names=["target", "input", "extra"],
        input_col="input",
        target_col="target",
        p_matrix=p_matrix,
        val_matrix=val_matrix,
    )
    restored = KernelCandidateSet.from_dict(payload)
    parametric_candidates = [
        c for c in restored.candidates if c.candidate_type == "parametric_learner"
    ]
    assert parametric_candidates
    for candidate in parametric_candidates:
        assert "shape_alpha" not in candidate.learner_parameters
        assert "rate_beta" not in candidate.learner_parameters
        assert "rate_lambda" not in candidate.learner_parameters
        if candidate.family == "gamma":
            assert "init_shape_alpha" in candidate.learner_parameters
            assert "init_rate_beta" in candidate.learner_parameters
        if candidate.family == "exponential":
            assert "init_rate_lambda" in candidate.learner_parameters

    result = fit_kernel_candidates(_fit_df(), restored)
    fits_by_id = {item.candidate.candidate_id: item for item in result.family_results}
    for candidate in parametric_candidates:
        fit_item = fits_by_id[candidate.candidate_id]
        assert fit_item.succeeded is True
        assert fit_item.error is None
        assert fit_item.fit_result is not None


def test_phase4_no_live_objects_in_candidate_descriptors() -> None:
    graph, p_matrix, val_matrix = _graph_payload()

    payload = candidate_set_from_tigramite_links(
        graph,
        var_names=["target", "input", "extra"],
        input_col="input",
        target_col="target",
        p_matrix=p_matrix,
        val_matrix=val_matrix,
    )

    for candidate in payload["candidates"]:
        assert isinstance(candidate, dict)
        assert isinstance(candidate["fixed_parameters"], dict)
        assert isinstance(candidate["learner_parameters"], dict)
        assert "fit" not in repr(candidate).lower()


def test_phase4_no_fitting_side_effect(monkeypatch: pytest.MonkeyPatch) -> None:
    graph, p_matrix, val_matrix = _graph_payload()

    def _fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("fit_kernel_candidates must not be called by adapter")

    monkeypatch.setattr("rtdfeatures.candidates.fit_kernel_candidates", _fail_if_called)

    payload = candidate_set_from_tigramite_links(
        graph,
        var_names=["target", "input", "extra"],
        input_col="input",
        target_col="target",
        p_matrix=p_matrix,
        val_matrix=val_matrix,
    )
    assert payload["candidates"]


def test_phase4_candidate_set_propagates_mark_allowlist() -> None:
    graph = [
        [["", "", ""], ["", "->", "->"]],
        [["", "", ""], ["", "", ""]],
    ]

    with pytest.warns(UserWarning, match="No usable lag evidence"):
        payload = candidate_set_from_tigramite_links(
            graph,
            var_names=["target", "input"],
            input_col="input",
            target_col="target",
            allowed_directed_marks=["o-o"],
        )

    assert payload["metadata"]["lag_evidence"] == []
    assert len(payload["candidates"]) == 2
    assert all(item["candidate_type"] == "baseline" for item in payload["candidates"])
    assert payload["baseline_names"] == ["no_lag", "best_single_lag"]


def test_phase4_candidate_set_applies_lag_step_range_filter() -> None:
    graph, p_matrix, val_matrix = _graph_payload()

    payload = candidate_set_from_tigramite_links(
        graph,
        var_names=["target", "input", "extra"],
        input_col="input",
        target_col="target",
        p_matrix=p_matrix,
        val_matrix=val_matrix,
        lag_step_range=(2, 2),
    )

    summary = payload["metadata"]["lag_evidence_summary"]
    assert summary["lag_steps"] == [2]
    assert summary["min_lag_steps"] == 2
    assert summary["max_lag_steps"] == 2


def test_phase4_candidate_set_accepts_plain_dict_payload_boundary() -> None:
    graph, p_matrix, val_matrix = _graph_payload()
    payload = candidate_set_from_tigramite_links(
        {"graph": graph},
        var_names={"0": "target", "1": "input", "2": "extra"},
        input_col="input",
        target_col="target",
        p_matrix={"p_matrix": p_matrix},
        val_matrix={"val_matrix": val_matrix},
    )

    assert payload["candidate_set_id"] == "tigramite:input->target"
    assert payload["metadata"]["lag_evidence_summary"]["lag_steps"] == [1, 2]
    assert payload["metadata"]["graph_metadata"]["p_matrix"] == p_matrix
    assert payload["metadata"]["graph_metadata"]["val_matrix"] == val_matrix


@pytest.mark.parametrize("bad_lag_step_range", [(2,), (), (1, 2, 3)])
def test_phase4_candidate_set_malformed_lag_step_range_raises_value_error(
    bad_lag_step_range: Any,
) -> None:
    graph, p_matrix, val_matrix = _graph_payload()

    with pytest.raises(ValueError, match="lag_step_range must be a 2-item sequence"):
        candidate_set_from_tigramite_links(
            graph,
            var_names=["target", "input", "extra"],
            input_col="input",
            target_col="target",
            p_matrix=p_matrix,
            val_matrix=val_matrix,
            lag_step_range=bad_lag_step_range,
        )
