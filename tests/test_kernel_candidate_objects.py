"""Tests for v0.7 kernel candidate object contracts."""

from __future__ import annotations

import json

import polars as pl
import pytest

from rtdfeatures.diagnostics import (
    KernelCandidate,
    KernelCandidateSet,
    KernelComparisonResult,
    KernelFamilyFitResult,
    KernelSelectionResult,
)
from rtdfeatures.kernels import UniformKernel


def _fixed_candidate(candidate_id: str = "fixed-1") -> KernelCandidate:
    return KernelCandidate(
        candidate_id=candidate_id,
        family="uniform",
        candidate_type="fixed_kernel",
        min_lag="0m",
        max_lag="30m",
        fixed_parameters={"max_lag_steps": 3, "min_lag_steps": 0, "dt": 60.0},
        metadata={"source": "unit-test"},
    )


def test_kernel_candidate_is_json_serializable() -> None:
    candidate = _fixed_candidate()
    payload = candidate.to_dict()
    encoded = json.dumps(payload)
    assert "fixed_kernel" in encoded


def test_candidate_type_validation_rejects_invalid_type() -> None:
    with pytest.raises(ValueError, match="candidate_type must be one of"):
        KernelCandidate(
            candidate_id="bad-type",
            family="simplex",
            candidate_type="not-valid",  # type: ignore[arg-type]
            min_lag="0m",
            max_lag="2h",
        )


def test_candidate_rejects_non_serializable_live_objects() -> None:
    with pytest.raises(ValueError, match="must be JSON-serializable"):
        KernelCandidate(
            candidate_id="bad-live-object",
            family="uniform",
            candidate_type="fixed_kernel",
            min_lag="0m",
            max_lag="30m",
            fixed_parameters={"kernel": UniformKernel(max_lag_steps=3, min_lag_steps=0, dt=60.0)},
        )


def test_candidate_set_rejects_empty_candidates() -> None:
    with pytest.raises(ValueError, match="at least one candidate"):
        KernelCandidateSet(
            candidate_set_id="set-1",
            input_col="feed",
            target_col="product",
            time_col="timestamp",
            candidates=(),
        )


def test_candidate_set_rejects_duplicate_candidate_ids() -> None:
    with pytest.raises(ValueError, match="requires unique candidate_id values"):
        KernelCandidateSet(
            candidate_set_id="set-dup",
            input_col="feed",
            target_col="product",
            time_col="timestamp",
            candidates=(_fixed_candidate("dup"), _fixed_candidate("dup")),
        )


def test_candidate_set_requires_input_target_time_metadata() -> None:
    with pytest.raises(ValueError, match="input_col must be a non-empty string"):
        KernelCandidateSet(
            candidate_set_id="set-bad-input",
            input_col="  ",
            target_col="product",
            time_col="timestamp",
            candidates=(_fixed_candidate(),),
        )
    with pytest.raises(ValueError, match="target_col must be a non-empty string"):
        KernelCandidateSet(
            candidate_set_id="set-bad-target",
            input_col="feed",
            target_col="",
            time_col="timestamp",
            candidates=(_fixed_candidate(),),
        )
    with pytest.raises(ValueError, match="time_col must be a non-empty string"):
        KernelCandidateSet(
            candidate_set_id="set-bad-time",
            input_col="feed",
            target_col="product",
            time_col=" ",
            candidates=(_fixed_candidate(),),
        )


def test_candidate_set_round_trip_preserves_metadata() -> None:
    candidate_set = KernelCandidateSet(
        candidate_set_id="set-round-trip",
        input_col="feed",
        target_col="product",
        time_col="timestamp",
        candidates=(
            _fixed_candidate("fixed-a"),
            KernelCandidate(
                candidate_id="baseline-a",
                family="no_lag",
                candidate_type="baseline",
                min_lag="0m",
                max_lag="30m",
                metadata={"tier": "baseline"},
            ),
            KernelCandidate(
                candidate_id="baseline-b",
                family="best_single_lag",
                candidate_type="baseline",
                min_lag="0m",
                max_lag="30m",
                metadata={"tier": "baseline"},
            ),
        ),
        baseline_names=("no_lag", "best_single_lag"),
        selection_metric="validation_loss",
        metadata={"run_id": "abc-123", "seed": 17},
    )

    restored = KernelCandidateSet.from_dict(candidate_set.to_dict())
    assert restored.metadata == {"run_id": "abc-123", "seed": 17}
    assert restored.baseline_names == ("no_lag", "best_single_lag")
    assert tuple(c.candidate_id for c in restored.candidates) == (
        "fixed-a",
        "baseline-a",
        "baseline-b",
    )


def test_candidate_set_rejects_unsupported_selection_metric() -> None:
    with pytest.raises(ValueError, match="selection_metric must be one of"):
        KernelCandidateSet(
            candidate_set_id="set-bad-metric",
            input_col="feed",
            target_col="product",
            time_col="timestamp",
            candidates=(_fixed_candidate(),),
            selection_metric="aic",
        )


def test_candidate_set_requires_baseline_names_to_match_baseline_candidates() -> None:
    with pytest.raises(ValueError, match="must reference candidate baseline families"):
        KernelCandidateSet(
            candidate_set_id="set-bad-baseline-names",
            input_col="feed",
            target_col="product",
            time_col="timestamp",
            candidates=(_fixed_candidate(),),
            baseline_names=("no_lag",),
        )


def test_family_fit_result_requires_explicit_failure_error() -> None:
    with pytest.raises(ValueError, match="failed results must include an explicit error"):
        KernelFamilyFitResult(
            candidate=_fixed_candidate(),
            fit_result=None,
            succeeded=False,
            error=None,
            is_parametric=False,
            is_empirical=False,
            is_baseline=False,
        )


def test_selection_result_allows_empty_recommendation() -> None:
    candidate_set = KernelCandidateSet(
        candidate_set_id="set-selection-none",
        input_col="feed",
        target_col="product",
        time_col="timestamp",
        candidates=(_fixed_candidate(),),
    )
    comparison = KernelComparisonResult(
        candidate_set=candidate_set,
        family_results=(),
        comparison_table=pl.DataFrame({"candidate_id": ["fixed-1"]}),
        warnings=("No candidate met reliability threshold.",),
        selection_summary={"selected": False},
    )

    selection = KernelSelectionResult(
        selected_candidate_id=None,
        selected_kernel=None,
        selected_fit_result=None,
        selection_reason=None,
        selection_warnings=("No recommendation.",),
        all_candidates=comparison,
    )
    assert selection.selected_candidate_id is None
