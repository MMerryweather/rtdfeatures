"""Tests for fold-aware out-of-fold feature evidence integration."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import polars as pl
import pytest

from rtdfeatures.diagnostics import (
    FeatureEvidenceReport,
    KernelCandidate,
    KernelCandidateSet,
    KernelComparisonResult,
    KernelSelectionResult,
)
from rtdfeatures.kernels import UniformKernel
from rtdfeatures.oof import ForwardChainingSplitConfig, fit_transform_oof


def _make_df(n_rows: int = 24) -> pl.DataFrame:
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return pl.DataFrame(
        {
            "timestamp": [t0 + timedelta(minutes=i) for i in range(n_rows)],
            "x": [float(i) for i in range(n_rows)],
            "y": [0.4 * float(i) + 2.0 for i in range(n_rows)],
        }
    )


def _candidate_set() -> KernelCandidateSet:
    return KernelCandidateSet(
        candidate_set_id="oof-evidence-tests",
        input_col="x",
        target_col="y",
        time_col="timestamp",
        candidates=(
            KernelCandidate(
                candidate_id="cand_a",
                family="fixed_delay",
                candidate_type="fixed_kernel",
                min_lag="0m",
                max_lag="2m",
                fixed_parameters={"delay_steps": 1},
            ),
        ),
    )


def _comparison(candidate_set: KernelCandidateSet, candidate_id: str) -> KernelComparisonResult:
    return KernelComparisonResult(
        candidate_set=candidate_set,
        family_results=(),
        comparison_table=pl.DataFrame(
            {
                "candidate_id": [candidate_id],
                "validation_loss": [0.1],
                "succeeded": [True],
            }
        ),
        warnings=(),
        selection_summary={},
    )


def _selection(
    comparison: KernelComparisonResult,
    candidate_id: str | None,
    *,
    kernel: UniformKernel | None = None,
) -> KernelSelectionResult:
    selected_kernel = kernel
    if candidate_id is not None and selected_kernel is None:
        selected_kernel = UniformKernel(
            min_lag_steps=0,
            max_lag_steps=1,
            dt=60.0,
            name=candidate_id,
        )
    return KernelSelectionResult(
        selected_candidate_id=candidate_id,
        selected_kernel=selected_kernel,
        selected_fit_result=None,
        selection_reason="deterministic test selection" if candidate_id is not None else None,
        selection_warnings=(),
        all_candidates=comparison,
    )


def _run_candidate_oof(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_second_fold: bool = False,
) -> Any:
    candidate_set = _candidate_set()
    calls = {"n": 0}

    def fake_fit_kernel_candidates(
        _train_df: pl.DataFrame, *_args: Any, **_kwargs: Any
    ) -> KernelComparisonResult:
        return _comparison(candidate_set, "cand_a")

    def fake_select_kernel_candidate(
        comparison_result: KernelComparisonResult,
        **_kwargs: Any,
    ) -> KernelSelectionResult:
        calls["n"] += 1
        if fail_second_fold and calls["n"] == 2:
            return _selection(comparison_result, None)
        return _selection(comparison_result, "cand_a")

    monkeypatch.setattr("rtdfeatures.oof.fit_kernel_candidates", fake_fit_kernel_candidates)
    monkeypatch.setattr("rtdfeatures.oof.select_kernel_candidate", fake_select_kernel_candidate)

    return fit_transform_oof(
        df=_make_df(),
        candidate_set=candidate_set,
        split_config=ForwardChainingSplitConfig(
            n_folds=2,
            min_train_size=8,
            validation_size=4,
            gap=0,
        ),
        input_col="x",
        target_col="y",
        time_col="timestamp",
        numeric_cols=["x"],
    )


def _fold_evidence_rows(report: FeatureEvidenceReport) -> list[dict[str, Any]]:
    rows = [
        item.metadata.get("fold_evidence")
        for item in report.feature_evidence
        if "fold_evidence" in item.metadata
    ]
    assert rows
    first = rows[0]
    assert isinstance(first, list)
    return first


def test_fold_evidence_count_matches_split_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _run_candidate_oof(monkeypatch)

    assert result.feature_evidence_report is not None
    fold_evidence = _fold_evidence_rows(result.feature_evidence_report)
    assert len(fold_evidence) == result.split_summary.n_folds


def test_oof_evidence_registry_builds_on_fresh_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _run_candidate_oof(monkeypatch)

    assert result.feature_evidence_report is not None
    generated_names = [name for name in result.features.columns if name != "timestamp"]
    evidence_names = [item.feature_name for item in result.feature_evidence_report.feature_evidence]
    assert set(evidence_names) == set(generated_names)


def test_fold_id_and_candidate_provenance_are_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _run_candidate_oof(monkeypatch)

    assert result.feature_evidence_report is not None
    fold_evidence = _fold_evidence_rows(result.feature_evidence_report)
    assert [row["fold_id"] for row in fold_evidence] == [0, 1]
    assert {row["candidate_id"] for row in fold_evidence} == {"cand_a"}
    assert {row["kernel_name"] for row in fold_evidence} == {"cand_a"}
    assert {row["evidence_scope"] for row in fold_evidence} == {"oof"}

    first_metadata = result.feature_evidence_report.feature_evidence[0].metadata
    assert first_metadata["evidence_scope"] == "oof"
    assert first_metadata["full_data_evidence_available"] is False


def test_feature_values_unchanged_by_evidence_integration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _run_candidate_oof(monkeypatch)

    def fake_build_feature_evidence(**_kwargs: Any) -> FeatureEvidenceReport:
        return FeatureEvidenceReport(
            feature_evidence=(),
            feature_count=0,
            kernel_count=0,
            source_columns=(),
            warning_summary={},
            evidence_summary_by_kernel={},
            evidence_summary_by_feature_family={},
        )

    monkeypatch.setattr("rtdfeatures.oof.build_feature_evidence", fake_build_feature_evidence)
    changed = _run_candidate_oof(monkeypatch)

    assert baseline.features.equals(changed.features)


def test_failed_fold_evidence_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _run_candidate_oof(monkeypatch, fail_second_fold=True)

    assert result.feature_evidence_report is not None
    fold_evidence = _fold_evidence_rows(result.feature_evidence_report)
    by_fold = {row["fold_id"]: row for row in fold_evidence}
    assert by_fold[1]["status"] == "failed"
    assert "No fold kernel selected from candidate comparison" in by_fold[1]["failure_reason"]
    assert by_fold[1]["kernel_name"] is None
    assert by_fold[1]["candidate_id"] is None


def test_repeated_candidate_id_preserves_fold_kernel_summaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_set = _candidate_set()
    calls = {"n": 0}
    fold_kernels = (
        UniformKernel(min_lag_steps=0, max_lag_steps=1, dt=60.0, name="cand_a"),
        UniformKernel(min_lag_steps=0, max_lag_steps=2, dt=60.0, name="cand_a"),
    )

    def fake_fit_kernel_candidates(
        _train_df: pl.DataFrame, *_args: Any, **_kwargs: Any
    ) -> KernelComparisonResult:
        return _comparison(candidate_set, "cand_a")

    def fake_select_kernel_candidate(
        comparison_result: KernelComparisonResult,
        **_kwargs: Any,
    ) -> KernelSelectionResult:
        idx = calls["n"]
        calls["n"] += 1
        return _selection(comparison_result, "cand_a", kernel=fold_kernels[idx])

    monkeypatch.setattr("rtdfeatures.oof.fit_kernel_candidates", fake_fit_kernel_candidates)
    monkeypatch.setattr("rtdfeatures.oof.select_kernel_candidate", fake_select_kernel_candidate)

    result = fit_transform_oof(
        df=_make_df(),
        candidate_set=candidate_set,
        split_config=ForwardChainingSplitConfig(
            n_folds=2,
            min_train_size=8,
            validation_size=4,
            gap=0,
        ),
        input_col="x",
        target_col="y",
        time_col="timestamp",
        numeric_cols=["x"],
    )

    assert result.feature_evidence_report is not None
    evidence_item = result.feature_evidence_report.feature_evidence[0]
    fold_kernel_summaries = evidence_item.metadata.get("fold_kernel_summaries")
    assert isinstance(fold_kernel_summaries, dict)
    cand_rows = fold_kernel_summaries.get("cand_a")
    assert isinstance(cand_rows, list)
    assert [row["fold_id"] for row in cand_rows] == [0, 1]
    assert cand_rows[0]["kernel_summary"] != cand_rows[1]["kernel_summary"]
    # Primary kernel summary remains deterministic (first successful fold), while
    # fold-aware summaries keep both fold-specific kernels explicit.
    assert evidence_item.kernel_summary == cand_rows[0]["kernel_summary"]
