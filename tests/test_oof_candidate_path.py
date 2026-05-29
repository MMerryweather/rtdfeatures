"""Tests for candidate-set out-of-fold feature generation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

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
        candidate_set_id="oof-candidate-tests",
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


def test_fold_candidate_comparison_uses_training_rows_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _make_df()
    candidate_set = _candidate_set()
    split = ForwardChainingSplitConfig(n_folds=2, min_train_size=8, validation_size=4, gap=0)
    observed_train_max_times: list[Any] = []

    def fake_fit_kernel_candidates(
        train_df: pl.DataFrame, *_args: Any, **_kwargs: Any
    ) -> KernelComparisonResult:
        observed_train_max_times.append(train_df.get_column("timestamp").max())
        return _comparison(candidate_set, "cand_a")

    def fake_select_kernel_candidate(
        comparison_result: KernelComparisonResult,
        **_kwargs: Any,
    ) -> KernelSelectionResult:
        return _selection(comparison_result, "cand_a")

    monkeypatch.setattr("rtdfeatures.oof.fit_kernel_candidates", fake_fit_kernel_candidates)
    monkeypatch.setattr("rtdfeatures.oof.select_kernel_candidate", fake_select_kernel_candidate)

    fit_transform_oof(
        df=df,
        candidate_set=candidate_set,
        split_config=split,
        input_col="x",
        target_col="y",
        time_col="timestamp",
        numeric_cols=["x"],
    )

    assert observed_train_max_times == [
        df.get_column("timestamp")[7],
        df.get_column("timestamp")[11],
    ]


def test_fold_selection_provenance_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_set = _candidate_set()

    def fake_fit_kernel_candidates(
        train_df: pl.DataFrame, *_args: Any, **_kwargs: Any
    ) -> KernelComparisonResult:
        assert train_df.height > 0
        return _comparison(candidate_set, "cand_a")

    def fake_select_kernel_candidate(
        comparison_result: KernelComparisonResult,
        **_kwargs: Any,
    ) -> KernelSelectionResult:
        return _selection(comparison_result, "cand_a")

    monkeypatch.setattr("rtdfeatures.oof.fit_kernel_candidates", fake_fit_kernel_candidates)
    monkeypatch.setattr("rtdfeatures.oof.select_kernel_candidate", fake_select_kernel_candidate)

    result = fit_transform_oof(
        df=_make_df(),
        candidate_set=candidate_set,
        split_config=ForwardChainingSplitConfig(
            n_folds=1,
            min_train_size=8,
            validation_size=4,
            gap=0,
        ),
        input_col="x",
        target_col="y",
        time_col="timestamp",
        numeric_cols=["x"],
    )

    fold_result = result.fold_results[0]
    assert fold_result["status"] == "succeeded"
    assert isinstance(fold_result["comparison_result"], KernelComparisonResult)
    assert isinstance(fold_result["selection_result"], KernelSelectionResult)
    selection_result = fold_result["selection_result"]
    assert selection_result is not None
    assert selection_result.selected_candidate_id == "cand_a"


def test_failed_fold_handling_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_set = _candidate_set()
    call_count = {"n": 0}

    def fake_fit_kernel_candidates(
        _train_df: pl.DataFrame, *_args: Any, **_kwargs: Any
    ) -> KernelComparisonResult:
        return _comparison(candidate_set, "cand_a")

    def fake_select_kernel_candidate(
        comparison_result: KernelComparisonResult,
        **_kwargs: Any,
    ) -> KernelSelectionResult:
        call_count["n"] += 1
        if call_count["n"] == 2:
            return _selection(comparison_result, None)
        return _selection(comparison_result, "cand_a")

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

    statuses = [entry["status"] for entry in result.fold_results]
    assert statuses == ["succeeded", "failed"]
    failed = result.fold_results[1]
    assert "failure_reason" in failed
    assert result.warnings


def test_null_output_for_failed_fold_validation_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_set = _candidate_set()
    call_count = {"n": 0}

    def fake_fit_kernel_candidates(
        _train_df: pl.DataFrame, *_args: Any, **_kwargs: Any
    ) -> KernelComparisonResult:
        return _comparison(candidate_set, "cand_a")

    def fake_select_kernel_candidate(
        comparison_result: KernelComparisonResult,
        **_kwargs: Any,
    ) -> KernelSelectionResult:
        call_count["n"] += 1
        if call_count["n"] == 2:
            return _selection(comparison_result, None)
        return _selection(comparison_result, "cand_a")

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

    generated_cols = [name for name in result.features.columns if name != "timestamp"]
    failed_fold_rows = [12, 13, 14, 15]
    for name in generated_cols:
        values = result.features.get_column(name).to_list()
        for row in failed_fold_rows:
            assert values[row] is None


def test_no_validation_leakage_through_candidate_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_set = _candidate_set()

    def fake_fit_kernel_candidates(
        train_df: pl.DataFrame, *_args: Any, **_kwargs: Any
    ) -> KernelComparisonResult:
        train_y_sum = float(train_df.get_column("y").sum())
        candidate_id = "cand_a" if train_y_sum < 200.0 else "cand_b"
        return _comparison(candidate_set, candidate_id)

    def fake_select_kernel_candidate(
        comparison_result: KernelComparisonResult,
        **_kwargs: Any,
    ) -> KernelSelectionResult:
        selected_id = str(comparison_result.comparison_table.get_column("candidate_id")[0])
        return _selection(comparison_result, selected_id)

    monkeypatch.setattr("rtdfeatures.oof.fit_kernel_candidates", fake_fit_kernel_candidates)
    monkeypatch.setattr("rtdfeatures.oof.select_kernel_candidate", fake_select_kernel_candidate)

    base_df = _make_df()
    changed_df = _make_df().with_columns(
        pl.when(pl.int_range(pl.len()) >= 8)
        .then(pl.col("y") * 100.0)
        .otherwise(pl.col("y"))
        .alias("y")
    )

    split = ForwardChainingSplitConfig(n_folds=1, min_train_size=8, validation_size=4, gap=0)
    base_result = fit_transform_oof(
        df=base_df,
        candidate_set=candidate_set,
        split_config=split,
        input_col="x",
        target_col="y",
        time_col="timestamp",
        numeric_cols=["x"],
    )
    changed_result = fit_transform_oof(
        df=changed_df,
        candidate_set=candidate_set,
        split_config=split,
        input_col="x",
        target_col="y",
        time_col="timestamp",
        numeric_cols=["x"],
    )

    base_selected = base_result.fold_results[0]["selection_result"]
    changed_selected = changed_result.fold_results[0]["selection_result"]
    assert base_selected is not None
    assert changed_selected is not None
    assert base_selected.selected_candidate_id == changed_selected.selected_candidate_id


def test_fold_exception_in_fit_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_set = _candidate_set()
    call_count = {"n": 0}

    def fake_fit_kernel_candidates(
        _train_df: pl.DataFrame, *_args: Any, **_kwargs: Any
    ) -> KernelComparisonResult:
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("comparison boom")
        return _comparison(candidate_set, "cand_a")

    def fake_select_kernel_candidate(
        comparison_result: KernelComparisonResult,
        **_kwargs: Any,
    ) -> KernelSelectionResult:
        return _selection(comparison_result, "cand_a")

    monkeypatch.setattr("rtdfeatures.oof.fit_kernel_candidates", fake_fit_kernel_candidates)
    monkeypatch.setattr("rtdfeatures.oof.select_kernel_candidate", fake_select_kernel_candidate)

    with pytest.raises(RuntimeError, match="comparison boom"):
        fit_transform_oof(
            df=_make_df(),
            candidate_set=candidate_set,
            split_config=ForwardChainingSplitConfig(n_folds=2, min_train_size=8, validation_size=4),
            input_col="x",
            target_col="y",
            time_col="timestamp",
            numeric_cols=["x"],
        )


def test_unsorted_input_raises_by_default() -> None:
    df = _make_df().reverse()
    with pytest.raises(ValueError, match="not sorted"):
        fit_transform_oof(
            df=df,
            candidate_set=_candidate_set(),
            split_config=ForwardChainingSplitConfig(n_folds=1, min_train_size=8, validation_size=4),
            input_col="x",
            target_col="y",
            time_col="timestamp",
            numeric_cols=["x"],
        )


def test_unsorted_input_order_by_time_opt_in() -> None:
    df = _make_df().reverse()
    result = fit_transform_oof(
        df=df,
        candidate_set=_candidate_set(),
        split_config=ForwardChainingSplitConfig(n_folds=1, min_train_size=8, validation_size=4),
        input_col="x",
        target_col="y",
        time_col="timestamp",
        numeric_cols=["x"],
        order_by_time=True,
    )
    assert result.features.get_column("timestamp").to_list() == sorted(
        df.get_column("timestamp").to_list()
    )


def test_validation_transform_uses_fold_history_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_set = _candidate_set()

    def fake_fit_kernel_candidates(
        _train_df: pl.DataFrame, *_args: Any, **_kwargs: Any
    ) -> KernelComparisonResult:
        return _comparison(candidate_set, "cand_a")

    def fake_select_kernel_candidate(
        comparison_result: KernelComparisonResult,
        **_kwargs: Any,
    ) -> KernelSelectionResult:
        return _selection(
            comparison_result,
            "cand_a",
            kernel=UniformKernel(min_lag_steps=0, max_lag_steps=2, dt=60.0, name="cand_a"),
        )

    monkeypatch.setattr("rtdfeatures.oof.fit_kernel_candidates", fake_fit_kernel_candidates)
    monkeypatch.setattr("rtdfeatures.oof.select_kernel_candidate", fake_select_kernel_candidate)

    result = fit_transform_oof(
        df=_make_df(),
        candidate_set=candidate_set,
        split_config=ForwardChainingSplitConfig(
            n_folds=1,
            min_train_size=8,
            validation_size=4,
            gap=0,
        ),
        input_col="x",
        target_col="y",
        time_col="timestamp",
        numeric_cols=["x"],
    )

    generated_cols = [name for name in result.features.columns if name != "timestamp"]
    validation_rows = [8, 9, 10, 11]
    for name in generated_cols:
        values = result.features.get_column(name).cast(pl.Float64).to_list()
        for idx in validation_rows:
            assert values[idx] is not None


def test_mixed_fold_candidates_union_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_set = _candidate_set()
    call_count = {"n": 0}

    def fake_fit_kernel_candidates(
        _train_df: pl.DataFrame, *_args: Any, **_kwargs: Any
    ) -> KernelComparisonResult:
        return _comparison(candidate_set, "cand_a")

    def fake_select_kernel_candidate(
        comparison_result: KernelComparisonResult,
        **_kwargs: Any,
    ) -> KernelSelectionResult:
        call_count["n"] += 1
        candidate_id = "cand_a" if call_count["n"] == 1 else "cand_b"
        return _selection(comparison_result, candidate_id)

    monkeypatch.setattr("rtdfeatures.oof.fit_kernel_candidates", fake_fit_kernel_candidates)
    monkeypatch.setattr("rtdfeatures.oof.select_kernel_candidate", fake_select_kernel_candidate)

    result = fit_transform_oof(
        df=_make_df(),
        candidate_set=candidate_set,
        split_config=ForwardChainingSplitConfig(n_folds=2, min_train_size=8, validation_size=4),
        input_col="x",
        target_col="y",
        time_col="timestamp",
        numeric_cols=["x"],
    )

    assert "cand_a_num_x_wmean" in result.features.columns
    assert "cand_b_num_x_wmean" in result.features.columns
    # Under schema union, coverage should count rows with any usable generated feature.
    assert result.split_summary.rows_with_features == 8
    assert result.split_summary.rows_without_features == result.features.height - 8


def test_select_candidate_per_fold_false_is_functional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_set = _candidate_set()

    def fake_fit_kernel_candidates(
        _train_df: pl.DataFrame, *_args: Any, **_kwargs: Any
    ) -> KernelComparisonResult:
        candidate = candidate_set.candidates[0]
        return KernelComparisonResult(
            candidate_set=candidate_set,
            family_results=(
                KernelFamilyFitResult(
                    candidate=candidate,
                    fit_result=None,
                    succeeded=True,
                    error=None,
                    is_parametric=False,
                    is_empirical=False,
                    is_baseline=False,
                    validation_loss=0.1,
                    evaluated_fixed_kernel=UniformKernel(
                        min_lag_steps=0,
                        max_lag_steps=1,
                        dt=60.0,
                        name="cand_a",
                    ),
                ),
            ),
            comparison_table=pl.DataFrame(
                {"candidate_id": ["cand_a"], "validation_loss": [0.1], "succeeded": [True]}
            ),
            warnings=(),
            selection_summary={},
        )

    def fail_if_called(*_args: Any, **_kwargs: Any) -> KernelSelectionResult:
        raise AssertionError("select_kernel_candidate must not be called")

    monkeypatch.setattr("rtdfeatures.oof.fit_kernel_candidates", fake_fit_kernel_candidates)
    monkeypatch.setattr("rtdfeatures.oof.select_kernel_candidate", fail_if_called)

    result = fit_transform_oof(
        df=_make_df(),
        candidate_set=candidate_set,
        select_candidate_per_fold=False,
        split_config=ForwardChainingSplitConfig(n_folds=1, min_train_size=8, validation_size=4),
        input_col="x",
        target_col="y",
        time_col="timestamp",
        numeric_cols=["x"],
    )

    assert result.fold_results[0]["status"] == "succeeded"
    assert result.fold_results[0]["selection_result"] is None


def test_fallback_without_validation_loss_column_does_not_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_set = _candidate_set()

    def fake_fit_kernel_candidates(
        _train_df: pl.DataFrame, *_args: Any, **_kwargs: Any
    ) -> KernelComparisonResult:
        candidate = candidate_set.candidates[0]
        return KernelComparisonResult(
            candidate_set=candidate_set,
            family_results=(
                KernelFamilyFitResult(
                    candidate=candidate,
                    fit_result=None,
                    succeeded=True,
                    error=None,
                    is_parametric=False,
                    is_empirical=False,
                    is_baseline=False,
                    validation_loss=None,
                    evaluated_fixed_kernel=UniformKernel(
                        min_lag_steps=0,
                        max_lag_steps=1,
                        dt=60.0,
                        name="cand_a",
                    ),
                ),
            ),
            comparison_table=pl.DataFrame(
                {
                    "candidate_id": ["cand_a"],
                    "succeeded": [True],
                }
            ),
            warnings=(),
            selection_summary={},
        )

    def fail_if_called(*_args: Any, **_kwargs: Any) -> KernelSelectionResult:
        raise AssertionError("select_kernel_candidate must not be called")

    monkeypatch.setattr("rtdfeatures.oof.fit_kernel_candidates", fake_fit_kernel_candidates)
    monkeypatch.setattr("rtdfeatures.oof.select_kernel_candidate", fail_if_called)

    result = fit_transform_oof(
        df=_make_df(),
        candidate_set=candidate_set,
        select_candidate_per_fold=False,
        split_config=ForwardChainingSplitConfig(n_folds=1, min_train_size=8, validation_size=4),
        input_col="x",
        target_col="y",
        time_col="timestamp",
        numeric_cols=["x"],
    )

    fold_result = result.fold_results[0]
    assert fold_result["status"] == "succeeded"
    assert fold_result["validation_loss"] is None
