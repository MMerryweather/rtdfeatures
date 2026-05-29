from __future__ import annotations

import polars as pl
import pytest

from rtdfeatures.candidates.contracts import (
    KernelCandidate,
    KernelCandidateSet,
    KernelCandidateType,
)
from rtdfeatures.diagnostics import (
    BaselineComparison,
    FitDiagnostics,
    IdentifiabilityReport,
    KernelComparisonResult,
    KernelFamilyFitResult,
    KernelFitResult,
)
from rtdfeatures.kernels import FixedDelayKernel, Kernel
from rtdfeatures.oof.reports import (
    RecoverableFoldError,
    _deterministic_fallback_candidate_id,
    _failed_fold_report,
    _resolve_selected_kernel_from_comparison,
)

_BASELINE: KernelCandidateType = "baseline"
_FIXED: KernelCandidateType = "fixed_kernel"


def _dummy_set() -> KernelCandidateSet:
    cand = KernelCandidate(
        candidate_id="dummy", family="no_lag", candidate_type=_BASELINE,
        fixed_parameters={}, min_lag=0, max_lag=0,
    )
    return KernelCandidateSet(
        candidate_set_id="test", input_col="in", target_col="out", time_col="t",
        candidates=(cand,), baseline_names=("no_lag",), metadata={},
    )


def test_recoverable_fold_error_can_be_raised() -> None:
    with pytest.raises(RecoverableFoldError):
        raise RecoverableFoldError("fold failed")


def test_failed_fold_report_has_zeroed_fields() -> None:
    report = _failed_fold_report(validation_rows=10)
    assert report.row_count == 10
    assert report.output_row_count == 10
    assert report.warmup_rows == 0
    assert report.feature_names == ()


def _make_kernel_fit_result() -> KernelFitResult:
    kernel = FixedDelayKernel(delay_steps=1, max_lag_steps=2, dt=1.0)
    fit_diag = FitDiagnostics(
        train_loss=0.5, validation_loss=0.6, input_variance=1.0, target_variance=1.0,
        kernel_weight_sum=1.0, mean_lag=1.0, p50_lag=1.0, p90_lag=2.0,
        tail_mass=0.5, boundary_mass_fraction=0.0,
    )
    id_report = IdentifiabilityReport(warnings=(), is_reliable=True)
    baselines = BaselineComparison(
        no_lag_validation_loss=1.0, best_single_lag_validation_loss=0.8,
        learned_validation_loss=0.6,
    )
    return KernelFitResult(
        kernel=kernel, fit_diagnostics=fit_diag, identifiability_report=id_report,
        baseline_comparison=baselines,
    )


def _make_family_result(
    candidate_id: str = "cand",
    fit_result: KernelFitResult | None = None,
    evaluated_kernel: Kernel | None = None,
) -> KernelFamilyFitResult:
    cand = KernelCandidate(
        candidate_id=candidate_id, family="test", candidate_type=_FIXED,
        fixed_parameters={"delay_steps": 1}, min_lag=0, max_lag=2,
    )
    return KernelFamilyFitResult(
        candidate=cand, fit_result=fit_result, succeeded=True, error=None,
        is_parametric=False, is_empirical=False, is_baseline=False,
        n_parameters=None, validation_loss=0.6, train_loss=0.5,
        warning_codes=(), evaluated_fixed_kernel=evaluated_kernel,
        fixed_baseline_comparison=None, evaluation_provenance=None,
    )


def test_resolve_selected_kernel_returns_none_when_no_match() -> None:
    fit_result = _make_kernel_fit_result()
    family_res = _make_family_result(candidate_id="other", fit_result=fit_result)
    comp_result = KernelComparisonResult(
        candidate_set=_dummy_set(), family_results=(family_res,),
        comparison_table=pl.DataFrame({"candidate_id": ["other"]}),
    )
    result = _resolve_selected_kernel_from_comparison(comp_result, "wanted")
    assert result is None


def test_resolve_selected_kernel_returns_fit_result_kernel_on_match() -> None:
    fit_result = _make_kernel_fit_result()
    family_res = _make_family_result(candidate_id="match", fit_result=fit_result)
    comp_result = KernelComparisonResult(
        candidate_set=_dummy_set(), family_results=(family_res,),
        comparison_table=pl.DataFrame({"candidate_id": ["match"]}),
    )
    result = _resolve_selected_kernel_from_comparison(comp_result, "match")
    assert result is fit_result.kernel


def test_resolve_selected_kernel_returns_evaluated_fixed_kernel() -> None:
    kernel = FixedDelayKernel(delay_steps=1, max_lag_steps=2, dt=1.0)
    family_res = _make_family_result(
        candidate_id="fix", fit_result=None, evaluated_kernel=kernel,
    )
    comp_result = KernelComparisonResult(
        candidate_set=_dummy_set(), family_results=(family_res,),
        comparison_table=pl.DataFrame({"candidate_id": ["fix"]}),
    )
    result = _resolve_selected_kernel_from_comparison(comp_result, "fix")
    assert result is kernel


def _make_comp(table: pl.DataFrame) -> KernelComparisonResult:
    return KernelComparisonResult(
        candidate_set=_dummy_set(), family_results=(),
        comparison_table=table,
    )


def test_deterministic_fallback_returns_none_on_empty_table() -> None:
    table = pl.DataFrame({"candidate_id": []}, schema={"candidate_id": pl.String})
    assert _deterministic_fallback_candidate_id(_make_comp(table)) is None


def test_deterministic_fallback_filters_succeeded_and_returns_none_if_empty() -> None:
    table = pl.DataFrame(
        {"candidate_id": ["a"], "succeeded": [False]},
        schema={"candidate_id": pl.String, "succeeded": pl.Boolean},
    )
    assert _deterministic_fallback_candidate_id(_make_comp(table)) is None


def test_deterministic_fallback_with_succeeded_and_validation_loss() -> None:
    table = pl.DataFrame(
        {"candidate_id": ["b", "a"], "validation_loss": [0.5, 0.3], "succeeded": [True, True]},
        schema={"candidate_id": pl.String, "validation_loss": pl.Float64, "succeeded": pl.Boolean},
    )
    result = _deterministic_fallback_candidate_id(_make_comp(table))
    assert result == "a"


def test_deterministic_fallback_with_null_validation_loss() -> None:
    table = pl.DataFrame(
        {"candidate_id": ["c", "d"], "validation_loss": [None, 0.4], "succeeded": [True, True]},
        schema={"candidate_id": pl.String, "validation_loss": pl.Float64, "succeeded": pl.Boolean},
    )
    result = _deterministic_fallback_candidate_id(_make_comp(table))
    assert result == "d"


def test_deterministic_fallback_without_validation_loss_sorts_by_id() -> None:
    table = pl.DataFrame(
        {"candidate_id": ["z", "a"], "succeeded": [True, True]},
        schema={"candidate_id": pl.String, "succeeded": pl.Boolean},
    )
    result = _deterministic_fallback_candidate_id(_make_comp(table))
    assert result == "a"


def test_deterministic_fallback_without_succeeded_column() -> None:
    table = pl.DataFrame({"candidate_id": ["b", "a"]}, schema={"candidate_id": pl.String})
    result = _deterministic_fallback_candidate_id(_make_comp(table))
    assert result == "a"


def test_deterministic_fallback_candidate_id_is_none() -> None:
    table = pl.DataFrame(
        {"candidate_id": [None]},
        schema={"candidate_id": pl.String},
    )
    result = _deterministic_fallback_candidate_id(_make_comp(table))
    assert result is None
