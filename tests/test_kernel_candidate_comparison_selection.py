"""Tests for kernel candidate comparison and optional selection."""

from __future__ import annotations

import math

import polars as pl

from rtdfeatures.candidates import (
    fit_kernel_candidates,
    kernel_comparison_compact_dict,
    kernel_comparison_compact_text,
    kernel_comparison_table,
    select_kernel_candidate,
)
from rtdfeatures.diagnostics import (
    BaselineComparison,
    FitDiagnostics,
    IdentifiabilityReport,
    KernelCandidate,
    KernelCandidateSet,
    KernelComparisonResult,
    KernelFamilyFitResult,
    KernelFitResult,
    KernelShapeSummary,
)
from rtdfeatures.kernels import UniformKernel


def _make_df(n_rows: int = 72) -> pl.DataFrame:
    ts = pl.datetime_range(
        start=pl.datetime(2024, 1, 1, 0, 0, 0),
        end=pl.datetime(2024, 1, 1, 0, 0, 0) + pl.duration(minutes=n_rows - 1),
        interval="1m",
        eager=True,
    )
    x = [float(i) for i in range(n_rows)]
    y = [0.8 * x[i - 2] + 0.2 * x[i - 1] if i >= 2 else float(i) for i in range(n_rows)]
    return pl.DataFrame({"ts": ts, "x": x, "y": y})


def _fit_result(
    *,
    validation_loss: float,
    train_loss: float = 0.2,
    reliable: bool = True,
    warnings: tuple[str, ...] = (),
    warning_codes: tuple[str, ...] = (),
    no_lag: float = 0.8,
    best_single_lag: float = 0.7,
    fit_provenance: dict[str, object] | None = None,
) -> KernelFitResult:
    return KernelFitResult(
        kernel=UniformKernel(min_lag_steps=0, max_lag_steps=2, dt=60.0, name="k"),
        fit_diagnostics=FitDiagnostics(
            train_loss=train_loss,
            validation_loss=validation_loss,
            input_variance=1.0,
            target_variance=1.0,
            kernel_weight_sum=1.0,
            mean_lag=1.0,
            p50_lag=1.0,
            p90_lag=2.0,
            tail_mass=0.1,
            boundary_mass_fraction=0.0,
        ),
        identifiability_report=IdentifiabilityReport(
            warnings=warnings,
            is_reliable=reliable,
            warning_codes=warning_codes,
        ),
        baseline_comparison=BaselineComparison(
            no_lag_validation_loss=no_lag,
            best_single_lag_validation_loss=best_single_lag,
            learned_validation_loss=validation_loss,
        ),
        kernel_shape_summary=KernelShapeSummary(
            normalized_entropy=0.8,
            max_weight=0.6,
            min_weight=0.1,
            concentration_hhi=0.4,
            effective_lag_count=2.0,
        ),
        fit_provenance=fit_provenance
        or {
            "loss": "huber",
            "huber_delta": 1.0,
            "validation_fraction": 0.2,
            "dt_seconds": 60.0,
            "total_valid_windows": 50,
            "validation_windows": 10,
        },
    )


def test_comparison_table_schema_failures_warnings_and_nulls() -> None:
    df = _make_df()
    candidate_set = KernelCandidateSet(
        candidate_set_id="schema",
        input_col="x",
        target_col="y",
        time_col="ts",
        candidates=(
            KernelCandidate(
                candidate_id="simplex-ok",
                family="simplex",
                candidate_type="empirical_learner",
                min_lag="0m",
                max_lag="3m",
                learner_parameters={"max_epochs": 10, "seed": 3},
            ),
            KernelCandidate(
                candidate_id="fixed-delay",
                family="fixed_delay",
                candidate_type="fixed_kernel",
                min_lag="0m",
                max_lag="3m",
                fixed_parameters={"delay_steps": 2},
            ),
            KernelCandidate(
                candidate_id="bad-family",
                family="nope",
                candidate_type="parametric_learner",
                min_lag="0m",
                max_lag="3m",
                learner_parameters={"max_epochs": 10, "seed": 4},
            ),
        ),
    )
    result = fit_kernel_candidates(df, candidate_set)
    table = kernel_comparison_table(result)

    assert table.columns == [
        "candidate_id",
        "family",
        "candidate_type",
        "succeeded",
        "validation_loss",
        "train_loss",
        "mean_lag",
        "p50_lag",
        "p90_lag",
        "tail_mass",
        "warning_count",
        "warning_codes",
        "n_parameters",
        "beats_no_lag",
        "beats_best_single_lag",
        "error",
    ]
    bad = table.filter(pl.col("candidate_id") == "bad-family").row(0, named=True)
    assert bad["succeeded"] is False
    assert "Unsupported learner family" in str(bad["error"])
    fixed = table.filter(pl.col("candidate_id") == "fixed-delay").row(0, named=True)
    fixed_family_result = next(
        family_result
        for family_result in result.family_results
        if family_result.candidate.candidate_id == "fixed-delay"
    )
    expected_fixed_beats_no_lag = None
    expected_fixed_beats_best_single_lag = None
    if (
        fixed_family_result.fixed_baseline_comparison is not None
        and fixed_family_result.validation_loss is not None
    ):
        expected_fixed_beats_no_lag = (
            fixed_family_result.validation_loss
            < fixed_family_result.fixed_baseline_comparison.no_lag_validation_loss
        )
        expected_fixed_beats_best_single_lag = (
            fixed_family_result.validation_loss
            < fixed_family_result.fixed_baseline_comparison.best_single_lag_validation_loss
        )
    assert fixed["train_loss"] is None
    assert fixed["mean_lag"] is None
    assert fixed["beats_no_lag"] is expected_fixed_beats_no_lag
    assert fixed["beats_best_single_lag"] is expected_fixed_beats_best_single_lag
    assert any(warn.startswith("bad-family:") for warn in result.warnings)


def test_comparison_table_order_and_compact_helpers_are_deterministic() -> None:
    df = _make_df()
    candidate_set = KernelCandidateSet(
        candidate_set_id="order",
        input_col="x",
        target_col="y",
        time_col="ts",
        candidates=(
            KernelCandidate(
                candidate_id="z-simplex",
                family="simplex",
                candidate_type="empirical_learner",
                min_lag="0m",
                max_lag="3m",
                learner_parameters={"max_epochs": 10, "seed": 7},
            ),
            KernelCandidate(
                candidate_id="a-simplex",
                family="simplex",
                candidate_type="empirical_learner",
                min_lag="0m",
                max_lag="3m",
                learner_parameters={"max_epochs": 10, "seed": 8},
            ),
        ),
    )
    result = fit_kernel_candidates(df, candidate_set)
    table_a = kernel_comparison_table(result)
    table_b = kernel_comparison_table(result)
    assert table_a.to_dicts() == table_b.to_dicts()

    compact_a = kernel_comparison_compact_dict(result)
    compact_b = kernel_comparison_compact_dict(result)
    assert compact_a == compact_b
    assert kernel_comparison_compact_text(result).startswith("candidate validation losses: ")


def test_select_kernel_candidate_blocks_fixed_without_fit_result() -> None:
    candidate_set = KernelCandidateSet(
        candidate_set_id="tie",
        input_col="x",
        target_col="y",
        time_col="ts",
        candidates=(
            KernelCandidate(
                candidate_id="gamma",
                family="gamma",
                candidate_type="parametric_learner",
                min_lag="0m",
                max_lag="3m",
                learner_parameters={"max_epochs": 10},
            ),
            KernelCandidate(
                candidate_id="fixed-delay",
                family="fixed_delay",
                candidate_type="fixed_kernel",
                min_lag="0m",
                max_lag="3m",
                fixed_parameters={"delay_steps": 2},
            ),
        ),
    )
    gamma_result = KernelFamilyFitResult(
        candidate=candidate_set.candidates[0],
        fit_result=_fit_result(validation_loss=0.50, no_lag=0.9, best_single_lag=0.8),
        succeeded=True,
        error=None,
        is_parametric=True,
        is_empirical=False,
        is_baseline=False,
        n_parameters=6,
        validation_loss=0.50,
        train_loss=0.45,
    )
    evaluated_fixed_kernel = UniformKernel(
        min_lag_steps=0,
        max_lag_steps=3,
        dt=60.0,
        name="fixed-delay-evaluated",
    )
    fixed_result = KernelFamilyFitResult(
        candidate=candidate_set.candidates[1],
        fit_result=None,
        succeeded=True,
        error=None,
        is_parametric=False,
        is_empirical=False,
        is_baseline=False,
        n_parameters=3,
        validation_loss=0.505,
        train_loss=None,
        evaluated_fixed_kernel=evaluated_fixed_kernel,
        evaluation_provenance={
            "loss": "huber",
            "huber_delta": 1.0,
            "validation_fraction": 0.2,
            "dt_seconds": 60.0,
            "total_valid_windows": 50,
            "validation_windows": 10,
        },
    )
    comparison_seed = fit_kernel_candidates(_make_df(), candidate_set)
    comparison = KernelComparisonResult(
        candidate_set=comparison_seed.candidate_set,
        family_results=(gamma_result, fixed_result),
        comparison_table=comparison_seed.comparison_table,
        warnings=(),
        selection_summary={},
    )
    selected = select_kernel_candidate(comparison, loss_tolerance_fraction=0.02)
    assert selected.selected_candidate_id == "gamma"
    assert gamma_result.fit_result is not None
    assert selected.selected_fit_result is gamma_result.fit_result
    assert selected.selected_kernel is gamma_result.fit_result.kernel


def test_select_kernel_candidate_uses_evaluated_fixed_kernel_from_fit_results() -> None:
    candidate_set = KernelCandidateSet(
        candidate_set_id="fixed-evaluated-selection",
        input_col="x",
        target_col="y",
        time_col="ts",
        candidates=(
            KernelCandidate(
                candidate_id="fixed-delay",
                family="fixed_delay",
                candidate_type="fixed_kernel",
                min_lag="0m",
                max_lag="3m",
                fixed_parameters={"delay_steps": 2},
            ),
        ),
    )
    fixed_result = KernelFamilyFitResult(
        candidate=candidate_set.candidates[0],
        fit_result=None,
        succeeded=True,
        error=None,
        is_parametric=False,
        is_empirical=False,
        is_baseline=False,
        n_parameters=3,
        validation_loss=0.2,
        evaluated_fixed_kernel=UniformKernel(
            min_lag_steps=0,
            max_lag_steps=3,
            dt=60.0,
            name="fixed-delay-evaluated",
        ),
        fixed_baseline_comparison=BaselineComparison(
            no_lag_validation_loss=0.9,
            best_single_lag_validation_loss=0.8,
            learned_validation_loss=0.2,
        ),
        evaluation_provenance={
            "loss": "huber",
            "huber_delta": 1.0,
            "validation_fraction": 0.2,
            "dt_seconds": 60.0,
            "total_valid_windows": 50,
            "validation_windows": 10,
        },
    )
    comparison = KernelComparisonResult(
        candidate_set=candidate_set,
        family_results=(fixed_result,),
        comparison_table=pl.DataFrame(),
        warnings=(),
        selection_summary={},
    )
    selected = select_kernel_candidate(comparison)
    assert selected.selected_candidate_id == "fixed-delay"
    assert selected.selected_fit_result is None
    assert selected.selected_kernel is fixed_result.evaluated_fixed_kernel


def test_select_kernel_candidate_blocks_fixed_missing_evaluation_provenance() -> None:
    candidate_set = KernelCandidateSet(
        candidate_set_id="fixed-missing-eval-provenance",
        input_col="x",
        target_col="y",
        time_col="ts",
        candidates=(
            KernelCandidate(
                candidate_id="fixed-delay",
                family="fixed_delay",
                candidate_type="fixed_kernel",
                min_lag="0m",
                max_lag="3m",
                fixed_parameters={"delay_steps": 2},
            ),
        ),
    )
    fixed_result = KernelFamilyFitResult(
        candidate=candidate_set.candidates[0],
        fit_result=None,
        succeeded=True,
        error=None,
        is_parametric=False,
        is_empirical=False,
        is_baseline=False,
        n_parameters=3,
        validation_loss=0.2,
        evaluated_fixed_kernel=UniformKernel(
            min_lag_steps=0,
            max_lag_steps=3,
            dt=60.0,
            name="fixed-delay-evaluated",
        ),
        fixed_baseline_comparison=BaselineComparison(
            no_lag_validation_loss=0.9,
            best_single_lag_validation_loss=0.8,
            learned_validation_loss=0.2,
        ),
        evaluation_provenance=None,
    )
    comparison = KernelComparisonResult(
        candidate_set=candidate_set,
        family_results=(fixed_result,),
        comparison_table=pl.DataFrame(),
        warnings=(),
        selection_summary={},
    )
    selected = select_kernel_candidate(comparison)
    assert selected.selected_candidate_id is None
    assert any(
        "Fixed-kernel candidate 'fixed-delay' is not selection-eligible; "
        "missing required evidence: evaluation_provenance." in warning
        for warning in selected.selection_warnings
    )


def test_select_kernel_candidate_blocks_incompatible_contexts() -> None:
    candidate_set = KernelCandidateSet(
        candidate_set_id="context-mismatch",
        input_col="x",
        target_col="y",
        time_col="ts",
        candidates=(
            KernelCandidate(
                candidate_id="gamma",
                family="gamma",
                candidate_type="parametric_learner",
                min_lag="0m",
                max_lag="3m",
                learner_parameters={"max_epochs": 10},
            ),
            KernelCandidate(
                candidate_id="fixed-delay",
                family="fixed_delay",
                candidate_type="fixed_kernel",
                min_lag="0m",
                max_lag="3m",
                fixed_parameters={"delay_steps": 2},
            ),
        ),
    )
    gamma_result = KernelFamilyFitResult(
        candidate=candidate_set.candidates[0],
        fit_result=_fit_result(validation_loss=0.20, no_lag=0.9, best_single_lag=0.8),
        succeeded=True,
        error=None,
        is_parametric=True,
        is_empirical=False,
        is_baseline=False,
        n_parameters=6,
        validation_loss=0.20,
        train_loss=0.18,
    )
    fixed_result = KernelFamilyFitResult(
        candidate=candidate_set.candidates[1],
        fit_result=None,
        succeeded=True,
        error=None,
        is_parametric=False,
        is_empirical=False,
        is_baseline=False,
        n_parameters=3,
        validation_loss=0.19,
        evaluated_fixed_kernel=UniformKernel(
            min_lag_steps=0,
            max_lag_steps=3,
            dt=60.0,
            name="fixed-delay-evaluated",
        ),
        fixed_baseline_comparison=BaselineComparison(
            no_lag_validation_loss=0.9,
            best_single_lag_validation_loss=0.8,
            learned_validation_loss=0.19,
        ),
        evaluation_provenance={
            "loss": "huber",
            "huber_delta": 1.0,
            "validation_fraction": 0.25,
            "dt_seconds": 60.0,
            "total_valid_windows": 50,
            "validation_windows": 12,
        },
    )
    comparison = KernelComparisonResult(
        candidate_set=candidate_set,
        family_results=(gamma_result, fixed_result),
        comparison_table=pl.DataFrame(),
        warnings=(),
        selection_summary={},
    )
    selected = select_kernel_candidate(comparison)
    assert selected.selected_candidate_id is None
    assert any(
        "candidate validation contexts are not comparable" in warning
        for warning in selected.selection_warnings
    )


def test_select_kernel_candidate_blocks_mixed_missing_context_signatures() -> None:
    candidate_set = KernelCandidateSet(
        candidate_set_id="context-missing-signature",
        input_col="x",
        target_col="y",
        time_col="ts",
        candidates=(
            KernelCandidate(
                candidate_id="gamma-with-signature",
                family="gamma",
                candidate_type="parametric_learner",
                min_lag="0m",
                max_lag="3m",
                learner_parameters={"max_epochs": 10},
            ),
            KernelCandidate(
                candidate_id="simplex-missing-signature",
                family="simplex",
                candidate_type="empirical_learner",
                min_lag="0m",
                max_lag="3m",
                learner_parameters={"max_epochs": 10},
            ),
        ),
    )
    gamma_result = KernelFamilyFitResult(
        candidate=candidate_set.candidates[0],
        fit_result=_fit_result(validation_loss=0.20, no_lag=0.9, best_single_lag=0.8),
        succeeded=True,
        error=None,
        is_parametric=True,
        is_empirical=False,
        is_baseline=False,
        n_parameters=6,
        validation_loss=0.20,
        train_loss=0.18,
    )
    simplex_result = KernelFamilyFitResult(
        candidate=candidate_set.candidates[1],
        fit_result=_fit_result(
            validation_loss=0.21,
            no_lag=0.9,
            best_single_lag=0.8,
            fit_provenance={"loss": "huber"},
        ),
        succeeded=True,
        error=None,
        is_parametric=False,
        is_empirical=True,
        is_baseline=False,
        n_parameters=4,
        validation_loss=0.21,
        train_loss=0.19,
    )
    comparison = KernelComparisonResult(
        candidate_set=candidate_set,
        family_results=(gamma_result, simplex_result),
        comparison_table=pl.DataFrame(),
        warnings=(),
        selection_summary={},
    )
    selected = select_kernel_candidate(comparison)
    assert selected.selected_candidate_id is None
    assert any(
        "candidate validation contexts are not comparable" in warning
        for warning in selected.selection_warnings
    )
    assert any(
        "Missing evaluation context/provenance signature for candidate(s): "
        "simplex-missing-signature." in warning
        for warning in selected.selection_warnings
    )


def test_select_kernel_candidate_accepts_validation_rows_signature_compat() -> None:
    candidate_set = KernelCandidateSet(
        candidate_set_id="context-validation-rows-compat",
        input_col="x",
        target_col="y",
        time_col="ts",
        candidates=(
            KernelCandidate(
                candidate_id="gamma-validation-rows",
                family="gamma",
                candidate_type="parametric_learner",
                min_lag="0m",
                max_lag="3m",
                learner_parameters={"max_epochs": 10},
            ),
            KernelCandidate(
                candidate_id="fixed-validation-windows",
                family="fixed_delay",
                candidate_type="fixed_kernel",
                min_lag="0m",
                max_lag="3m",
                fixed_parameters={"delay_steps": 2},
            ),
        ),
    )
    gamma_result = KernelFamilyFitResult(
        candidate=candidate_set.candidates[0],
        fit_result=_fit_result(
            validation_loss=0.20,
            no_lag=0.9,
            best_single_lag=0.8,
            fit_provenance={
                "loss": "huber",
                "huber_delta": 1.0,
                "validation_fraction": 0.2,
                "dt_seconds": 60.0,
                "total_valid_windows": 50,
                "validation_rows": 10,
            },
        ),
        succeeded=True,
        error=None,
        is_parametric=True,
        is_empirical=False,
        is_baseline=False,
        n_parameters=6,
        validation_loss=0.20,
        train_loss=0.18,
    )
    fixed_result = KernelFamilyFitResult(
        candidate=candidate_set.candidates[1],
        fit_result=None,
        succeeded=True,
        error=None,
        is_parametric=False,
        is_empirical=False,
        is_baseline=False,
        n_parameters=3,
        validation_loss=0.30,
        evaluated_fixed_kernel=UniformKernel(
            min_lag_steps=0,
            max_lag_steps=3,
            dt=60.0,
            name="fixed-delay-evaluated",
        ),
        fixed_baseline_comparison=BaselineComparison(
            no_lag_validation_loss=0.9,
            best_single_lag_validation_loss=0.8,
            learned_validation_loss=0.30,
        ),
        evaluation_provenance={
            "loss": "huber",
            "huber_delta": 1.0,
            "validation_fraction": 0.2,
            "dt_seconds": 60.0,
            "total_valid_windows": 50,
            "validation_windows": 10,
        },
    )
    comparison = KernelComparisonResult(
        candidate_set=candidate_set,
        family_results=(gamma_result, fixed_result),
        comparison_table=pl.DataFrame(),
        warnings=(),
        selection_summary={},
    )

    selected = select_kernel_candidate(comparison)
    assert selected.selected_candidate_id == "gamma-validation-rows"
    assert not any(
        "candidate validation contexts are not comparable" in warning
        for warning in selected.selection_warnings
    )


def test_select_kernel_candidate_synthetic_gamma_and_delay_and_weak_data() -> None:
    candidate_set = KernelCandidateSet(
        candidate_set_id="synth",
        input_col="x",
        target_col="y",
        time_col="ts",
        candidates=(
            KernelCandidate(
                candidate_id="gamma",
                family="gamma",
                candidate_type="parametric_learner",
                min_lag="0m",
                max_lag="3m",
                learner_parameters={"max_epochs": 10},
            ),
            KernelCandidate(
                candidate_id="fixed-delay",
                family="fixed_delay",
                candidate_type="fixed_kernel",
                min_lag="0m",
                max_lag="3m",
                fixed_parameters={"delay_steps": 2},
            ),
        ),
    )
    gamma_win = KernelFamilyFitResult(
        candidate=candidate_set.candidates[0],
        fit_result=_fit_result(validation_loss=0.2, no_lag=0.7, best_single_lag=0.6),
        succeeded=True,
        error=None,
        is_parametric=True,
        is_empirical=False,
        is_baseline=False,
        n_parameters=6,
        validation_loss=0.2,
        train_loss=0.19,
    )
    delay_win = KernelFamilyFitResult(
        candidate=candidate_set.candidates[1],
        fit_result=None,
        succeeded=True,
        error=None,
        is_parametric=False,
        is_empirical=False,
        is_baseline=False,
        n_parameters=3,
        validation_loss=0.25,
        evaluation_provenance={
            "loss": "huber",
            "huber_delta": 1.0,
            "validation_fraction": 0.2,
            "dt_seconds": 60.0,
            "total_valid_windows": 50,
            "validation_windows": 10,
        },
    )
    comparison_seed = fit_kernel_candidates(_make_df(), candidate_set)
    gamma_selected = KernelComparisonResult(
        candidate_set=comparison_seed.candidate_set,
        family_results=(gamma_win, delay_win),
        comparison_table=comparison_seed.comparison_table,
        warnings=(),
        selection_summary={},
    )
    assert select_kernel_candidate(gamma_selected).selected_candidate_id == "gamma"

    gamma_weaker = KernelFamilyFitResult(
        candidate=candidate_set.candidates[0],
        fit_result=_fit_result(validation_loss=0.5),
        succeeded=True,
        error=None,
        is_parametric=True,
        is_empirical=False,
        is_baseline=False,
        n_parameters=6,
        validation_loss=0.5,
        train_loss=0.45,
    )
    delay_best = KernelComparisonResult(
        candidate_set=comparison_seed.candidate_set,
        family_results=(
            gamma_weaker,
            KernelFamilyFitResult(**{**delay_win.__dict__, "validation_loss": 0.2}),
        ),
        comparison_table=comparison_seed.comparison_table,
        warnings=(),
        selection_summary={},
    )
    assert select_kernel_candidate(delay_best).selected_candidate_id == "gamma"

    weak_gamma = KernelFamilyFitResult(
        candidate=candidate_set.candidates[0],
        fit_result=_fit_result(
            validation_loss=math.inf,
            reliable=False,
            warnings=("weak signal",),
            warning_codes=("WEAK",),
        ),
        succeeded=True,
        error=None,
        is_parametric=True,
        is_empirical=False,
        is_baseline=False,
        n_parameters=6,
        validation_loss=math.inf,
        train_loss=math.inf,
    )
    weak_delay = KernelFamilyFitResult(**{**delay_win.__dict__, "validation_loss": math.inf})
    weak = KernelComparisonResult(
        candidate_set=comparison_seed.candidate_set,
        family_results=(
            weak_gamma,
            weak_delay,
        ),
        comparison_table=comparison_seed.comparison_table,
        warnings=("weak",),
        selection_summary={},
    )
    none_selected = select_kernel_candidate(weak)
    assert none_selected.selected_candidate_id is None
    assert "No strong recommendation" in " ".join(none_selected.selection_warnings)


def test_acceptance_pure_delay_synthetic_selects_fixed_delay() -> None:
    candidate_set = KernelCandidateSet(
        candidate_set_id="pure-delay-selection",
        input_col="x",
        target_col="y",
        time_col="ts",
        candidates=(
            KernelCandidate(
                candidate_id="gamma",
                family="gamma",
                candidate_type="parametric_learner",
                min_lag="0m",
                max_lag="3m",
                learner_parameters={"max_epochs": 10},
            ),
            KernelCandidate(
                candidate_id="fixed-delay",
                family="fixed_delay",
                candidate_type="fixed_kernel",
                min_lag="0m",
                max_lag="3m",
                fixed_parameters={"delay_steps": 2},
            ),
        ),
    )
    gamma_result = KernelFamilyFitResult(
        candidate=candidate_set.candidates[0],
        fit_result=_fit_result(validation_loss=0.28, no_lag=0.55, best_single_lag=0.48),
        succeeded=True,
        error=None,
        is_parametric=True,
        is_empirical=False,
        is_baseline=False,
        n_parameters=6,
        validation_loss=0.28,
        train_loss=0.25,
    )
    fixed_result = KernelFamilyFitResult(
        candidate=candidate_set.candidates[1],
        fit_result=None,
        succeeded=True,
        error=None,
        is_parametric=False,
        is_empirical=False,
        is_baseline=False,
        n_parameters=3,
        validation_loss=0.20,
        train_loss=None,
        evaluated_fixed_kernel=UniformKernel(
            min_lag_steps=0,
            max_lag_steps=3,
            dt=60.0,
            name="fixed-delay-evaluated",
        ),
        fixed_baseline_comparison=BaselineComparison(
            no_lag_validation_loss=0.55,
            best_single_lag_validation_loss=0.48,
            learned_validation_loss=0.20,
        ),
        evaluation_provenance={
            "loss": "huber",
            "huber_delta": 1.0,
            "validation_fraction": 0.2,
            "dt_seconds": 60.0,
            "total_valid_windows": 50,
            "validation_windows": 10,
        },
    )
    comparison = KernelComparisonResult(
        candidate_set=candidate_set,
        family_results=(gamma_result, fixed_result),
        comparison_table=pl.DataFrame(),
        warnings=(),
        selection_summary={},
    )
    selected = select_kernel_candidate(comparison, loss_tolerance_fraction=0.01)
    assert selected.selected_candidate_id == "fixed-delay"
    assert selected.selected_fit_result is None
    assert selected.selected_kernel is fixed_result.evaluated_fixed_kernel


def test_kernel_comparison_compact_dict_ignores_non_finite_losses_for_best_id() -> None:
    candidate_set = KernelCandidateSet(
        candidate_set_id="compact-finite",
        input_col="x",
        target_col="y",
        time_col="ts",
        candidates=(
            KernelCandidate(
                candidate_id="finite-good",
                family="simplex",
                candidate_type="empirical_learner",
                min_lag="0m",
                max_lag="3m",
                learner_parameters={"max_epochs": 10},
            ),
            KernelCandidate(
                candidate_id="nan-loss",
                family="simplex",
                candidate_type="empirical_learner",
                min_lag="0m",
                max_lag="3m",
                learner_parameters={"max_epochs": 10},
            ),
            KernelCandidate(
                candidate_id="inf-loss",
                family="simplex",
                candidate_type="empirical_learner",
                min_lag="0m",
                max_lag="3m",
                learner_parameters={"max_epochs": 10},
            ),
        ),
    )
    finite = KernelFamilyFitResult(
        candidate=candidate_set.candidates[0],
        fit_result=_fit_result(validation_loss=0.31),
        succeeded=True,
        error=None,
        is_parametric=False,
        is_empirical=True,
        is_baseline=False,
        n_parameters=4,
        validation_loss=0.31,
        train_loss=0.28,
    )
    nan_loss = KernelFamilyFitResult(
        candidate=candidate_set.candidates[1],
        fit_result=_fit_result(validation_loss=math.nan),
        succeeded=True,
        error=None,
        is_parametric=False,
        is_empirical=True,
        is_baseline=False,
        n_parameters=4,
        validation_loss=math.nan,
        train_loss=0.28,
    )
    inf_loss = KernelFamilyFitResult(
        candidate=candidate_set.candidates[2],
        fit_result=_fit_result(validation_loss=math.inf),
        succeeded=True,
        error=None,
        is_parametric=False,
        is_empirical=True,
        is_baseline=False,
        n_parameters=4,
        validation_loss=math.inf,
        train_loss=0.28,
    )
    comparison_seed = fit_kernel_candidates(_make_df(), candidate_set)
    comparison = KernelComparisonResult(
        candidate_set=comparison_seed.candidate_set,
        family_results=(finite, nan_loss, inf_loss),
        comparison_table=comparison_seed.comparison_table,
        warnings=(),
        selection_summary={},
    )
    compact = kernel_comparison_compact_dict(comparison)
    assert compact["best_candidate_id_by_validation_loss"] == "finite-good"
