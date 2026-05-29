"""Tests for bootstrap summary builders."""

from __future__ import annotations

import pytest

from rtdfeatures.bootstrap import build_kernel_bootstrap_summary
from rtdfeatures.bootstrap.summaries import (
    bootstrap_lag_interval_table,
    bootstrap_lag_summary_samples_table,
    bootstrap_parameter_interval_table,
    bootstrap_parameter_samples_table,
    bootstrap_summary_compact_dict,
    bootstrap_summary_compact_text,
    bootstrap_weight_interval_table,
    bootstrap_weight_samples_table,
)
from rtdfeatures.candidates.contracts import KernelCandidateType
from rtdfeatures.diagnostics import (
    DEFAULT_BOOTSTRAP_INTERVAL_QUANTILES,
    BootstrapLagSummarySample,
    BootstrapParameterSample,
    BootstrapResult,
    BootstrapWeightSample,
)

_CAND_FIXED: KernelCandidateType = "fixed_kernel"
_CAND_EMPIRICAL: KernelCandidateType = "empirical_learner"
_CAND_PARAM: KernelCandidateType = "parametric_learner"
_CAND_BASELINE: KernelCandidateType = "baseline"


def _bootstrap_result() -> BootstrapResult:
    return BootstrapResult(
        n_bootstrap=5,
        n_succeeded=3,
        n_failed=2,
        failures=(
            {"bootstrap_id": 3, "error": "RuntimeError: fail"},
            {"bootstrap_id": 4, "error": "RuntimeError: fail"},
        ),
        weight_samples=(
            BootstrapWeightSample(
                bootstrap_id=0, candidate_id="cand", lag_step=0, lag_time=0.0, weight=0.10
            ),
            BootstrapWeightSample(
                bootstrap_id=1, candidate_id="cand", lag_step=0, lag_time=0.0, weight=0.20
            ),
            BootstrapWeightSample(
                bootstrap_id=2, candidate_id="cand", lag_step=0, lag_time=0.0, weight=0.30
            ),
            BootstrapWeightSample(
                bootstrap_id=0, candidate_id="cand", lag_step=1, lag_time=1.0, weight=0.90
            ),
            BootstrapWeightSample(
                bootstrap_id=1, candidate_id="cand", lag_step=1, lag_time=1.0, weight=0.80
            ),
            BootstrapWeightSample(
                bootstrap_id=2, candidate_id="cand", lag_step=1, lag_time=1.0, weight=0.70
            ),
        ),
        parameter_samples=(
            BootstrapParameterSample(
                bootstrap_id=0,
                candidate_id="cand",
                parameter_name="rate_lambda",
                parameter_value=1.0,
            ),
            BootstrapParameterSample(
                bootstrap_id=1,
                candidate_id="cand",
                parameter_name="rate_lambda",
                parameter_value=2.0,
            ),
            BootstrapParameterSample(
                bootstrap_id=2,
                candidate_id="cand",
                parameter_name="rate_lambda",
                parameter_value=3.0,
            ),
        ),
        lag_summary_samples=(
            BootstrapLagSummarySample(
                bootstrap_id=0,
                candidate_id="cand",
                mean_lag=1.0,
                p50_lag=1.0,
                p90_lag=2.0,
                tail_mass=0.10,
            ),
            BootstrapLagSummarySample(
                bootstrap_id=1,
                candidate_id="cand",
                mean_lag=2.0,
                p50_lag=2.0,
                p90_lag=3.0,
                tail_mass=0.20,
            ),
            BootstrapLagSummarySample(
                bootstrap_id=2,
                candidate_id="cand",
                mean_lag=3.0,
                p50_lag=3.0,
                p90_lag=4.0,
                tail_mass=0.30,
            ),
        ),
        family_selection_counts={"fam_a": 2, "fam_b": 1},
        warnings=(),
        bootstrap_config={"candidate_id": "cand"},
    )


def test_weight_parameter_and_lag_intervals_are_deterministic() -> None:
    boot = _bootstrap_result()
    first, first_warnings = build_kernel_bootstrap_summary(boot, candidate_id="cand")
    second, second_warnings = build_kernel_bootstrap_summary(boot, candidate_id="cand")

    assert first == second
    assert first_warnings == second_warnings
    assert first.mean_lag_interval == pytest.approx((1.05, 2.95))
    assert first.p50_lag_interval == pytest.approx((1.05, 2.95))
    assert first.p90_lag_interval == pytest.approx((2.05, 3.95))
    assert first.tail_mass_interval == pytest.approx((0.105, 0.295))

    by_lag = {item.lag_step: item for item in first.weight_interval_by_lag}
    assert by_lag[0].lower == pytest.approx(0.105)
    assert by_lag[0].upper == pytest.approx(0.295)
    assert by_lag[1].lower == pytest.approx(0.705)
    assert by_lag[1].upper == pytest.approx(0.895)

    by_name = {item.parameter_name: item for item in first.parameter_interval_by_name}
    assert by_name["rate_lambda"].lower == pytest.approx(1.05)
    assert by_name["rate_lambda"].upper == pytest.approx(2.95)
    assert by_name["rate_lambda"].n_samples == 3


def test_failed_iterations_are_excluded_but_failure_counts_remain() -> None:
    boot = _bootstrap_result()
    summary, _warnings = build_kernel_bootstrap_summary(boot, candidate_id="cand")

    assert len(summary.weight_interval_by_lag) == 2
    assert boot.n_failed == 2
    assert len(boot.failures) == 2


def test_too_few_successes_warning() -> None:
    boot = _bootstrap_result()
    _summary, warnings = build_kernel_bootstrap_summary(
        boot,
        candidate_id="cand",
        min_successes_for_warning=4,
    )
    assert "BOOTSTRAP_TOO_FEW_SUCCESSES" in warnings


def test_family_instability_warning() -> None:
    boot = _bootstrap_result()
    _summary, warnings = build_kernel_bootstrap_summary(
        boot,
        candidate_id="cand",
        family_stability_threshold=0.8,
    )
    assert "BOOTSTRAP_FAMILY_UNSTABLE" in warnings


def test_interval_touches_boundary_warning() -> None:
    boot = _bootstrap_result()
    _summary, warnings = build_kernel_bootstrap_summary(
        boot,
        candidate_id="cand",
        lag_bounds=(1.05, 4.5),
    )
    assert "BOOTSTRAP_INTERVAL_TOUCHES_BOUNDARY" in warnings


def test_empty_summary_is_valid() -> None:
    empty = BootstrapResult(
        n_bootstrap=4,
        n_succeeded=0,
        n_failed=4,
        failures=(
            {"bootstrap_id": 0, "error": "ValueError: fail"},
            {"bootstrap_id": 1, "error": "ValueError: fail"},
            {"bootstrap_id": 2, "error": "ValueError: fail"},
            {"bootstrap_id": 3, "error": "ValueError: fail"},
        ),
        weight_samples=(),
        parameter_samples=(),
        lag_summary_samples=(),
        family_selection_counts={},
        warnings=(),
        bootstrap_config={},
    )
    summary, warnings = build_kernel_bootstrap_summary(empty)
    assert summary.weight_interval_by_lag == ()
    assert summary.parameter_interval_by_name == ()
    assert summary.stability_score == 0.0
    assert "BOOTSTRAP_TOO_FEW_SUCCESSES" in warnings


def test_default_quantiles_match_contract() -> None:
    assert DEFAULT_BOOTSTRAP_INTERVAL_QUANTILES == (0.025, 0.975)


def test_build_summary_raises_on_bad_interval_quantiles() -> None:
    boot = _bootstrap_result()
    with pytest.raises(ValueError, match="interval_quantiles"):
        build_kernel_bootstrap_summary(boot, interval_quantiles=(-0.1, 0.9))
    with pytest.raises(ValueError, match="interval_quantiles"):
        build_kernel_bootstrap_summary(boot, interval_quantiles=(0.5, 0.3))


def test_build_summary_raises_on_bad_min_successes() -> None:
    boot = _bootstrap_result()
    with pytest.raises(ValueError, match="min_successes_for_warning"):
        build_kernel_bootstrap_summary(boot, min_successes_for_warning=0)


def test_build_summary_raises_on_bad_stability_threshold() -> None:
    boot = _bootstrap_result()
    with pytest.raises(ValueError, match="family_stability_threshold"):
        build_kernel_bootstrap_summary(boot, family_stability_threshold=0.0)


def test_build_summary_too_few_successes_and_unstable_warning() -> None:
    boot = _bootstrap_result()
    _summary, warnings = build_kernel_bootstrap_summary(
        boot,
        candidate_id="cand",
        min_successes_for_warning=10,
        family_stability_threshold=0.8,
    )
    assert "BOOTSTRAP_TOO_FEW_SUCCESSES" in warnings
    assert "BOOTSTRAP_FAMILY_UNSTABLE" in warnings


def test_bootstrap_weight_interval_table() -> None:
    boot = _bootstrap_result()
    table = bootstrap_weight_interval_table(boot, candidate_id="cand")
    assert "candidate_id" in table.columns
    assert "lag_step" in table.columns
    assert "weight_estimate" in table.columns
    assert table.height >= 2


def test_bootstrap_parameter_interval_table() -> None:
    boot = _bootstrap_result()
    table = bootstrap_parameter_interval_table(boot, candidate_id="cand")
    assert "parameter_name" in table.columns
    assert table.height >= 1


def test_bootstrap_lag_interval_table() -> None:
    boot = _bootstrap_result()
    table = bootstrap_lag_interval_table(boot, candidate_id="cand")
    assert "metric" in table.columns
    assert table.height >= 1


def test_bootstrap_weight_samples_table() -> None:
    boot = _bootstrap_result()
    table = bootstrap_weight_samples_table(boot, candidate_id="cand")
    assert table.height >= 2
    assert "bootstrap_id" in table.columns


def test_bootstrap_parameter_samples_table() -> None:
    boot = _bootstrap_result()
    table = bootstrap_parameter_samples_table(boot, candidate_id="cand")
    assert table.height >= 1
    assert "parameter_name" in table.columns


def test_bootstrap_lag_summary_samples_table() -> None:
    boot = _bootstrap_result()
    table = bootstrap_lag_summary_samples_table(boot, candidate_id="cand")
    assert table.height >= 1
    assert "mean_lag" in table.columns


def test_bootstrap_summary_compact_dict_and_text() -> None:
    boot = _bootstrap_result()
    d = bootstrap_summary_compact_dict(boot, candidate_id="cand")
    assert d["n_bootstrap"] == 5
    assert d["candidate_ids"] == ("cand",)
    assert len(d["lag_metrics"]) >= 1

    text = bootstrap_summary_compact_text(boot, candidate_id="cand")
    assert "bootstrap summary: " in text
    assert "n_bootstrap=5" in text


def test_select_bootstrap_candidate_empty_raises() -> None:
    from rtdfeatures.bootstrap.summaries import _select_bootstrap_candidate

    with pytest.raises(ValueError, match="at least one candidate"):
        _select_bootstrap_candidate(succeeded_losses=[], loss_tolerance_fraction=0.1)


def test_select_bootstrap_candidate_negative_tolerance_raises() -> None:
    from rtdfeatures.bootstrap.summaries import _select_bootstrap_candidate

    with pytest.raises(ValueError, match="non-negative"):
        _select_bootstrap_candidate(
            succeeded_losses=[(None, 1.0)],  # type: ignore[list-item]
            loss_tolerance_fraction=-0.1,
        )


def test_select_bootstrap_candidate_tolerance_pool_empty_falls_back() -> None:
    from rtdfeatures.bootstrap.summaries import _select_bootstrap_candidate
    from rtdfeatures.candidates.contracts import KernelCandidate
    from rtdfeatures.diagnostics import KernelFamilyFitResult
    from rtdfeatures.kernels import UniformKernel

    c1 = KernelCandidate(
        candidate_id="c1", family="f1", candidate_type=_CAND_FIXED,
        fixed_parameters={"delay_steps": 1}, min_lag=0, max_lag=2,
    )
    r1 = KernelFamilyFitResult(
        candidate=c1, fit_result=None, succeeded=True, error=None,
        is_parametric=False, is_empirical=False, is_baseline=False,
        n_parameters=None, validation_loss=0.5, train_loss=None,
        warning_codes=(), evaluated_fixed_kernel=UniformKernel(max_lag_steps=2, dt=1.0),
        fixed_baseline_comparison=None, evaluation_provenance=None,
    )
    c2 = KernelCandidate(
        candidate_id="c2", family="f2", candidate_type=_CAND_PARAM,
        fixed_parameters={}, min_lag=0, max_lag=2,
        learner_parameters={"family": "gamma"},
    )
    r2 = KernelFamilyFitResult(
        candidate=c2, fit_result=None, succeeded=True, error=None,
        is_parametric=True, is_empirical=False, is_baseline=False,
        n_parameters=2, validation_loss=0.1, train_loss=None,
        warning_codes=(), evaluated_fixed_kernel=None,
        fixed_baseline_comparison=None, evaluation_provenance=None,
    )
    result = _select_bootstrap_candidate(
        succeeded_losses=[(r1, 0.5), (r2, 0.1)],
        loss_tolerance_fraction=0.0,
    )
    assert result is not None


def test_bootstrap_simplicity_rank_all_types() -> None:
    from rtdfeatures.bootstrap.summaries import _bootstrap_simplicity_rank
    from rtdfeatures.candidates.contracts import KernelCandidate
    from rtdfeatures.diagnostics import KernelFamilyFitResult

    def _make_result(
        candidate_type: KernelCandidateType, is_baseline: bool = False,
    ) -> KernelFamilyFitResult:
        fixed_p: dict[str, object] = (
            {"delay_steps": 1} if candidate_type == "fixed_kernel" else {}
        )
        learner_p: dict[str, object] = (
            {"family": "gamma"}
            if candidate_type in {"empirical_learner", "parametric_learner"}
            else {}
        )
        cand = KernelCandidate(
            candidate_id="x", family="f", candidate_type=candidate_type,
            fixed_parameters=fixed_p, learner_parameters=learner_p, min_lag=0, max_lag=2,
        )
        return KernelFamilyFitResult(
            candidate=cand, fit_result=None, succeeded=True, error=None,
            is_parametric=False, is_empirical=False, is_baseline=is_baseline,
            n_parameters=None, validation_loss=None, train_loss=None,
            warning_codes=(), evaluated_fixed_kernel=None,
            fixed_baseline_comparison=None, evaluation_provenance=None,
        )

    assert _bootstrap_simplicity_rank(_make_result(_CAND_FIXED)) == 1
    assert _bootstrap_simplicity_rank(_make_result(_CAND_EMPIRICAL)) == 2
    assert _bootstrap_simplicity_rank(_make_result(_CAND_PARAM)) == 3
    assert _bootstrap_simplicity_rank(_make_result(_CAND_FIXED, is_baseline=True)) == 4


def test_interval_empty_values() -> None:
    from rtdfeatures.bootstrap.summaries import _interval
    lower, upper = _interval([], 0.025, 0.975)
    assert lower != lower  # nan


def test_stability_score_zero_n_bootstrap() -> None:
    from rtdfeatures.bootstrap.summaries import _stability_score
    from rtdfeatures.diagnostics import BootstrapResult
    boot = BootstrapResult(
        n_bootstrap=0, n_succeeded=0, n_failed=0, failures=(),
        weight_samples=(), parameter_samples=(), lag_summary_samples=(),
        family_selection_counts={}, warnings=(), bootstrap_config={},
    )
    assert _stability_score(boot) is None


def test_interval_touches_boundary_none_bounds() -> None:
    from rtdfeatures.bootstrap.summaries import (
        _interval_touches_boundary,
        build_kernel_bootstrap_summary,
    )
    boot = _bootstrap_result()
    summary, _ = build_kernel_bootstrap_summary(boot, candidate_id="cand")
    assert not _interval_touches_boundary(summary, lag_bounds=None)


def test_interval_touches_boundary_with_nan_skips() -> None:
    from rtdfeatures.bootstrap.summaries import _interval_touches_boundary
    from rtdfeatures.diagnostics import KernelBootstrapSummary

    summary = KernelBootstrapSummary(
        mean_lag_interval=(float("nan"), float("nan")),
        p50_lag_interval=(2.0, 5.0),
        p90_lag_interval=(3.0, 6.0),
        tail_mass_interval=(0.0, 0.5),
        weight_interval_by_lag=(),
        parameter_interval_by_name=(),
        stability_score=1.0,
    )
    assert not _interval_touches_boundary(summary, lag_bounds=(1.5, 10.0))
    assert _interval_touches_boundary(summary, lag_bounds=(2.0, 10.0))


def test_loss_delta_fraction() -> None:
    from rtdfeatures.bootstrap.summaries import _loss_delta_fraction
    assert _loss_delta_fraction(1.0, 1.5) == pytest.approx(0.5)
    assert _loss_delta_fraction(0.0, 0.5) == pytest.approx(500000000000.0)
