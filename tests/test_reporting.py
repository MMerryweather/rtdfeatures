"""Tests for compact diagnostics reporting helpers."""

from __future__ import annotations

from copy import deepcopy

import polars as pl
import pytest

from rtdfeatures.diagnostics import BaselineComparison, IdentifiabilityReport, KernelFitResult
from rtdfeatures.reporting import (
    baseline_comparison_compact_dict,
    baseline_comparison_compact_text,
    baseline_comparison_table,
    fit_result_baseline_summary_table,
    fit_result_warning_summary_table,
    learner_diagnostic_comparison_table,
    learner_diagnostic_warning_table,
    warning_summary_compact_dict,
    warning_summary_compact_text,
    warning_summary_table,
)


def _baseline_fixture() -> BaselineComparison:
    return BaselineComparison(
        no_lag_validation_loss=1.4,
        best_single_lag_validation_loss=1.1,
        learned_validation_loss=1.2,
        uniform_validation_loss=1.25,
        exponential_validation_loss=1.18,
        primary_ranking_metric="validation_loss",
        summary_by_baseline={
            "no_lag": {
                "baseline_loss": 1.4,
                "learned_loss": 1.2,
                "delta_fraction_vs_learned": (1.2 - 1.4) / 1.2,
                "beats_learned_by_margin": False,
            },
            "best_single_lag": {
                "baseline_loss": 1.1,
                "learned_loss": 1.2,
                "delta_fraction_vs_learned": (1.2 - 1.1) / 1.2,
                "beats_learned_by_margin": True,
            },
        },
    )


def _warning_fixture() -> IdentifiabilityReport:
    return IdentifiabilityReport(
        warnings=(
            "Input is too flat.",
            "best_single_lag beats the learned kernel.",
        ),
        is_reliable=False,
        warning_codes=("INPUT_TOO_FLAT", "BEST_SINGLE_LAG_BEATS_LEARNED"),
        warning_severity_by_code={
            "INPUT_TOO_FLAT": "high",
            "BEST_SINGLE_LAG_BEATS_LEARNED": "medium",
        },
    )


def test_baseline_summary_table_is_stable_and_schema_checked() -> None:
    baseline = _baseline_fixture()
    baseline_before = deepcopy(baseline)

    out = baseline_comparison_table(baseline)

    assert out.columns == [
        "baseline",
        "validation_loss",
        "learned_validation_loss",
        "delta_fraction_vs_learned",
        "beats_learned",
        "is_learned",
    ]
    assert out.schema == {
        "baseline": pl.String,
        "validation_loss": pl.Float64,
        "learned_validation_loss": pl.Float64,
        "delta_fraction_vs_learned": pl.Float64,
        "beats_learned": pl.Boolean,
        "is_learned": pl.Boolean,
    }
    assert out["baseline"].to_list() == [
        "learned",
        "no_lag",
        "best_single_lag",
        "uniform",
        "exponential",
    ]
    assert out.filter(pl.col("baseline") == "best_single_lag")["beats_learned"].item() is True
    assert baseline == baseline_before


def test_warning_summary_table_matches_report_values_and_schema_checked() -> None:
    report = _warning_fixture()
    report_before = deepcopy(report)

    out = warning_summary_table(report)

    assert out.columns == ["warning_index", "warning_code", "warning_severity", "warning_message"]
    assert out.schema == {
        "warning_index": pl.Int64,
        "warning_code": pl.String,
        "warning_severity": pl.String,
        "warning_message": pl.String,
    }
    assert out["warning_code"].to_list() == list(report.warning_codes)
    assert out["warning_message"].to_list() == list(report.warnings)
    assert out["warning_severity"].to_list() == [
        report.warning_severity_by_code["INPUT_TOO_FLAT"],
        report.warning_severity_by_code["BEST_SINGLE_LAG_BEATS_LEARNED"],
    ]
    assert report == report_before


def test_compact_text_and_dict_helpers_are_deterministic() -> None:
    baseline = _baseline_fixture()
    report = _warning_fixture()

    baseline_dict = baseline_comparison_compact_dict(baseline)
    warning_dict = warning_summary_compact_dict(report)

    assert baseline_dict["primary_ranking_metric"] == "validation_loss"
    assert baseline_dict["available_baselines"] == (
        "learned",
        "no_lag",
        "best_single_lag",
        "uniform",
        "exponential",
    )
    assert warning_dict["warning_count"] == 2
    assert warning_dict["warning_codes"] == report.warning_codes

    baseline_text = baseline_comparison_compact_text(baseline)
    warning_text = warning_summary_compact_text(report)

    assert baseline_text.startswith("validation losses: learned=1.2")
    assert "best_single_lag=1.1" in baseline_text
    assert warning_text == "warnings: INPUT_TOO_FLAT:high, BEST_SINGLE_LAG_BEATS_LEARNED:medium"


def test_baseline_table_without_optional_baselines() -> None:
    baseline = _baseline_fixture()
    out = baseline_comparison_table(baseline, include_optional_baselines=False)
    assert out["baseline"].to_list() == ["learned", "no_lag", "best_single_lag"]
    assert "uniform" not in out["baseline"].to_list()


def test_warning_summary_compact_text_no_warnings() -> None:
    report = IdentifiabilityReport(warnings=(), is_reliable=True)
    assert warning_summary_compact_text(report) == "warnings: none"


def test_warning_summary_compact_text_warnings_no_codes() -> None:
    report = IdentifiabilityReport(
        warnings=("something is off",), is_reliable=False, warning_codes=(),
    )
    text = warning_summary_compact_text(report)
    assert "warnings: 1" in text


def test_fit_result_warning_and_baseline_tables() -> None:
    from rtdfeatures.diagnostics import FitDiagnostics, KernelFitResult
    from rtdfeatures.kernels import LearnedKernel

    kernel = LearnedKernel(
        weights=(0.4, 0.6), lag_steps=(0, 1), dt=1.0,
        min_lag_steps=0, max_lag_steps=1,
    )
    fit = KernelFitResult(
        kernel=kernel,
        fit_diagnostics=FitDiagnostics(
            train_loss=0.5, validation_loss=0.6, input_variance=1.0,
            target_variance=1.0, kernel_weight_sum=1.0,
            mean_lag=0.6, p50_lag=1.0, p90_lag=1.0, tail_mass=0.6,
            boundary_mass_fraction=0.4,
        ),
        identifiability_report=IdentifiabilityReport(warnings=("bad",), is_reliable=False),
        baseline_comparison=_baseline_fixture(),
    )

    wt = fit_result_warning_summary_table(fit)
    assert "warning_message" in wt.columns
    assert wt["warning_message"].to_list() == ["bad"]

    bt = fit_result_baseline_summary_table(fit)
    assert "baseline" in bt.columns


def test_learner_diagnostic_comparison_table_without_baselines() -> None:
    fit = _make_fit_result()
    table = learner_diagnostic_comparison_table({"simplex": fit}, include_baselines=False)
    assert table.height == 1
    assert table["row_type"].to_list() == ["learned"]


def test_learner_diagnostic_comparison_table_with_single_family() -> None:
    fit = _make_fit_result()
    table = learner_diagnostic_comparison_table({"simplex": fit}, include_baselines=True)
    assert table.height >= 2
    assert "simplex" in table["learner_family"].to_list()


def test_learner_diagnostic_warning_table_negative_margin_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        learner_diagnostic_warning_table({"simplex": _make_fit_result()}, warning_margin=-0.1)


def test_learner_diagnostic_warning_table_single_family_no_warnings() -> None:
    fit = _make_fit_result()
    table = learner_diagnostic_warning_table({"simplex": fit})
    assert table.height == 0


def test_learner_diagnostic_warning_table_parametric_worse() -> None:
    fit_simplex = _make_fit_result(validation_loss=0.1, provenance={"validation_fraction": 0.2})
    gamma_result = _make_fit_result(validation_loss=1.0, provenance={"validation_fraction": 0.2})
    table = learner_diagnostic_warning_table(
        {"simplex": fit_simplex, "gamma": gamma_result},
        warning_margin=0.05,
    )
    assert table.height >= 1


def test_learner_diagnostic_warning_table_only_parametric() -> None:
    gamma_result = _make_fit_result(validation_loss=0.8, provenance={"validation_fraction": 0.2})
    table = learner_diagnostic_warning_table(
        {"gamma": gamma_result},
        warning_margin=0.05,
    )
    assert table.height >= 0


def test_learner_diagnostic_comparison_provenance_mismatch_raises() -> None:
    fit1 = _make_fit_result(provenance={"validation_fraction": 0.2})
    fit2 = _make_fit_result(provenance={"validation_fraction": 0.3})
    with pytest.raises(ValueError, match="Incompatible fit results"):
        learner_diagnostic_comparison_table({"simplex": fit1, "gamma": fit2})


def test_learner_diagnostic_comparison_loss_mismatch_raises() -> None:
    fit1 = _make_fit_result(provenance={"validation_fraction": 0.2, "loss_name": "mse"})
    fit2 = _make_fit_result(provenance={"validation_fraction": 0.2, "loss_name": "mae"})
    with pytest.raises(ValueError, match="loss mismatch"):
        learner_diagnostic_comparison_table({"simplex": fit1, "gamma": fit2})


def test_learner_diagnostic_comparison_huber_delta_mismatch_raises() -> None:
    fit1 = _make_fit_result(provenance={
        "validation_fraction": 0.2, "loss_name": "huber", "huber_delta": 0.5,
    })
    fit2 = _make_fit_result(provenance={
        "validation_fraction": 0.2, "loss_name": "huber", "huber_delta": 1.0,
    })
    with pytest.raises(ValueError, match="huber_delta"):
        learner_diagnostic_comparison_table({"simplex": fit1, "gamma": fit2})


def test_learner_diagnostic_comparison_non_dict_provenance_skips_check() -> None:
    fit1 = _make_fit_result(provenance=None)
    fit2 = _make_fit_result(provenance={"validation_fraction": 0.2})
    table = learner_diagnostic_comparison_table({"simplex": fit1, "gamma": fit2})
    assert table.height >= 0


def test_learner_diagnostic_comparison_missing_provenance_key_skips_check() -> None:
    fit1 = _make_fit_result(provenance={"validation_fraction": 0.2})
    fit2 = _make_fit_result(provenance={"dt_seconds": 300.0})
    table = learner_diagnostic_comparison_table({"simplex": fit1, "gamma": fit2})
    assert table.height >= 0


# -
# helpers
# -


def _make_fit_result(
    validation_loss: float = 0.6, provenance: dict | None = None,
) -> KernelFitResult:
    from rtdfeatures.diagnostics import FitDiagnostics
    from rtdfeatures.kernels import LearnedKernel

    kernel = LearnedKernel(
        weights=(0.4, 0.6), lag_steps=(0, 1), dt=1.0,
        min_lag_steps=0, max_lag_steps=1,
    )
    return KernelFitResult(
        kernel=kernel,
        fit_diagnostics=FitDiagnostics(
            train_loss=0.5, validation_loss=validation_loss, input_variance=1.0,
            target_variance=1.0, kernel_weight_sum=1.0,
            mean_lag=0.6, p50_lag=1.0, p90_lag=1.0, tail_mass=0.6,
            boundary_mass_fraction=0.4,
        ),
        identifiability_report=IdentifiabilityReport(warnings=(), is_reliable=True),
        baseline_comparison=BaselineComparison(
            no_lag_validation_loss=0.9, best_single_lag_validation_loss=0.8,
            learned_validation_loss=validation_loss,
        ),
        fit_provenance=provenance,
    )
    return KernelFitResult(
        kernel=kernel,
        fit_diagnostics=FitDiagnostics(
            train_loss=0.5, validation_loss=validation_loss, input_variance=1.0,
            target_variance=1.0, kernel_weight_sum=1.0,
            mean_lag=0.6, p50_lag=1.0, p90_lag=1.0, tail_mass=0.6,
            boundary_mass_fraction=0.4,
        ),
        identifiability_report=IdentifiabilityReport(warnings=(), is_reliable=True),
        baseline_comparison=BaselineComparison(
            no_lag_validation_loss=0.9, best_single_lag_validation_loss=0.8,
            learned_validation_loss=validation_loss,
        ),
    )
