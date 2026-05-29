"""Tests for learner-family diagnostic comparison reporting."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import cast

import polars as pl
import pytest

from rtdfeatures.diagnostics import (
    BaselineComparison,
    FitDiagnostics,
    IdentifiabilityReport,
    KernelFitResult,
)
from rtdfeatures.kernels import UniformKernel
from rtdfeatures.learners import GammaKernelLearner, SimplexKernelLearner
from rtdfeatures.reporting import (
    DEFAULT_PARAMETRIC_DIAGNOSTIC_WARNING_MARGIN,
    learner_diagnostic_comparison_table,
    learner_diagnostic_warning_table,
)


def _fit_result(
    *,
    learned_loss: float,
    no_lag: float,
    best_single_lag: float,
    fit_provenance: dict[str, float | int | str] | None = None,
) -> KernelFitResult:
    kernel = UniformKernel(
        max_lag_steps=2,
        min_lag_steps=0,
        dt=60.0,
        name="fixture",
    )
    diagnostics = FitDiagnostics(
        train_loss=max(learned_loss * 0.9, 1e-9),
        validation_loss=learned_loss,
        input_variance=1.0,
        target_variance=1.0,
        kernel_weight_sum=1.0,
        mean_lag=60.0,
        p50_lag=60.0,
        p90_lag=120.0,
        tail_mass=0.3,
        boundary_mass_fraction=0.2,
    )
    baseline = BaselineComparison(
        no_lag_validation_loss=no_lag,
        best_single_lag_validation_loss=best_single_lag,
        learned_validation_loss=learned_loss,
        uniform_validation_loss=learned_loss + 0.02,
        exponential_validation_loss=learned_loss + 0.01,
        summary_by_baseline={},
    )
    return KernelFitResult(
        kernel=kernel,
        fit_diagnostics=diagnostics,
        identifiability_report=IdentifiabilityReport(warnings=(), is_reliable=True),
        baseline_comparison=baseline,
        fit_provenance=fit_provenance
        or {
            "validation_fraction": 0.2,
            "dt_seconds": 60.0,
            "total_valid_windows": 100,
        },
    )


def test_comparison_table_includes_configured_families_and_baselines() -> None:
    fits = {
        "simplex": _fit_result(learned_loss=0.90, no_lag=1.10, best_single_lag=0.95),
        "gamma": _fit_result(learned_loss=0.93, no_lag=1.05, best_single_lag=0.91),
        "exponential": _fit_result(learned_loss=0.96, no_lag=1.03, best_single_lag=0.92),
    }

    out = learner_diagnostic_comparison_table(fits)

    assert out.columns == [
        "learner_family",
        "row_type",
        "candidate",
        "validation_loss",
        "delta_fraction_vs_learned",
        "is_parametric_family",
    ]
    assert out.filter(pl.col("row_type") == "learned")["learner_family"].to_list() == [
        "simplex",
        "gamma",
        "exponential",
    ]
    assert set(out.filter(pl.col("row_type") == "baseline")["candidate"].to_list()) >= {
        "no_lag",
        "best_single_lag",
    }
    assert "winner" not in " ".join(out.columns).lower()
    assert "select" not in " ".join(out.columns).lower()
    assert "recommend" not in " ".join(out.columns).lower()


def test_warning_table_is_deterministic_and_thresholded() -> None:
    fits = {
        "simplex": _fit_result(learned_loss=0.80, no_lag=1.00, best_single_lag=0.90),
        "gamma": _fit_result(learned_loss=1.00, no_lag=0.90, best_single_lag=0.85),
        "exponential": _fit_result(learned_loss=1.02, no_lag=0.92, best_single_lag=0.89),
    }

    warnings = learner_diagnostic_warning_table(fits)

    assert warnings.height >= 4
    assert warnings["warning_code"].to_list().count("PARAMETRIC_WORSE_THAN_SIMPLEX") == 2
    assert set(warnings["reference"].to_list()) >= {"simplex", "no_lag", "best_single_lag"}
    assert warnings["warning_margin"].to_list() == [
        pytest.approx(DEFAULT_PARAMETRIC_DIAGNOSTIC_WARNING_MARGIN)
    ] * warnings.height
    min_delta = cast(float, warnings["delta_fraction_worse"].min())
    assert min_delta >= DEFAULT_PARAMETRIC_DIAGNOSTIC_WARNING_MARGIN
    schema_text = " ".join(warnings.columns).lower()
    assert "winner" not in schema_text
    assert "select" not in schema_text
    assert "recommend" not in schema_text


def test_warning_table_respects_override_margin() -> None:
    fits = {
        "simplex": _fit_result(learned_loss=0.95, no_lag=1.05, best_single_lag=0.98),
        "gamma": _fit_result(learned_loss=1.00, no_lag=0.99, best_single_lag=0.98),
    }

    default_warnings = learner_diagnostic_warning_table(fits)
    strict_warnings = learner_diagnostic_warning_table(fits, warning_margin=0.10)

    assert default_warnings.height > strict_warnings.height
    assert strict_warnings.height == 0


def test_comparison_table_raises_for_incompatible_fit_provenance() -> None:
    fits = {
        "simplex": _fit_result(learned_loss=0.90, no_lag=1.10, best_single_lag=0.95),
        "gamma": _fit_result(
            learned_loss=0.93,
            no_lag=1.05,
            best_single_lag=0.91,
            fit_provenance={
                "validation_fraction": 0.3,
                "dt_seconds": 60.0,
                "total_valid_windows": 100,
            },
        ),
    }

    with pytest.raises(ValueError, match="fit_provenance\\['validation_fraction'\\] mismatch"):
        learner_diagnostic_comparison_table(fits)


def test_warning_table_raises_for_incompatible_fit_provenance() -> None:
    fits = {
        "simplex": _fit_result(learned_loss=0.90, no_lag=1.10, best_single_lag=0.95),
        "gamma": _fit_result(
            learned_loss=0.93,
            no_lag=1.05,
            best_single_lag=0.91,
            fit_provenance={
                "validation_fraction": 0.2,
                "dt_seconds": 60.0,
                "total_valid_windows": 101,
            },
        ),
    }

    with pytest.raises(ValueError, match="fit_provenance\\['total_valid_windows'\\] mismatch"):
        learner_diagnostic_warning_table(fits)


def test_comparison_table_raises_for_incompatible_loss_name() -> None:
    fits = {
        "simplex": _fit_result(
            learned_loss=0.90,
            no_lag=1.10,
            best_single_lag=0.95,
            fit_provenance={
                "validation_fraction": 0.2,
                "dt_seconds": 60.0,
                "total_valid_windows": 100,
                "loss_name": "mse",
            },
        ),
        "gamma": _fit_result(
            learned_loss=0.93,
            no_lag=1.05,
            best_single_lag=0.91,
            fit_provenance={
                "validation_fraction": 0.2,
                "dt_seconds": 60.0,
                "total_valid_windows": 100,
                "loss_name": "huber",
            },
        ),
    }

    with pytest.raises(ValueError, match="loss mismatch"):
        learner_diagnostic_comparison_table(fits)


def test_warning_table_raises_for_incompatible_huber_delta() -> None:
    fits = {
        "simplex": _fit_result(
            learned_loss=0.90,
            no_lag=1.10,
            best_single_lag=0.95,
            fit_provenance={
                "validation_fraction": 0.2,
                "dt_seconds": 60.0,
                "total_valid_windows": 100,
                "loss_name": "huber",
                "huber_delta": 1.0,
            },
        ),
        "gamma": _fit_result(
            learned_loss=0.93,
            no_lag=1.05,
            best_single_lag=0.91,
            fit_provenance={
                "validation_fraction": 0.2,
                "dt_seconds": 60.0,
                "total_valid_windows": 100,
                "loss_name": "huber",
                "huber_delta": 1.5,
            },
        ),
    }

    with pytest.raises(ValueError, match="fit_provenance\\['huber_delta'\\] mismatch"):
        learner_diagnostic_warning_table(fits)


def test_warning_delta_uses_reference_relative_denominator() -> None:
    fits = {
        "simplex": _fit_result(learned_loss=0.90, no_lag=1.10, best_single_lag=0.95),
        "gamma": _fit_result(learned_loss=1.00, no_lag=0.98, best_single_lag=0.97),
    }

    warnings = learner_diagnostic_warning_table(fits, warning_margin=0.0201)

    row = warnings.filter(
        (pl.col("learner_family") == "gamma")
        & (pl.col("warning_code") == "PARAMETRIC_WORSE_THAN_BASELINE")
        & (pl.col("reference") == "no_lag")
    )
    assert row.height == 1
    assert row["delta_fraction_worse"][0] == pytest.approx((1.00 - 0.98) / 0.98)


def test_real_huber_fit_outputs_include_huber_delta_and_are_compared() -> None:
    n = 180
    input_series = [0.0] * n
    for idx in range(n):
        input_series[idx] = float(idx % 7) / 7.0
    target_series = [0.0] * n
    for idx in range(2, n):
        target_series[idx] = 0.6 * input_series[idx - 1] + 0.4 * input_series[idx - 2]

    frame = pl.DataFrame(
        {
            "t": [datetime(2024, 1, 1) + timedelta(minutes=idx) for idx in range(n)],
            "x": input_series,
            "y": target_series,
        }
    )

    simplex_fit = SimplexKernelLearner(
        max_lag=7,
        min_lag=1,
        seed=211,
        loss="huber",
        huber_delta=1.0,
    ).fit(frame, time_col="t", input_col="x", target_col="y")
    gamma_fit = GammaKernelLearner(
        max_lag=7,
        min_lag=1,
        seed=211,
        loss="huber",
        huber_delta=1.5,
    ).fit(frame, time_col="t", input_col="x", target_col="y")

    assert simplex_fit.fit_provenance is not None
    assert gamma_fit.fit_provenance is not None
    assert simplex_fit.fit_provenance["huber_delta"] == pytest.approx(1.0)
    assert gamma_fit.fit_provenance["huber_delta"] == pytest.approx(1.5)

    with pytest.raises(ValueError, match="fit_provenance\\['huber_delta'\\] mismatch"):
        learner_diagnostic_warning_table({"simplex": simplex_fit, "gamma": gamma_fit})
