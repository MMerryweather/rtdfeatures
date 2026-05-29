"""Tests for shared learner identifiability behavior."""

from __future__ import annotations

import numpy as np

from rtdfeatures.diagnostics import FitDiagnostics
from rtdfeatures.learners import (
    ExponentialKernelLearner,
    GammaKernelLearner,
    SimplexKernelLearner,
)
from rtdfeatures.learners._identifiability import (
    WARNING_DEFINITIONS,
    build_identifiability_report,
)
from rtdfeatures.synthetic import make_exponential_kernel_dataset


def _fit_diagnostics(
    *,
    train_loss: float,
    validation_loss: float,
    input_variance: float = 1.0,
    target_variance: float = 1.0,
) -> FitDiagnostics:
    return FitDiagnostics(
        train_loss=train_loss,
        validation_loss=validation_loss,
        input_variance=input_variance,
        target_variance=target_variance,
        kernel_weight_sum=1.0,
        mean_lag=1.0,
        p50_lag=1.0,
        p90_lag=2.0,
        tail_mass=0.1,
        boundary_mass_fraction=0.1,
    )


def test_identifiability_warning_definitions_are_stable() -> None:
    assert tuple(WARNING_DEFINITIONS.items()) == (
        ("INPUT_TOO_FLAT", ("Input is too flat.", "high")),
        ("TARGET_TOO_FLAT", ("Target signal is too flat.", "high")),
        (
            "WEAK_NO_LAG_IMPROVEMENT",
            ("Target signal appears noisy or weakly explained.", "medium"),
        ),
        ("LARGE_VALIDATION_GAP", ("Validation loss is much worse than training loss.", "high")),
        ("BOUNDARY_PILED_KERNEL", ("Kernel piles mass at the lag boundary.", "medium")),
        ("DIFFUSE_KERNEL", ("Kernel is too diffuse to interpret confidently.", "medium")),
        (
            "BEST_SINGLE_LAG_BEATS_LEARNED",
            ("best_single_lag beats the learned kernel.", "medium"),
        ),
        (
            "UNIFORM_BASELINE_BEATS_LEARNED",
            ("uniform baseline beats the learned kernel.", "medium"),
        ),
        (
            "EXPONENTIAL_BASELINE_BEATS_LEARNED",
            ("exponential baseline beats the learned kernel.", "medium"),
        ),
    )


def test_identifiability_report_matches_flat_input_warning() -> None:
    fit_diagnostics = _fit_diagnostics(
        train_loss=1.0, validation_loss=0.9, input_variance=1e-12
    )
    report = build_identifiability_report(
        fit_diagnostics=fit_diagnostics,
        learned_weights=np.asarray([0.2, 0.8], dtype=np.float64),
        no_lag_validation_loss=1.5,
        best_single_lag_validation_loss=0.95,
        uniform_validation_loss=1.0,
        exponential_validation_loss=1.0,
    )

    assert "INPUT_TOO_FLAT" in report.warning_codes
    assert "Input is too flat." in report.warnings


def test_identifiability_report_flags_baseline_beating_learned() -> None:
    report = build_identifiability_report(
        fit_diagnostics=_fit_diagnostics(train_loss=1.0, validation_loss=1.0),
        learned_weights=np.asarray([0.2, 0.8], dtype=np.float64),
        no_lag_validation_loss=1.5,
        best_single_lag_validation_loss=0.7,
        uniform_validation_loss=0.7,
        exponential_validation_loss=0.6,
    )

    assert "BEST_SINGLE_LAG_BEATS_LEARNED" in report.warning_codes
    assert "UNIFORM_BASELINE_BEATS_LEARNED" in report.warning_codes
    assert "EXPONENTIAL_BASELINE_BEATS_LEARNED" in report.warning_codes


def test_simplex_gamma_exponential_fit_still_return_identifiability_reports() -> None:
    synthetic = make_exponential_kernel_dataset(seed=223, n_rows=260, dt=60.0, noise_std=0.03)
    meta = synthetic.true_kernels["input_signal->target_signal"]

    simplex_fit = SimplexKernelLearner(
        max_lag=meta["max_lag"],
        min_lag=meta["min_lag"],
        dt="60s",
        seed=223,
        max_epochs=220,
    ).fit(
        synthetic.data,
        input_col="input_signal",
        target_col="target_signal",
        time_col="time",
    )
    gamma_fit = GammaKernelLearner(
        max_lag=meta["max_lag"],
        min_lag=meta["min_lag"],
        dt="60s",
        seed=223,
        max_epochs=220,
    ).fit(
        synthetic.data,
        input_col="input_signal",
        target_col="target_signal",
        time_col="time",
    )
    exponential_fit = ExponentialKernelLearner(
        max_lag=meta["max_lag"],
        min_lag=meta["min_lag"],
        dt="60s",
        seed=223,
        max_epochs=220,
    ).fit(
        synthetic.data,
        input_col="input_signal",
        target_col="target_signal",
        time_col="time",
    )

    for fit in (simplex_fit, gamma_fit, exponential_fit):
        report = fit.identifiability_report
        assert isinstance(report.warnings, tuple)
        assert isinstance(report.warning_codes, tuple)
        assert isinstance(report.warning_severity_by_code, dict)
