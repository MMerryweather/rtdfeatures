"""Behavior baselines for learner fit-result shape and constructor params."""

from __future__ import annotations

import inspect
from dataclasses import fields

import numpy as np
import polars as pl
import pytest

from rtdfeatures.diagnostics import BaselineComparison, KernelFitResult
from rtdfeatures.features import KernelFeatureBuilder, TransformResult
from rtdfeatures.features import builder as builder_module
from rtdfeatures.kernels import FixedDelayKernel
from rtdfeatures.learners import (
    ExponentialKernelLearner,
    GammaKernelLearner,
    SimplexKernelLearner,
)
from rtdfeatures.synthetic import (
    make_exponential_kernel_dataset,
    make_gamma_kernel_dataset,
    make_single_delay_dataset,
)


def _assert_fit_result_baseline_shape(fit: KernelFitResult) -> None:
    assert isinstance(fit, KernelFitResult)
    assert fit.kernel is not None
    assert fit.fit_diagnostics is not None
    assert fit.identifiability_report is not None
    assert fit.baseline_comparison is not None
    assert fit.kernel_shape_summary is not None
    assert fit.fit_data_coverage_summary is not None
    assert fit.fit_provenance is not None
    assert np.isfinite(fit.fit_diagnostics.validation_loss)
    assert np.isfinite(fit.baseline_comparison.learned_validation_loss)


def _assert_exact_constructor_params(obj: type, *, expected: set[str]) -> None:
    sig = inspect.signature(obj)
    actual = set(sig.parameters.keys()) - {"self"}
    assert actual == expected


def _assert_keyword_only_signature_snapshot(
    obj: type,
    *,
    expected_order: tuple[str, ...],
    expected_defaults: dict[str, object],
) -> None:
    sig = inspect.signature(obj)
    parameters = list(sig.parameters.values())
    assert tuple(p.name for p in parameters) == expected_order
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in parameters)
    for name, default in expected_defaults.items():
        assert sig.parameters[name].default == default


def test_simplex_fit_result_baseline_shape() -> None:
    synthetic = make_single_delay_dataset(n_rows=240, dt=1.0, seed=401, noise_std=0.02)
    fit = SimplexKernelLearner(max_lag=8, min_lag=0, dt="1s", seed=401, max_epochs=80).fit(
        synthetic.data,
        input_col="input_signal",
        target_col="target_signal",
        time_col="time",
    )
    _assert_fit_result_baseline_shape(fit)


def test_gamma_fit_result_baseline_shape() -> None:
    synthetic = make_gamma_kernel_dataset(seed=402, n_rows=260, dt=60.0, noise_std=0.02)
    meta = synthetic.true_kernels["input_signal->target_signal"]
    fit = GammaKernelLearner(
        max_lag=meta["max_lag"],
        min_lag=meta["min_lag"],
        dt="60s",
        seed=402,
        max_epochs=90,
    ).fit(
        synthetic.data,
        input_col="input_signal",
        target_col="target_signal",
        time_col="time",
    )
    _assert_fit_result_baseline_shape(fit)


def test_exponential_fit_result_baseline_shape() -> None:
    synthetic = make_exponential_kernel_dataset(seed=403, n_rows=260, dt=60.0, noise_std=0.02)
    meta = synthetic.true_kernels["input_signal->target_signal"]
    fit = ExponentialKernelLearner(
        max_lag=meta["max_lag"],
        min_lag=meta["min_lag"],
        dt="60s",
        seed=403,
        max_epochs=90,
    ).fit(
        synthetic.data,
        input_col="input_signal",
        target_col="target_signal",
        time_col="time",
    )
    _assert_fit_result_baseline_shape(fit)


def test_simplex_gamma_exponential_public_constructor_signatures_snapshot() -> None:
    _assert_exact_constructor_params(
        SimplexKernelLearner,
        expected={
            "max_lag",
            "min_lag",
            "dt",
            "loss",
            "smoothness_penalty",
            "seed",
            "validation_fraction",
            "learning_rate",
            "max_epochs",
            "huber_delta",
        },
    )
    _assert_exact_constructor_params(
        GammaKernelLearner,
        expected={
            "max_lag",
            "min_lag",
            "dt",
            "loss",
            "smoothness_penalty",
            "seed",
            "validation_fraction",
            "learning_rate",
            "max_epochs",
            "huber_delta",
            "init_shape_alpha",
            "init_rate_beta",
        },
    )
    _assert_exact_constructor_params(
        ExponentialKernelLearner,
        expected={
            "max_lag",
            "min_lag",
            "dt",
            "loss",
            "smoothness_penalty",
            "seed",
            "validation_fraction",
            "learning_rate",
            "max_epochs",
            "huber_delta",
            "init_rate_lambda",
        },
    )


def test_simplex_gamma_exponential_public_constructor_keyword_only_defaults_snapshot() -> None:
    _assert_keyword_only_signature_snapshot(
        SimplexKernelLearner,
        expected_order=(
            "max_lag",
            "min_lag",
            "dt",
            "loss",
            "smoothness_penalty",
            "seed",
            "validation_fraction",
            "learning_rate",
            "max_epochs",
            "huber_delta",
        ),
        expected_defaults={
            "min_lag": 0,
            "dt": None,
            "loss": "huber",
            "smoothness_penalty": 0.0,
            "seed": None,
            "validation_fraction": 0.2,
            "learning_rate": 0.05,
            "max_epochs": 800,
            "huber_delta": 1.0,
        },
    )
    _assert_keyword_only_signature_snapshot(
        GammaKernelLearner,
        expected_order=(
            "max_lag",
            "min_lag",
            "dt",
            "loss",
            "smoothness_penalty",
            "seed",
            "validation_fraction",
            "learning_rate",
            "max_epochs",
            "huber_delta",
            "init_shape_alpha",
            "init_rate_beta",
        ),
        expected_defaults={
            "min_lag": 0,
            "dt": None,
            "loss": "huber",
            "smoothness_penalty": 0.0,
            "seed": None,
            "validation_fraction": 0.2,
            "learning_rate": 0.05,
            "max_epochs": 800,
            "huber_delta": 1.0,
            "init_shape_alpha": 2.0,
            "init_rate_beta": None,
        },
    )
    _assert_keyword_only_signature_snapshot(
        ExponentialKernelLearner,
        expected_order=(
            "max_lag",
            "min_lag",
            "dt",
            "loss",
            "smoothness_penalty",
            "seed",
            "validation_fraction",
            "learning_rate",
            "max_epochs",
            "huber_delta",
            "init_rate_lambda",
        ),
        expected_defaults={
            "min_lag": 0,
            "dt": None,
            "loss": "huber",
            "smoothness_penalty": 0.0,
            "seed": None,
            "validation_fraction": 0.2,
            "learning_rate": 0.05,
            "max_epochs": 800,
            "huber_delta": 1.0,
            "init_rate_lambda": None,
        },
    )


def test_kernel_fit_result_and_baseline_comparison_field_snapshots() -> None:
    fit_result_fields = tuple(field.name for field in fields(KernelFitResult))
    baseline_fields = tuple(field.name for field in fields(BaselineComparison))
    assert fit_result_fields == (
        "kernel",
        "fit_diagnostics",
        "identifiability_report",
        "baseline_comparison",
        "kernel_shape_summary",
        "fit_data_coverage_summary",
        "fit_provenance",
    )
    assert baseline_fields == (
        "no_lag_validation_loss",
        "best_single_lag_validation_loss",
        "learned_validation_loss",
        "uniform_validation_loss",
        "exponential_validation_loss",
        "primary_ranking_metric",
        "summary_by_baseline",
    )


def _ms00_3_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "t": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x": [10.0, 20.0, 30.0, 40.0, 50.0],
            "cat": ["A", "B", "A", "B", "A"],
        }
    )


def _ms00_3_result() -> TransformResult:
    builder = KernelFeatureBuilder(
        kernels={"k": FixedDelayKernel(delay_steps=1, max_lag_steps=2, dt=1.0)},
        time_col="t",
        numeric_cols=["x"],
        category_cols=["cat"],
    )
    return builder.transform_result(_ms00_3_df())


def test_transform_result_baseline_feature_names_and_registry_names() -> None:
    result = _ms00_3_result()
    expected_feature_names = [
        "k_num_x_wmean",
        "k_num_x_wstd",
        "k_num_x_wsum",
        "k_cat_cat_A_frac",
        "k_cat_cat_B_frac",
        "k_cat_cat_entropy",
        "k_age_mean",
        "k_age_p50",
        "k_age_p90",
        "k_age_tail_gt_threshold",
    ]
    non_time_names = [name for name in result.features.columns if name != "t"]
    assert non_time_names == expected_feature_names
    assert list(result.report.feature_names) == expected_feature_names
    assert [spec.name for spec in result.feature_registry.specs] == expected_feature_names


def test_transform_result_baseline_report_fields() -> None:
    result = _ms00_3_result()
    expected_fields = {
        "row_count",
        "output_row_count",
        "warmup_rows",
        "feature_names",
        "missing_rows_by_feature",
        "zero_denominator_rows_by_feature",
        "missing_fraction_by_feature",
        "missing_rows_by_kernel",
        "missing_fraction_by_kernel",
        "zero_denominator_rows_by_kernel",
        "warmup_unusable_summary",
        "collision_naming_summary",
    }
    actual_fields = {f.name for f in fields(type(result.report))}
    assert actual_fields == expected_fields
    assert result.report.row_count == 5
    assert result.report.output_row_count == 5
    assert result.report.warmup_rows == 2
    assert result.report.missing_rows_by_feature["k_num_x_wmean"] == 2
    assert result.report.zero_denominator_rows_by_feature["k_num_x_wmean"] == 0


def test_feature_values_are_stable_for_fixed_delay_kernel() -> None:
    result = _ms00_3_result()
    wmean = result.features["k_num_x_wmean"].to_list()
    assert np.isnan(wmean[0])
    assert np.isnan(wmean[1])
    assert wmean[2:] == pytest.approx([20.0, 30.0, 40.0])


def test_ms06_phase1_execution_struct_fields_exist() -> None:
    execution_result_fields = {field.name for field in fields(builder_module._ExecutionResult)}
    feature_computation_fields = {
        field.name for field in fields(builder_module._FeatureComputation)
    }
    assert "feature_registry" in execution_result_fields
    assert "feature_specs" in feature_computation_fields


def test_ms06_phase1_one_spec_per_non_time_feature_in_feature_order() -> None:
    result = _ms00_3_result()
    non_time_feature_names = [name for name in result.features.columns if name != "t"]
    spec_names = [spec.name for spec in result.feature_registry.specs]
    assert spec_names == non_time_feature_names
    assert len(spec_names) == len(set(spec_names))
