"""legacy milestone contract tests for v0.8 bootstrap objects."""

from __future__ import annotations

from dataclasses import fields

import polars as pl

from rtdfeatures.diagnostics import (
    BOOTSTRAP_WARNING_CODES,
    DEFAULT_BOOTSTRAP_INTERVAL_QUANTILES,
    BootstrapLagSummarySample,
    BootstrapParameterSample,
    BootstrapResult,
    BootstrapWeightSample,
    KernelBootstrapSummary,
    ParameterUncertaintySummary,
    WeightUncertaintySummary,
    bootstrap_lag_summary_samples_schema,
    bootstrap_parameter_samples_schema,
    bootstrap_weight_samples_schema,
    parameter_uncertainty_summary_schema,
    weight_uncertainty_summary_schema,
)


def test_bootstrap_dataclass_fields_are_stable() -> None:
    assert [field.name for field in fields(BootstrapWeightSample)] == [
        "bootstrap_id",
        "candidate_id",
        "lag_step",
        "lag_time",
        "weight",
    ]
    assert [field.name for field in fields(BootstrapParameterSample)] == [
        "bootstrap_id",
        "candidate_id",
        "parameter_name",
        "parameter_value",
    ]
    assert [field.name for field in fields(BootstrapLagSummarySample)] == [
        "bootstrap_id",
        "candidate_id",
        "mean_lag",
        "p50_lag",
        "p90_lag",
        "tail_mass",
    ]
    assert [field.name for field in fields(ParameterUncertaintySummary)] == [
        "parameter_name",
        "estimate",
        "lower",
        "upper",
        "bootstrap_std",
        "n_samples",
    ]
    assert [field.name for field in fields(WeightUncertaintySummary)] == [
        "lag_step",
        "lag_time",
        "weight_estimate",
        "lower",
        "upper",
        "bootstrap_std",
    ]
    assert [field.name for field in fields(KernelBootstrapSummary)] == [
        "mean_lag_interval",
        "p50_lag_interval",
        "p90_lag_interval",
        "tail_mass_interval",
        "weight_interval_by_lag",
        "parameter_interval_by_name",
        "stability_score",
    ]
    assert [field.name for field in fields(BootstrapResult)] == [
        "n_bootstrap",
        "n_succeeded",
        "n_failed",
        "failures",
        "weight_samples",
        "parameter_samples",
        "lag_summary_samples",
        "family_selection_counts",
        "warnings",
        "bootstrap_config",
    ]


def test_bootstrap_warning_codes_and_quantiles_are_stable() -> None:
    assert BOOTSTRAP_WARNING_CODES == (
        "BOOTSTRAP_TOO_FEW_SUCCESSES",
        "BOOTSTRAP_WEIGHT_UNSTABLE",
        "BOOTSTRAP_PARAMETER_UNSTABLE",
        "BOOTSTRAP_PARAMETER_PROVENANCE_MISSING",
        "BOOTSTRAP_LAG_SUMMARY_UNSTABLE",
        "BOOTSTRAP_FAMILY_UNSTABLE",
        "BOOTSTRAP_INTERVAL_TOUCHES_BOUNDARY",
        "BOOTSTRAP_VALIDATION_WINDOW_CHANGED",
        "BOOTSTRAP_CONTEXT_MISMATCH",
        "BOOTSTRAP_BLOCK_LENGTH_INVALID",
    )
    assert DEFAULT_BOOTSTRAP_INTERVAL_QUANTILES == (0.025, 0.975)


def test_bootstrap_schema_helpers_are_deterministic() -> None:
    assert bootstrap_weight_samples_schema() == {
        "bootstrap_id": pl.Int64,
        "candidate_id": pl.String,
        "lag_step": pl.Int64,
        "lag_time": pl.Float64,
        "weight": pl.Float64,
    }
    assert bootstrap_parameter_samples_schema() == {
        "bootstrap_id": pl.Int64,
        "candidate_id": pl.String,
        "parameter_name": pl.String,
        "parameter_value": pl.Float64,
    }
    assert bootstrap_lag_summary_samples_schema() == {
        "bootstrap_id": pl.Int64,
        "candidate_id": pl.String,
        "mean_lag": pl.Float64,
        "p50_lag": pl.Float64,
        "p90_lag": pl.Float64,
        "tail_mass": pl.Float64,
    }
    assert parameter_uncertainty_summary_schema() == {
        "parameter_name": pl.String,
        "estimate": pl.Float64,
        "lower": pl.Float64,
        "upper": pl.Float64,
        "bootstrap_std": pl.Float64,
        "n_samples": pl.Int64,
    }
    assert weight_uncertainty_summary_schema() == {
        "lag_step": pl.Int64,
        "lag_time": pl.Float64,
        "weight_estimate": pl.Float64,
        "lower": pl.Float64,
        "upper": pl.Float64,
        "bootstrap_std": pl.Float64,
    }


def test_bootstrap_objects_are_exported() -> None:
    expected = {
        "BOOTSTRAP_WARNING_CODES",
        "DEFAULT_BOOTSTRAP_INTERVAL_QUANTILES",
        "BootstrapResult",
        "BootstrapWeightSample",
        "BootstrapParameterSample",
        "BootstrapLagSummarySample",
        "KernelBootstrapSummary",
        "ParameterUncertaintySummary",
        "WeightUncertaintySummary",
        "bootstrap_weight_samples_schema",
        "bootstrap_parameter_samples_schema",
        "bootstrap_lag_summary_samples_schema",
        "parameter_uncertainty_summary_schema",
        "weight_uncertainty_summary_schema",
    }
    import rtdfeatures.diagnostics as _diag
    assert expected.issubset(set(dir(_diag)))
    # Confirm objects can also be imported via the bootstrap module directly
    from rtdfeatures.bootstrap import (
        BOOTSTRAP_WARNING_CODES,
        DEFAULT_BOOTSTRAP_INTERVAL_QUANTILES,
    )
    assert BOOTSTRAP_WARNING_CODES is not None
    assert DEFAULT_BOOTSTRAP_INTERVAL_QUANTILES is not None
