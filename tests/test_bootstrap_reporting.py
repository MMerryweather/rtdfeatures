"""Tests for bootstrap reporting helpers."""

from __future__ import annotations

from copy import deepcopy

import polars as pl

from rtdfeatures.bootstrap import (
    bootstrap_lag_interval_table,
    bootstrap_lag_summary_samples_table,
    bootstrap_parameter_interval_table,
    bootstrap_parameter_samples_table,
    bootstrap_summary_compact_dict,
    bootstrap_summary_compact_text,
    bootstrap_weight_interval_table,
    bootstrap_weight_samples_table,
)
from rtdfeatures.diagnostics import (
    BootstrapLagSummarySample,
    BootstrapParameterSample,
    BootstrapResult,
    BootstrapWeightSample,
)


def _result_fixture() -> BootstrapResult:
    return BootstrapResult(
        n_bootstrap=4,
        n_succeeded=3,
        n_failed=1,
        failures=(
            {"bootstrap_id": 3, "candidate_id": "cand_b", "error": "RuntimeError: fail"},
        ),
        weight_samples=(
            BootstrapWeightSample(
                bootstrap_id=1,
                candidate_id="cand_b",
                lag_step=1,
                lag_time=1.0,
                weight=0.8,
            ),
            BootstrapWeightSample(
                bootstrap_id=0,
                candidate_id="cand_a",
                lag_step=0,
                lag_time=0.0,
                weight=0.2,
            ),
            BootstrapWeightSample(
                bootstrap_id=0,
                candidate_id="cand_a",
                lag_step=1,
                lag_time=1.0,
                weight=0.8,
            ),
            BootstrapWeightSample(
                bootstrap_id=1,
                candidate_id="cand_a",
                lag_step=0,
                lag_time=0.0,
                weight=0.3,
            ),
            BootstrapWeightSample(
                bootstrap_id=1,
                candidate_id="cand_a",
                lag_step=1,
                lag_time=1.0,
                weight=0.7,
            ),
            BootstrapWeightSample(
                bootstrap_id=0,
                candidate_id="cand_b",
                lag_step=1,
                lag_time=1.0,
                weight=0.9,
            ),
        ),
        parameter_samples=(
            BootstrapParameterSample(
                bootstrap_id=1,
                candidate_id="cand_a",
                parameter_name="shape_alpha",
                parameter_value=2.1,
            ),
            BootstrapParameterSample(
                bootstrap_id=0,
                candidate_id="cand_a",
                parameter_name="shape_alpha",
                parameter_value=2.0,
            ),
            BootstrapParameterSample(
                bootstrap_id=1,
                candidate_id="cand_a",
                parameter_name="rate_beta",
                parameter_value=1.1,
            ),
            BootstrapParameterSample(
                bootstrap_id=0,
                candidate_id="cand_a",
                parameter_name="rate_beta",
                parameter_value=1.0,
            ),
            BootstrapParameterSample(
                bootstrap_id=2,
                candidate_id="cand_b",
                parameter_name="rate_lambda",
                parameter_value=0.6,
            ),
            BootstrapParameterSample(
                bootstrap_id=3,
                candidate_id="cand_b",
                parameter_name="shape_alpha",
                parameter_value=None,
            ),
        ),
        lag_summary_samples=(
            BootstrapLagSummarySample(
                bootstrap_id=1,
                candidate_id="cand_a",
                mean_lag=1.5,
                p50_lag=1.0,
                p90_lag=2.0,
                tail_mass=0.2,
            ),
            BootstrapLagSummarySample(
                bootstrap_id=0,
                candidate_id="cand_a",
                mean_lag=1.0,
                p50_lag=1.0,
                p90_lag=1.8,
                tail_mass=0.1,
            ),
            BootstrapLagSummarySample(
                bootstrap_id=0,
                candidate_id="cand_b",
                mean_lag=1.9,
                p50_lag=1.7,
                p90_lag=2.4,
                tail_mass=0.4,
            ),
            BootstrapLagSummarySample(
                bootstrap_id=1,
                candidate_id="cand_b",
                mean_lag=2.0,
                p50_lag=1.8,
                p90_lag=2.5,
                tail_mass=0.45,
            ),
        ),
        family_selection_counts={"gamma": 2, "simplex": 1},
        warnings=("BOOTSTRAP_FAMILY_UNSTABLE",),
        bootstrap_config={"candidate_set_id": "set1"},
    )


def test_sample_table_schemas_and_order_are_stable() -> None:
    result = _result_fixture()

    weight = bootstrap_weight_samples_table(result)
    assert weight.columns == ["bootstrap_id", "candidate_id", "lag_step", "lag_time", "weight"]
    assert weight.schema == {
        "bootstrap_id": pl.Int64,
        "candidate_id": pl.String,
        "lag_step": pl.Int64,
        "lag_time": pl.Float64,
        "weight": pl.Float64,
    }
    assert weight.select(["candidate_id", "bootstrap_id", "lag_step"]).to_dicts() == [
        {"candidate_id": "cand_a", "bootstrap_id": 0, "lag_step": 0},
        {"candidate_id": "cand_a", "bootstrap_id": 0, "lag_step": 1},
        {"candidate_id": "cand_a", "bootstrap_id": 1, "lag_step": 0},
        {"candidate_id": "cand_a", "bootstrap_id": 1, "lag_step": 1},
        {"candidate_id": "cand_b", "bootstrap_id": 0, "lag_step": 1},
        {"candidate_id": "cand_b", "bootstrap_id": 1, "lag_step": 1},
    ]

    parameter = bootstrap_parameter_samples_table(result)
    assert parameter.columns == [
        "bootstrap_id",
        "candidate_id",
        "parameter_name",
        "parameter_value",
    ]
    assert parameter.schema == {
        "bootstrap_id": pl.Int64,
        "candidate_id": pl.String,
        "parameter_name": pl.String,
        "parameter_value": pl.Float64,
    }
    assert parameter.filter(pl.col("parameter_value").is_null()).to_dicts() == [
        {
            "bootstrap_id": 3,
            "candidate_id": "cand_b",
            "parameter_name": "shape_alpha",
            "parameter_value": None,
        }
    ]
    assert parameter.select(["candidate_id", "bootstrap_id", "parameter_name"]).to_dicts()[0] == {
        "candidate_id": "cand_a",
        "bootstrap_id": 0,
        "parameter_name": "rate_beta",
    }

    lag = bootstrap_lag_summary_samples_table(result)
    assert lag.columns == [
        "bootstrap_id",
        "candidate_id",
        "mean_lag",
        "p50_lag",
        "p90_lag",
        "tail_mass",
    ]
    assert lag.schema == {
        "bootstrap_id": pl.Int64,
        "candidate_id": pl.String,
        "mean_lag": pl.Float64,
        "p50_lag": pl.Float64,
        "p90_lag": pl.Float64,
        "tail_mass": pl.Float64,
    }
    assert lag.select(["candidate_id", "bootstrap_id"]).to_dicts() == [
        {"candidate_id": "cand_a", "bootstrap_id": 0},
        {"candidate_id": "cand_a", "bootstrap_id": 1},
        {"candidate_id": "cand_b", "bootstrap_id": 0},
        {"candidate_id": "cand_b", "bootstrap_id": 1},
    ]


def test_interval_tables_schema_empty_and_failure_heavy_behavior() -> None:
    empty = BootstrapResult(
        n_bootstrap=3,
        n_succeeded=0,
        n_failed=3,
        failures=(
            {"bootstrap_id": 0, "error": "RuntimeError: fail"},
            {"bootstrap_id": 1, "error": "RuntimeError: fail"},
            {"bootstrap_id": 2, "error": "RuntimeError: fail"},
        ),
        weight_samples=(),
        parameter_samples=(),
        lag_summary_samples=(),
        family_selection_counts={},
        warnings=(),
        bootstrap_config={},
    )

    weight_out = bootstrap_weight_interval_table(empty)
    assert weight_out.columns == [
        "candidate_id",
        "lag_step",
        "lag_time",
        "weight_estimate",
        "lower",
        "upper",
        "bootstrap_std",
        "n_samples",
    ]
    assert weight_out.schema == {
        "candidate_id": pl.String,
        "lag_step": pl.Int64,
        "lag_time": pl.Float64,
        "weight_estimate": pl.Float64,
        "lower": pl.Float64,
        "upper": pl.Float64,
        "bootstrap_std": pl.Float64,
        "n_samples": pl.Int64,
    }
    assert weight_out.height == 0

    param_out = bootstrap_parameter_interval_table(empty)
    assert param_out.columns == [
        "candidate_id",
        "parameter_name",
        "estimate",
        "lower",
        "upper",
        "bootstrap_std",
        "n_samples",
    ]
    assert param_out.height == 0

    lag_out = bootstrap_lag_interval_table(empty)
    assert lag_out.columns == [
        "candidate_id",
        "metric",
        "estimate",
        "lower",
        "upper",
        "bootstrap_std",
        "n_samples",
    ]
    assert lag_out.height == 0


def test_compact_helpers_are_deterministic_and_non_mutating() -> None:
    result = _result_fixture()
    before = deepcopy(result)

    first_dict = bootstrap_summary_compact_dict(result)
    second_dict = bootstrap_summary_compact_dict(result)
    assert first_dict == second_dict
    assert first_dict["candidate_ids"] == ("cand_a", "cand_b")
    assert [row["metric"] for row in first_dict["lag_metrics"][:4]] == [
        "mean_lag",
        "p50_lag",
        "p90_lag",
        "tail_mass",
    ]

    first_text = bootstrap_summary_compact_text(result)
    second_text = bootstrap_summary_compact_text(result)
    assert first_text == second_text
    assert first_text.startswith("bootstrap summary: n_bootstrap=4, n_succeeded=3, n_failed=1")

    cand_b_weight = bootstrap_weight_interval_table(result, candidate_id="cand_b")
    assert cand_b_weight["candidate_id"].to_list() == ["cand_b"]
    assert cand_b_weight["n_samples"].to_list() == [2]

    assert result == before
