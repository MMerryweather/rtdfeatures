"""legacy milestone contract tests for v0.3 docs updates."""

from __future__ import annotations

import inspect
from dataclasses import fields

from tests.simulation_harness.contracts import GeneratorOutput
from tests.simulation_harness.scenarios import (
    make_closed_loop_crushing_dataset,
    make_flotation_bank_dataset,
    make_plug_flow_dataset,
    make_plug_flow_spread_dataset,
    make_tank_dataset,
    make_toy_full_plant_dataset,
)

from rtdfeatures.diagnostics import FitDiagnostics, TransformReport


def test_docs_example_tank_helper_runs() -> None:
    out = make_tank_dataset(n_rows=120, dt=1.0, seed=7, feed_mass=100.0)

    assert isinstance(out, GeneratorOutput)
    assert out.data.height == 120
    assert "tank" in out.true_kernels
    assert out.scenario["name"] == "tank"


def test_documented_helper_signatures_and_return_shape() -> None:
    strict_signature_helpers = {
        make_tank_dataset: ("n_rows", "dt", "seed", "feed_mass"),
        make_plug_flow_dataset: ("n_rows", "dt", "seed", "feed_mass"),
        make_plug_flow_spread_dataset: ("n_rows", "dt", "seed", "feed_mass"),
        make_flotation_bank_dataset: ("n_rows", "dt", "seed", "feed_mass", "n_cells"),
    }
    runtime_keyword_only_helpers = (
        make_closed_loop_crushing_dataset,
        make_toy_full_plant_dataset,
    )

    for helper, names in strict_signature_helpers.items():
        signature = inspect.signature(helper)
        assert tuple(signature.parameters.keys()) == names
        for parameter in signature.parameters.values():
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY

        out = helper(n_rows=20, dt=1.0, seed=1, feed_mass=100.0)
        assert isinstance(out, GeneratorOutput)
        assert hasattr(out, "data")
        assert hasattr(out, "true_kernels")
        assert hasattr(out, "genealogy")
        assert hasattr(out, "scenario")

    for helper in runtime_keyword_only_helpers:
        out = helper(n_rows=20, dt=1.0, seed=1, feed_mass=100.0)
        assert isinstance(out, GeneratorOutput)
        assert hasattr(out, "data")
        assert hasattr(out, "true_kernels")
        assert hasattr(out, "genealogy")
        assert hasattr(out, "scenario")


def test_transform_and_fit_diagnostics_fields_remain_stable() -> None:
    transform_fields = {field.name for field in fields(TransformReport)}
    fit_fields = {field.name for field in fields(FitDiagnostics)}

    assert transform_fields >= {
        "row_count",
        "output_row_count",
        "warmup_rows",
        "feature_names",
        "missing_rows_by_feature",
        "missing_rows_by_kernel",
        "zero_denominator_rows_by_feature",
        "zero_denominator_rows_by_kernel",
        "warmup_unusable_summary",
        "collision_naming_summary",
    }
    assert fit_fields == {
        "train_loss",
        "validation_loss",
        "input_variance",
        "target_variance",
        "kernel_weight_sum",
        "mean_lag",
        "p50_lag",
        "p90_lag",
        "tail_mass",
        "boundary_mass_fraction",
    }
