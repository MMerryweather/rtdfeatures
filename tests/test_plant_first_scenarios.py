from __future__ import annotations

from typing import cast

import polars as pl
import pytest
from examples._support.plant_first_scenarios import (
    _kernel_metadata,
    _normalise,
    core_scenario_fixtures,
    make_mini_flowsheet_dataset,
)


def _assert_regular_grid(data: pl.DataFrame, *, expected_dt_seconds: float) -> None:
    deltas = data["time"].diff().drop_nulls().dt.total_seconds().to_list()
    assert deltas
    for delta in deltas:
        assert float(delta) == pytest.approx(expected_dt_seconds, abs=1e-9)


def test_core_scenario_fixtures_are_deterministic_and_configured() -> None:
    fixtures = core_scenario_fixtures()
    assert len(fixtures) == 6

    for fixture in fixtures:
        first = fixture.dataset_factory()
        second = fixture.dataset_factory()
        assert first.scenario == second.scenario
        assert first.true_kernels == second.true_kernels
        assert first.data.equals(second.data)
        assert "input_signal->target_signal" in first.true_kernels
        _assert_regular_grid(first.data, expected_dt_seconds=float(first.scenario["dt"]))


def test_mini_flowsheet_is_deterministic_and_has_expected_columns() -> None:
    first = make_mini_flowsheet_dataset()
    second = make_mini_flowsheet_dataset()

    assert first.scenario == second.scenario
    assert first.true_kernels == second.true_kernels
    assert first.data.equals(second.data)
    _assert_regular_grid(first.data, expected_dt_seconds=float(first.scenario["dt"]))

    assert first.data.columns == [
        "time",
        "feed_mass",
        "feed_copper_grade",
        "ore_type",
        "crusher_output_mass",
        "ball_mill_product_mass",
        "cyclone_overflow_mass",
        "cyclone_underflow_recycle_mass",
        "flotation_bank_1_mass",
        "flotation_bank_2_mass",
        "flotation_bank_3_mass",
        "cleaner_product_mass",
        "cleaner_copper_grade",
        "cleaner_recovered_copper_mass",
    ]
    assert first.data["ore_type"].n_unique() == 2
    assert set(first.data["ore_type"].unique().to_list()) == {"A", "B"}


def test_mini_flowsheet_transition_metadata_and_single_transition() -> None:
    dataset = make_mini_flowsheet_dataset()
    params = dataset.scenario["params"]
    transition_row = int(params["ore_transition_row"])
    assert transition_row == int(round(0.45 * int(dataset.scenario["n_rows"])))

    ore_type = dataset.data["ore_type"].to_list()
    assert all(value == "B" for value in ore_type[:transition_row])
    assert all(value == "A" for value in ore_type[transition_row:])

    transition_count = sum(
        1 for idx in range(1, len(ore_type)) if ore_type[idx - 1] != ore_type[idx]
    )
    assert transition_count == 1


def test_mini_flowsheet_kernel_metadata_and_positive_mass_outputs() -> None:
    dataset = make_mini_flowsheet_dataset()
    kernel = dataset.true_kernels["feed_copper_grade->cleaner_copper_grade"]
    assert kernel["min_lag"] >= 0
    assert sum(kernel["weights"]) == pytest.approx(1.0, abs=1e-12)
    assert kernel["p90_lag"] >= kernel["p50_lag"]
    assert kernel["mean_lag"] >= 0.0

    assert cast(float, dataset.data["cleaner_product_mass"].min()) >= 0.0
    assert cast(float, dataset.data["cleaner_recovered_copper_mass"].min()) >= 0.0
    assert cast(float, dataset.data["feed_mass"].min()) > 0.0


def test_normalise_zero_sum_raises_value_error() -> None:
    with pytest.raises(ValueError, match="weights must sum to a positive value"):
        _normalise([0.0, 0.0, 0.0])


def test_normalise_empty_list_raises_value_error() -> None:
    with pytest.raises(ValueError, match="weights must sum to a positive value"):
        _normalise([])


def test_kernel_metadata_mismatched_lag_weight_lengths_raises_value_error() -> None:
    with pytest.raises(ValueError, match="lag_steps and weights must have equal non-zero length"):
        _kernel_metadata([1, 2, 3], [0.5], dt=1.0)


def test_make_mini_flowsheet_dataset_non_positive_n_rows_raises_value_error() -> None:
    with pytest.raises(ValueError, match="n_rows must be positive"):
        make_mini_flowsheet_dataset(n_rows=0)
    with pytest.raises(ValueError, match="n_rows must be positive"):
        make_mini_flowsheet_dataset(n_rows=-10)


def test_make_mini_flowsheet_dataset_non_positive_dt_raises_value_error() -> None:
    with pytest.raises(ValueError, match="dt must be positive"):
        make_mini_flowsheet_dataset(dt=0.0)
    with pytest.raises(ValueError, match="dt must be positive"):
        make_mini_flowsheet_dataset(dt=-5.0)
