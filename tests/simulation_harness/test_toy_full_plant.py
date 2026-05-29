from __future__ import annotations

import polars as pl
import pytest
from tests.simulation_harness.genealogy import (
    assert_genealogy_mass_closure,
    validate_genealogy_schema,
)
from tests.simulation_harness.scenarios import make_toy_full_plant_dataset


def _assert_regular_grid(data: pl.DataFrame, *, dt: float, tol: float = 1e-9) -> None:
    diffs = data["time"].diff().drop_nulls().to_list()
    assert diffs
    for d in diffs:
        assert d == pytest.approx(dt, abs=tol)


def test_stage_level_shape_and_grid_contract() -> None:
    out = make_toy_full_plant_dataset()
    data = out.data

    assert data.columns == [
        "time",
        "feed_mass",
        "feed_grade",
        "target_grade",
        "crusher_output_mass",
        "ball_mill_product_mass",
        "cyclone_overflow_mass",
        "cyclone_underflow_recycle_mass",
        "flotation_bank_1_mass",
        "flotation_bank_2_mass",
        "flotation_bank_3_mass",
        "cleaner_product_mass",
    ]
    _assert_regular_grid(data, dt=float(out.scenario["dt"]))
    assert data.height == int(out.scenario["n_rows"])
    assert all(v >= 0.0 for v in data["cleaner_product_mass"].to_list())


def test_final_genealogy_closure_after_warmup() -> None:
    out = make_toy_full_plant_dataset()
    validate_genealogy_schema(out.genealogy)
    assert_genealogy_mass_closure(out.genealogy, tol=1e-9)

    contribution = (
        out.genealogy.filter(~pl.col("is_warmup"))
        .group_by("output_time")
        .agg(pl.col("contribution_fraction").sum().alias("frac_sum"))
        .sort("output_time")
    )
    assert contribution.height > 0
    for v in contribution["frac_sum"].to_list():
        assert v == pytest.approx(1.0, abs=1e-9)


def test_final_effective_kernel_is_causal_non_negative_and_broader() -> None:
    out = make_toy_full_plant_dataset()
    final_kernel = out.true_kernels["toy_full_plant_final_effective"]
    crusher = out.true_kernels["toy_full_plant_open_loop_crusher"]
    cyclone = out.true_kernels["toy_full_plant_cyclone_delay"]
    cleaner = out.true_kernels["toy_full_plant_cleaner"]

    assert final_kernel["min_lag"] >= 0
    assert sum(final_kernel["weights"]) == pytest.approx(1.0, abs=1e-12)
    assert all(w >= 0.0 for w in final_kernel["weights"])

    final_spread = final_kernel["p90_lag"] - final_kernel["p50_lag"]
    assert final_spread > (crusher["p90_lag"] - crusher["p50_lag"])
    assert final_spread > (cyclone["p90_lag"] - cyclone["p50_lag"])
    assert final_spread > (cleaner["p90_lag"] - cleaner["p50_lag"])


def test_deterministic_regression_fixture_output() -> None:
    a = make_toy_full_plant_dataset(seed=42)
    b = make_toy_full_plant_dataset(seed=42)

    assert a.scenario == b.scenario
    assert a.true_kernels == b.true_kernels
    assert a.data.equals(b.data)
    assert a.genealogy.equals(b.genealogy)

    final_kernel = a.true_kernels["toy_full_plant_final_effective"]
    assert final_kernel["lag_steps"][:6] == [1, 2, 3, 4, 5, 6]
    assert final_kernel["weights"][:3] == pytest.approx(
        [0.0, 0.0, 0.0], rel=0.0, abs=1e-18
    )
    assert a.data["target_grade"][120] == pytest.approx(0.9899520246495053, abs=1e-12)
