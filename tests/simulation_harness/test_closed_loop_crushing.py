from __future__ import annotations

import polars as pl
import pytest
from tests.simulation_harness.genealogy import (
    assert_genealogy_mass_closure,
    validate_genealogy_schema,
)
from tests.simulation_harness.scenarios import make_closed_loop_crushing_dataset


def test_closed_loop_mass_closure() -> None:
    out = make_closed_loop_crushing_dataset()
    data = out.data

    discharge = data["crusher_screen_discharge_mass"].to_list()
    product = data["product_mass"].to_list()
    recycle = data["recycle_mass"].to_list()

    assert all(m >= 0.0 for m in recycle)
    assert all(m >= 0.0 for m in product)
    assert max(recycle) < 10.0

    for d, p, r in zip(discharge, product, recycle):
        assert p + r == pytest.approx(d, abs=1e-12)


def test_closed_loop_genealogy_contribution_closure() -> None:
    out = make_closed_loop_crushing_dataset()
    validate_genealogy_schema(out.genealogy)
    assert_genealogy_mass_closure(out.genealogy, tol=1e-9)

    contribution = (
        out.genealogy.filter(~pl.col("is_warmup"))
        .group_by("output_time")
        .agg(pl.col("contribution_mass").sum().alias("sum_mass"))
        .sort("output_time")
    )
    assert contribution.height > 0
    for val in contribution["sum_mass"].to_list():
        assert val == pytest.approx(1.0, abs=1e-9)


def test_effective_kernel_tail_mass_exceeds_open_loop_pass() -> None:
    out = make_closed_loop_crushing_dataset()
    effective = out.true_kernels["closed_loop_crushing_product_effective"]
    crusher_pass = out.true_kernels["closed_loop_crushing_crusher_pass"]

    tail_threshold = 8
    effective_tail = sum(
        w
        for lag, w in zip(effective["lag_steps"], effective["weights"])
        if lag >= tail_threshold
    )
    open_loop_tail = sum(
        w
        for lag, w in zip(crusher_pass["lag_steps"], crusher_pass["weights"])
        if lag >= tail_threshold
    )

    assert effective["max_lag"] <= 64
    assert effective_tail > open_loop_tail


def test_closed_loop_is_deterministic_for_fixed_seed() -> None:
    a = make_closed_loop_crushing_dataset(seed=42)
    b = make_closed_loop_crushing_dataset(seed=42)

    assert a.scenario == b.scenario
    assert a.true_kernels == b.true_kernels
    assert a.data.equals(b.data)
    assert a.genealogy.equals(b.genealogy)
