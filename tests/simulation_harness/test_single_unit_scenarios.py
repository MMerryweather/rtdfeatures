from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import polars as pl
import pytest
from tests.simulation_harness.genealogy import (
    assert_genealogy_mass_closure,
    validate_genealogy_schema,
)
from tests.simulation_harness.scenarios import (
    make_flotation_bank_dataset,
    make_plug_flow_dataset,
    make_plug_flow_spread_dataset,
    make_tank_dataset,
)


def _normalized_entropy(weights: list[float]) -> float:
    n = len(weights)
    if n <= 1:
        return 0.0
    entropy = -sum(w * math.log(w) for w in weights if w > 0.0)
    return entropy / math.log(n)


def _assert_regular_grid(data: pl.DataFrame, *, dt: float, tol: float = 1e-9) -> None:
    diffs = data["time"].diff().drop_nulls().to_list()
    assert diffs
    for d in diffs:
        assert d == pytest.approx(dt, abs=tol)


def test_tank_kernel_metadata_and_target_convolution() -> None:
    out = make_tank_dataset()
    kernel = out.true_kernels["tank"]

    expected_mean = sum(k * w for k, w in zip(kernel["lag_steps"], kernel["weights"]))
    tail_mass = sum(w for k, w in zip(kernel["lag_steps"], kernel["weights"]) if k >= 16)
    expected_tail_mass = (0.8**16) * (1.0 - 0.8**17) / (1.0 - 0.8**33)

    assert kernel["mean_lag"] == pytest.approx(expected_mean, abs=1e-12)
    assert tail_mass == pytest.approx(expected_tail_mass, abs=1e-6)

    row = 40
    expected_target = sum(
        kernel["weights"][idx] * out.data["feed_grade"][row - lag]
        for idx, lag in enumerate(kernel["lag_steps"])
        if row - lag >= 0
    )
    assert out.data["target_grade"][row] == pytest.approx(expected_target, abs=1e-12)


def test_plug_flow_and_spread_kernels_follow_defaults() -> None:
    plug = make_plug_flow_dataset()
    spread = make_plug_flow_spread_dataset()

    plug_kernel = plug.true_kernels["plug_flow"]
    spread_kernel = spread.true_kernels["plug_flow_spread"]

    assert plug_kernel["lag_steps"] == [6]
    assert plug_kernel["weights"] == pytest.approx([1.0], abs=1e-12)
    assert spread_kernel["lag_steps"] == [5, 6, 7]
    assert spread_kernel["weights"] == pytest.approx([0.2, 0.6, 0.2], abs=1e-12)
    assert abs(plug_kernel["p50_lag"] - 6) <= 1


def test_flotation_bank_is_broader_than_plug_flow() -> None:
    plug = make_plug_flow_dataset()
    bank = make_flotation_bank_dataset()

    plug_weights = plug.true_kernels["plug_flow"]["weights"]
    bank_weights = bank.true_kernels["flotation_bank"]["weights"]

    assert _normalized_entropy(bank_weights) > _normalized_entropy(plug_weights)
    assert max(bank_weights) < max(plug_weights)
    assert bank.scenario["params"]["n_cells"] == 3


def test_small_fixture_hand_computed_convolution_values() -> None:
    spread = make_plug_flow_spread_dataset(n_rows=10)
    tg = spread.data["target_grade"].to_list()
    fg = spread.data["feed_grade"].to_list()

    assert tg[5] == pytest.approx(0.2 * fg[0], abs=1e-12)
    assert tg[6] == pytest.approx(0.2 * fg[1] + 0.6 * fg[0], abs=1e-12)
    assert tg[7] == pytest.approx(0.2 * fg[2] + 0.6 * fg[1] + 0.2 * fg[0], abs=1e-12)


@pytest.mark.parametrize(
    "factory",
    [
        make_tank_dataset,
        make_plug_flow_dataset,
        make_plug_flow_spread_dataset,
        make_flotation_bank_dataset,
    ],
)
def test_scenarios_contract_readiness(factory: Callable[[], Any]) -> None:
    out = factory()
    assert out.data.columns[:4] == ["time", "feed_mass", "feed_grade", "target_grade"]
    _assert_regular_grid(out.data, dt=float(out.scenario["dt"]))
    validate_genealogy_schema(out.genealogy)
    assert_genealogy_mass_closure(out.genealogy, tol=1e-9)
