from __future__ import annotations

import pytest
from tests.simulation_harness.kernels import (
    build_lag_grid,
    convolve_discrete_kernels,
    normalize_weights,
)


def test_normalize_weights_sums_to_one() -> None:
    w = normalize_weights([2.0, 3.0, 5.0])
    assert sum(w) == pytest.approx(1.0, abs=1e-12)


def test_build_lag_grid_inclusive_bounds() -> None:
    assert build_lag_grid(2, 5) == [2, 3, 4, 5]


def test_convolution_matches_expected_discrete_mass() -> None:
    lags, weights = convolve_discrete_kernels([0, 1], [0.5, 0.5], [1], [1.0])
    assert lags == [1, 2]
    assert weights == pytest.approx([0.5, 0.5], abs=1e-12)
