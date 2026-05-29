from __future__ import annotations

import pytest
from tests.simulation_harness.params import (
    validate_lag_bounds,
    validate_non_negative_mass,
    validate_positive_timestep,
)


def test_negative_mass_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        validate_non_negative_mass(-0.1)


def test_invalid_lag_bounds_raise() -> None:
    with pytest.raises(ValueError, match=">="):
        validate_lag_bounds(4, 3)


def test_non_positive_timestep_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        validate_positive_timestep(0.0)
