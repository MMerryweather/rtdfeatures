from __future__ import annotations

import polars as pl
import pytest
from tests.simulation_harness.genealogy import assert_genealogy_mass_closure


def test_genealogy_mass_closure_passes_for_non_warmup_rows() -> None:
    df = pl.DataFrame(
        {
            "output_time": [10, 10, 11, 11],
            "source_time": [4, 5, 5, 6],
            "unit": ["u1", "u1", "u1", "u1"],
            "path": ["a", "b", "a", "b"],
            "source_mass": [1.0, 1.0, 1.0, 1.0],
            "contribution_mass": [0.4, 0.6, 0.3, 0.7],
            "contribution_fraction": [0.4, 0.6, 0.3, 0.7],
            "is_warmup": [False, False, False, False],
        }
    )
    assert_genealogy_mass_closure(df, tol=1e-9)


def test_genealogy_mass_closure_raises_when_sum_is_not_one() -> None:
    df = pl.DataFrame(
        {
            "output_time": [10, 10],
            "source_time": [4, 5],
            "unit": ["u1", "u1"],
            "path": ["a", "b"],
            "source_mass": [1.0, 1.0],
            "contribution_mass": [0.2, 0.6],
            "contribution_fraction": [0.2, 0.6],
            "is_warmup": [False, False],
        }
    )
    with pytest.raises(ValueError, match="sum to one"):
        assert_genealogy_mass_closure(df)
