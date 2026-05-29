"""Genealogy contribution-table utilities for harness fixtures."""

from __future__ import annotations

import polars as pl

GENEALOGY_COLUMNS = [
    "output_time",
    "source_time",
    "unit",
    "path",
    "source_mass",
    "contribution_mass",
    "contribution_fraction",
    "is_warmup",
]


def empty_genealogy() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "output_time": pl.Series([], dtype=pl.Int64),
            "source_time": pl.Series([], dtype=pl.Int64),
            "unit": pl.Series([], dtype=pl.String),
            "path": pl.Series([], dtype=pl.String),
            "source_mass": pl.Series([], dtype=pl.Float64),
            "contribution_mass": pl.Series([], dtype=pl.Float64),
            "contribution_fraction": pl.Series([], dtype=pl.Float64),
            "is_warmup": pl.Series([], dtype=pl.Boolean),
        }
    )


def validate_genealogy_schema(df: pl.DataFrame) -> None:
    if df.columns != GENEALOGY_COLUMNS:
        raise ValueError("genealogy must use exact long-table schema")


def assert_genealogy_mass_closure(df: pl.DataFrame, *, tol: float = 1e-9) -> None:
    validate_genealogy_schema(df)
    non_warmup = df.filter(~pl.col("is_warmup"))
    if non_warmup.is_empty():
        return

    sums = non_warmup.group_by("output_time").agg(
        pl.col("contribution_fraction").sum().alias("frac_sum")
    )
    bad = sums.filter((pl.col("frac_sum") - 1.0).abs() > tol)
    if bad.height > 0:
        raise ValueError("contribution_fraction must sum to one for each non-warmup output_time")
