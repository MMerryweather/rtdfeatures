from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from rtdfeatures import KernelFeatureBuilder
from rtdfeatures.features.registry import FeatureRegistry, FeatureSpec
from rtdfeatures.kernels import FixedDelayKernel


def _spec(name: str, source_col: str, family: str, metric: str) -> FeatureSpec:
    return FeatureSpec(
        name=name,
        kernel_name="k",
        source_col=source_col,
        family=family,
        metric=metric,
        category_level=None,
        lag_steps=(0, 1, 2),
        kernel_summary={"name": "k", "max_lag_steps": 2},
    )


def _baseline_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "t": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x": [10.0, 20.0, 30.0, 40.0, 50.0],
            "cat": ["A", "B", "A", "B", "A"],
        }
    )


def _baseline_result() -> tuple[pl.DataFrame, tuple[str, ...]]:
    builder = KernelFeatureBuilder(
        kernels={"k": FixedDelayKernel(delay_steps=1, max_lag_steps=2, dt=1.0)},
        time_col="t",
        numeric_cols=["x"],
        category_cols=["cat"],
    )
    result = builder.transform_result(_baseline_df())
    non_time_cols = tuple(name for name in result.features.columns if name != "t")
    return result.features, non_time_cols


def test_feature_registry_names_returns_ordered_names() -> None:
    registry = FeatureRegistry(
        specs=(
            _spec("k_num_x_wmean", "x", "numeric", "wmean"),
            _spec("k_num_x_wstd", "x", "numeric", "wstd"),
            _spec("k_age_mean", "_kernel_", "age", "mean"),
        )
    )
    assert registry.names() == ("k_num_x_wmean", "k_num_x_wstd", "k_age_mean")


def test_feature_registry_to_frame_has_one_row_per_spec() -> None:
    registry = FeatureRegistry(
        specs=(
            _spec("k_num_x_wmean", "x", "numeric", "wmean"),
            _spec("k_num_x_wstd", "x", "numeric", "wstd"),
        )
    )
    frame = registry.to_frame()
    assert frame.height == len(registry.specs)
    assert frame.columns == [
        "name",
        "kernel_name",
        "source_col",
        "family",
        "metric",
        "category_level",
        "lag_steps",
        "kernel_summary",
    ]


def test_feature_registry_to_frame_preserves_order() -> None:
    registry = FeatureRegistry(
        specs=(
            _spec("k_num_x_wstd", "x", "numeric", "wstd"),
            _spec("k_num_x_wmean", "x", "numeric", "wmean"),
            _spec("k_num_x_wsum", "x", "numeric", "wsum"),
        )
    )
    frame = registry.to_frame()
    assert frame["name"].to_list() == [spec.name for spec in registry.specs]


def test_feature_registry_to_frame_handles_empty_registry() -> None:
    frame = FeatureRegistry(specs=()).to_frame()
    assert frame.height == 0
    assert frame.columns == [
        "name",
        "kernel_name",
        "source_col",
        "family",
        "metric",
        "category_level",
        "lag_steps",
        "kernel_summary",
    ]
    assert frame.schema == {
        "name": pl.Utf8,
        "kernel_name": pl.Utf8,
        "source_col": pl.Utf8,
        "family": pl.Utf8,
        "metric": pl.Utf8,
        "category_level": pl.Utf8,
        "lag_steps": pl.List(pl.Int64),
        "kernel_summary": pl.Object,
    }


def test_feature_registry_to_frame_schema_matches_empty_and_non_empty() -> None:
    empty = FeatureRegistry(specs=()).to_frame()
    non_empty = FeatureRegistry(
        specs=(_spec("k_num_x_wmean", "x", "numeric", "wmean"),)
    ).to_frame()
    assert non_empty.schema == empty.schema


def test_feature_builder_private_blocks_preserve_baseline_feature_names() -> None:
    _, non_time_cols = _baseline_result()
    assert non_time_cols == (
        "k_num_x_wmean",
        "k_num_x_wstd",
        "k_num_x_wsum",
        "k_cat_cat_A_frac",
        "k_cat_cat_B_frac",
        "k_cat_cat_entropy",
        "k_age_mean",
        "k_age_p50",
        "k_age_p90",
        "k_age_tail_gt_threshold",
    )


def test_feature_builder_private_blocks_preserve_baseline_feature_values() -> None:
    features, _ = _baseline_result()
    wmean = features["k_num_x_wmean"].to_list()
    assert np.isnan(wmean[0])
    assert np.isnan(wmean[1])
    assert wmean[2:] == pytest.approx([20.0, 30.0, 40.0])
