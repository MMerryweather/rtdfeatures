"""Unit tests for feature accumulation internals."""

from __future__ import annotations

import numpy as np
import pytest

from rtdfeatures.features.accumulator import FeatureAccumulator
from rtdfeatures.features.registry import FeatureSpec


def _spec(*, name: str, kernel_name: str = "k1") -> FeatureSpec:
    return FeatureSpec(
        name=name,
        kernel_name=kernel_name,
        source_col="x",
        family="numeric",
        metric="wmean",
        category_level=None,
        lag_steps=(1,),
        kernel_summary={"name": kernel_name},
    )


def test_feature_accumulator_adds_array_and_spec() -> None:
    acc = FeatureAccumulator(n_rows=3)
    values = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    spec = _spec(name="k1_num_x_wmean")

    acc.add(name="k1_num_x_wmean", values=values, spec=spec)

    assert np.array_equal(acc.arrays["k1_num_x_wmean"], values)
    assert acc.specs == [spec]
    assert acc.zero_denominator_rows_by_feature["k1_num_x_wmean"] == 0


def test_feature_accumulator_rejects_duplicate_feature_name() -> None:
    acc = FeatureAccumulator(n_rows=2)
    values = np.array([1.0, 2.0], dtype=np.float64)
    spec = _spec(name="k1_num_x_wmean")
    acc.add(name="k1_num_x_wmean", values=values, spec=spec)

    with pytest.raises(ValueError, match="Generated feature name collision"):
        acc.add(name="k1_num_x_wmean", values=values, spec=spec)


def test_feature_accumulator_rejects_spec_name_mismatch() -> None:
    acc = FeatureAccumulator(n_rows=2)
    values = np.array([1.0, 2.0], dtype=np.float64)

    with pytest.raises(ValueError, match="Feature name mismatch"):
        acc.add(name="k1_num_x_wmean", values=values, spec=_spec(name="different"))


def test_feature_accumulator_rejects_wrong_length_values() -> None:
    acc = FeatureAccumulator(n_rows=3)
    values = np.array([1.0, 2.0], dtype=np.float64)
    spec = _spec(name="k1_num_x_wmean")

    with pytest.raises(ValueError, match="length mismatch"):
        acc.add(name="k1_num_x_wmean", values=values, spec=spec)


def test_feature_accumulator_counts_missing_rows() -> None:
    acc = FeatureAccumulator(n_rows=4)
    values = np.array([1.0, np.nan, np.inf, 4.0], dtype=np.float64)
    spec = _spec(name="k1_num_x_wmean")

    acc.add(name="k1_num_x_wmean", values=values, spec=spec)

    assert acc.missing_rows_by_feature["k1_num_x_wmean"] == 2


def test_feature_accumulator_ensure_kernel_rejects_empty_name() -> None:
    acc = FeatureAccumulator(n_rows=1)

    with pytest.raises(ValueError, match="kernel names must be non-empty"):
        acc.ensure_kernel("")


def test_feature_accumulator_ensure_kernel_is_idempotent() -> None:
    acc = FeatureAccumulator(n_rows=1)

    acc.ensure_kernel("k1")
    acc.ensure_kernel("k1")

    assert acc.kernel_feature_names["k1"] == []
    assert list(acc.kernel_feature_names.keys()) == ["k1"]


def test_feature_accumulator_tracks_kernel_feature_names() -> None:
    acc = FeatureAccumulator(n_rows=2)
    values = np.array([1.0, 2.0], dtype=np.float64)

    acc.add(
        name="k1_num_x_wmean",
        values=values,
        spec=_spec(name="k1_num_x_wmean", kernel_name="k1"),
    )
    acc.add(name="k1_num_x_wstd", values=values, spec=_spec(name="k1_num_x_wstd", kernel_name="k1"))
    acc.ensure_kernel("k2")

    assert acc.kernel_feature_names["k1"] == ["k1_num_x_wmean", "k1_num_x_wstd"]
    assert acc.kernel_feature_names["k2"] == []


def test_feature_accumulator_preserves_insertion_order() -> None:
    acc = FeatureAccumulator(n_rows=2)
    values = np.array([1.0, 2.0], dtype=np.float64)

    acc.add(name="f1", values=values, spec=_spec(name="f1", kernel_name="k1"))
    acc.add(name="f2", values=values, spec=_spec(name="f2", kernel_name="k1"))
    acc.add(name="f3", values=values, spec=_spec(name="f3", kernel_name="k1"))

    assert list(acc.arrays.keys()) == ["f1", "f2", "f3"]
    assert [spec.name for spec in acc.specs] == ["f1", "f2", "f3"]
    assert acc.kernel_feature_names["k1"] == ["f1", "f2", "f3"]


def test_feature_accumulator_finalizes_missing_fractions() -> None:
    acc = FeatureAccumulator(n_rows=4)
    acc.add(
        name="f1",
        values=np.array([1.0, np.nan, 3.0, 4.0], dtype=np.float64),
        spec=_spec(name="f1", kernel_name="k1"),
    )
    acc.add(
        name="f2",
        values=np.array([1.0, 2.0, 3.0, np.nan], dtype=np.float64),
        spec=_spec(name="f2", kernel_name="k2"),
    )

    acc.finalize_missing_fractions()

    assert acc.missing_fraction_by_feature["f1"] == 0.25
    assert acc.missing_fraction_by_feature["f2"] == 0.25


def test_feature_accumulator_finalize_missing_fractions_zero_rows() -> None:
    acc = FeatureAccumulator(n_rows=0)
    acc.missing_rows_by_feature["f1"] = 1

    acc.finalize_missing_fractions()

    assert acc.missing_fraction_by_feature["f1"] == 0.0
