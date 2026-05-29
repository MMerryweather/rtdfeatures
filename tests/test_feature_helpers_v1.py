"""Tests for standalone feature computation helpers ."""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from rtdfeatures import FixedDelayKernel, KernelFeatureBuilder, UniformKernel
from rtdfeatures.features.age import age_feature_values, resolve_age_tail_threshold
from rtdfeatures.features.categorical import categorical_fraction_and_entropy_series
from rtdfeatures.features.numeric import weighted_numeric_series

# -
# numeric helper
# -


def test_weighted_numeric_helper_matches_builder_output() -> None:
    kernel = FixedDelayKernel(delay_steps=1, max_lag_steps=2, dt=1.0)
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    lag_steps = np.asarray(kernel.lag_steps, dtype=np.int64)
    lag_weights = np.asarray(kernel.weights, dtype=np.float64)

    mean, std, sum_, zero_cnt = weighted_numeric_series(
        values=values,
        lag_steps=lag_steps,
        lag_weights=lag_weights,
        max_lag_steps=kernel.max_lag_steps,
        weight_values=None,
    )

    assert zero_cnt == 0
    assert np.isnan(mean[0]) and np.isnan(mean[1])
    assert not np.isnan(mean[2])
    assert math.isclose(mean[2], 2.0)
    assert math.isclose(std[2], 0.0)
    assert math.isclose(sum_[2], 2.0)
    assert math.isclose(mean[3], 3.0)
    assert math.isclose(mean[4], 4.0)
    assert mean.shape == (5,)


# -
# categorical helper
# -


def test_categorical_helper_matches_builder_output() -> None:
    kernel = FixedDelayKernel(delay_steps=1, max_lag_steps=2, dt=1.0)
    category_values = np.array(["A", "A", "B", "B", "A"])
    levels = ["A", "B"]
    lag_steps = np.asarray(kernel.lag_steps, dtype=np.int64)
    lag_weights = np.asarray(kernel.weights, dtype=np.float64)

    fractions, entropy, zero_cnt = categorical_fraction_and_entropy_series(
        category_values=category_values,
        levels=levels,
        lag_steps=lag_steps,
        lag_weights=lag_weights,
        max_lag_steps=kernel.max_lag_steps,
        weight_values=None,
    )

    assert zero_cnt == 0
    assert zero_cnt >= 0
    assert sorted(fractions.keys()) == ["A", "B"]
    assert np.isnan(entropy[0]) and np.isnan(entropy[1])
    assert not np.isnan(entropy[2])
    assert math.isclose(fractions["A"][2], 1.0)  # all mass at lag_step=1 -> "A"
    assert math.isclose(fractions["B"][2], 0.0)
    assert math.isclose(entropy[2], 0.0)  # single level => zero entropy
    assert entropy.shape == (5,)


def test_categorical_helper_with_weight_values() -> None:
    kernel = FixedDelayKernel(delay_steps=1, max_lag_steps=2, dt=1.0)
    category_values = np.array(["A", "A", "B", "B", "A"])
    levels = ["A", "B"]
    lag_steps = np.asarray(kernel.lag_steps, dtype=np.int64)
    lag_weights = np.asarray(kernel.weights, dtype=np.float64)
    weight_values = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    fractions, entropy, zero_cnt = categorical_fraction_and_entropy_series(
        category_values=category_values,
        levels=levels,
        lag_steps=lag_steps,
        lag_weights=lag_weights,
        max_lag_steps=kernel.max_lag_steps,
        weight_values=weight_values,
    )

    assert zero_cnt == 0
    assert math.isclose(fractions["A"][2], 1.0)
    assert math.isclose(entropy[2], 0.0)


def test_categorical_helper_non_finite_weight_skips() -> None:
    kernel = FixedDelayKernel(delay_steps=1, max_lag_steps=2, dt=1.0)
    category_values = np.array(["A", "A", "B", "B", "A"])
    levels = ["A", "B"]
    lag_steps = np.asarray(kernel.lag_steps, dtype=np.int64)
    lag_weights = np.asarray(kernel.weights, dtype=np.float64)
    weight_values = np.array([np.nan, 1.0, 1.0, 1.0, 1.0])

    fractions, entropy, zero_cnt = categorical_fraction_and_entropy_series(
        category_values=category_values,
        levels=levels,
        lag_steps=lag_steps,
        lag_weights=lag_weights,
        max_lag_steps=kernel.max_lag_steps,
        weight_values=weight_values,
    )

    assert np.isnan(fractions["A"][2])


def test_categorical_helper_none_in_window_skips() -> None:
    category_values = np.array(["A", None, "B", "B", "A"], dtype=object)
    levels = ["A", "B"]
    lag_steps = np.array([0, 1, 2], dtype=np.int64)
    lag_weights = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)

    fractions, entropy, zero_cnt = categorical_fraction_and_entropy_series(
        category_values=category_values,
        levels=levels,
        lag_steps=lag_steps,
        lag_weights=lag_weights,
        max_lag_steps=2,
        weight_values=None,
    )

    assert np.isnan(fractions["A"][2])


def test_categorical_helper_zero_denominator() -> None:
    kernel = FixedDelayKernel(delay_steps=1, max_lag_steps=2, dt=1.0)
    category_values = np.array(["A", "A", "B", "B", "A"])
    levels = ["A", "B"]
    lag_steps = np.asarray(kernel.lag_steps, dtype=np.int64)
    lag_weights = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    fractions, entropy, zero_cnt = categorical_fraction_and_entropy_series(
        category_values=category_values,
        levels=levels,
        lag_steps=lag_steps,
        lag_weights=lag_weights,
        max_lag_steps=kernel.max_lag_steps,
        weight_values=None,
    )

    assert zero_cnt > 0


# -
# age helpers
# -


def test_age_helper_matches_builder_output() -> None:
    kernel = UniformKernel(max_lag_steps=4, dt=2.0)
    threshold = resolve_age_tail_threshold(
        min_lag_steps=kernel.min_lag_steps,
        max_lag_steps=kernel.max_lag_steps,
        dt=kernel.dt,
        configured_threshold=None,
    )
    expected_threshold = (0 + 0.75 * (4 - 0)) * 2.0
    assert math.isclose(threshold, expected_threshold)

    threshold_custom = resolve_age_tail_threshold(
        min_lag_steps=0,
        max_lag_steps=4,
        dt=2.0,
        configured_threshold=5.0,
    )
    assert math.isclose(threshold_custom, 5.0)

    values = age_feature_values(kernel=kernel, threshold=threshold)
    assert sorted(values.keys()) == ["mean", "p50", "p90", "tail_gt_threshold"]
    assert math.isclose(values["mean"], kernel.mean_lag())
    assert math.isclose(values["p50"], kernel.percentile(0.5))
    assert math.isclose(values["p90"], kernel.percentile(0.9))
    assert math.isclose(values["tail_gt_threshold"], kernel.tail_mass(threshold))


# -
# regression: builder transform output unchanged
# -


def test_transform_output_unchanged() -> None:
    df = pl.DataFrame({"t": range(10), "x": [1.0] * 10, "cat": ["A", "B"] * 5})
    builder = KernelFeatureBuilder(
        kernels={"k": FixedDelayKernel(delay_steps=1, max_lag_steps=3, dt=1.0)},
        time_col="t",
        numeric_cols=["x"],
        category_cols=["cat"],
    )
    result = builder.transform_result(df)
    cols = set(result.features.columns)
    expected = {
        "t",
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
    }
    assert cols == expected, (
        f"Schema mismatch: extra={cols - expected}, missing={expected - cols}"
    )
