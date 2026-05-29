"""Contract tests for feature name helpers — names must match exact v1 patterns."""

from __future__ import annotations

import pytest

from rtdfeatures.features.names import (
    age_feature_name,
    categorical_entropy_feature_name,
    categorical_fraction_feature_name,
    numeric_feature_name,
)


class TestNumericFeatureNames:
    def test_numeric_feature_names_are_stable(self) -> None:
        assert numeric_feature_name("k", "feed", "wmean") == "k_num_feed_wmean"
        assert numeric_feature_name("k", "feed", "wstd") == "k_num_feed_wstd"
        assert numeric_feature_name("k", "feed", "wsum") == "k_num_feed_wsum"

    def test_empty_kernel_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="kernel_name"):
            numeric_feature_name("", "feed", "wmean")

    def test_empty_source_col_rejected(self) -> None:
        with pytest.raises(ValueError, match="source_col"):
            numeric_feature_name("k", "", "wmean")

    def test_invalid_numeric_metric_rejected(self) -> None:
        with pytest.raises(ValueError, match="metric"):
            numeric_feature_name("k", "feed", "invalid")


class TestCategoricalFeatureNames:
    def test_categorical_feature_names_are_stable(self) -> None:
        assert categorical_fraction_feature_name("k", "source", "A") == "k_cat_source_A_frac"
        assert categorical_entropy_feature_name("k", "source") == "k_cat_source_entropy"

    def test_empty_kernel_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="kernel_name"):
            categorical_fraction_feature_name("", "source", "A")
        with pytest.raises(ValueError, match="kernel_name"):
            categorical_entropy_feature_name("", "source")

    def test_empty_source_col_rejected(self) -> None:
        with pytest.raises(ValueError, match="source_col"):
            categorical_fraction_feature_name("k", "", "A")
        with pytest.raises(ValueError, match="source_col"):
            categorical_entropy_feature_name("k", "")


class TestAgeFeatureNames:
    def test_age_feature_names_are_stable(self) -> None:
        assert age_feature_name("k", "mean") == "k_age_mean"
        assert age_feature_name("k", "p50") == "k_age_p50"
        assert age_feature_name("k", "p90") == "k_age_p90"
        assert age_feature_name("k", "tail_gt_threshold") == "k_age_tail_gt_threshold"

    def test_empty_kernel_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="kernel_name"):
            age_feature_name("", "mean")

    def test_invalid_age_metric_rejected(self) -> None:
        with pytest.raises(ValueError, match="metric"):
            age_feature_name("k", "invalid")
