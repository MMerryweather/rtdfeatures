from __future__ import annotations

import polars as pl

from rtdfeatures.features import KernelFeatureBuilder
from rtdfeatures.kernels import FixedDelayKernel


def _make_builder() -> KernelFeatureBuilder:
    builder = KernelFeatureBuilder(
        kernels={"k1": FixedDelayKernel(delay_steps=1, max_lag_steps=2, dt=1.0, name="k1")},
        time_col="t",
        numeric_cols=["x"],
        category_cols=["cat"],
    )
    builder.category_levels_by_col = {"cat": ("A", "B")}
    return builder


def _make_builder_without_levels() -> KernelFeatureBuilder:
    return KernelFeatureBuilder(
        kernels={"k1": FixedDelayKernel(delay_steps=1, max_lag_steps=2, dt=1.0, name="k1")},
        time_col="t",
        numeric_cols=["x"],
        category_cols=["cat"],
    )


def _make_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "t": [0, 1, 2, 3, 4],
            "x": [10.0, 20.0, 30.0, 40.0, 50.0],
            "cat": ["A", "A", "B", "B", "A"],
        }
    )


def test_transform_result_registry_specs_match_generated_features() -> None:
    result = _make_builder().transform_result(_make_df())
    feature_cols = [name for name in result.features.columns if name != "t"]
    spec_names = [spec.name for spec in result.feature_registry.specs]
    assert spec_names == feature_cols


def test_numeric_feature_specs_are_created_during_generation() -> None:
    result = _make_builder().transform_result(_make_df())
    numeric_specs = [spec for spec in result.feature_registry.specs if spec.family == "numeric"]
    assert {spec.metric for spec in numeric_specs} == {"wmean", "wstd", "wsum"}
    assert {spec.source_col for spec in numeric_specs} == {"x"}


def test_categorical_feature_specs_are_created_during_generation() -> None:
    result = _make_builder().transform_result(_make_df())
    cat_specs = [spec for spec in result.feature_registry.specs if spec.family == "categorical"]
    frac_specs = [spec for spec in cat_specs if spec.metric == "frac"]
    entropy_specs = [spec for spec in cat_specs if spec.metric == "entropy"]
    assert {spec.category_level for spec in frac_specs} == {"A", "B"}
    assert len(entropy_specs) == 1
    assert entropy_specs[0].category_level is None


def test_age_feature_specs_are_created_during_generation() -> None:
    result = _make_builder().transform_result(_make_df())
    age_specs = [spec for spec in result.feature_registry.specs if spec.family == "age"]
    assert {spec.metric for spec in age_specs} == {"mean", "p50", "p90", "tail_gt_threshold"}
    assert {spec.source_col for spec in age_specs} == {"__kernel__"}


def test_feature_registry_order_matches_feature_table_order() -> None:
    result = _make_builder().transform_result(_make_df())
    non_time_cols = [name for name in result.features.columns if name != "t"]
    assert [spec.name for spec in result.feature_registry.specs] == non_time_cols


def test_diagnose_feature_evidence_filters_specs_by_exact_name_without_parsing() -> None:
    builder = _make_builder()
    result = builder.transform_result(_make_df())
    selected_name = "k1_cat_cat_A_frac"
    report = builder.diagnose_feature_evidence(feature_names=[selected_name, "k1_cat_cat_frac"])
    assert report.feature_count == 1
    assert [item.feature_name for item in report.feature_evidence] == [selected_name]
    assert selected_name in [spec.name for spec in result.feature_registry.specs]


def test_build_feature_registry_filters_exact_names_without_transform_state() -> None:
    builder = _make_builder()
    selected_names = [
        "k1_num_x_wmean",
        "k1_cat_cat_A_frac",
        "k1_cat_cat_entropy",
    ]

    registry = builder._build_feature_registry(
        feature_names=[*selected_names, "k1_num_x", "k1_cat_cat_frac"]
    )

    assert [spec.name for spec in registry.specs] == selected_names


def test_build_feature_registry_infers_requested_categorical_fraction_specs_on_fresh_builder(
) -> None:
    builder = _make_builder_without_levels()
    selected_name = "k1_cat_cat_A_frac"

    registry = builder._build_feature_registry(feature_names=[selected_name])

    assert [spec.name for spec in registry.specs] == [selected_name]
    assert registry.specs[0].family == "categorical"
    assert registry.specs[0].metric == "frac"
    assert registry.specs[0].category_level == "A"
