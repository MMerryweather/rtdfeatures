from __future__ import annotations

import polars as pl

from rtdfeatures.diagnostics import (
    BaselineComparison,
    FeatureEvidenceReport,
    FitDataCoverageSummary,
    FitDiagnostics,
    IdentifiabilityReport,
    KernelFitResult,
    KernelShapeSummary,
)
from rtdfeatures.features import (
    KernelFeatureBuilder,
    build_feature_evidence,
    feature_evidence_compact_dict,
    feature_evidence_compact_text,
    feature_evidence_table,
)
from rtdfeatures.kernels import FixedDelayKernel, LearnedKernel


def _make_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "t": [0, 1, 2, 3, 4],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "cat": ["A", "A", "B", "B", "A"],
        }
    )


def _make_builder() -> KernelFeatureBuilder:
    kernel = FixedDelayKernel(delay_steps=1, max_lag_steps=2, dt=1.0, name="k1")
    return KernelFeatureBuilder(
        kernels={"k1": kernel},
        time_col="t",
        numeric_cols=["x"],
        category_cols=["cat"],
    )


def _make_fit_result() -> KernelFitResult:
    kernel = LearnedKernel(
        weights=(0.0, 1.0, 0.0),
        lag_steps=(0, 1, 2),
        dt=1.0,
        min_lag_steps=0,
        max_lag_steps=2,
        name="learned_k1",
    )
    return KernelFitResult(
        kernel=kernel,
        fit_diagnostics=FitDiagnostics(
            train_loss=0.1,
            validation_loss=0.2,
            input_variance=1.0,
            target_variance=1.0,
            kernel_weight_sum=1.0,
            mean_lag=1.0,
            p50_lag=1.0,
            p90_lag=2.0,
            tail_mass=0.0,
            boundary_mass_fraction=0.0,
        ),
        identifiability_report=IdentifiabilityReport(
            warnings=("weak identifiability",),
            is_reliable=False,
            warning_codes=("IDENT_WEAK",),
            warning_severity_by_code={"IDENT_WEAK": "medium"},
        ),
        baseline_comparison=BaselineComparison(
            no_lag_validation_loss=0.25,
            best_single_lag_validation_loss=0.23,
            learned_validation_loss=0.2,
        ),
        kernel_shape_summary=KernelShapeSummary(
            normalized_entropy=0.0,
            max_weight=1.0,
            min_weight=0.0,
            concentration_hhi=1.0,
            effective_lag_count=1.0,
        ),
        fit_data_coverage_summary=FitDataCoverageSummary(
            total_rows=100,
            valid_windows=80,
            train_windows=64,
            validation_windows=16,
            retained_row_fraction=0.8,
            retained_window_fraction=0.8,
        ),
        fit_provenance={"fit_result_id": "fit-k1"},
    )


def test_transform_values_and_columns_unchanged_with_evidence() -> None:
    builder = _make_builder()
    df = _make_df()

    features_before = builder.transform(df)
    _ = builder.diagnose_feature_evidence()
    features_after = builder.transform(df)

    assert features_before.columns == features_after.columns
    assert features_before.equals(features_after)


def test_raw_kernel_evidence_fallback_and_count_matches_features() -> None:
    builder = _make_builder()
    _ = builder.transform(_make_df())

    report = builder.diagnose_feature_evidence()
    transform_report = builder.last_transform_report

    assert isinstance(report, FeatureEvidenceReport)
    assert transform_report is not None
    assert report.feature_count == len(report.feature_evidence)
    assert report.feature_count == len(transform_report.feature_names)
    assert {item.interpretation for item in report.feature_evidence} == {"unknown"}
    assert {item.evidence_completeness for item in report.feature_evidence} == {"kernel_only"}


def test_feature_families_include_numeric_categorical_entropy_and_age() -> None:
    builder = _make_builder()
    _ = builder.transform(_make_df())

    report = build_feature_evidence(
        builder=builder, feature_registry=builder._build_feature_registry(),
    )
    families = {item.feature_family for item in report.feature_evidence}

    assert "numeric_wmean" in families
    assert "numeric_wstd" in families
    assert "numeric_wsum" in families
    assert "categorical_fraction" in families
    assert "categorical_entropy" in families
    assert "age_mean" in families
    assert "age_p50" in families


def test_fit_and_bootstrap_evidence_and_warning_propagation() -> None:
    builder = _make_builder()
    _ = builder.transform(_make_df())

    report = build_feature_evidence(
        builder=builder,
        feature_registry=builder._build_feature_registry(),
        fit_result_by_kernel={"k1": _make_fit_result()},
        candidate_id_by_kernel={"k1": "cand-k1"},
        bootstrap_summary_by_kernel={"k1": {"n_bootstrap": 100}},
    )

    assert {item.evidence_completeness for item in report.feature_evidence} == {"full_evidence"}
    assert all(item.fit_result_id == "fit-k1" for item in report.feature_evidence)
    assert report.warning_summary["IDENT_WEAK"] == report.feature_count


def test_fit_only_evidence_completeness_is_fit_evidence() -> None:
    builder = _make_builder()
    _ = builder.transform(_make_df())

    report = build_feature_evidence(
        builder=builder,
        feature_registry=builder._build_feature_registry(),
        fit_result_by_kernel={"k1": _make_fit_result()},
    )

    assert {item.evidence_completeness for item in report.feature_evidence} == {"fit_evidence"}


def test_categorical_fraction_parsing_handles_level_with_underscores() -> None:
    df = pl.DataFrame(
        {
            "t": [0, 1, 2, 3],
            "x": [1.0, 2.0, 3.0, 4.0],
            "mode_state": ["A_state", "A_state", "B_state_v2", "B_state_v2"],
        }
    )
    kernel = FixedDelayKernel(delay_steps=1, max_lag_steps=2, dt=1.0, name="k1")
    builder = KernelFeatureBuilder(
        kernels={"k1": kernel},
        time_col="t",
        numeric_cols=["x"],
        category_cols=["mode_state"],
    )
    features = builder.transform(df)
    report = build_feature_evidence(
        builder=builder,
        feature_registry=builder._build_feature_registry(feature_names=tuple(features.columns[1:])),
    )

    frac_sources = {
        item.source_col
        for item in report.feature_evidence
        if item.feature_family == "categorical_fraction"
    }
    assert frac_sources == {"mode_state"}


def test_kernel_count_reflects_participating_kernels_for_subset() -> None:
    kernel_a = FixedDelayKernel(delay_steps=1, max_lag_steps=2, dt=1.0, name="k_a")
    kernel_b = FixedDelayKernel(delay_steps=2, max_lag_steps=2, dt=1.0, name="k_b")
    builder = KernelFeatureBuilder(
        kernels={"k_a": kernel_a, "k_b": kernel_b},
        time_col="t",
        numeric_cols=["x"],
        category_cols=[],
    )
    _ = builder.transform(_make_df())

    report = build_feature_evidence(
        builder=builder,
        feature_registry=builder._build_feature_registry(
            feature_names=("k_a_num_x_wmean", "k_a_num_x_wstd")
        ),
    )

    assert report.kernel_count == 1


def test_feature_evidence_parsing_handles_reserved_tokens_in_kernel_name() -> None:
    kernel_tokenized = FixedDelayKernel(
        delay_steps=1,
        max_lag_steps=2,
        dt=1.0,
        name="k_num_cat_age",
    )
    kernel_plain = FixedDelayKernel(delay_steps=2, max_lag_steps=2, dt=1.0, name="plain")
    builder = KernelFeatureBuilder(
        kernels={"k_num_cat_age": kernel_tokenized, "plain": kernel_plain},
        time_col="t",
        numeric_cols=["x"],
        category_cols=["cat"],
    )
    features = builder.transform(_make_df())
    report = build_feature_evidence(
        builder=builder,
        feature_registry=builder._build_feature_registry(feature_names=tuple(features.columns[1:])),
    )

    tokenized_rows = [
        item for item in report.feature_evidence if item.kernel_name == "k_num_cat_age"
    ]
    plain_rows = [item for item in report.feature_evidence if item.kernel_name == "plain"]

    assert tokenized_rows
    assert plain_rows
    assert all(
        item.feature_name.startswith("k_num_cat_age_") for item in tokenized_rows
    )
    assert all(item.feature_name.startswith("plain_") for item in plain_rows)


def test_table_schema_missing_optional_and_deterministic_ordering() -> None:
    builder = _make_builder()
    _ = builder.transform(_make_df())
    report = builder.diagnose_feature_evidence()

    table = feature_evidence_table(report)

    assert table.columns == [
        "feature_name",
        "source_col",
        "feature_family",
        "kernel_name",
        "kernel_family",
        "interpretation",
        "evidence_completeness",
        "warning_count",
        "warning_codes",
        "has_fit_evidence",
        "has_comparison_evidence",
        "has_bootstrap_evidence",
    ]
    assert table["feature_name"].to_list() == sorted(table["feature_name"].to_list())
    assert table["has_fit_evidence"].to_list() == [False] * report.feature_count
    assert table["has_bootstrap_evidence"].to_list() == [False] * report.feature_count


def test_compact_helpers_stable_and_non_mutating() -> None:
    builder = _make_builder()
    _ = builder.transform(_make_df())
    report = builder.diagnose_feature_evidence()
    original_count = report.feature_count

    compact = feature_evidence_compact_dict(report)
    text = feature_evidence_compact_text(report)

    assert compact["feature_count"] == original_count
    assert "kernel_only" in text
    assert report.feature_count == original_count


def test_fit_only_comparison_flag_is_false_in_table_and_compact() -> None:
    builder = _make_builder()
    _ = builder.transform(_make_df())
    report = build_feature_evidence(
        builder=builder,
        feature_registry=builder._build_feature_registry(),
        fit_result_by_kernel={"k1": _make_fit_result()},
    )

    table = feature_evidence_table(report)
    assert table["has_fit_evidence"].to_list() == [True] * report.feature_count
    assert table["has_comparison_evidence"].to_list() == [False] * report.feature_count

    compact = feature_evidence_compact_dict(report)
    assert all(
        row["has_fit_evidence"] is True
        for row in compact["by_feature_name"].values()
    )
    assert all(
        row["has_comparison_evidence"] is False
        for row in compact["by_feature_name"].values()
    )


def test_comparison_bootstrap_without_fit_sets_comparison_flag_true() -> None:
    builder = _make_builder()
    _ = builder.transform(_make_df())
    report = build_feature_evidence(
        builder=builder,
        feature_registry=builder._build_feature_registry(),
        candidate_id_by_kernel={"k1": "cand-k1"},
        bootstrap_summary_by_kernel={"k1": {"n_bootstrap": 100}},
    )

    assert {item.evidence_completeness for item in report.feature_evidence} == {
        "bootstrap_evidence"
    }
    table = feature_evidence_table(report)
    assert table["has_fit_evidence"].to_list() == [False] * report.feature_count
    assert table["has_bootstrap_evidence"].to_list() == [True] * report.feature_count
    assert table["has_comparison_evidence"].to_list() == [True] * report.feature_count

    compact = feature_evidence_compact_dict(report)
    assert all(
        row["has_fit_evidence"] is False
        for row in compact["by_feature_name"].values()
    )
    assert all(
        row["has_bootstrap_evidence"] is True
        for row in compact["by_feature_name"].values()
    )
    assert all(
        row["has_comparison_evidence"] is True
        for row in compact["by_feature_name"].values()
    )


