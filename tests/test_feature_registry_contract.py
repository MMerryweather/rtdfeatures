"""Tests: FeatureSpec, FeatureRegistry, TransformResult, builder polish."""

from __future__ import annotations

import polars as pl
import pytest

from rtdfeatures.diagnostics import FeatureEvidenceReport, TransformReport
from rtdfeatures.features import (
    KernelFeatureBuilder,
    build_feature_evidence,
)
from rtdfeatures.features.registry import FeatureRegistry, FeatureSpec, TransformResult
from rtdfeatures.kernels import FixedDelayKernel

# -
# helpers
# -

def _simple_df() -> pl.DataFrame:
    return pl.DataFrame({
        "t": [0, 1, 2, 3, 4],
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "cat": ["A", "A", "B", "B", "A"],
    })


def _simple_builder() -> KernelFeatureBuilder:
    return KernelFeatureBuilder(
        kernels={"k1": FixedDelayKernel(delay_steps=1, max_lag_steps=2, dt=1.0, name="k1")},
        time_col="t",
        numeric_cols=["x"],
        category_cols=["cat"],
    )


# -
# FeatureSpec
# -

class TestFeatureSpec:
    def test_construction(self) -> None:
        spec = FeatureSpec(
            name="k1_num_x_wmean",
            kernel_name="k1",
            source_col="x",
            family="numeric",
            metric="wmean",
            category_level=None,
            lag_steps=(1,),
            kernel_summary={"mean_lag": 1.0},
        )
        assert spec.name == "k1_num_x_wmean"
        assert spec.kernel_name == "k1"
        assert spec.source_col == "x"
        assert spec.family == "numeric"
        assert spec.metric == "wmean"
        assert spec.category_level is None
        assert spec.lag_steps == (1,)
        assert spec.kernel_summary["mean_lag"] == 1.0

    def test_is_frozen(self) -> None:
        spec = FeatureSpec(
            name="k1_num_x_wmean",
            kernel_name="k1",
            source_col="x",
            family="numeric",
            metric="wmean",
            category_level=None,
            lag_steps=(1,),
            kernel_summary={},
        )
        with pytest.raises(BaseException):
            spec.name = "other"  # type: ignore[misc]

    def test_category_level_is_none_for_numeric(self) -> None:
        spec = FeatureSpec(
            name="k1_num_x_wmean",
            kernel_name="k1",
            source_col="x",
            family="numeric",
            metric="wmean",
            category_level=None,
            lag_steps=(1,),
            kernel_summary={},
        )
        assert spec.category_level is None

    def test_category_level_is_set_for_cat_frac(self) -> None:
        spec = FeatureSpec(
            name="k1_cat_cat_A_frac",
            kernel_name="k1",
            source_col="cat",
            family="categorical",
            metric="frac",
            category_level="A",
            lag_steps=(1,),
            kernel_summary={},
        )
        assert spec.category_level == "A"


# -
# FeatureRegistry
# -

class TestFeatureRegistry:
    def test_empty_registry(self) -> None:
        reg = FeatureRegistry(specs=())
        assert len(reg) == 0
        assert list(reg) == []

    def test_len_and_iter(self) -> None:
        specs = (
            FeatureSpec(
                name="a", kernel_name="k1", source_col="x", family="numeric",
                metric="wmean", category_level=None, lag_steps=(1,), kernel_summary={},
            ),
            FeatureSpec(
                name="b", kernel_name="k1", source_col="x", family="numeric",
                metric="wstd", category_level=None, lag_steps=(1,), kernel_summary={},
            ),
        )
        reg = FeatureRegistry(specs=specs)
        assert len(reg) == 2
        names = [s.name for s in reg]
        assert names == ["a", "b"]

    def test_filter_by_kernel_name(self) -> None:
        specs = (
            FeatureSpec(
                name="k1_num_x_wmean", kernel_name="k1", source_col="x",
                family="numeric", metric="wmean", category_level=None,
                lag_steps=(1,), kernel_summary={},
            ),
            FeatureSpec(
                name="k2_num_x_wmean", kernel_name="k2", source_col="x",
                family="numeric", metric="wmean", category_level=None,
                lag_steps=(2,), kernel_summary={},
            ),
        )
        reg = FeatureRegistry(specs=specs)
        k1_only = reg.filter(kernel_name="k1")
        assert len(k1_only) == 1
        assert k1_only.specs[0].kernel_name == "k1"

    def test_filter_by_source_col(self) -> None:
        specs = (
            FeatureSpec(
                name="k1_num_x_wmean", kernel_name="k1", source_col="x",
                family="numeric", metric="wmean", category_level=None,
                lag_steps=(1,), kernel_summary={},
            ),
            FeatureSpec(
                name="k1_num_y_wmean", kernel_name="k1", source_col="y",
                family="numeric", metric="wmean", category_level=None,
                lag_steps=(1,), kernel_summary={},
            ),
        )
        reg = FeatureRegistry(specs=specs)
        y_only = reg.filter(source_col="y")
        assert len(y_only) == 1
        assert y_only.specs[0].source_col == "y"

    def test_filter_by_family(self) -> None:
        specs = (
            FeatureSpec(
                name="k1_num_x_wmean", kernel_name="k1", source_col="x",
                family="numeric", metric="wmean", category_level=None,
                lag_steps=(1,), kernel_summary={},
            ),
            FeatureSpec(
                name="k1_age_mean", kernel_name="k1", source_col="_kernel_",
                family="age", metric="mean", category_level=None,
                lag_steps=(1,), kernel_summary={},
            ),
        )
        reg = FeatureRegistry(specs=specs)
        age_only = reg.filter(family="age")
        assert len(age_only) == 1
        assert age_only.specs[0].family == "age"

    def test_filter_by_metric(self) -> None:
        specs = (
            FeatureSpec(
                name="k1_num_x_wmean", kernel_name="k1", source_col="x",
                family="numeric", metric="wmean", category_level=None,
                lag_steps=(1,), kernel_summary={},
            ),
            FeatureSpec(
                name="k1_num_x_wstd", kernel_name="k1", source_col="x",
                family="numeric", metric="wstd", category_level=None,
                lag_steps=(1,), kernel_summary={},
            ),
        )
        reg = FeatureRegistry(specs=specs)
        std_only = reg.filter(metric="wstd")
        assert len(std_only) == 1
        assert std_only.specs[0].metric == "wstd"

    def test_filter_no_match_returns_empty(self) -> None:
        reg = FeatureRegistry(specs=(
            FeatureSpec(
                name="a", kernel_name="k1", source_col="x", family="numeric",
                metric="wmean", category_level=None, lag_steps=(1,), kernel_summary={},
            ),
        ))
        assert len(reg.filter(kernel_name="nonexistent")) == 0

    def test_filter_with_none_defaults(self) -> None:
        specs = (
            FeatureSpec(
                name="a", kernel_name="k1", source_col="x", family="numeric",
                metric="wmean", category_level=None, lag_steps=(1,), kernel_summary={},
            ),
            FeatureSpec(
                name="b", kernel_name="k2", source_col="y", family="age",
                metric="mean", category_level=None, lag_steps=(1,), kernel_summary={},
            ),
        )
        reg = FeatureRegistry(specs=specs)
        assert len(reg.filter()) == 2


# -
# TransformResult
# -

class TestTransformResult:
    def test_bundles_features_report_registry(self) -> None:
        df = _simple_df()
        builder = _simple_builder()
        result = builder.transform_result(df)

        assert isinstance(result, TransformResult)
        assert isinstance(result.features, pl.DataFrame)
        assert isinstance(result.report, TransformReport)
        assert isinstance(result.feature_registry, FeatureRegistry)

    def test_feature_registry_count_matches_column_count(self) -> None:
        df = _simple_df()
        builder = _simple_builder()
        result = builder.transform_result(df)

        n_feature_cols = len(result.features.columns) - 1  # exclude time_col
        assert len(result.feature_registry) == n_feature_cols

    def test_feature_names_match_registry(self) -> None:
        df = _simple_df()
        builder = _simple_builder()
        result = builder.transform_result(df)

        feature_cols = [c for c in result.features.columns if c != "t"]
        registry_names = [s.name for s in result.feature_registry]
        assert sorted(feature_cols) == sorted(registry_names)


# -
# transform / transform_with_report / transform_result
# -

class TestBuilderNewMethods:
    def test_transform_still_returns_plain_dataframe(self) -> None:
        df = _simple_df()
        builder = _simple_builder()
        out = builder.transform(df)
        assert isinstance(out, pl.DataFrame)

    def test_transform_with_report_returns_tuple(self) -> None:
        df = _simple_df()
        builder = _simple_builder()
        features, report = builder.transform_with_report(df)
        assert isinstance(features, pl.DataFrame)
        assert isinstance(report, TransformReport)

    def test_transform_with_report_preserves_last_transform_report(self) -> None:
        df = _simple_df()
        builder = _simple_builder()
        features, report = builder.transform_with_report(df)
        assert builder.last_transform_report is report

    def test_transform_result_preserves_last_transform_report(self) -> None:
        df = _simple_df()
        builder = _simple_builder()
        result = builder.transform_result(df)
        assert builder.last_transform_report is result.report

    def test_repeated_transforms_do_not_corrupt_reports(self) -> None:
        builder = _simple_builder()
        df = _simple_df()

        r1_features, r1_report = builder.transform_with_report(df)
        assert builder.last_transform_report is r1_report

        r2_features, r2_report = builder.transform_with_report(df)
        assert builder.last_transform_report is r2_report

        assert r1_report is not r2_report
        assert r1_report.row_count == r2_report.row_count
        assert r1_report.feature_names == r2_report.feature_names

    def test_transform_and_transform_result_feature_tables_match(self) -> None:
        df = _simple_df()
        builder = _simple_builder()

        plain = builder.transform(df)
        result = builder.transform_result(df)

        assert plain.columns == result.features.columns
        assert plain.equals(result.features)

    def test_diagnose_transform_consistent_with_transform_with_report(self) -> None:
        df = _simple_df()
        builder = _simple_builder()

        _, report = builder.transform_with_report(df)
        diag = builder.diagnose_transform(df)

        assert report.feature_names == diag.feature_names
        assert report.warmup_rows == diag.warmup_rows
        assert report.missing_rows_by_feature == diag.missing_rows_by_feature

    def test_last_transform_report_available_after_all_three_methods(self) -> None:
        df = _simple_df()
        builder = _simple_builder()

        builder.transform(df)
        assert builder.last_transform_report is not None

        builder.transform_with_report(df)
        assert builder.last_transform_report is not None

        builder.transform_result(df)
        assert builder.last_transform_report is not None


# -
# Evidence from registry
# -

class TestEvidenceFromRegistry:
    def test_evidence_with_fit_results_from_registry(self) -> None:
        """Verify evidence produced via registry can carry fit info."""
        df = _simple_df()
        builder = _simple_builder()
        _ = builder.transform(df)

        result = builder.transform_result(df)
        report = build_feature_evidence(
            builder=builder,
            feature_registry=result.feature_registry,
            fit_result_by_kernel={"k1": None},  # type: ignore[dict-item]  # no actual fit
        )

        assert isinstance(report, FeatureEvidenceReport)
        assert report.feature_count == len(result.feature_registry)

    def test_build_evidence_with_registry_rejects_unknown_feature(self) -> None:
        bad_registry = FeatureRegistry(specs=(
            FeatureSpec(
                name="nonexistent_feature",
                kernel_name="nonexistent",
                source_col="x",
                family="numeric",
                metric="wmean",
                category_level=None,
                lag_steps=(1,),
                kernel_summary={},
            ),
        ))
        builder = _simple_builder()
        with pytest.raises(ValueError, match="references unknown kernel"):
            build_feature_evidence(
                builder=builder,
                feature_registry=bad_registry,
            )

    def test_evidence_with_full_fit_from_registry(self) -> None:
        """Integration: evidence from registry with fit_result_by_kernel."""
        df = _simple_df()
        builder = _simple_builder()
        _ = builder.transform(df)

        from rtdfeatures.diagnostics import (
            BaselineComparison,
            FitDataCoverageSummary,
            FitDiagnostics,
            IdentifiabilityReport,
            KernelFitResult,
            KernelShapeSummary,
        )
        from rtdfeatures.kernels import LearnedKernel

        fit_kernel = LearnedKernel(
            weights=(0.0, 1.0, 0.0),
            lag_steps=(0, 1, 2),
            dt=1.0,
            min_lag_steps=0,
            max_lag_steps=2,
            name="learned_k1",
        )
        fit_result = KernelFitResult(
            kernel=fit_kernel,
            fit_diagnostics=FitDiagnostics(
                train_loss=0.1, validation_loss=0.2,
                input_variance=1.0, target_variance=1.0,
                kernel_weight_sum=1.0, mean_lag=1.0,
                p50_lag=1.0, p90_lag=2.0, tail_mass=0.0,
                boundary_mass_fraction=0.0,
            ),
            identifiability_report=IdentifiabilityReport(
                warnings=("weak identifiability",), is_reliable=False,
                warning_codes=("IDENT_WEAK",),
                warning_severity_by_code={"IDENT_WEAK": "medium"},
            ),
            baseline_comparison=BaselineComparison(
                no_lag_validation_loss=0.25,
                best_single_lag_validation_loss=0.23,
                learned_validation_loss=0.2,
            ),
            kernel_shape_summary=KernelShapeSummary(
                normalized_entropy=0.0, max_weight=1.0, min_weight=0.0,
                concentration_hhi=1.0, effective_lag_count=1.0,
            ),
            fit_data_coverage_summary=FitDataCoverageSummary(
                total_rows=100, valid_windows=80,
                train_windows=64, validation_windows=16,
                retained_row_fraction=0.8, retained_window_fraction=0.8,
            ),
            fit_provenance={"fit_result_id": "fit-k1"},
        )

        result = builder.transform_result(df)
        report = build_feature_evidence(
            builder=builder,
            feature_registry=result.feature_registry,
            fit_result_by_kernel={"k1": fit_result},
        )
        assert all(e.fit_result_id == "fit-k1" for e in report.feature_evidence
                   if e.kernel_name == "k1")


# -
# Underscore-heavy names (underscores in kernel names, column names, levels)
# -

class TestUnderscoreNames:
    def test_underscore_kernel_name(self) -> None:
        kernel = FixedDelayKernel(delay_steps=1, max_lag_steps=2, dt=1.0, name="my_kernel_v2")
        builder = KernelFeatureBuilder(
            kernels={"my_kernel_v2": kernel},
            time_col="t",
            numeric_cols=["x"],
        )
        df = _simple_df()
        result = builder.transform_result(df)

        assert len(result.feature_registry) == 7  # 3 numeric + 4 age
        assert all(s.kernel_name == "my_kernel_v2" for s in result.feature_registry)
        assert all(s.name.startswith("my_kernel_v2_") for s in result.feature_registry)

    def test_underscore_source_column(self) -> None:
        df = pl.DataFrame({
            "t": [0, 1, 2, 3],
            "my_input_col": [1.0, 2.0, 3.0, 4.0],
        })
        kernel = FixedDelayKernel(delay_steps=1, max_lag_steps=2, dt=1.0, name="k")
        builder = KernelFeatureBuilder(
            kernels={"k": kernel},
            time_col="t",
            numeric_cols=["my_input_col"],
        )
        result = builder.transform_result(df)
        num_specs = [s for s in result.feature_registry if s.family == "numeric"]
        assert all(s.source_col == "my_input_col" for s in num_specs)
        assert all("my_input_col" in s.name for s in num_specs)

    def test_underscore_category_column_and_levels(self) -> None:
        df = pl.DataFrame({
            "t": [0, 1, 2, 3],
            "x": [1.0, 2.0, 3.0, 4.0],
            "mode_state": ["A_fine", "A_fine", "B_coarse", "B_coarse"],
        })
        kernel = FixedDelayKernel(delay_steps=1, max_lag_steps=2, dt=1.0, name="k")
        builder = KernelFeatureBuilder(
            kernels={"k": kernel},
            time_col="t",
            numeric_cols=["x"],
            category_cols=["mode_state"],
        )
        result = builder.transform_result(df)

        frac_specs = [s for s in result.feature_registry if s.metric == "frac"]
        assert len(frac_specs) == 2  # two levels
        assert all(s.source_col == "mode_state" for s in frac_specs)
        assert all(s.category_level in ("A_fine", "B_coarse") for s in frac_specs)

    def test_underscore_kernel_column_level_evidence_via_registry(self) -> None:
        df = pl.DataFrame({
            "t": [0, 1, 2, 3],
            "my_signal": [1.0, 2.0, 3.0, 4.0],
            "my_cat_var": ["high_val", "low_val", "high_val", "low_val"],
        })
        kernel = FixedDelayKernel(delay_steps=1, max_lag_steps=2, dt=1.0, name="my_kernel")
        builder = KernelFeatureBuilder(
            kernels={"my_kernel": kernel},
            time_col="t",
            numeric_cols=["my_signal"],
            category_cols=["my_cat_var"],
        )
        _ = builder.transform(df)

        result = builder.transform_result(df)
        evidence = build_feature_evidence(
            builder=builder,
            feature_registry=result.feature_registry,
        )
        assert evidence.feature_count == len(result.feature_registry)
        for item in evidence.feature_evidence:
            assert item.kernel_name == "my_kernel"
        frac_items = [
            e for e in evidence.feature_evidence
            if e.feature_family == "categorical_fraction"
        ]
        assert len(frac_items) == 2
        assert all("my_cat_var" == e.source_col for e in frac_items)


# -
# Edge cases
# -

class TestEdgeCases:
    def test_no_category_cols_produces_no_cat_features(self) -> None:
        df = _simple_df()
        builder = KernelFeatureBuilder(
            kernels={"k1": FixedDelayKernel(delay_steps=1, max_lag_steps=2, dt=1.0, name="k1")},
            time_col="t",
            numeric_cols=["x"],
            category_cols=[],
        )
        result = builder.transform_result(df)
        cat_specs = [s for s in result.feature_registry if s.family == "categorical"]
        assert len(cat_specs) == 0

    def test_no_numeric_cols_still_produces_age_features(self) -> None:
        df = pl.DataFrame({"t": [0, 1, 2, 3]})
        builder = KernelFeatureBuilder(
            kernels={"k1": FixedDelayKernel(delay_steps=1, max_lag_steps=2, dt=1.0, name="k1")},
            time_col="t",
            numeric_cols=[],
        )
        result = builder.transform_result(df)
        assert len(result.feature_registry) == 4  # age features only
        assert all(s.family == "age" for s in result.feature_registry)

    def test_multiple_kernels_produce_multiple_registry_entries(self) -> None:
        df = _simple_df()
        k1 = FixedDelayKernel(delay_steps=1, max_lag_steps=2, dt=1.0, name="k1")
        k2 = FixedDelayKernel(delay_steps=2, max_lag_steps=2, dt=1.0, name="k2")
        builder = KernelFeatureBuilder(
            kernels={"k1": k1, "k2": k2},
            time_col="t",
            numeric_cols=["x"],
            category_cols=["cat"],
        )
        result = builder.transform_result(df)
        assert len(result.feature_registry) == len(result.features.columns) - 1
        assert len({s.kernel_name for s in result.feature_registry}) == 2

    def test_feature_evidence_from_registry_can_operate_without_last_transform_report(self) -> None:
        """Evidence using registry doesn't need last_transform_report."""
        df = _simple_df()
        builder = _simple_builder()
        result = builder.transform_result(df)

        # Clear the report to prove registry path is self-sufficient.
        builder.last_transform_report = None

        evidence = build_feature_evidence(
            builder=builder,
            feature_registry=result.feature_registry,
        )
        assert evidence.feature_count == len(result.feature_registry)


def test_feature_registry_round_trip_via_dataclasses_dict() -> None:
    """FeatureRegistry round-trips through dataclasses.asdict()."""
    import dataclasses

    registry = FeatureRegistry(specs=(
        FeatureSpec(
            name="k1_num_x_wmean",
            kernel_name="k1",
            source_col="x",
            family="numeric",
            metric="wmean",
            category_level=None,
            lag_steps=(1,),
            kernel_summary={"mean_lag": 1.0},
        ),
        FeatureSpec(
            name="k1_cat_A_score",
            kernel_name="k1",
            source_col="cat",
            family="categorical",
            metric="score",
            category_level="A",
            lag_steps=(1,),
            kernel_summary={"mean_lag": 1.0},
        ),
    ))
    as_dict = {"specs": tuple(
        FeatureSpec(**dataclasses.asdict(s)) for s in registry.specs
    )}
    restored = FeatureRegistry(**as_dict)
    assert restored == registry


def test_transform_report_round_trips_through_dataclasses() -> None:
    import dataclasses

    from rtdfeatures.diagnostics import TransformReport

    report = TransformReport(
        row_count=10,
        output_row_count=10,
        warmup_rows=3,
        feature_names=("k_num_x_wmean",),
        missing_rows_by_feature={},
        zero_denominator_rows_by_feature={},
    )
    as_dict = dataclasses.asdict(report)
    restored = TransformReport(**as_dict)
    assert restored == report
