"""API contract tests for V1 stable public API.

Confirms that KernelFeatureBuilder methods return the documented types
and that no method is spuriously marked deprecated.
"""

from __future__ import annotations

import warnings

import polars as pl

from rtdfeatures import (
    FixedDelayKernel,
    KernelFeatureBuilder,
    TransformResult,
)
from rtdfeatures.diagnostics import (
    FeatureEvidenceReport,
    TransformReport,
)


def _make_builder_and_df() -> tuple[KernelFeatureBuilder, pl.DataFrame]:
    """Return a builder and small DataFrame for contract tests."""
    kernel = FixedDelayKernel(delay_steps=3, max_lag_steps=5, dt=1.0)
    builder = KernelFeatureBuilder(
        kernels={"fixed": kernel},
        time_col="t",
        numeric_cols=["x"],
    )
    df = pl.DataFrame(
        {
            "t": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "x": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )
    return builder, df


class TestTransformReturnsPolarsDataFrame:
    def test_transform_returns_polars_dataframe(self) -> None:
        builder, df = _make_builder_and_df()
        result = builder.transform(df)
        assert isinstance(result, pl.DataFrame), (
            f"Expected pl.DataFrame, got {type(result)}"
        )


class TestTransformWithReportReturnsFeaturesAndReportTuple:
    def test_transform_with_report_returns_features_and_report_tuple(self) -> None:
        builder, df = _make_builder_and_df()
        features, report = builder.transform_with_report(df)
        assert isinstance(features, pl.DataFrame), (
            f"Expected pl.DataFrame, got {type(features)}"
        )
        assert isinstance(report, TransformReport), (
            f"Expected TransformReport, got {type(report)}"
        )


class TestTransformResultReturnsTransformResultWithRegistry:
    def test_transform_result_returns_transform_result_with_registry(self) -> None:
        builder, df = _make_builder_and_df()
        result = builder.transform_result(df)
        assert isinstance(result, TransformResult), (
            f"Expected TransformResult, got {type(result)}"
        )
        assert isinstance(result.features, pl.DataFrame)
        assert isinstance(result.report, TransformReport)
        from rtdfeatures import FeatureRegistry
        assert isinstance(result.feature_registry, FeatureRegistry)
        assert len(result.feature_registry) > 0


class TestDiagnoseTransformReturnsTransformReport:
    def test_diagnose_transform_returns_transform_report(self) -> None:
        builder, df = _make_builder_and_df()
        report = builder.diagnose_transform(df)
        assert isinstance(report, TransformReport), (
            f"Expected TransformReport, got {type(report)}"
        )


class TestDiagnoseFeatureEvidenceAfterTransformResult:
    def test_diagnose_feature_evidence_after_transform_result(self) -> None:
        builder, df = _make_builder_and_df()
        builder.transform_result(df)
        evidence = builder.diagnose_feature_evidence()
        assert isinstance(evidence, FeatureEvidenceReport), (
            f"Expected FeatureEvidenceReport, got {type(evidence)}"
        )


class TestTransformWithReportIsNotDeprecated:
    def test_transform_with_report_is_not_deprecated(self) -> None:
        builder, df = _make_builder_and_df()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            builder.transform_with_report(df)
        deprecation_warnings = [
            x for x in w if issubclass(x.category, DeprecationWarning)
        ]
        assert not deprecation_warnings, (
            f"transform_with_report emitted DeprecationWarning: "
            f"{[str(x.message) for x in deprecation_warnings]}"
        )
