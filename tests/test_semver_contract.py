"""Semantic versioning contract snapshot tests.

Protects the stable V1 API from accidental breaking changes by
snapshotting constructor signatures, dataclass fields, generated
feature names, version consistency, and deprecation-state.
"""

from __future__ import annotations

import dataclasses
import inspect
import re
from pathlib import Path

import polars as pl

from rtdfeatures import (
    DelayedExponentialKernel,
    DelayedExponentialKernelLearner,
    ErlangKernel,
    ErlangKernelLearner,
    ExponentialKernel,
    ExponentialKernelLearner,
    FeatureRegistry,
    FeatureSpec,
    FixedDelayKernel,
    FixedDelayKernelLearner,
    GammaKernel,
    GammaKernelLearner,
    Kernel,
    KernelFeatureBuilder,
    LogNormalKernel,
    LogNormalKernelLearner,
    SimplexKernelLearner,
    TransformResult,
    UniformKernel,
    UniformKernelLearner,
)
from rtdfeatures.diagnostics import (
    BaselineComparison,
    FeatureEvidence,
    FeatureEvidenceReport,
    FitDataCoverageSummary,
    FitDiagnostics,
    IdentifiabilityReport,
    KernelFitResult,
    KernelShapeSummary,
    TransformReport,
)

# -
# Constructor signature snapshot test — Constructor signature snapshot test
# -


def _assert_params(obj: type, *, expected: set[str]) -> None:
    sig = inspect.signature(obj)
    actual = set(sig.parameters.keys()) - {"self"}
    missing = expected - actual
    extra = actual - expected
    assert not missing, (
        f"{obj.__name__}._init_ is missing expected parameters: "
        f"{sorted(missing)}"
    )
    assert not extra, (
        f"{obj.__name__}._init_ has unexpected parameters: "
        f"{sorted(extra)}"
    )


class TestConstructorSignatures:
    """Snapshot constructor signatures for root-exported classes only."""

    def test_kernel(self) -> None:
        _assert_params(
            Kernel,
            expected={
                "weights",
                "lag_steps",
                "dt",
                "min_lag_steps",
                "max_lag_steps",
                "name",
            },
        )

    def test_fixed_delay_kernel(self) -> None:
        _assert_params(
            FixedDelayKernel,
            expected={"delay_steps", "max_lag_steps", "dt", "min_lag_steps", "name"},
        )

    def test_uniform_kernel(self) -> None:
        _assert_params(
            UniformKernel,
            expected={"max_lag_steps", "dt", "min_lag_steps", "name"},
        )

    def test_gamma_kernel(self) -> None:
        _assert_params(
            GammaKernel,
            expected={
                "shape_alpha",
                "rate_beta",
                "max_lag_steps",
                "dt",
                "min_lag_steps",
                "name",
            },
        )

    def test_exponential_kernel(self) -> None:
        _assert_params(
            ExponentialKernel,
            expected={
                "rate_lambda",
                "max_lag_steps",
                "dt",
                "min_lag_steps",
                "name",
            },
        )

    def test_delayed_exponential_kernel(self) -> None:
        _assert_params(
            DelayedExponentialKernel,
            expected={
                "delay",
                "rate_lambda",
                "max_lag_steps",
                "dt",
                "min_lag_steps",
                "name",
            },
        )

    def test_erlang_kernel(self) -> None:
        _assert_params(
            ErlangKernel,
            expected={
                "shape_k",
                "rate_beta",
                "max_lag_steps",
                "dt",
                "min_lag_steps",
                "name",
            },
        )

    def test_lognormal_kernel(self) -> None:
        _assert_params(
            LogNormalKernel,
            expected={
                "log_mu",
                "log_sigma",
                "max_lag_steps",
                "dt",
                "min_lag_steps",
                "name",
            },
        )

    def test_simplex_kernel_learner(self) -> None:
        _assert_params(
            SimplexKernelLearner,
            expected={
                "max_lag",
                "min_lag",
                "dt",
                "loss",
                "smoothness_penalty",
                "seed",
                "validation_fraction",
                "learning_rate",
                "max_epochs",
                "huber_delta",
            },
        )

    def test_gamma_kernel_learner(self) -> None:
        _assert_params(
            GammaKernelLearner,
            expected={
                "max_lag",
                "min_lag",
                "dt",
                "loss",
                "smoothness_penalty",
                "seed",
                "validation_fraction",
                "learning_rate",
                "max_epochs",
                "huber_delta",
                "init_shape_alpha",
                "init_rate_beta",
            },
        )

    def test_exponential_kernel_learner(self) -> None:
        _assert_params(
            ExponentialKernelLearner,
            expected={
                "max_lag",
                "min_lag",
                "dt",
                "loss",
                "smoothness_penalty",
                "seed",
                "validation_fraction",
                "learning_rate",
                "max_epochs",
                "huber_delta",
                "init_rate_lambda",
            },
        )

    def test_delayed_exponential_kernel_learner(self) -> None:
        _assert_params(
            DelayedExponentialKernelLearner,
            expected={
                "max_lag",
                "min_lag",
                "dt",
                "loss",
                "smoothness_penalty",
                "seed",
                "validation_fraction",
                "learning_rate",
                "max_epochs",
                "huber_delta",
                "init_delay",
                "init_rate_lambda",
            },
        )

    def test_lognormal_kernel_learner(self) -> None:
        _assert_params(
            LogNormalKernelLearner,
            expected={
                "max_lag",
                "min_lag",
                "dt",
                "loss",
                "smoothness_penalty",
                "seed",
                "validation_fraction",
                "learning_rate",
                "max_epochs",
                "huber_delta",
                "init_log_mu",
                "init_log_sigma",
            },
        )

    def test_erlang_kernel_learner(self) -> None:
        _assert_params(
            ErlangKernelLearner,
            expected={
                "max_lag",
                "min_lag",
                "dt",
                "loss",
                "smoothness_penalty",
                "seed",
                "validation_fraction",
                "learning_rate",
                "max_epochs",
                "huber_delta",
                "shape_k_candidates",
                "init_rate_beta",
            },
        )

    def test_fixed_delay_kernel_learner(self) -> None:
        _assert_params(
            FixedDelayKernelLearner,
            expected={
                "max_lag",
                "min_lag",
                "dt",
                "loss",
                "smoothness_penalty",
                "seed",
                "validation_fraction",
                "learning_rate",
                "max_epochs",
                "huber_delta",
            },
        )

    def test_uniform_kernel_learner(self) -> None:
        _assert_params(
            UniformKernelLearner,
            expected={
                "max_lag",
                "min_lag",
                "dt",
                "loss",
                "smoothness_penalty",
                "seed",
                "validation_fraction",
                "learning_rate",
                "max_epochs",
                "huber_delta",
            },
        )

    def test_kernel_feature_builder(self) -> None:
        _assert_params(
            KernelFeatureBuilder,
            expected={
                "kernels",
                "time_col",
                "numeric_cols",
                "category_cols",
                "weight_col",
                "age_tail_threshold",
            },
        )

    def test_feature_registry(self) -> None:
        _assert_params(FeatureRegistry, expected={"specs"})

    def test_feature_spec(self) -> None:
        _assert_params(
            FeatureSpec,
            expected={
                "name",
                "kernel_name",
                "source_col",
                "family",
                "metric",
                "category_level",
                "lag_steps",
                "kernel_summary",
            },
        )

    def test_transform_result(self) -> None:
        _assert_params(
            TransformResult,
            expected={"features", "report", "feature_registry"},
        )


# -
# Result dataclass field snapshot test — Result dataclass field snapshot test
# -


def _assert_fields(obj: type, *, expected: set[str]) -> None:
    fields = dataclasses.fields(obj)
    actual = {f.name for f in fields}
    missing = expected - actual
    extra = actual - expected
    assert not missing, (
        f"{obj.__name__} is missing expected fields: {sorted(missing)}"
    )
    assert not extra, (
        f"{obj.__name__} has unexpected fields: {sorted(extra)}"
    )


class TestResultDataclassFields:
    """Snapshot public dataclass fields for stable diagnostic result types."""

    def test_transform_report(self) -> None:
        _assert_fields(
            TransformReport,
            expected={
                "row_count",
                "output_row_count",
                "warmup_rows",
                "feature_names",
                "missing_rows_by_feature",
                "zero_denominator_rows_by_feature",
                "missing_fraction_by_feature",
                "missing_rows_by_kernel",
                "missing_fraction_by_kernel",
                "zero_denominator_rows_by_kernel",
                "warmup_unusable_summary",
                "collision_naming_summary",
            },
        )

    def test_feature_evidence_report(self) -> None:
        _assert_fields(
            FeatureEvidenceReport,
            expected={
                "feature_evidence",
                "feature_count",
                "kernel_count",
                "source_columns",
                "warning_summary",
                "evidence_summary_by_kernel",
                "evidence_summary_by_feature_family",
            },
        )

    def test_feature_evidence(self) -> None:
        _assert_fields(
            FeatureEvidence,
            expected={
                "feature_name",
                "source_col",
                "feature_family",
                "kernel_name",
                "kernel_family",
                "kernel_summary",
                "fit_result_id",
                "candidate_id",
                "baseline_summary",
                "identifiability_warnings",
                "bootstrap_summary",
                "interpretation",
                "evidence_completeness",
                "metadata",
            },
        )

    def test_kernel_fit_result(self) -> None:
        _assert_fields(
            KernelFitResult,
            expected={
                "kernel",
                "fit_diagnostics",
                "identifiability_report",
                "baseline_comparison",
                "kernel_shape_summary",
                "fit_data_coverage_summary",
                "fit_provenance",
            },
        )

    def test_fit_diagnostics(self) -> None:
        _assert_fields(
            FitDiagnostics,
            expected={
                "train_loss",
                "validation_loss",
                "input_variance",
                "target_variance",
                "kernel_weight_sum",
                "mean_lag",
                "p50_lag",
                "p90_lag",
                "tail_mass",
                "boundary_mass_fraction",
            },
        )

    def test_identifiability_report(self) -> None:
        _assert_fields(
            IdentifiabilityReport,
            expected={
                "warnings",
                "is_reliable",
                "warning_codes",
                "warning_severity_by_code",
            },
        )

    def test_baseline_comparison(self) -> None:
        _assert_fields(
            BaselineComparison,
            expected={
                "no_lag_validation_loss",
                "best_single_lag_validation_loss",
                "learned_validation_loss",
                "uniform_validation_loss",
                "exponential_validation_loss",
                "primary_ranking_metric",
                "summary_by_baseline",
            },
        )

    def test_kernel_shape_summary(self) -> None:
        _assert_fields(
            KernelShapeSummary,
            expected={
                "normalized_entropy",
                "max_weight",
                "min_weight",
                "concentration_hhi",
                "effective_lag_count",
            },
        )

    def test_fit_data_coverage_summary(self) -> None:
        _assert_fields(
            FitDataCoverageSummary,
            expected={
                "total_rows",
                "valid_windows",
                "train_windows",
                "validation_windows",
                "retained_row_fraction",
                "retained_window_fraction",
            },
        )


# -
# Generated feature naming snapshot test — Generated feature naming snapshot test
# -


def _make_simple_numeric_df(n: int = 10) -> pl.DataFrame:
    return pl.DataFrame({"t": range(n), "x": [1.0] * n})


def _make_numeric_and_cat_df(n: int = 10) -> pl.DataFrame:
    return pl.DataFrame({"t": range(n), "x": [1.0] * n, "cat": ["A", "B"] * (n // 2)})


def _make_two_kernel_builder() -> KernelFeatureBuilder:
    return KernelFeatureBuilder(
        kernels={
            "k1": FixedDelayKernel(delay_steps=1, max_lag_steps=3, dt=1.0),
            "k2": FixedDelayKernel(delay_steps=2, max_lag_steps=3, dt=1.0),
        },
        time_col="t",
        numeric_cols=["x"],
    )


class TestFeatureNaming:
    """Snapshot generated feature names from KernelFeatureBuilder."""

    def test_numeric_feature_names(self) -> None:
        df = _make_simple_numeric_df()
        builder = KernelFeatureBuilder(
            kernels={"k": FixedDelayKernel(delay_steps=1, max_lag_steps=3, dt=1.0)},
            time_col="t",
            numeric_cols=["x"],
        )
        result = builder.transform_result(df)
        names = list(result.features.columns)
        expected = {
            "t",
            "k_num_x_wmean",
            "k_num_x_wstd",
            "k_num_x_wsum",
            "k_age_mean",
            "k_age_p50",
            "k_age_p90",
            "k_age_tail_gt_threshold",
        }
        assert expected.issubset(set(names)), (
            f"Missing expected feature names. "
            f"Expected subset: {sorted(expected)}. Got: {sorted(names)}"
        )

    def test_categorical_feature_names(self) -> None:
        df = _make_numeric_and_cat_df()
        builder = KernelFeatureBuilder(
            kernels={"k": FixedDelayKernel(delay_steps=1, max_lag_steps=3, dt=1.0)},
            time_col="t",
            numeric_cols=["x"],
            category_cols=["cat"],
        )
        result = builder.transform_result(df)
        names = set(result.features.columns)
        assert "k_cat_cat_A_frac" in names, names
        assert "k_cat_cat_B_frac" in names, names
        assert "k_cat_cat_entropy" in names, names

    def test_numeric_and_categorical_feature_names(self) -> None:
        df = _make_numeric_and_cat_df()
        builder = KernelFeatureBuilder(
            kernels={"k": FixedDelayKernel(delay_steps=1, max_lag_steps=3, dt=1.0)},
            time_col="t",
            numeric_cols=["x"],
            category_cols=["cat"],
        )
        result = builder.transform_result(df)
        names = set(result.features.columns)
        expected_numeric = {"k_num_x_wmean", "k_num_x_wstd", "k_num_x_wsum"}
        expected_cat = {"k_cat_cat_A_frac", "k_cat_cat_B_frac", "k_cat_cat_entropy"}
        expected_age = {
            "k_age_mean",
            "k_age_p50",
            "k_age_p90",
            "k_age_tail_gt_threshold",
        }
        assert expected_numeric.issubset(names), (
            f"Missing numeric features. Got: {sorted(names)}"
        )
        assert expected_cat.issubset(names), (
            f"Missing categorical features. Got: {sorted(names)}"
        )
        assert expected_age.issubset(names), (
            f"Missing age features. Got: {sorted(names)}"
        )

    def test_two_kernel_feature_names(self) -> None:
        df = _make_simple_numeric_df()
        builder = _make_two_kernel_builder()
        result = builder.transform_result(df)
        names = set(result.features.columns)
        for prefix in ("k1", "k2"):
            assert f"{prefix}_num_x_wmean" in names, names
            assert f"{prefix}_num_x_wstd" in names, names
            assert f"{prefix}_num_x_wsum" in names, names
            assert f"{prefix}_age_mean" in names, names
            assert f"{prefix}_age_p50" in names, names
            assert f"{prefix}_age_p90" in names, names
            assert f"{prefix}_age_tail_gt_threshold" in names, names

    def test_age_feature_names(self) -> None:
        df = _make_simple_numeric_df()
        builder = KernelFeatureBuilder(
            kernels={"k": FixedDelayKernel(delay_steps=1, max_lag_steps=3, dt=1.0)},
            time_col="t",
            numeric_cols=["x"],
        )
        result = builder.transform_result(df)
        names = set(result.features.columns)
        expected_age = {
            "k_age_mean",
            "k_age_p50",
            "k_age_p90",
            "k_age_tail_gt_threshold",
        }
        assert expected_age.issubset(names), (
            f"Missing age features. Got: {sorted(names)}"
        )


# -
# Version consistency test — Version consistency test
# -


def _read_package_version() -> str:
    """Read version from pyproject.toml."""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    text = pyproject.read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match is not None, "Could not find version in pyproject.toml"
    return match.group(1)


def test_changelog_version_matches_package() -> None:
    version = _read_package_version()
    changelog = (
        Path(__file__).resolve().parent.parent / "CHANGELOG.md"
    ).read_text()
    assert version in changelog, (
        f"Version {version} not found in CHANGELOG.md"
    )


def test_release_notes_version_matches_package() -> None:
    version = _read_package_version()
    release_notes = (
        Path(__file__).resolve().parent.parent / "docs/RELEASE_NOTES.md"
    ).read_text()
    assert version in release_notes, (
        f"Version {version} not found in docs/RELEASE_NOTES.md"
    )


# -
# Deprecation consistency test — Deprecation consistency test
# -


def test_no_false_deprecation_text_in_docs() -> None:
    """Scan user-facing docs for 'deprecated' or 'Deprecated'.

    The preferred V1 state is no deprecations. Lines that explicitly
    say 'not deprecated' are affirmations and are excluded.
    """
    root = Path(__file__).resolve().parent.parent / "docs"
    excluded_dirs = {"plans"}
    excluded_files = {"deprecation-policy.md"}
    hits: list[str] = []
    for path in sorted(root.rglob("*.md")):
        rel = path.relative_to(root.parent)
        if any(part in excluded_dirs for part in path.relative_to(root).parts):
            continue
        if path.name in excluded_files:
            continue
        text = path.read_text()
        for i, line in enumerate(text.splitlines(), 1):
            if re.search(r"(?i)\bdeprecated\b", line):
                if "not deprecated" in line.lower():
                    continue
                hits.append(f"{rel}:{i}: {line.strip()}")
    assert not hits, (
        f"Found deprecation text in user-facing docs ({len(hits)} hits):\n"
        + "\n".join(hits)
    )
