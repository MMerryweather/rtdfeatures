"""Architecture and source-hygiene invariants for internal simplification milestone"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import polars as pl

import rtdfeatures
from rtdfeatures import (
    FeatureRegistry,
    FixedDelayKernel,
    FixedDelayKernelLearner,
    Kernel,
    KernelFeatureBuilder,
    TransformResult,
    UniformKernelLearner,
)

_RUNTIME_SOURCE_FILES = tuple(sorted(Path("src").rglob("*.py")))


def _tiny_result() -> TransformResult:
    df = pl.DataFrame(
        {
            "t": [1.0, 2.0, 3.0, 4.0],
            "x": [10.0, 20.0, 30.0, 40.0],
            "cat": ["A", "B", "A", "B"],
        }
    )
    builder = KernelFeatureBuilder(
        kernels={"k": FixedDelayKernel(delay_steps=1, max_lag_steps=2, dt=1.0)},
        time_col="t",
        numeric_cols=["x"],
        category_cols=["cat"],
    )
    return builder.transform_result(df)


def test_sklearn_adapter_is_not_root_exported() -> None:
    assert "KernelFeatureTransformer" not in dir(rtdfeatures)


def test_all_root_kernel_exports_are_kernel_subclasses() -> None:
    for name in rtdfeatures.__all__:
        if not name.endswith("Kernel"):
            continue
        exported = getattr(rtdfeatures, name)
        assert isinstance(exported, type)
        assert issubclass(exported, Kernel)


def test_transform_result_contains_features_report_and_registry() -> None:
    result = _tiny_result()
    assert isinstance(result, TransformResult)
    assert isinstance(result.features, pl.DataFrame)
    assert result.report is not None
    assert isinstance(result.feature_registry, FeatureRegistry)


def test_feature_registry_specs_match_generated_feature_names() -> None:
    result = _tiny_result()
    generated_names = [name for name in result.features.columns if name != "t"]
    registry_names = [spec.name for spec in result.feature_registry.specs]
    assert registry_names == generated_names


def test_runtime_source_contains_no_work_package_markers() -> None:
    pattern = re.compile(r"work[- ]package", re.IGNORECASE)
    for path in _RUNTIME_SOURCE_FILES:
        text = path.read_text(encoding="utf-8")
        assert pattern.search(text) is None, f"work-package marker found in {path}"


def test_runtime_source_contains_no_old_version_markers() -> None:
    pattern = re.compile(r"for v0\.")
    for path in _RUNTIME_SOURCE_FILES:
        text = path.read_text(encoding="utf-8")
        assert pattern.search(text) is None, f"Old version marker found in {path}"


def test_fixed_uniform_learners_use_shared_fit_assembly_path() -> None:
    fixed_source = inspect.getsource(FixedDelayKernelLearner.fit)
    uniform_source = inspect.getsource(UniformKernelLearner.fit)
    assert "evaluate_baselines(" in fixed_source
    assert "assemble_kernel_fit_result(" in fixed_source
    assert "evaluate_weight_vector_losses(" in fixed_source
    assert "evaluate_baselines(" in uniform_source
    assert "assemble_kernel_fit_result(" in uniform_source
    assert "evaluate_weight_vector_losses(" in uniform_source
