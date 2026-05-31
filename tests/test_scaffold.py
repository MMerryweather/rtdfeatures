"""Scaffold-level tests for public contract exports."""

import importlib

import rtdfeatures


def test_public_api_exports_exist() -> None:
    root_expected = {
        "DelayedExponentialKernel",
        "ExponentialKernel",
        "ExponentialKernelLearner",
        "FeatureRegistry",
        "FeatureSpec",
        "FixedDelayKernel",
        "GammaKernel",
        "GammaKernelLearner",
        "Kernel",
        "KernelFeatureBuilder",
        "SimplexKernelLearner",
        "TransformResult",
        "UniformKernel",
    }
    submodule_expected_diag = {
        "BaselineComparison",
        "FitDiagnostics",
        "IdentifiabilityReport",
        "KernelFitResult",
        "TransformReport",
    }
    import rtdfeatures.diagnostics as _diag
    assert submodule_expected_diag.issubset(set(dir(_diag)))
    import rtdfeatures.kernels as _kern
    assert {"LearnedKernel", "Kernel"}.issubset(set(dir(_kern)))
    # Verify root namespace is not a junk drawer
    root_public = set(rtdfeatures.__all__)
    assert root_public == root_expected, (
        f"Root __all__ mismatch: extra={root_public - root_expected}, "
        f"missing={root_expected - root_public}"
    )

    # Specialist V1 objects stay importable from submodules without
    # becoming root-level semver commitments.
    from rtdfeatures.kernels import ErlangKernel, LogNormalKernel
    from rtdfeatures.learners import (
        DelayedExponentialKernelLearner,
        ErlangKernelLearner,
        FixedDelayKernelLearner,
        LogNormalKernelLearner,
        UniformKernelLearner,
    )

    assert ErlangKernel is not None
    assert LogNormalKernel is not None
    assert DelayedExponentialKernelLearner is not None
    assert ErlangKernelLearner is not None
    assert FixedDelayKernelLearner is not None
    assert LogNormalKernelLearner is not None
    assert UniformKernelLearner is not None


def test_architecture_modules_import() -> None:
    modules = [
        "rtdfeatures.kernels",
        "rtdfeatures.learners",
        "rtdfeatures.features",
        "rtdfeatures.diagnostics",
        "rtdfeatures.baselines",
        "rtdfeatures.utils",
        "rtdfeatures.synthetic",
    ]
    for module_name in modules:
        module = importlib.import_module(module_name)
        assert module is not None
