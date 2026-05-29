"""Scaffold-level tests for public contract exports."""

import importlib

import rtdfeatures


def test_public_api_exports_exist() -> None:
    root_expected = {
        "DelayedExponentialKernel",
        "DelayedExponentialKernelLearner",
        "ErlangKernel",
        "ErlangKernelLearner",
        "ExponentialKernel",
        "ExponentialKernelLearner",
        "FeatureRegistry",
        "FeatureSpec",
        "FixedDelayKernel",
        "FixedDelayKernelLearner",
        "GammaKernel",
        "GammaKernelLearner",
        "Kernel",
        "KernelFeatureBuilder",
        "LogNormalKernel",
        "LogNormalKernelLearner",
        "SimplexKernelLearner",
        "TransformResult",
        "UniformKernel",
        "UniformKernelLearner",
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
