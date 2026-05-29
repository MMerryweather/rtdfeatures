"""Public package exports for rtdfeatures.

Stable V1.0 API — core workflow objects only.
Advanced/non-core objects live in their respective submodules and are
importable via ``from rtdfeatures.<submodule> import <name>``.
"""

try:
    from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
    from importlib.metadata import version as _version

    __version__ = _version("rtdfeatures")
except _PackageNotFoundError:
    __version__ = "0.0.0"

from rtdfeatures.features import KernelFeatureBuilder
from rtdfeatures.features.registry import FeatureRegistry, FeatureSpec, TransformResult
from rtdfeatures.kernels import (
    DelayedExponentialKernel,
    ErlangKernel,
    ExponentialKernel,
    FixedDelayKernel,
    GammaKernel,
    Kernel,
    LogNormalKernel,
    UniformKernel,
)
from rtdfeatures.learners import (
    DelayedExponentialKernelLearner,
    ErlangKernelLearner,
    ExponentialKernelLearner,
    FixedDelayKernelLearner,
    GammaKernelLearner,
    LogNormalKernelLearner,
    SimplexKernelLearner,
    UniformKernelLearner,
)

__all__ = [
    "DelayedExponentialKernel",
    "DelayedExponentialKernelLearner",
    "ErlangKernelLearner",
    "ErlangKernel",
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
]
