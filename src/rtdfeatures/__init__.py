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
    ExponentialKernel,
    FixedDelayKernel,
    GammaKernel,
    Kernel,
    UniformKernel,
)
from rtdfeatures.learners import (
    ExponentialKernelLearner,
    GammaKernelLearner,
    SimplexKernelLearner,
)

__all__ = [
    "Kernel",
    "FixedDelayKernel",
    "UniformKernel",
    "GammaKernel",
    "ExponentialKernel",
    "DelayedExponentialKernel",
    "SimplexKernelLearner",
    "GammaKernelLearner",
    "ExponentialKernelLearner",
    "KernelFeatureBuilder",
    "FeatureRegistry",
    "FeatureSpec",
    "TransformResult",
]
