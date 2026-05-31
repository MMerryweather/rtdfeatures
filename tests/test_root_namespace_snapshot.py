"""Root namespace snapshot test.

Ensures that ``dir(rtdfeatures)`` (excluding dunder/private names and
submodule entries that appear as side effects of ``from X import Y``)
contains exactly the expected set of stable core objects.

If the namespace grows or shrinks unexpectedly, this test fails until
the expected set is explicitly updated — preventing accidental public
API expansion or removal.
"""

from __future__ import annotations

import rtdfeatures

# Submodule names that may appear in dir() due to import side effects
# from other tests or internal transitive imports. These are not part
# of the stable public API but are unavoidable because Python adds
# them to the module namespace when any code in the process executes
# ``from rtdfeatures.<submodule> import ...``.
_SUBMODULE_ENTRIES = frozenset({
    "bootstrap",
    "baselines",
    "benchmarks",
    "candidates",
    "diagnostics",
    "features",
    "integrations",
    "kernels",
    "learners",
    "oof",
    "reporting",
    "synthetic",
    "utils",
})

# The only public names (non-dunder, non-submodule) that should appear
# in dir(rtdfeatures). If you add a new stable export, update this set.
EXPECTED_STABLE_NAMES = frozenset({
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
})


def test_root_namespace_snapshot() -> None:
    all_names = set(dir(rtdfeatures))
    # Drop dunder/private names
    public = {n for n in all_names if not n.startswith("_")}
    # Drop submodule side-effect entries
    public -= _SUBMODULE_ENTRIES

    extra = public - EXPECTED_STABLE_NAMES
    missing = EXPECTED_STABLE_NAMES - public

    assert not extra, (
        f"Unexpected public names in root namespace: "
        f"{sorted(extra)}.\n"
        f"Either prune them from __init__.py or add them to "
        f"EXPECTED_STABLE_NAMES in this test."
    )
    assert not missing, (
        f"Expected stable names missing from root namespace: "
        f"{sorted(missing)}.\n"
        f"Ensure they are exported from __init__.py and listed in "
        f"EXPECTED_STABLE_NAMES."
    )


def test_package_import_is_fast() -> None:
    """Cold import of rtdfeatures must complete in under 2 seconds."""
    import subprocess
    import sys

    code = "import time; t0=time.monotonic(); import rtdfeatures; print(time.monotonic()-t0)"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    elapsed = float(result.stdout.strip())
    assert elapsed < 2.0, f"Package import took {elapsed:.2f}s (>2.0s threshold)"
