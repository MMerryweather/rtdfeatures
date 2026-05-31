# rtdfeatures 1.0.0 Release Gate

## Commit

TBD until merge

## Date

Sun May 31 17:21:11 ChST 2026

## Environment

- Python 3.13.5
- rtdfeatures 1.0.0
- Install mode: editable install with dev/examples/sklearn extras

## Commands Run

| Command | Result |
|---|---|
| `ruff check .` | PASS |
| `mypy src tests` | PASS |
| `pytest -m "not external_data" -v` | PASS — 823 passed, 3 deselected, 2 xfailed |
| `python -m build` | PASS |
| `twine check dist/*` | PASS |

## Targeted checks

| Command | Result |
|---|---|
| `pytest tests/test_semver_contract.py -v` | PASS — 33 passed |
| `pytest tests/test_public_api_docs_contract.py -v` | PASS — 3 passed |
| `pytest tests/test_sklearn_adapter.py -v` | PASS — 26 passed |
| `python examples/08_sklearn_adapter.py` | PASS |

## Root API

Stable V1 root exports:

- `Kernel`
- `FixedDelayKernel`
- `UniformKernel`
- `GammaKernel`
- `ExponentialKernel`
- `DelayedExponentialKernel`
- `SimplexKernelLearner`
- `GammaKernelLearner`
- `ExponentialKernelLearner`
- `KernelFeatureBuilder`
- `FeatureRegistry`
- `FeatureSpec`
- `TransformResult`

## Result

Release gate passed. No failed checks are waived.

## Deferred Items

- Learner fit-pipeline simplification remains deferred to V1.1.
- Specialised kernels and learners remain available from submodules but are not root-stable V1 exports.

## TestPyPI Validation

Not part of the automated release gate. Release artefacts are validated through clean wheel and sdist install checks before trusted publishing.
