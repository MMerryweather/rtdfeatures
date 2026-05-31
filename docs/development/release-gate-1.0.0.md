# rtdfeatures 1.0.0 Release Gate

## Commit

TBD until merge

## Date

Sun May 31 18:01:19 ChST 2026

## Environment

- Python 3.12.3
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

## Publish Workflow Readiness

- `publish.yml` triggers on GitHub Release publication.
- Build job validates release tag matches `pyproject.toml` version.
- Wheel and sdist are built once and uploaded as workflow artefacts.
- Wheel install is validated in a clean virtual environment.
- Sdist install is validated in a clean virtual environment.
- Final publish job uses GitHub environment `pypi`.
- Final publish job uses PyPI trusted publishing via OIDC with `id-token: write`.
- No PyPI API token or password secret is required.

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
