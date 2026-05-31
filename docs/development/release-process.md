# Release Process

Internal changes should preserve the golden path: fit or define kernel → generate `TransformResult` → inspect features/report/registry. See `architecture-principles.md`.


## Current process

1. Changes are merged to `main` via pull request.
2. Version is bumped in `pyproject.toml` following semver.
3. CHANGELOG.md is updated with the new version's additions, changes, and fixes.
4. A GitHub Release is published for tag `v<version>`.
5. The `publish.yml` workflow builds artifacts once, validates wheel and sdist installs in clean environments, and publishes to PyPI through trusted publishing.

## Pre-release checks

- `pytest -m "not external_data"` passes
- `ruff check src/ tests/` passes
- `mypy src/ tests/` passes
- Security workflows configured for this repository plan are green on `main`
- Dependency audit checks pass on release-bound pull requests
- Release notes are reviewed

## API stability and versioning

- **Stable API** changes (removals, renames) require a major version bump.
  Additive changes (new classes, methods, fields) are allowed in minor releases.
- **Provisional APIs** may change or be removed in minor releases with migration
  notes.
- **Internal APIs** have no compatibility promise.

See [API stability policy](../api/stability.md) for the full tier definitions.

## Release classifiers

- **Release candidates** (e.g. `1.0.0rc1`) must use the
  `Development Status :: 4 - Beta` classifier in `pyproject.toml`.
- **Final stable releases** (e.g. `1.0.0`) must use the
  `Development Status :: 5 - Production/Stable` classifier.
- The `Development Status :: 3 - Alpha` classifier is not used in this release
  path.

## Publishing workflow

1. **Trigger** — Publishing a GitHub Release starts `.github/workflows/publish.yml`.
2. **Release tag/version guard** — The `build` job verifies the GitHub Release tag `v<version>` matches `version` in `pyproject.toml` before running the build.
3. **Build and integrity check** — The workflow builds wheel and sdist once and runs `twine check`.
4. **Artifact validation** — The workflow installs the built wheel and sdist in separate clean virtual environments, validates import path/version from the installed distribution, and runs a minimal import smoke check from outside the checkout.
5. **Environment-gated trusted publish** — If both validation jobs pass, the final `publish` job enters GitHub environment `pypi` for approval, then publishes the validated artifacts to PyPI using OIDC trusted publishing. No PyPI token/password secret is required in normal operation.

## Security Governance

- `SECURITY.md` defines the vulnerability reporting channel and response expectations.
- Security scanning workflows are enabled according to repository-plan capabilities.
- Dependency audit runs in CI via `pip-audit` in `.github/workflows/ci.yml`.

## TestPyPI

A TestPyPI dry run is not currently part of the automated release process. Releases are validated through clean wheel and sdist install jobs before trusted publishing to PyPI.

### Local pre-release checks

Use these commands locally before creating a release:

```bash
python -m pip install --upgrade pip
pip install -e ".[dev,examples,sklearn]"
pytest -m "not external_data" -v
ruff check src/ tests/
mypy src/ tests/
python -m build
twine check dist/*
```

## First PyPI release checklist

1. Confirm PyPI pending trusted publisher is configured:
   - project: `rtdfeatures`
   - owner: `MMerryweather`
   - repository: `rtdfeatures`
   - workflow: `publish.yml`
   - environment: `pypi`
2. Confirm GitHub environment `pypi` exists and has the intended reviewer policy.
3. Confirm `pyproject.toml` version matches the intended release tag.
4. Run the full local release gate.
5. Merge release-prep PR to `main`.
6. Create annotated tag `v<version>`.
7. Publish a GitHub Release for that tag.
8. Approve the `pypi` environment deployment.
9. Verify install from PyPI in a clean virtual environment.

## Version history

See [CHANGELOG.md](../../CHANGELOG.md) for the full version history.
