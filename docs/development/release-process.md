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
2. **Build and integrity check** — The workflow builds wheel and sdist once and runs `twine check`.
3. **Artifact validation** — The workflow installs the built wheel and sdist in separate clean virtual environments, validates import path/version from the installed distribution, and runs a minimal import smoke check from outside the checkout.
4. **Trusted publish** — If both validation jobs pass, the workflow publishes the validated artifacts to PyPI using trusted publishing.

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
pip install -e ".[dev]"
pytest -m "not external_data" -v
ruff check src/ tests/
mypy src/ tests/
python -m build
twine check dist/*
```

## Version history

See [CHANGELOG.md](../../CHANGELOG.md) for the full version history.
