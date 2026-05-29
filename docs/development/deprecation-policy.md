# Deprecation Policy

## Versioning

`rtdfeatures` follows semantic versioning (`MAJOR.MINOR.PATCH`).

- **Major** (`v1.0`, `v2.0`): breaking public API changes
- **Minor** (`v1.1`, `v1.2`): additive, non-breaking changes
- **Patch** (`v1.0.1`, `v1.0.2`): bug fixes and internal improvements

## What counts as public API

The public API is everything exported from `rtdfeatures.__init__` (i.e. listed in
`__all__`). Any name there is subject to the stability policy.

Code in private modules (prefixed with `_` or not listed in `__all__`) is internal
and may change without notice.

## What is stable

- All classes, functions, and constants in `rtdfeatures.__all__`
- `polars.DataFrame` input/output contract
- Constrained kernel semantics (causal, non-negative, sum-to-one, bounded lag)
- Diagnostic result object fields (additive extensions only)

## What is not stable (provisional)

- Objects marked as provisional in their docstrings or design docs
- Interoperability primitives (tigramite integration, etc.)
- Internal module structure (imports from `rtdfeatures.private_module`)
- Benchmark-layer API

## Deprecation timeline

- Breaking changes are announced one minor version in advance.
- Deprecated names emit a `FutureWarning` for at least one minor version before removal.
- The deprecation notice includes the replacement path.
- Deprecated names remain importable from their submodule path until the next major version.

## How to propose API changes

1. Open an issue describing the proposed change, motivation, and migration path.
2. If the change is breaking, mark the issue with the `breaking` label.
3. Discuss in the next community / maintainer review cycle.
4. Update the relevant docs and deprecation notices before the change lands in a release.
5. Breaking changes must be documented in the changelog with a clear migration guide.
