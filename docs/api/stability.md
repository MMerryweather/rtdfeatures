# API Stability Policy

Internal changes should preserve the golden path: fit or define kernel → generate `TransformResult` → inspect features/report/registry. See `../development/architecture-principles.md`.

## Versioning

`rtdfeatures` follows semantic versioning (`MAJOR.MINOR.PATCH`).

- **Major** (`v1.0`, `v2.0`): breaking public API changes
- **Minor** (`v1.1`, `v1.2`): additive, non-breaking changes
- **Patch** (`v1.0.1`, `v1.0.2`): bug fixes and internal improvements

## Deprecation policy

See the [dedicated deprecation policy](../development/deprecation-policy.md) for the full timeline and process.

## How to propose API changes

1. Open an issue describing the proposed change, motivation, and migration path.
2. If the change is breaking, mark the issue with the `breaking` label.
3. Discuss in the next community / maintainer review cycle.
4. Update the relevant docs and deprecation notices before the change lands in a release.
5. Breaking changes must be documented in the changelog with a clear migration guide.

## Stable V1 API

Removals or renames in the stable API require a **major version bump**. Additive changes
(new classes, methods, fields) are allowed in minor releases.

- Names exported from `rtdfeatures.__all__`
- Constructor signatures of root-exported classes
- `KernelFeatureBuilder.transform`
- `KernelFeatureBuilder.transform_result`
- `KernelFeatureBuilder.augment_cols`
- `KernelFeatureBuilder.diagnose_feature_evidence`
- `TransformResult`, `FeatureRegistry`, `FeatureSpec`
- Documented diagnostic/result dataclasses used by the root workflow
- Default generated feature naming convention

**Stable root imports:**

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

The stable root import list is intentionally smaller than the full package
surface. More specialised kernels and learners remain available from their
submodules but are not part of the root-level V1 stability promise.

## Provisional V1 API

Provisional APIs may change or be removed in **minor releases** with migration notes.
They are usable but not covered by the major-version stability guarantee.

- `rtdfeatures.bootstrap` subpackage
- `rtdfeatures.candidates` subpackage
- `rtdfeatures.oof` subpackage
- Reporting helpers (`rtdfeatures.reporting`)
- Optional integrations, including `rtdfeatures.integrations.sklearn`

## Internal API

Internal APIs have **no compatibility promise**. They may change at any time without
notice.

- Private helpers (modules prefixed with `_` or not in `__all__`)
- Private dataclasses
- Test helpers
- Generated gallery implementation details
- Scripts not documented as stable user API

## Current status

Current version: `1.0.0`. The core learning, feature generation, and diagnostics API
is stable. Interoperability primitives are provisional.

## Versioning commitments

- Root namespace removals require a major version bump.
- Default feature-name changes require a major version bump unless the new names are opt-in.
- Stable API changes follow semantic versioning (breaking = major, additive = minor).
- Provisional APIs may change in minor releases with migration notes.
- Internal APIs have no compatibility promise.
