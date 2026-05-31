# rtdfeatures 1.0.0

## Summary

`1.0.0` is the final stable release of `rtdfeatures`, a library that learns constrained causal kernels from regular-grid process time series and converts them into auditable Polars feature tables with diagnostics.

## Who this is for

- Process engineers and metallurgists building lag-aware process features.
- Industrial data scientists and machine-learning engineers preparing feature tables for downstream modelling.
- Anyone working with regularly-gridded process data who needs interpretable, constrained lag features without committing to a final predictive model.

## What is included

- Constrained empirical kernel learning via `SimplexKernelLearner`.
- Parametric kernel learning via `GammaKernelLearner` and `ExponentialKernelLearner`.
- Multi-pair shared kernel learning via `SharedSimplexKernelLearner`.
- Fixed and parametric kernel families under `rtdfeatures.kernels`.
- `KernelFeatureBuilder` for deterministic Polars feature generation with transform diagnostics and feature evidence.
- `TransformResult` for auditable feature table, report, and registry output.
- Baseline comparisons (`no_lag`, `best_single_lag`, `uniform`, `exponential`).
- Candidate comparison framework with information criteria and cross-validation.
- Bootstrap uncertainty estimation.
- Out-of-fold (OOF) feature generation with leakage-aware fold splitting.
- Feature evidence: structured metadata per generated column.
- Diagnostics: `FitDiagnostics`, `IdentifiabilityReport`, `BaselineComparison`, `TransformReport`, `KernelShapeSummary`, `FitDataCoverageSummary`.
- Scikit-learn integration via optional `sklearn` extra.
- Repository documentation, examples, CI, and build/publish scaffolding.

## Installation

```bash
pip install rtdfeatures
```

Optional extras are documented in `docs/install.md`.

## Stable public API

The following names are exported from `rtdfeatures.__init__` and form the stable V1 API. Removals or renames require a major version bump.

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

Specialised kernels and learners that are not root-exported remain available from
`rtdfeatures.kernels` and `rtdfeatures.learners`. They are usable, but the
root-level V1 stability promise applies only to the stable public API list above.

## Advanced and provisional APIs

The following subpackages and modules are usable but not covered by the major-version stability guarantee. They may change or be removed in minor releases with migration notes.

- `rtdfeatures.bootstrap` — kernel bootstrap uncertainty
- `rtdfeatures.candidates` — kernel candidate selection and comparison
- `rtdfeatures.oof` — out-of-fold feature generation
- `rtdfeatures.reporting` — diagnostic report helpers
- `rtdfeatures.integrations.sklearn` — scikit-learn adapter (optional extra)

## Known limitations

- Input data must use a regular time grid. Irregular or missing timestamps raise by default.
- Operations are batch-oriented; online/streaming feature generation is out of scope.
- Final predictive modelling is out of scope.
- Plant-wide topology/genealogy modelling is out of scope.
- Learned kernels are constrained lag relationships; they do not prove causality.
- RTD interpretation requires independent process/tracer/topology/SME evidence.
- Warmup rows before the maximum lag is satisfied produce null features.

## Validation summary

The repository includes automated tests for package metadata, root namespace snapshots, API contracts (constructor signatures, dataclass fields, generated feature names), schema stability, feature naming, feature registry behaviour, parametric and empirical kernel learners, OOF generation, feature evidence, and benchmark extraction. All tests pass on the `main` branch with `pytest -m "not external_data"`.

## Migration notes

This is the first stable V1 release. No migration from a prior stable version is required.

## Citation

Use `CITATION.cff` for citation metadata.
