# nRTD Benchmark Layers And Usage Boundaries

This page defines how external nRTD-derived benchmark material is used in this
repository without expanding product scope.

Reference: benchmark and simulation harness specification.

The package still follows a narrow loop:

1. learn the lag kernel
2. validate the kernel
3. generate kernel-based features
4. stop there

## Three Validation Layers

Use benchmark inputs in this order of authority.

### Layer 1: Repo-owned synthetic oracle tests

Use this layer for required correctness checks and release gates.

- source of truth: synthetic fixtures with exact known kernels
- expected use: baseline comparisons, identifiability warnings, and
  feature-generation contract checks
- decision rule: if Layer 1 fails, treat as a package regression

readiness and release-gate plan note for this repository state:

- strict tank/flotation recovery-threshold checks currently tracked in
  `tests/test_integration_readiness.py` are explicit non-blocking readiness
  guards for known `SimplexKernelLearner` capacity limits in `v0.1`
- these guards keep target thresholds visible and deterministic while avoiding
  false release blocking for the current narrow wedge
- they are placeholders for future, more expressive learner work and should not
  be interpreted as simulator or final-model claims

### Layer 2: Small nRTD analytical and literature fixtures

Use this layer to sanity-check known RTD-style analytical behavior.

- expected fixtures include `HSA_000` analytical examples
- expected use: optional benchmark checks that compare package behavior against
  known analytical/literature references
- decision rule: use as supplementary evidence; do not replace Layer 1

`HSA_000` analytical fixtures are useful for known RTD-style benchmark checks,
especially for verifying causal non-negative kernel behavior on controlled
examples.

For integration-readiness tests:

- mark these checks with `pytest.mark.external_data`
- skip clearly when benchmark fixture files are not present
- exclude these checks by default in CI with `pytest -m "not external_data"`

### Layer 3: Optional nRTD experimental learned RTD references

Use this layer only as contextual comparison.

- expected use: optional reporting references for learned RTD trends
- non-authoritative: these are references, not ground truth for this package
- decision rule: never override Layer 1 conclusions with Layer 3 outcomes

## Terminology Boundaries

- `kernel` is the package-generic term for learned lag/response weights.
- `RTD kernel` may be used only when physically justified by the benchmark
  context.
- `response kernel` may be used when discussing external literature naming.

These terms are related but not interchangeable. Public package behavior stays
domain-neutral and uses `kernel` by default.

## Scope Boundaries

External benchmarks supplement, but do not replace, repo-owned synthetic
positive examples.

Use fixture selection in this order for integration work:

1. required CI checks: harness-owned synthetic fixtures (`tank`, `plug_flow`,
   `flotation_bank`, `toy_full_plant`)
2. optional checks: external nRTD analytical fixtures when files are present

This package does not adopt nRTD's CNN API or multi-compartment learning API.
The benchmark layer is evidence support for kernel-learning validation, not a
request to mirror external modelling families.

The benchmark layer does not imply support for:

- forecasting frameworks
- plant simulation or digital twin behavior
- final predictive modelling workflows

## Citation And License Notes

When distributing extracted fixtures:

- cite the nRTD paper and the nRTD project archive in benchmark docs
- preserve the Zenodo dataset MIT license notice in extracted fixture metadata
- preserve citation/NOTICE files alongside fixtures

Repository fixture manifests and notices should point to:

- nRTD paper citation record (as documented in project benchmark metadata)
- nRTD Zenodo archive record used for extracted fixtures
