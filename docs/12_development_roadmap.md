# Development Roadmap

This is the concise roadmap.

## Versioned Roadmap

### `v0.1`

- kernel object model
- single-pair constrained learner
- baseline comparison (`no_lag`, `best_single_lag`)
- kernel-weighted feature generation
- diagnostics and synthetic validation

### `v0.2`

- shared-kernel fit contracts
- pairwise shared-kernel workflow additions

### `v0.3`

- synthetic harness and diagnostics expansion

### `v0.5`

- constrained parametric learners
- parametric conversion and fit provenance

### `v0.6`

- direct parametric kernel objects and family expansion

### `v0.7`

- candidate and comparison objects

### `v0.8`

- bootstrap uncertainty contracts and summaries

### `v0.9`

- feature evidence contract and evidence reporting only
- interpretation/evidence label taxonomy contract
- no public OOF execution helper in this version

### `v0.95`

- out-of-fold safeguards and execution path hardening
- public OOF split contracts and `fit_transform_oof(...)`

### `v1.0`

- optional Tigramite output adapter boundary
- no Tigramite hard dependency in core package

### `v1.5`

- interoperability primitives (`KernelSpec`, `FeatureSpec`, `KernelRegistry`, `KernelFeaturePlan`)
- kernel composition API (`compose_kernels`)
- observation and role metadata (`ObservationSemantics`, `ColumnRole`)
- reproducibility artifact (`FeaturePlanManifest`)
- external provenance fields in feature evidence

### `v1.6`

- feature-name policy
- feature-plan validation report
- feature-plan dry run
- evidence join helpers

### `v1.7`

- uncertainty propagation through composed kernels
- kernel similarity and drift diagnostics

## Post-`v1.0` Boundary Rule

Post-`v1.0` scope does not add graph traversal, process-compiler logic, or
final-model workflows to `rtdfeatures`.

Post-`v1.0` work adds kernel/spec/plan/evidence primitives so external compilers
can plan and pass validated work into `rtdfeatures`.

See:
[13_post_v1_rtdfeatures_roadmap.md](13_post_v1_rtdfeatures_roadmap.md)
