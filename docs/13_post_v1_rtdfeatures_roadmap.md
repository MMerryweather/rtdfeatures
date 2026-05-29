# Post-V1.0 `rtdfeatures` Roadmap

## Purpose

Post-`v1.0` work keeps `rtdfeatures` as a kernel-feature engine.

It adds interoperability primitives so external planners can compile process
hypotheses into executable feature plans, while `rtdfeatures` validates and
executes kernel-based feature generation with evidence.

## Stable Boundary

`rtdfeatures` owns:

- constrained kernel objects and learners
- kernel comparison and bootstrap uncertainty contracts
- kernel-weighted feature generation
- feature evidence and reproducibility artifacts
- kernel/spec/plan contracts for execution
- kernel algebra independent of topology ownership

`rtdfeatures` does not own:

- graph traversal
- flowsheet graph schema ownership
- regime detection
- motif mining
- TDA workflows
- SINDy workflows
- conformal prediction
- final predictive modelling
- MLOps orchestration
- historian connectors
- real-time serving
- process-specific API templates

## Dependency Direction

```text
rtdfeatures <- processfeaturecompiler <- external MLOps
```

`processfeaturecompiler` may depend on `rtdfeatures`.
`rtdfeatures` must not depend on `processfeaturecompiler`.

## Interoperability Primitives

### `KernelSpec`

A serialisable kernel description contract.
It declares a kernel without running learner fitting.

### `FeatureSpec`

A serialisable feature request contract.
It defines source column, kernel reference, feature family, aggregations,
optional weight column, and metadata.

### `KernelRegistry`

A named registry for validated kernels/specs used by feature plans.
It supports stable lookup and validation before execution.

### `KernelFeaturePlan`

A serialisable batch of feature requests.
It orchestrates existing feature-generation semantics and must not change
`transform()` return behavior.

### `compose_kernels`

Kernel algebra for sequential-memory composition via convolution.
This is in scope as kernel math; graph path discovery is out of scope.

### `ObservationSemantics`

Optional machine-readable timestamp/sampling semantics for validation context.
Examples: `instant`, `interval_start`, `interval_end`, `interval_midpoint`,
`online`, `grab`, `composite`, `event`, `derived`.

### `ColumnRole`

Optional machine-readable column roles for leakage and compatibility checks.
Examples: `input_tag`, `target`, `lab_label`, `event`, `setpoint`,
`controller_output`, `downstream_measurement`, `future_known`, `forbidden`.

### `FeaturePlanManifest`

A reproducibility artifact for plan execution metadata.
Typical fields include plan hash, schema hashes, package versions, warnings,
and output schema summary.

### External Provenance Fields

Feature evidence may include optional external provenance fields such as:

- `source_node`
- `target_node`
- `graph_edge`
- `graph_path`
- `edge_kernel_name`
- `path_kernel_name`
- `feature_plan_id`
- `external_hypothesis_id`
- `interpretation_class`
- `observation_semantics`

All provenance extensions remain optional and must not add hard graph
dependencies.

## Recommended Sequence

### `v1.5`

- `KernelSpec`
- `FeatureSpec`
- `KernelRegistry`
- `KernelFeaturePlan`
- `compose_kernels`
- `ObservationSemantics`
- `ColumnRole`
- `FeaturePlanManifest`
- external provenance fields in `FeatureEvidence`

### `v1.6`

- feature-name policy
- feature-plan validation report
- feature-plan dry run
- evidence join helpers

### `v1.7`

- uncertainty propagation through composed kernels
- kernel similarity and drift diagnostics

## Non-Goal Reminder

Post-`v1.0` additions are constrained interoperability primitives.
No post-`v1.0` item should promise graph execution, process-compiler
implementation, or final modelling workflows inside `rtdfeatures`.
