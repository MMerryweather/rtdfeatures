# Package Architecture

Internal changes should preserve the golden path: fit or define kernel → generate `TransformResult` → inspect features/report/registry. See `development/architecture-principles.md`.
Identifiability warnings are shared learner diagnostics, not simplex-specific behaviour. Warning policy and report construction live in a private helper module so empirical and parametric learners use one interpretation path.
Learner fit-result assembly is centralised. Learner classes are orchestration layers: prepare data, optimise parameters or weights, then delegate baseline evaluation and result assembly to private helpers.
Learner classes do not inherit from each other merely to share helpers. Shared optimisation and validation mechanics live in private functions. Public learner classes remain thin orchestration layers around preparation, optimisation, baseline evaluation, and result assembly.
Feature-generation internals use an accumulator to keep arrays, metadata, missingness, and per-kernel feature lists together. Parallel dictionaries should not be manually updated in multiple places.

## Stable Core Packages (Current `v1.0`)

Current package layout under `src/rtdfeatures/`:

- `kernels/` (kernel objects and parametric kernel families)
- `learners/` (simplex/parametric learners and shared fit orchestration)
- `features/` (feature generation, registry, and evidence helpers)
- `diagnostics/` (fit/transform diagnostics contracts and warning helpers)
- `candidates/` (candidate contracts, fitting, and selection)
- `bootstrap/` (blocked bootstrap contracts, sampling, and summaries)
- `oof/` (forward-chaining splits, OOF generation, and reports)
- `integrations/` (optional adapter boundary; core package must not require adapters)
- top-level support modules: `baselines.py`, `reporting.py`, `synthetic.py`, `utils.py`

`docs/development/architecture.md` is the implementation-facing map; this
document defines the contract-level architecture and boundaries.

## Public API Surface In `v0.1`

- `Kernel`
- `LearnedKernel`
- `FixedDelayKernel`
- `UniformKernel`
- `SimplexKernelLearner`
- `KernelFeatureBuilder`
- `KernelFitResult`
- `FitDiagnostics`
- `IdentifiabilityReport`
- `BaselineComparison`
- `TransformReport`

## Public API Surface Additions In `v0.2`

- `SharedPairFitResult`
- `SharedKernelFitResult`

## Public API Surface Additions In `v0.5`

- `GammaKernelLearner`
- `ExponentialKernelLearner`

## Public API Surface Additions In `v0.7`

- `KernelCandidate`
- `KernelCandidateSet`
- `KernelFamilyFitResult`
- `KernelComparisonResult`
- `KernelSelectionResult`

`v0.7` additions are object-contract primitives for candidate comparison.
`KernelFitResult` remains unchanged and is reused within
`KernelFamilyFitResult` where applicable.

## Public API Surface Additions In `v0.8`

- `BootstrapResult`
- `BootstrapWeightSample`
- `BootstrapParameterSample`
- `BootstrapLagSummarySample`
- `KernelBootstrapSummary`
- `ParameterUncertaintySummary`
- `WeightUncertaintySummary`
- `BOOTSTRAP_WARNING_CODES`
- `DEFAULT_BOOTSTRAP_INTERVAL_QUANTILES`
- `bootstrap_weight_samples_schema()`
- `bootstrap_parameter_samples_schema()`
- `bootstrap_lag_summary_samples_schema()`
- `parameter_uncertainty_summary_schema()`
- `weight_uncertainty_summary_schema()`

`v0.8` bootstrap uncertainty includes contract objects, warning codes, table
schemas, interval-default constants, and blocked-bootstrap execution utilities.

## Public API Surface Additions In `v0.9`

- `FeatureEvidence`
- `FeatureEvidenceReport`
- `FEATURE_INTERPRETATION_LABELS`
- `FEATURE_EVIDENCE_COMPLETENESS_LABELS`

`v0.9` first introduced feature-evidence contract objects and
aggregate evidence reports. Later `v0.9` updates kept transform behavior unchanged and added
execution helpers in `features.py`:
`build_feature_evidence()`, `feature_evidence_table()`,
`feature_evidence_compact_dict()`, and `feature_evidence_compact_text()`.

`FeatureEvidence` label validation is fail-closed (`ValueError` for unknown
labels). `FeatureEvidenceReport` validates summary-shape consistency, including
`feature_count == len(feature_evidence)`.

## Repository Layout (Current `v1.0`)

```text
rtdfeatures/
  pyproject.toml
  src/
    rtdfeatures/
      __init__.py
      baselines.py
      reporting.py
      synthetic.py
      utils.py
      kernels/
      learners/
      features/
      diagnostics/
      candidates/
      bootstrap/
      oof/
      integrations/
  tests/
    test_kernels.py
    test_learners.py
    test_parametric_learners_contract.py
    test_parametric_learners.py
    test_features.py
    test_diagnostics.py
```

This layout reflects the current repository structure. Internal boundaries may
evolve, but public contracts must remain additive and stable.

## Required Dependencies For `v1.0`

- `polars`
- `numpy`
- `torch`

## Development Tooling

- `pytest`
- `ruff`
- `mypy`

## Boundary Rules

- Keep the product wedge narrow: learn lag kernels, validate kernel quality,
  generate feature tables and diagnostics, then stop.
- Keep terminology and APIs domain-neutral; avoid domain-specific control or
  forecasting scope.
- Preserve constrained learned-kernel semantics: causal, non-negative,
  sum-to-one, bounded lag.
- Public data interface remains `polars.DataFrame` in and `polars.DataFrame`
  out.
- Optional integrations remain adapter-only boundaries; external algorithms and
  runtime dependencies do not become hard requirements for core flows.

## Deferred Or Optional Pieces

- Additional baselines beyond `no_lag` and `best_single_lag`
- Extra runtime dependencies justified only by deferred features

`v0.5` module/class intent:

- `learners/` holds learner families, including `GammaKernelLearner` and
  `ExponentialKernelLearner`
- `kernels/parametric.py` holds parametric-kernel construction and provenance
  helper utilities
- direct parametric conversion helpers are deterministic kernel-construction
  utilities only; learner fitting logic remains in `learners/`
- learners still emit constrained `Kernel` objects and `KernelFitResult`
- diagnostics extensions for parametric fits remain additive to existing report
  objects

Plotting decision for `v0.2`:

- plotting helpers remain deferred
- no plotting dependency is added to the core install in `v0.2`
- if plotting is introduced later, it must be via explicit optional extras

## Public API Surface Additions In `v0.95`

- `OutOfFoldKernelFeatureResult`
- `OutOfFoldSplitSummary`
- `ForwardChainingSplitConfig`
- `ForwardChainingFoldSplit`
- `generate_forward_chaining_splits()`
- `fit_transform_oof(...)`

`v0.95` defines OOF contract objects and deterministic
forward-chaining split generation with explicit leakage guardrails. Later `v0.95` updates add
public OOF execution through `fit_transform_oof(...)`, covering both
single-learner and candidate-set paths while keeping the package scoped to
kernel learning/validation and feature generation.

## Optional Integration Boundary In `v1.0` (Tigramite Adapter)

Optional adapter contract additions:

- `TigramiteLagCandidateResult`
- Tigramite warning-code constants:
  `TIGRAMITE_NO_LINKS_FOUND`, `TIGRAMITE_LAG_RANGE_EMPTY`,
  `TIGRAMITE_CONTEMPORANEOUS_LINK_IGNORED`,
  `TIGRAMITE_GRAPH_MARK_UNSUPPORTED`,
  `TIGRAMITE_KERNEL_SUPPORT_THRESHOLD_BREACH`,
  `TIGRAMITE_VALUES_NOT_KERNEL_WEIGHTS`, `TIGRAMITE_PAYLOAD_SHAPE_INVALID`,
  `TIGRAMITE_VARIABLE_NAME_MISSING`

Architecture boundary rules:

- Tigramite is not a hard dependency of `rtdfeatures`
- core install and core tests remain Tigramite-free
- Tigramite runs outside `rtdfeatures`; only result payloads cross the boundary
- adapter input is graph/value/p-value-style payloads, not live estimator
  objects
- adapter output is candidate lag evidence only (for downstream constrained
  kernel fitting inside `rtdfeatures`)
- Tigramite statistics are never interpreted as constrained kernel weights

## External Compiler Boundary

`processfeaturecompiler` may depend on `rtdfeatures`.
`rtdfeatures` must not depend on `processfeaturecompiler`.

## Dependency Direction

```text
rtdfeatures <- processfeaturecompiler <- external MLOps
```

## No Hard Integration Dependencies

- no Databricks dependency
- no SageMaker dependency
- no MLflow dependency
- no Tigramite hard dependency
- no graph library hard dependency

## Optional Integration Policy

Adapters consume external outputs and convert them into `rtdfeatures`-native
contracts. Adapters do not run or embed external algorithms inside
`rtdfeatures`.
