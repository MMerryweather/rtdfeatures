# API Design

Internal changes should preserve the golden path: fit or define kernel → generate `TransformResult` → inspect features/report/registry. See `development/architecture-principles.md`.

## `v0.1` API Goals

Expose one clean learning path and one clean feature-generation path.

## Learning API

```python
learner = SimplexKernelLearner(max_lag="6h", min_lag="10m", loss="huber")
fit = learner.fit(
    df,
    input_col="feed_signal",
    target_col="product_signal",
    time_col="timestamp",
    order_by_time=False,
)
```

## Feature Generation API

```python
builder = KernelFeatureBuilder(
    kernels={"learned": fit.kernel},
    time_col="timestamp",
    numeric_cols=["temperature", "pressure"],
    category_cols=["operating_mode"],
    weight_col="flowrate",
    age_tail_threshold=None,
)

features = builder.transform(df)
augmented = builder.augment_cols(df)
report = builder.diagnose_transform(df)
```

When `age_tail_threshold=None`, the builder uses the default tail-mass threshold
from the diagnostics contract.

## Return Objects

- `fit` is `KernelFitResult`
- `features` is `pl.DataFrame`
- `augmented` is `pl.DataFrame`
- `report` is `TransformReport`

## Post-v1.0 Planning API (Provisional)

These API shapes are forward-facing contract sketches, not implementation
promises for this phase.

```python
kernel_spec = KernelSpec(...)
feature_spec = FeatureSpec(...)
registry = KernelRegistry(...)
plan = KernelFeaturePlan(...)
```

Planning objects describe kernels/features in serializable form so external
planners can hand execution requests into `rtdfeatures`.

## Executing A Feature Plan (Provisional)

```python
features = execute_feature_plan(df, plan)
report = validate_feature_plan(df, plan)
manifest = feature_plan_manifest(plan, df)
```

These names may evolve, but boundary intent is stable:

- execute kernel-based feature generation
- emit diagnostics/report/manifest artifacts
- keep dataframe I/O contract

## Bootstrap Object Contract In `v0.8`

Public bootstrap objects:

- `BootstrapResult`
- `BootstrapWeightSample`
- `BootstrapParameterSample`
- `BootstrapLagSummarySample`
- `KernelBootstrapSummary`
- `ParameterUncertaintySummary`
- `WeightUncertaintySummary`

Bootstrap schema helper API:

- `bootstrap_weight_samples_schema()`
- `bootstrap_parameter_samples_schema()`
- `bootstrap_lag_summary_samples_schema()`
- `parameter_uncertainty_summary_schema()`
- `weight_uncertainty_summary_schema()`

Bootstrap quantile contract:

- intervals use deterministic quantiles from `DEFAULT_BOOTSTRAP_INTERVAL_QUANTILES`
- default interval quantiles are `(0.025, 0.975)`

## Feature Evidence API In `v0.9`

Public feature-evidence objects:

- `FeatureEvidence`
- `FeatureEvidenceReport`

Label constants:

- `FEATURE_INTERPRETATION_LABELS`
- `FEATURE_EVIDENCE_COMPLETENESS_LABELS`

Allowed interpretation labels:

- `material_memory`
- `process_response`
- `statistical_pattern`
- `unknown`

Allowed completeness labels:

- `kernel_only`
- `fit_evidence`
- `comparison_evidence`
- `bootstrap_evidence`
- `full_evidence`

Contract notes:

- evidence metadata is additive and descriptive; it does not alter
  `KernelFeatureBuilder.transform()` output columns or values
- missing optional evidence is represented with `None` for:
  `fit_result_id`, `candidate_id`, `baseline_summary`,
  `identifiability_warnings`, `bootstrap_summary`
- invalid labels for `interpretation` and `evidence_completeness` are rejected
  with `ValueError`
- Early `v0.9` releases defined object/label contracts; later updates added execution helpers:
  `build_feature_evidence()`, `feature_evidence_table()`,
  `feature_evidence_compact_dict()`, and `feature_evidence_compact_text()`

## Kernel Inspection API

All kernel variants should support:

- `summary()`
- `mean_lag()`
- `percentile(q)`
- `tail_mass(threshold)`
- `validate()`

## Kernel Composition API (Provisional)

```python
path_kernel = compose_kernels([k1, k2, k3])
```

Kernel composition is kernel algebra only. Path discovery remains outside this
package boundary.

## Observation Semantics API (Provisional)

```python
semantics = {
    "lab_fe": ObservationSemantics(
        timestamp_type="interval_end",
        sample_basis="composite",
        weighting_basis="dry_mass",
    )
}
```

Semantics metadata supports validation and warnings; it does not imply automatic
resampling.

## Column Role API (Provisional)

```python
roles = {
    "feed_fe": "input_tag",
    "recovery": "target",
    "future_lab_result": "forbidden",
}
```

Roles support leakage guardrails and compatibility checks.

## Boundary Examples

Not part of `rtdfeatures` public API:

- no `ProcessGraph` ownership
- no `GraphFeaturePlanner` ownership
- no motif mining API
- no final model fitting API

## Optional Tigramite Adapter Contract In `v1.0`

This adapter surface is optional. `rtdfeatures` does not require Tigramite for
install, import, or core execution.

Boundary:

- Tigramite causal discovery is run outside `rtdfeatures`
- `rtdfeatures` adapter helpers consume already-produced graph/value/p-value
  outputs only
- adapter helpers accept plain dict/array-like payloads
- adapter helpers do not require Tigramite estimator instances
- adapter helpers do not run or wrap Tigramite algorithms
- adapter outputs candidate evidence only
- adapter does not convert Tigramite statistics into kernel weights

Required contract object:

- `TigramiteLagCandidateResult`

Required fields:

- `source_col`
- `target_col`
- `lag_steps`
- `min_lag_step`
- `max_lag_step`
- `link_values`
- `p_values`
- `graph_marks`
- `source`
- `warnings`
- `metadata`

Lag-sign and mark handling contract:

- lag sign follows Tigramite-style outputs and is interpreted into candidate lag
  steps only; no kernel-weight inference is performed from sign/magnitude
- contemporaneous links (`lag == 0`) are not converted into lag candidates and
  emit `TIGRAMITE_CONTEMPORANEOUS_LINK_IGNORED`
- unsupported graph marks are skipped and emit
  `TIGRAMITE_GRAPH_MARK_UNSUPPORTED`

Warning taxonomy:

- `TIGRAMITE_NO_LINKS_FOUND`
- `TIGRAMITE_LAG_RANGE_EMPTY`
- `TIGRAMITE_CONTEMPORANEOUS_LINK_IGNORED`
- `TIGRAMITE_GRAPH_MARK_UNSUPPORTED`
- `TIGRAMITE_KERNEL_SUPPORT_THRESHOLD_BREACH`
- `TIGRAMITE_VALUES_NOT_KERNEL_WEIGHTS`
- `TIGRAMITE_PAYLOAD_SHAPE_INVALID`
- `TIGRAMITE_VARIABLE_NAME_MISSING`

## Design Rules

- Use clear names over compact names
- Keep dataframes as the public surface
- Follow tidy-style output principles
- `transform()` returns only generated columns plus `time_col`
- Diagnostics are separate from the main dataframe return value
- Builder methods accept `order_by_time=False` with the same sorting contract as learner methods

## Deferred API

### Synthetic Helper Boundary For `v0.3`

`rtdfeatures.synthetic` exposes deterministic helper datasets for kernel learning
and feature-generation workflows.

Public helper contracts in this boundary:

- `make_single_delay_dataset`
- `make_spread_delay_dataset`
- `make_noisy_identifiable_dataset`
- `make_weak_identifiability_dataset`
- `make_multi_pair_dataset`
- `make_missing_window_dataset`
- `make_boundary_kernel_dataset`
- `make_diffuse_kernel_dataset`
- `make_baseline_challenge_dataset`

Each helper returns `SyntheticDataset` with:

- `data: pl.DataFrame`
- `true_kernels: dict[str, KernelMetadata]`
- `scenario: SyntheticScenario`

Benchmark-layer usage remains separate from these helpers. See
[benchmarks/nrtd_benchmark_layers.md](benchmarks/nrtd_benchmark_layers.md).

nRTD experimental learned RTDs are references, not synthetic helper ground truth.

This helper boundary does not define learner families. Learner-family
availability is defined in the learning API and learners reference docs.

## Shared Learning API In `v0.2`

```python
shared = SharedSimplexKernelLearner(max_lag="6h", min_lag="10m", loss="huber")
fit = shared.fit(
    df,
    input_cols=["feed_a", "feed_b"],
    target_cols=["prod_a", "prod_b"],
    time_col="timestamp",
    pair_names=None,
    order_by_time=False,
)
```

Contract:

- `input_cols` and `target_cols` are positional zip pairs
- lengths must be equal and non-zero, otherwise `ValueError`
- default pair id: `"{input}->{target}"`, unless `pair_names` overrides
- return type: `SharedKernelFitResult` (aggregate per-pair outcomes)
- per-pair successful fits remain `KernelFitResult`

## Parametric Learning API In `v0.5`

```python
gamma = GammaKernelLearner(
    max_lag="6h",
    min_lag="10m",
    loss="huber",
    init_shape_alpha=2.0,
    init_rate_beta=None,
)
gamma_fit = gamma.fit(
    df,
    input_col="feed_signal",
    target_col="product_signal",
    time_col="timestamp",
    order_by_time=False,
)
```

```python
exp = ExponentialKernelLearner(
    max_lag="6h",
    min_lag="10m",
    loss="huber",
    init_rate_lambda=None,
)
exp_fit = exp.fit(
    df,
    input_col="feed_signal",
    target_col="product_signal",
    time_col="timestamp",
    order_by_time=False,
)
```

Parametric contract:

- `gamma_fit` and `exp_fit` are `KernelFitResult`
- successful fits expose a standard constrained discrete `LearnedKernel`
- parameter fields are positive (`shape_alpha`, `rate_beta`, `rate_lambda`)
- gamma zero-lag rule: when `min_lag=0`, `shape_alpha` must be `> 1.0`;
  violating this raises `ValueError`
- gamma zero-only lag windows (`min_lag=0`, `max_lag=0`) are invalid and raise
  `ValueError` during `fit()`
- when `None`, init-rate defaults are deterministic from the lag window and
  inferred or provided `dt`
- sorting, scaling, validation split, and missing-window handling match
  `SimplexKernelLearner`
- invalid parameter values or failed conversion to discrete weights raise clear
  errors and do not silently fall back

Direct parametric construction contract:

- direct construction converts supplied family parameters into a validated
  discrete `Kernel`
- direct construction does not fit parameters and does not create a
  `KernelFitResult`
- kernels from either path are consumed through `KernelFeatureBuilder`
- this API surface remains feature-engineering focused, not final prediction or
  causal discovery

## Choosing Simplex Versus Parametric Learners (`v0.5`)

Use learner families deliberately; the package does not perform automated model
selection.

- Start with `SimplexKernelLearner` when you need fewer structural assumptions
  about kernel shape.
- Use a parametric learner (`GammaKernelLearner`,
  `ExponentialKernelLearner`, `DelayedExponentialKernelLearner`,
  `LogNormalKernelLearner`, or `ErlangKernelLearner`) when you have a
  domain-informed reason to prefer that constrained family.
- Compare learner outputs with diagnostics and baseline metrics before using a
  kernel for feature generation.

Practical safety rule:

- simplex learning is usually safer when parametric fits show weak
  identifiability, boundary-piled mass, or limited validation improvement over
  `best_single_lag`.

## Candidate Comparison Object API (`v0.7`)

`v0.7` adds object contracts for comparing kernel candidates without adding
downstream predictive-model selection.

Public result objects:

- `KernelCandidate`
- `KernelCandidateSet`
- `KernelFamilyFitResult`
- `KernelComparisonResult`
- `KernelSelectionResult`

Descriptor contract:

- `KernelCandidate` is serializable and must not store live learner/kernel
  instances.
- `candidate_type` is one of:
  `fixed_kernel`, `empirical_learner`, `parametric_learner`, `baseline`.
- candidate and metadata payloads remain transport-friendly JSON data.

Set contract:

- `KernelCandidateSet` is scoped to one `input_col` / `target_col` / `time_col`
  triple.
- candidate IDs are unique and candidate lists are non-empty.

Relationship to existing fit results:

- `KernelFitResult` schema and semantics are unchanged.
- `KernelFamilyFitResult.fit_result` reuses `KernelFitResult` for successful
  learner candidates and allows `None` for fixed-kernel/baseline rows.
- `KernelFamilyFitResult` additionally exposes comparison-facing fields:
  `validation_loss`, `train_loss`, `warning_codes`, and
  `evaluated_fixed_kernel`.
- failures remain explicit results and are not dropped.

Selection contract:

- `KernelSelectionResult` is optional and kernel-only.
- learner candidates use baseline/identifiability gates from `fit_result`.
- fixed-kernel candidates can be selected without `fit_result` only when
  explicit fixed evidence fields are present:
  `evaluated_fixed_kernel`, `fixed_baseline_comparison`, and
  `evaluation_provenance`.
- mixed comparability signatures fail closed: when successful non-baseline
  finite-loss candidates include both present and missing context signatures,
  selection returns no recommendation with explicit warnings.
- no `v0.7` object selects a final downstream predictive model.

## Out-Of-Fold API In `v0.95`

Public OOF contract objects:

- `OutOfFoldKernelFeatureResult`
- `OutOfFoldSplitSummary`

Public split utility:

- `ForwardChainingSplitConfig`
- `ForwardChainingFoldSplit`
- `generate_forward_chaining_splits(n_rows, config)`

Public OOF execution helper:

- `fit_transform_oof(...)`

`ForwardChainingSplitConfig` fields:

- `n_folds`
- `min_train_size`
- `validation_size`
- `gap` (optional, default `0`)
- `max_train_size` (optional)

Contract notes:

- OOF utilities support leakage-safe fold generation for feature workflows
- `fit_transform_oof(...)` executes fold-aware kernel fitting and feature
  generation without training downstream predictive models
- `fit_transform_oof(...)` is a public export and supports single-learner and
  candidate-set OOF workflows
- it does not perform downstream model training
- fold metadata is deterministic and reproducible
- invalid split configurations fail closed with clear `ValueError` messages
