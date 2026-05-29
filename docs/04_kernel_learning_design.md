# Kernel Learning Design

## Goal

Learn one constrained kernel from one input signal and one target signal.

## `v0.1` Learner

- `SimplexKernelLearner`
- One `input_col`
- One `target_col`
- Static kernel
- CPU execution
- Deterministic seed support

`SharedSimplexKernelLearner` is introduced in `v0.2` as shared coordination of
independent one-input/one-target fits.

## `v0.5` Parametric Learners

`v0.5` adds parametric learner variants for the same one-input/one-target
contract:

- `GammaKernelLearner`
- `ExponentialKernelLearner`
- `DelayedExponentialKernelLearner`
- `LogNormalKernelLearner`
- `ErlangKernelLearner`

These learners estimate family parameters, then convert them to a discrete
kernel on the admissible lag grid. They do not introduce a separate
feature-generation path.

## Learners Versus Kernel Specs

`KernelSpec` is not a learner.

`FeatureSpec` is not a learner.

`KernelFeaturePlan` is not a learner.

Learners return `KernelFitResult`.

Direct kernel construction returns a validated `Kernel` object, not a
`KernelFitResult`.

Feature plans may reference kernels and feature requests, but they do not imply
fitting unless explicitly paired with learner/candidate evaluation workflows in
versioned plans.

## Input Contract

- `df: pl.DataFrame`
- `input_col: str`
- `target_col: str`
- `time_col: str`
- `max_lag`
- Optional `dt`
- Optional `min_lag`
- Optional smoothness penalty
- Optional deterministic seed
- Optional `order_by_time: bool = False`

## Time Handling

- The data must be on a regular time grid
- If `dt` is omitted, infer it from `time_col`
- If `dt` is supplied, validate that it matches the observed grid
- If the grid is irregular, raise a clear error
- If the data is unsorted, raise unless `order_by_time=True`

## Fit Contract

```python
fit = learner.fit(...)
fit.kernel
fit.fit_diagnostics
fit.identifiability_report
fit.baseline_comparison
```

`KernelFitResult` is the learner's primary return object.

`v0.5` contract choice: parametric learners return `KernelFitResult` directly
for successful fits. Any parametric details are additive diagnostics metadata,
not a replacement fit-result type.

## `v0.2` Shared Fit Contract

`SharedSimplexKernelLearner.fit(...)` takes `input_cols` and `target_cols`
using positional zip semantics:

- pair `i` is `(input_cols[i], target_cols[i])`
- `input_cols` and `target_cols` must have equal length
- unequal lengths raise `ValueError`
- empty lists raise `ValueError`

Pair naming rules:

- default pair id is `"{input_col}->{target_col}"`
- optional explicit `pair_names` may override default ids
- explicit names must be non-empty and unique

Return type:

- `SharedKernelFitResult`, an aggregate object
- ordered pair outcomes are preserved in positional zip order
- each pair is represented as `SharedPairFitResult`
- pair failures are explicit and do not hide successful pair results

## Learned Kernel Contract

A learned kernel stores:

- `weights`
- `lag_steps`
- `dt`
- `min_lag_steps`
- `max_lag_steps`
- Optional `name`

Fit provenance belongs in `KernelFitResult`, not on the kernel object itself.

All kernel variants must support:

- `summary()`
- `mean_lag()`
- `percentile(q)`
- `tail_mass(threshold)`
- `validate()`

`tail_mass(threshold)` is valid because kernels store `dt` and lag steps.

## Constraints

Learned kernels must be:

- Causal
- Non-negative
- Sum-to-one
- Bounded by `max_lag`

For parametric learners, family parameters must map to kernels that still
satisfy these constraints after discretisation.

Reference parameterisation:

```python
raw_theta = torch.nn.Parameter(torch.zeros(n_lags))
weights = torch.softmax(masked_theta, dim=0)
```

## Fitting Rules

- Prevent future leakage by design
- Support `min_lag`
- Scale inputs and targets before optimisation
- Prefer robust scaling for industrial sensor data
- Drop or mask training windows with missing input/target values rather than heavily imputing them
- Default loss: `huber`
- Optional loss: `mse`
- Use blocked time-based validation, not random row splits
- Primary ranking metric: validation loss for the configured learner loss
- Lower validation loss is better

`v0.5` parametric learners follow the same loss, scaling, validation split,
missing-window handling, and sorting contract as simplex learners.

## Learning Boundary

Kernel learning remains a one-input/one-target contract unless explicitly
expanded by a versioned plan.

Graph-derived paths, regime labels, or external process-feature compiler
outputs may define candidate pairs or feature specs, but they do not change
learner semantics in this package boundary.

## `v0.5` Parametric Parameter Contract

### `GammaKernelLearner` fields

- `shape_alpha` (> `0`)
- `rate_beta` (> `0`)
- `init_shape_alpha` (default `2.0`, must be > `0`)
- `init_rate_beta` (default derived from lag window midpoint, must be > `0`)
- if admissible lag steps include zero lag (`min_lag=0`), gamma fitting
  requires `shape_alpha > 1.0`; invalid initial shape raises `ValueError`
- a zero-only lag grid (`min_lag=0` and `max_lag=0`) is rejected early with
  `ValueError` because no valid gamma mass can be allocated

### `ExponentialKernelLearner` fields

- `rate_lambda` (> `0`)
- `init_rate_lambda` (default inverse of lag window midpoint, must be > `0`)

### Shared parameter defaults and checks

- `min_lag`, `max_lag`, and `dt` define admissible lag steps.
- Initial parameter defaults must be deterministic from fit inputs when not
  explicitly provided.
- Non-positive parameters or invalid lag-window settings raise `ValueError`.

## `v0.5` Parametric-To-Discrete Conversion

For each admissible lag step `k` with lag time `tau_k`:

1. evaluate the family density or mass proxy at `tau_k`
2. clamp negatives to `0`
3. normalise the vector to sum to `1`

The resulting weights define a standard constrained `LearnedKernel` with
`lag_steps`, `dt`, `min_lag_steps`, and `max_lag_steps`.

If conversion yields zero total mass or non-finite values, fitting fails with a
clear error; it does not silently fall back to another learner.

Direct parametric kernel construction uses the same conversion rules but skips
the fitting stage entirely. In that path, callers provide family parameters and
receive a validated `Kernel` object directly; no `KernelFitResult` is produced.

## Baselines In `v0.2`

- `no_lag`
- `best_single_lag`
- `uniform`
- `exponential`

These are baseline scoring methods for validation-loss comparison, not
parametric learner families.

## Identifiability Checks

- Flat input
- Flat or noisy target
- Boundary-piled kernels
- Diffuse kernels
- Validation loss much worse than training loss
- `best_single_lag` outperforming the learned kernel

Parametric learners use the same identifiability checks after conversion to the
discrete kernel representation.

## Post-v1.0 Related Objects

Post-v1.0 planning and comparison workflows may refer to objects such as:

- `KernelCandidate`
- `KernelComparisonConfig`
- `KernelEvaluationResult`
- internal fit/evaluation context metadata
- `BootstrapResult`

The learner contract in this document remains unchanged. Full object-level
details belong in diagnostics/API contracts.
