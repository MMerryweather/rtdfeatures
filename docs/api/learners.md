# Learners API

## SimplexKernelLearner

Empirical constrained kernel learner (one-input/one-target).

```python
from rtdfeatures import SimplexKernelLearner

learner = SimplexKernelLearner(
    max_lag="20m",
    min_lag=0,
    dt=None,
    loss="huber",               # "huber" or "mse"
    smoothness_penalty=0.0,
    seed=None,
    validation_fraction=0.2,
    learning_rate=0.05,
    max_epochs=800,
    huber_delta=1.0,
)
fit = learner.fit(df, input_col="in", target_col="out", time_col="t", order_by_time=False)
```

**Returns:** `KernelFitResult`

## FixedDelayKernelLearner

Deterministic fixed-delay learner (single active lag).

```python
from rtdfeatures.learners import FixedDelayKernelLearner

fixed = FixedDelayKernelLearner(max_lag="20m", min_lag=0)
fit = fixed.fit(df, input_col="in", target_col="out", time_col="t", order_by_time=False)
```

**Returns:** `KernelFitResult`

## UniformKernelLearner

Deterministic uniform-window learner (equal mass across the active lag window).

```python
from rtdfeatures.learners import UniformKernelLearner

uniform = UniformKernelLearner(max_lag="20m", min_lag=0)
fit = uniform.fit(df, input_col="in", target_col="out", time_col="t", order_by_time=False)
```

**Returns:** `KernelFitResult`

## SharedSimplexKernelLearner

Multi-pair shared execution over positional input/target lists.

```python
from rtdfeatures.learners import SharedSimplexKernelLearner

shared = SharedSimplexKernelLearner(max_lag="40m", min_lag="10m", loss="huber")
fit = shared.fit(
    df, input_cols=["a", "b"], target_cols=["c", "d"],
    time_col="t", pair_names=None, order_by_time=False,
)
```

**Returns:** `SharedKernelFitResult`

Methods: `pair_ids()`, `get_pair(id)`, `get_pair_result(id)`, `summary()`, `to_kernels()`

## GammaKernelLearner

Parametric gamma kernel learner.

```python
from rtdfeatures import GammaKernelLearner

gamma = GammaKernelLearner(max_lag="6h", min_lag="10m", loss="huber",
                           init_shape_alpha=2.0, init_rate_beta=None)
fit = gamma.fit(df, input_col="in", target_col="out", time_col="t")
```

**Returns:** `KernelFitResult` with parametric provenance in `fit_provenance`.

## ExponentialKernelLearner

Parametric exponential kernel learner.

```python
from rtdfeatures import ExponentialKernelLearner

exp = ExponentialKernelLearner(max_lag="6h", min_lag="10m", loss="huber",
                               init_rate_lambda=None)
fit = exp.fit(df, input_col="in", target_col="out", time_col="t")
```

**Returns:** `KernelFitResult` with parametric provenance in `fit_provenance`.

## DelayedExponentialKernelLearner

Parametric delayed-exponential kernel learner.

```python
from rtdfeatures.learners import DelayedExponentialKernelLearner

delayed_exp = DelayedExponentialKernelLearner(
    max_lag="6h", min_lag="10m", loss="huber", init_delay=None, init_rate_lambda=None
)
fit = delayed_exp.fit(df, input_col="in", target_col="out", time_col="t")
```

**Returns:** `KernelFitResult` with parametric provenance in `fit_provenance`.

## LogNormalKernelLearner

Parametric lognormal kernel learner.

```python
from rtdfeatures.learners import LogNormalKernelLearner

lognormal = LogNormalKernelLearner(
    max_lag="6h", min_lag="10m", loss="huber", init_log_mu=None, init_log_sigma=0.5
)
fit = lognormal.fit(df, input_col="in", target_col="out", time_col="t")
```

**Returns:** `KernelFitResult` with parametric provenance in `fit_provenance`.

## ErlangKernelLearner

Parametric Erlang kernel learner.

```python
from rtdfeatures.learners import ErlangKernelLearner

erlang = ErlangKernelLearner(
    max_lag="6h", min_lag="10m", loss="huber", shape_k_candidates=None, init_rate_beta=None
)
fit = erlang.fit(df, input_col="in", target_col="out", time_col="t")
```

**Returns:** `KernelFitResult` with parametric provenance in `fit_provenance`.

### Direct-construction kernels

1. **Direct construction** — instantiate the kernel class directly e.g.
   `DelayedExponentialKernel(delay=2.0, rate_lambda=0.5, ...)` for use with
   `KernelFeatureBuilder`.
2. **Fixed candidates** — include them in `KernelCandidateSet` comparisons with
   `candidate_type="fixed_kernel"` and the appropriate `fixed_parameters`.

See [kernels.md](kernels.md) for the full list of available parametric families
and [comparing-kernels.md](../user-guide/comparing-kernels.md) for fixed candidate usage.

## KernelFitResult

Return type from `fit()`:

Identifiability warnings are shared learner diagnostics, not simplex-specific behaviour. Warning policy and report construction live in a private helper module so empirical and parametric learners use one interpretation path.
Learner fit-result assembly is centralised. Learner classes are orchestration layers: prepare data, optimise parameters or weights, then delegate baseline evaluation and result assembly to private helpers.
Learner classes do not inherit from each other merely to share helpers. Shared optimisation and validation mechanics live in private functions. Public learner classes remain thin orchestration layers around preparation, optimisation, baseline evaluation, and result assembly.

- `kernel` — the learned `LearnedKernel`
- `fit_diagnostics` — `FitDiagnostics` (train/validation loss, lag stats)
- `identifiability_report` — `IdentifiabilityReport` (warnings)
- `baseline_comparison` — `BaselineComparison` (baseline losses)
- `fit_provenance` — `dict` (fitting metadata, parametric provenance)
- `kernel_shape_summary` — `KernelShapeSummary` (optional)
- `fit_data_coverage_summary` — `FitDataCoverageSummary` (optional)
