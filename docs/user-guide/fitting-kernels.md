# Fitting Kernels

## SimplexKernelLearner (empirical)

The simplest learner fits one input signal to one target signal using simplex-constrained optimisation:

```python
from rtdfeatures import SimplexKernelLearner

learner = SimplexKernelLearner(
    max_lag="20m",
    min_lag="0m",
    loss="huber",          # or "mse"
    smoothness_penalty=0.0,
    seed=None,
)
fit = learner.fit(
    df,
    input_col="feed_signal",
    target_col="product_signal",
    time_col="time",
)
```

`fit` is a `KernelFitResult` containing:

- `fit.kernel` — the learned `LearnedKernel`
- `fit.fit_diagnostics` — training/validation loss, lag statistics
- `fit.identifiability_report` — warnings about kernel trustworthiness
- `fit.baseline_comparison` — comparison against `no_lag`, `best_single_lag`, and other baselines

## Parametric learners

Use when you have a domain-informed reason to prefer a specific kernel family
or a fixed structural assumption. Fitted non-simplex learners include fixed
delay, uniform-window, and parametric families (gamma, exponential, delayed
exponential, lognormal, and erlang):

```python
from rtdfeatures import (
    DelayedExponentialKernelLearner,
    ErlangKernelLearner,
    ExponentialKernelLearner,
    GammaKernelLearner,
    LogNormalKernelLearner,
)

gamma = GammaKernelLearner(max_lag="6h", min_lag="10m")
fit = gamma.fit(df, input_col="in", target_col="out", time_col="t")
```

Parametric learners return the same `KernelFitResult` type. Additional parametric metadata lives in `fit.fit_provenance`.

| Learner | Family | When to use |
|---|---|---|
| `SimplexKernelLearner` | Empirical (no shape assumption) | Default — safest when shape is uncertain |
| `FixedDelayKernelLearner` | Fixed delay (single lag step) | Known dominant transport delay with negligible spread |
| `UniformKernelLearner` | Uniform window | Delay is known to be spread approximately evenly over a lag band |
| `GammaKernelLearner` | Gamma distribution | Positively skewed, single-peak lag |
| `ExponentialKernelLearner` | Exponential distribution | Memoryless decay, strong recent dominance |
| `DelayedExponentialKernelLearner` | Delayed exponential | Dead-time onset followed by decay |
| `LogNormalKernelLearner` | Lognormal distribution | Right-skewed spread with heavier tail |
| `ErlangKernelLearner` | Erlang distribution | Staged, multi-compartment-like transit |

Direct kernel constructors are still available when you want fixed
hand-specified parameters instead of fitted parameters:

```python
from rtdfeatures import DelayedExponentialKernel

kernel = DelayedExponentialKernel(delay=2.0, rate_lambda=0.5,
                                  min_lag_steps=1, max_lag_steps=12, dt=60.0)
```

These can be passed to `KernelFeatureBuilder` or included as `candidate_type="fixed_kernel"` entries in `KernelCandidateSet` comparisons.

## Shared multi-pair fitting

Fit multiple input/target pairs with a single API call:

```python
from rtdfeatures.learners import SharedSimplexKernelLearner

shared = SharedSimplexKernelLearner(max_lag="40m", min_lag="10m")
fit = shared.fit(
    df,
    input_cols=["feed_a", "feed_b"],
    target_cols=["prod_a", "prod_b"],
    time_col="time",
)
kernels = fit.to_kernels()  # dict[str, LearnedKernel]
```

## Choosing simplex vs parametric

- Start with `SimplexKernelLearner` when you need fewer structural assumptions.
- Use `FixedDelayKernelLearner` or `UniformKernelLearner` when that structure is
  justified a priori.
- Use a parametric learner when the assumed family shape is physically or
  operationally justified.
- Compare diagnostics and baselines before selecting a kernel for feature generation.
- Simplex is safer when parametric fits show weak identifiability, boundary-piled mass, or limited improvement over `best_single_lag`.

## See also

- [Comparing kernels](comparing-kernels.md) — baseline and candidate comparison
- [04_kernel_learning_design.md](../04_kernel_learning_design.md) — normative learning contract
- [08_api_design.md](../08_api_design.md) — full API reference with all parameters
