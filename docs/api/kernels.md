# Kernels API

## Kernel

Base class for all constrained causal kernels.

```python
from rtdfeatures import Kernel

kernel = Kernel(
    weights=(0.2, 0.5, 0.3),
    lag_steps=(0, 1, 2),
    dt=60.0,
    min_lag_steps=0,
    max_lag_steps=2,
    name="my_kernel",
)
```

**Properties:** `weights`, `lag_steps`, `dt`, `min_lag_steps`, `max_lag_steps`, `name`

**Methods:**

- `validate()` — check all constraints, raise on violation
- `summary() -> dict` — compact kernel description
- `mean_lag() -> float` — weighted mean lag in time units
- `percentile(q) -> float` — weighted lag percentile for `q in [0, 1]`
- `tail_mass(threshold) -> float` — weight mass above `threshold` time units

## LearnedKernel

Subclass of `Kernel` returned by learners. Adds fit provenance.

```python
from rtdfeatures import LearnedKernel
```

Same interface as `Kernel`.

## FixedDelayKernel

Single-step delay with all weight at one lag step.

```python
from rtdfeatures import FixedDelayKernel

kernel = FixedDelayKernel(delay_step=5, dt=60.0, name="fixed_5min")
```

## UniformKernel

Equal weight across all lag steps.

```python
from rtdfeatures import UniformKernel

kernel = UniformKernel(min_lag_steps=0, max_lag_steps=10, dt=60.0)
```

## Parametric kernels

```python
from rtdfeatures import GammaKernel, ExponentialKernel, DelayedExponentialKernel
from rtdfeatures.kernels import ErlangKernel, LogNormalKernel
```

These convert family parameters onto a discrete lag grid. They do not perform fitting — use the corresponding learner for that.
Parametric kernel families are registered in one place. Family metadata, parameter validation, and weight generation should not be duplicated across switches, summaries, and constructors.
Explicit parametric kernels share internal initialisation and summary helpers. Public constructors remain separate and readable; internal validation, lag-grid construction, weight generation, and summary extension are centralised.

| Kernel | Parameters | Family |
|---|---|---|
| `GammaKernel` | `shape_alpha`, `rate_beta` | Gamma distribution |
| `ExponentialKernel` | `rate_lambda` | Exponential distribution |
| `DelayedExponentialKernel` | `delay`, `rate_lambda` | Shifted exponential |
| `ErlangKernel` | `shape_k`, `rate_beta` | Erlang (integer-shape gamma) |
| `LogNormalKernel` | `log_mu`, `log_sigma` | Log-normal distribution |

**Learner availability:** All five families are available as fixed/explicit kernels, but only `GammaKernel` and `ExponentialKernel` have dedicated learners in V1 (`GammaKernelLearner`, `ExponentialKernelLearner`). `DelayedExponentialKernel`, `LogNormalKernel`, and `ErlangKernel` can still be used as fixed kernels or as `candidate_type="fixed_kernel"` entries in `KernelCandidateSet` comparisons. See [learners.md](learners.md) for learner-specific documentation.
