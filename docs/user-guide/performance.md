# Performance

## Expected row counts

rtdfeatures is designed for regular-grid process time series typically ranging
from hundreds to hundreds of thousands of rows. Datasets with fewer than ~100
rows are too short to learn a meaningful kernel. Datasets with more than ~10^6
rows may benefit from sampling or lazy-mode pre-aggregation before calling
`transform()`.

## Lag-window cost

The number of lag steps `(max_lag - min_lag + 1)` directly controls the number
of learned kernel parameters. For simplex learners the parameter count equals
the number of lag steps. For parametric learners (gamma, exponential) the
parameter count is fixed (2–3), but the convolution window still widens with
`max_lag`.

- **Fitting**: simplex `O(n_rows * n_lag_steps * n_epochs)`; parametric
  `O(n_rows * n_lag_steps)` per epoch but epochs converge in similar order.
- **Transform**: `O(n_rows * n_lag_steps * n_features)` — linear in the lag
  window.
- **Recommendation**: keep `max_lag` as tight as the physical process
  justifies. A lag window of 10–50 steps is typical for most process
  applications.

## Feature-count cost

Each feature column added to `KernelFeatureBuilder` multiplies the transform
cost linearly. If you have 20 input columns and request weighted mean, weighted
std, and kernel age for each, the transform cost is approximately
`60 × O(n_rows * n_lag_steps)`.

## Categorical cardinality risk

Categorical features generate one output column per category level
(`O(n_levels)` per source column). High-cardinality categoricals (100+
distinct levels) can produce thousands of feature columns, increasing memory
and transform time proportionally.

**Recommendation**: group rare levels into an "Other" category, or limit
categorical features to columns with fewer than ~50 distinct levels.

## Memory behaviour

- The input Polars DataFrame is the primary memory footprint.
- Feature generation adds one column per generated feature.
  Approximate overhead: `n_rows × n_features × 8 bytes` for float columns.
- Large lag windows (`max_lag >> 1000`) increase internal convolution buffers.
- Diagnose memory via `TransformReport` (row count, feature count).

## CPU-only expectations

All core operations run on CPU via Polars and NumPy. No GPU is required or
expected.

## Torch use is for optimisation only

PyTorch is used **only** for the simplex-learner gradient-based optimisation
(`torch.nn.Parameter`, `Adam`, `loss.backward()`). GPU acceleration is neither
required nor enabled by default. The `CUDA_VISIBLE_DEVICES` environment
variable is set to `""` in all examples to suppress irrelevant CUDA warnings
from stale local drivers.

## What is not optimised yet

- No GPU acceleration for `transform()` (it runs via Polars on CPU).
- No multi-threaded kernel fitting (each learner runs single-threaded on CPU).
- No lazy-mode chaining inside `transform()` (the result is materialised).
- No sparse handling for high-cardinality categorical features.
- No incremental / streaming transform for datasets that exceed RAM.

These are not planned for the initial release but may be addressed in future versions if
the use case demands it.

## Smoke benchmark

A lightweight smoke benchmark is available at
`benchmarks/smoke_transform_performance.py`. It fits a simplex kernel and
times `transform()` across several dataset sizes to document order-of-growth.
