# Quickstart

This walkthrough fits a kernel from an input signal to a target signal, validates it against baselines, and generates lag-aware features.

```python
from rtdfeatures import KernelFeatureBuilder, SimplexKernelLearner
from rtdfeatures.synthetic import make_single_delay_dataset

# --- 1. Synthetic data on a regular grid ---
dataset = make_single_delay_dataset(n_rows=120, dt=60.0, seed=7)
df = dataset.data

# --- 2. Learn a constrained kernel ---
learner = SimplexKernelLearner(max_lag="20m")
fit = learner.fit(
    df,
    input_col="input_signal",
    target_col="target_signal",
    time_col="time",
)

# --- 3. Inspect diagnostics ---
print("Validation loss:", fit.fit_diagnostics.validation_loss)
print("Mean lag (s):", fit.kernel.mean_lag())
print("Identifiability warnings:", fit.identifiability_report.warnings)
print("Baseline comparison:", fit.baseline_comparison.summary_by_baseline)

# --- 4. Generate features ---
builder = KernelFeatureBuilder(
    kernels={"learned": fit.kernel},
    time_col="time",
    numeric_cols=["input_signal"],
)
result = builder.transform_result(df)
features = result.features
report = result.report
registry = result.feature_registry

# --- 5. Inspect results ---
print(f"Generated features: {[c for c in features.columns if c != 'time']}")
print(f"Feature columns: {report.feature_names}")
print(f"Warmup rows: {report.warmup_rows}")
print(f"Feature registry entries: {len(registry)}")

# --- 6. Output ---
# Features is a polars.DataFrame with time_col + generated feature cols.
# Warmup rows (before max lag is satisfied) contain null feature values.
assert "time" in features.columns
assert "learned_num_input_signal_wmean" in features.columns
assert report.row_count == df.height
```

TransformResult is the preferred auditable output. It keeps the feature table, transform diagnostics, and feature registry together.

## What just happened

1. **Learn**: `SimplexKernelLearner` found a non-negative, sum-to-one, causal weighting over the last 20 minutes of input history that best predicts the current target.
2. **Validate**: `fit_diagnostics`, `identifiability_report`, and `baseline_comparison` tell you whether the learned kernel is trustworthy.
3. **Generate**: `KernelFeatureBuilder.transform_result()` applied the kernel to produce a weighted moving average (and other feature families) as a new Polars DataFrame, along with a `TransformReport` and a `FeatureRegistry` for full auditability.

## Next steps

- [Fitting kernels](user-guide/fitting-kernels.md) — parametric learners, shared multi-pair fitting
- [Comparing kernels](user-guide/comparing-kernels.md) — baseline and candidate comparison
- [Generating features](user-guide/generating-features.md) — categorical features, weight columns, age features
- [Concepts: kernels and RTDs](concepts/kernels-and-rtds.md) — understanding what a kernel represents
