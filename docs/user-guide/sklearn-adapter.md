# sklearn adapter

`KernelFeatureTransformer` wraps the core API into a scikit-learn compatible transformer so you can use it in sklearn pipelines, cross-validation, and hyperparameter-search workflows.

Import from the integrations module:

```python
from rtdfeatures.integrations.sklearn import KernelFeatureTransformer
```

The transformer is **not** exported from the root `rtdfeatures` namespace. This keeps sklearn an optional dependency.

## When to use this

- You already use sklearn Pipelines, `GridSearchCV`, or `cross_validate` and want to slot kernel-based feature generation into that workflow.
- You want a quick way to convert a `pandas.DataFrame` with a time column into lag-weighted features without manually wiring `KernelFeatureBuilder`.
- You are prototyping with fixed (pre-specified) kernels and do not need the learner's diagnostic output.

## When to use the core API instead

- You need fit diagnostics, baseline comparisons, identifiability reports, or feature evidence — the core `SimplexKernelLearner.fit()` and `KernelFeatureBuilder.transform_result()` return structured result objects that carry this information; the sklearn adapter returns only the feature table.
- You need out-of-fold feature generation for leakage control — use the core out-of-fold utilities directly.
- You prefer a Polars-native workflow and do not need sklearn integration.

## Install

The adapter requires `scikit-learn >= 1.3` and `pandas >= 2.0`. Install the optional `sklearn` extra:

```bash
pip install rtdfeatures[sklearn]
```

For development installs:

```bash
pip install -e ".[sklearn]"
```

## Fixed-kernel mode

Pass a pre-built kernel (or dict of kernels) — no learning happens during `fit()`.

```python
from rtdfeatures.kernels import FixedDelayKernel
from rtdfeatures.integrations.sklearn import KernelFeatureTransformer

kernel = FixedDelayKernel(delay_steps=6, max_lag_steps=10, dt=1.0)
transformer = KernelFeatureTransformer(
    time_col="time",
    kernels={"feature": kernel},
    numeric_cols=["input_signal"],
)
transformer.fit(df)
result = transformer.transform(df)
```

Only `fit()` validates columns and stores the kernel dict. The actual feature generation happens in `transform()`.

## Learner mode

Pass a learner instance — `fit()` learns a kernel from the data.

```python
from rtdfeatures.learners import SimplexKernelLearner
from rtdfeatures.integrations.sklearn import KernelFeatureTransformer

learner = SimplexKernelLearner(max_lag="10m", max_epochs=20, seed=42)
transformer = KernelFeatureTransformer(
    time_col="time",
    learner=learner,
    input_col="input_signal",
    target_col="target_signal",
    numeric_cols=["input_signal"],
)
transformer.fit(df)
result = transformer.transform(df)
```

During `fit()` the learner fits the kernel. The learned kernel is stored under `self.kernels_` with the key `self.kernel_name` (default `"learned"`).

The sklearn adapter has separate fit and transform column requirements. Fit may require target columns for kernel learning; transform only requires columns needed to generate features.
Optional integrations are thin adapters around the core API. They adapt external ecosystem conventions without owning core feature-generation semantics.

## Pipeline example

```python
from sklearn.pipeline import Pipeline
from rtdfeatures.kernels import FixedDelayKernel
from rtdfeatures.integrations.sklearn import KernelFeatureTransformer

kernel = FixedDelayKernel(delay_steps=6, max_lag_steps=10, dt=1.0)
pipeline = Pipeline([
    ("features", KernelFeatureTransformer(
        time_col="time",
        kernels={"feature": kernel},
        numeric_cols=["input_signal"],
    )),
])
pipeline.fit(df)
result = pipeline.transform(df)
```

## Return types

Set `return_type` to control output format:

| `return_type` | Output type |
|---|---|
| `"pandas"` (default) | `pandas.DataFrame` |
| `"polars"` | `polars.DataFrame` |
| `"numpy"` | `numpy.ndarray` |

## Leakage warning

The sklearn adapter does not automatically make time-series cross-validation leakage-safe. If you learn kernels before splitting data, you may leak future information. Use time-aware splits and prefer the package out-of-fold utilities when generating training features from learned kernels.

## Limitations

- The adapter exposes only the feature-generation surface of the package. Diagnostics, baseline comparisons, identifiability reports, and feature evidence are not returned — use the core API when you need those.
- Input must be a named DataFrame (pandas or Polars). numpy arrays are rejected.
- Learner mode requires both `input_col` and `target_col`.
- Exactly one of `learner` or `kernels` must be provided, never both.
- The learned kernel(s) and transform report from the most recent call to `transform()` are stored as `self.fit_result_`, `self.kernels_`, `self.last_transform_report_`, and `self.feature_registry_` for optional inspection.
