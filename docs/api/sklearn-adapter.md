# KernelFeatureTransformer API

## Import path

```python
from rtdfeatures.integrations.sklearn import KernelFeatureTransformer
```

The class is **not** exported from the root `rtdfeatures` namespace so that sklearn remains an optional dependency.

## Constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `time_col` | `str` | (required) | Name of the time column in the input DataFrame. |
| `learner` | `Any \| None` | `None` | A learner instance (e.g. `SimplexKernelLearner`). Mutually exclusive with `kernels`. |
| `kernels` | `dict[str, Any] \| None` | `None` | A dict of pre-built kernels keyed by name. Mutually exclusive with `learner`. |
| `input_col` | `str \| None` | `None` | Input signal column name — required in learner mode. |
| `target_col` | `str \| None` | `None` | Target signal column name — required in learner mode. |
| `numeric_cols` | `list[str] \| None` | `None` | Numeric columns to generate lag-weighted features for. |
| `category_cols` | `list[str] \| None` | `None` | Categorical columns to generate fraction features for. |
| `weight_col` | `str \| None` | `None` | Optional weight column passed through to `KernelFeatureBuilder`. |
| `kernel_name` | `str` | `"learned"` | Key used to store the learned kernel in `self.kernels_`. |
| `order_by_time` | `bool` | `False` | Sort input by time column before fitting/transforming. |
| `return_type` | `str` | `"pandas"` | Output format: `"pandas"`, `"polars"`, or `"numpy"`. |
| `include_time_col` | `bool` | `False` | Whether the time column appears in the output. |
| `passthrough` | `bool` | `False` | If `True`, original columns are kept and feature columns are appended. |

## Fixed-kernel mode

When `kernels` is provided, `fit()` validates columns and stores the kernel dict as `self.kernels_`. No learning occurs. `transform()` builds features using the stored kernels.

```python
transformer = KernelFeatureTransformer(
    time_col="time",
    kernels={"feature": kernel},
    numeric_cols=["input_signal"],
)
transformer.fit(df)
result = transformer.transform(df)
```

## Learner mode

When `learner` is provided, `fit()` calls `learner.fit()` and stores the result as `self.fit_result_` and the learned kernel dict as `self.kernels_` under the key `self.kernel_name`.

```python
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

The sklearn adapter has separate fit and transform column requirements. Fit may require target columns for kernel learning; transform only requires columns needed to generate features.
Optional integrations are thin adapters around the core API. They adapt external ecosystem conventions without owning core feature-generation semantics.

## Attributes after fit

| Attribute | Type | Description |
|---|---|---|
| `kernels_` | `dict` | Kernel dict used for transform (pre-built or learned). |
| `fit_result_` | `KernelFitResult \| None` | Fit result from learner (only in learner mode). |
| `feature_names_in_` | `np.ndarray` | Column names from the input. |
| `n_features_in_` | `int` | Number of input columns. |

## Attributes after transform

| Attribute | Type | Description |
|---|---|---|
| `feature_names_out_` | `np.ndarray` | Names of the output columns. |
| `last_transform_report_` | `TransformReport` | Diagnostics from the most recent `transform()` call. |
| `feature_registry_` | `FeatureRegistry` | Structured metadata per generated column. |

## Return types

| `return_type` | Output type | Notes |
|---|---|---|
| `"pandas"` | `pandas.DataFrame` | Default; integrates with sklearn pipelines. |
| `"polars"` | `polars.DataFrame` | Native format; no index reset. |
| `"numpy"` | `numpy.ndarray` | Column order matches `feature_names_out_`. |

## Limitations

- Not exported from `rtdfeatures` root — must import from `rtdfeatures.integrations.sklearn`.
- Input must be a named pandas or Polars DataFrame — numpy arrays are rejected.
- Learner mode requires both `input_col` and `target_col`.
- Exactly one of `learner` or `kernels` must be provided.
- Diagnostics, baseline comparisons, and feature evidence are not returned — use the core API when those are needed.
