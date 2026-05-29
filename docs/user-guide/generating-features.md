# Generating Features

`KernelFeatureBuilder` applies one or more validated kernels to produce lag-aware feature columns.

TransformResult is the preferred auditable output. It keeps the feature table, transform diagnostics, and feature registry together.

## Basic usage

```python
from rtdfeatures import KernelFeatureBuilder

builder = KernelFeatureBuilder(
    kernels={"learned": fit.kernel},
    time_col="time",
    numeric_cols=["temperature", "pressure"],
    category_cols=["operating_mode"],
    weight_col="flowrate",          # optional
    age_tail_threshold=None,        # uses default tail-mass threshold
)

# Primary: auditable workflow with registry + report
result = builder.transform_result(df)
features = result.features
report = result.report
registry = result.feature_registry

# Simple: feature table only
features = builder.transform(df)
```

## Feature families

For each numeric column, the builder generates:

| Feature | Description |
|---|---|
| `{kernel}_num_{col}_wmean` | Weighted mean over lag window |
| `{kernel}_num_{col}_wstd` | Weighted standard deviation |
| `{kernel}_num_{col}_wsum` | Weighted sum |

For each categorical column:

| Feature | Description |
|---|---|
| `{kernel}_cat_{col}_{level}_frac` | Fraction of each category level |
| `{kernel}_cat_{col}_entropy` | Entropy of the categorical distribution |

For each kernel:

| Feature | Description |
|---|---|
| `{kernel}_age_mean` | Kernel mean lag |
| `{kernel}_age_p50` | Kernel median lag |
| `{kernel}_age_p90` | Kernel 90th percentile lag |
| `{kernel}_age_tail_gt_threshold` | Kernel tail mass above threshold |

## Weight column

When `weight_col` is provided (e.g. mass flow), weighted features use:

```
wmean_t = sum w_k * m_{t-k} * x_{t-k} / sum w_k * m_{t-k}
```

Without `weight_col`:

```
wmean_t = sum w_k * x_{t-k}
```

This makes `weight_col` appropriate for flow, throughput, or mass — any row-level contribution measure.

## Multiple kernels

Pass multiple kernels to generate features from different lag shapes in one call:

```python
builder = KernelFeatureBuilder(
    kernels={"simplex": simplex_fit.kernel, "gamma": gamma_fit.kernel},
    time_col="time",
    numeric_cols=["temperature"],
)
```

The output includes columns for both `simplex_num_temperature_wmean` and `gamma_num_temperature_wmean`.

## Warmup rows

Rows before `max_lag` are satisfied produce `null` values. The row count is preserved. Check `TransformReport.warmup_rows` for the count.

## Renaming features

The `rename` parameter allows you to remap generated feature names:

```python
builder = KernelFeatureBuilder(
    kernels={"learned": fit.kernel},
    time_col="time",
    numeric_cols=["feed"],
    rename={"learned_num_feed_wmean": "feed_lag_mean"},
)
```

Only specified feature names are renamed; unmapped names keep their default.

## See also

- [Out-of-fold generation](out-of-fold.md) — leakage-safe feature generation
- [Categorical genealogy](categorical-genealogy.md) — working with categorical source fractions
- [Feature evidence](feature-evidence.md) — attaching provenance to generated features
- [05_feature_generation_design.md](../05_feature_generation_design.md) — normative feature definitions
