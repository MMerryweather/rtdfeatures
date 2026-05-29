# Features API

## KernelFeatureBuilder

Main feature generation class.

Generated feature arrays and `FeatureSpec` metadata share one source of truth.
Specs are created at feature-generation time; registry construction must not
parse feature names to recover metadata.

```python
from rtdfeatures import KernelFeatureBuilder

builder = KernelFeatureBuilder(
    kernels={"learned": my_kernel},
    time_col="time",
    numeric_cols=["temperature", "pressure"],
    category_cols=["operating_mode"],
    weight_col=None,               # optional
    age_tail_threshold=None,       # optional, defaults to diagnostics contract
)
```

### Methods

#### Primary workflows

**`transform_result(df, order_by_time=False) -> TransformResult`**

TransformResult is the preferred auditable output. It keeps the feature table, transform diagnostics, and feature registry together. Returns a `TransformResult` containing `features`, `report`, and `feature_registry`. Updates `builder.last_transform_report`.

**`transform(df, order_by_time=False) -> pl.DataFrame`**

Returns a feature table containing the time column plus generated feature columns. Updates `builder.last_transform_report`. Use this when you only need the feature table without audit metadata.

#### Secondary methods

**`transform_with_report(df, order_by_time=False) -> tuple[pl.DataFrame, TransformReport]`**

Returns `(features, report)`. Updates `builder.last_transform_report`. This method is supported in V1 and is not deprecated.

**`augment_cols(df, order_by_time=False) -> pl.DataFrame`**

Preserves all original columns and appends generated features.

**`diagnose_transform(df, order_by_time=False) -> TransformReport`**

Returns transform diagnostics. Does not return feature columns. Updates `builder.last_transform_report`.

## TransformReport

Fields: `row_count`, `output_row_count`, `warmup_rows`, `feature_names`, `missing_rows_by_feature`, `zero_denominator_rows_by_feature`, `warmup_unusable_summary`, `collision_naming_summary`.

## Output schema

Generated columns follow:

```
{kernel}_num_{col}_wmean       # weighted mean
{kernel}_num_{col}_wstd        # weighted std dev
{kernel}_num_{col}_wsum        # weighted sum
{kernel}_cat_{col}_{level}_frac  # categorical fraction per level
{kernel}_cat_{col}_entropy     # categorical entropy
{kernel}_age_mean              # kernel mean lag
{kernel}_age_p50               # kernel median lag
{kernel}_age_p90               # kernel 90th percentile lag
{kernel}_age_tail_gt_threshold # kernel tail mass
```

## Feature evidence helpers

`FeatureRegistry` convenience helpers:

- `registry.names()` returns ordered feature names.
- `registry.to_frame()` returns one row per feature spec as a Polars table.

```python
from rtdfeatures.features import (
    build_feature_evidence,
    feature_evidence_table,
    feature_evidence_compact_dict,
    feature_evidence_compact_text,
)
```

See [evidence docs](evidence.md) for details.
