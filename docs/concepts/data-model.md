# Data Model

## Table format

All public inputs and outputs are `polars.DataFrame`.

## Input contract

- One row per timestep
- One `time_col` with a regular time grid
- Single aligned table (upstream orchestration handles joins, resampling, cleaning)
- Numeric columns already sufficiently cleaned for modelling
- Categorical columns already standardised for modelling

## Time grid

- If `dt` is omitted, it is inferred from the `time_col` when the grid is regular
- If `dt` is supplied, it must match the observed grid
- Irregular grids raise a clear error — no imputation
- `max_lag` and `min_lag` accept duration-like strings: `"5m"`, `"30m"`, `"2h"`, `"1d"`

## Row ordering

- Unsorted input raises by default
- Set `order_by_time=True` to opt into automatic sorting by `time_col`
- The package never silently reorders input

## Warmup

Features generated from a kernel require `max_lag` worth of history. Rows before this history window are **warmup rows** and produce `null` feature values. The row count is preserved — warmup rows are not dropped.

## Lag windows

- `min_lag` / `max_lag` define the admissible lag window (converted to integer steps via `dt`)
- `min_lag` excludes very recent history when physically justified (e.g. sensor mixing delays)
- Weighted features only use lags within `[min_lag, max_lag]`

## Missing data

- Rows with missing input or target values are excluded from training windows during fitting (dropped, not imputed)
- Feature generation emits `null` for any row where the kernel window contains missing values or the denominator is zero
- `TransformReport.missing_rows_by_feature` quantifies null coverage

## Categorical handling

- Categorical columns produce fraction features per level and an entropy feature
- Fraction features use kernel-weighted counts of each category level over the lag window
- See [Categorical genealogy](../user-guide/categorical-genealogy.md) for workflow details

## Output contract

- `transform()` returns only `[time_col + generated feature cols]`
- `augment_cols()` preserves original columns and appends generated features
- `diagnose_transform()` returns a `TransformReport` with diagnostics (does not change the feature output)

## Feature naming

Generated features follow a deterministic naming pattern:

```
{kernel_name}_num_{column}_wmean     # weighted mean
{kernel_name}_num_{column}_wstd      # weighted std
{kernel_name}_num_{column}_wsum      # weighted sum
{kernel_name}_cat_{column}_{level}_frac  # categorical fraction
{kernel_name}_cat_{column}_entropy   # categorical entropy
{kernel_name}_age_mean               # kernel mean lag
{kernel_name}_age_p50                # kernel median lag
{kernel_name}_age_p90                # kernel 90th percentile lag
```

## See also

- [06_data_model.md](../06_data_model.md) — normative data contract reference
- [05_feature_generation_design.md](../05_feature_generation_design.md) — feature definitions and algebra
