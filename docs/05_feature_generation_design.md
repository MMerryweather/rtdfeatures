# Feature Generation Design

## Main Builder

- `KernelFeatureBuilder`
- Accepts any validated `Kernel` object, including learned and user-supplied fixed kernels
- Returns Polars-native outputs only

## Public Methods

- `transform(df, order_by_time=False) -> pl.DataFrame`
- `augment_cols(df, order_by_time=False) -> pl.DataFrame`
- `diagnose_transform(df, order_by_time=False) -> TransformReport`

`transform()` returns only `[time_col + generated feature cols]`.

`augment_cols()` preserves the original columns and appends generated features.

`transform()` and `diagnose_transform()` share the same internal execution path so diagnostics do not require a second independent implementation. `last_transform_report` caches the most recent report.

## Feature Generation Surfaces

Primary surface:

- `KernelFeatureBuilder` methods (`transform`, `augment_cols`, `diagnose_transform`)

Planned orchestration surface:

- `KernelFeaturePlan` executed through builder-compatible semantics

`KernelFeaturePlan` is orchestration over existing `KernelFeatureBuilder`
semantics. It must not change `transform()` return behavior.

## Row Alignment

- Keep the same row count as the input
- Preserve `time_col`
- Emit `null` for warmup rows
- Emit `null` when a denominator is zero or unusable
- Do not emit partial-window values

## Feature Families In `v0.1`

- `{kernel}_num_{column}_wmean`
- `{kernel}_num_{column}_wstd`
- `{kernel}_num_{column}_wsum`
- `{kernel}_cat_{column}_{level}_frac`
- `{kernel}_cat_{column}_entropy`
- `{kernel}_age_mean`
- `{kernel}_age_p50`
- `{kernel}_age_p90`
- `{kernel}_age_tail_gt_threshold`

## Feature Definitions

Let `w_k` be kernel weights and `x_t` the signal value.

### Numeric Weighted Mean

Without `weight_col`:

```text
wmean_t = sum_k w_k x_{t-k}
```

With `weight_col = m_t`:

```text
wmean_t =
    sum_k w_k m_{t-k} x_{t-k}
    / sum_k w_k m_{t-k}
```

### Numeric Weighted Sum

Without `weight_col`:

```text
wsum_t = sum_k w_k x_{t-k}
```

With `weight_col = m_t`:

```text
wsum_t = sum_k w_k m_{t-k} x_{t-k}
```

### Numeric Weighted Standard Deviation

Use the same denominator as `wmean`.

```text
wstd_t =
    sqrt(
        sum_k a_{t,k} (x_{t-k} - wmean_t)^2
        / sum_k a_{t,k}
    )
```

Where:

- `a_{t,k} = w_k` without `weight_col`
- `a_{t,k} = w_k m_{t-k}` with `weight_col`

### Categorical Fraction

For category level `L`:

```text
frac_t(L) =
    sum_k a_{t,k} I(category_{t-k} = L)
    / sum_k a_{t,k}
```

### Categorical Entropy

For category fractions `p_t(L)` across levels:

```text
entropy_t = - sum_L p_t(L) log(p_t(L))
```

Use `0` contribution for terms where `p_t(L) = 0`.

### Age Features

These come directly from the kernel:

- `age_mean`: kernel mean lag
- `age_p50`: kernel median lag
- `age_p90`: kernel 90th percentile lag
- `age_tail_gt_threshold`: kernel mass above a configured threshold

If `age_tail_threshold` is not supplied, use the default `tail_mass` threshold
defined in [07_validation_and_diagnostics.md](07_validation_and_diagnostics.md):

```text
min_lag + 0.75 * (max_lag - min_lag)
```

For static kernels these are constant across rows and still emitted as columns on every row.

## Deferred Features

- `wmin`
- `wmax`
- `recent_old_delta`
- `dominant_frac`

## FeatureSpec

A `FeatureSpec` is a serializable request to generate kernel-weighted features.

Proposed fields:

- `source_col`
- `kernel`
- `feature_family`
- `aggregations`
- `weight_col`
- `metadata`

`FeatureSpec` describes what to generate; it does not define learner fitting.

## KernelFeaturePlan

A `KernelFeaturePlan` groups kernel references and feature specs for one
execution boundary.

Proposed fields:

- `kernels`
- `specs`
- `observation_semantics`
- `column_roles`
- `metadata`

The plan remains domain-neutral and constrained to kernel feature generation.

## FeaturePlan Dry Run

Planned dry-run behavior:

- validate plan-level structure and references
- return expected output schema
- return warnings
- do not compute features

## Feature Naming Policy

Graph/planner-driven usage needs stable naming. A candidate convention is:

```text
{scope}__{source}__{kernel}__{feature_family}__{aggregation}
```

Exact naming-policy freeze remains versioned follow-up work.

## Observation Semantics In Feature Generation

- `rtdfeatures` does not resample time series
- it may warn when observation semantics are missing
- it may validate role/semantics compatibility when metadata is provided
- interval semantics labels are not converted into new timestamps unless a
  future explicit API adds that behavior

## Feature Evidence Contract (`v0.9`)

Feature evidence is a diagnostics/reporting contract attached to generated
feature definitions. It does not change feature values and does not select
downstream predictive-model features.

`FeatureEvidence` fields:

- `feature_name`
- `source_col`
- `feature_family`
- `kernel_name`
- `kernel_family`
- `kernel_summary`
- `fit_result_id`
- `candidate_id`
- `baseline_summary`
- `identifiability_warnings`
- `bootstrap_summary`
- `interpretation`
- `evidence_completeness`
- `metadata`

Interpretation labels (`interpretation`):

- `material_memory`
- `process_response`
- `statistical_pattern`
- `unknown`

Evidence completeness labels (`evidence_completeness`):

- `kernel_only`
- `fit_evidence`
- `comparison_evidence`
- `bootstrap_evidence`
- `full_evidence`

Missing optional evidence representation:

- use `None` for unavailable optional evidence fields:
  `fit_result_id`, `candidate_id`, `baseline_summary`,
  `identifiability_warnings`, `bootstrap_summary`

Validation contract:

- invalid `interpretation` or `evidence_completeness` labels must raise
  `ValueError` (fail closed)

## Out-Of-Fold Feature Generation Contract (`v0.95`)

Out-of-fold generation is a kernel-fitting and feature-generation workflow only.
It is not final predictive model training.

Forward-chaining is the default split strategy.

Public contract objects:

- `OutOfFoldKernelFeatureResult`
- `OutOfFoldSplitSummary`

`OutOfFoldKernelFeatureResult` fields:

- `features`
- `fold_results`
- `fold_reports`
- `combined_transform_report`
- `feature_evidence_report`
- `split_summary`
- `warnings`

`OutOfFoldSplitSummary` fields:

- `n_folds`
- `split_strategy`
- `fold_boundaries`
- `min_train_rows`
- `validation_rows_total`
- `rows_with_features`
- `rows_without_features`
- `warnings`

Warmup and row-output behavior follow the same transform contract:

- validation rows are transformed directly under the existing transform semantics
- kernel fitting still uses training rows only; validation rows are never used for fit
- fold warmup rows can remain `null` when lag windows are unavailable
- output rows remain aligned to input rows
- OOF validation rows are generated without using future rows
