# Out-of-Fold Feature Generation

Standard feature generation uses a kernel fitted on all available data. This leaks future information into every feature row — each row's features embed knowledge from the full time series.

Out-of-fold (OOF) generation prevents this by fitting kernels on chronologically earlier rows and generating features for held-out future rows.

## When to use OOF

- You are building features for a downstream model that will be evaluated on temporally separated train/test splits
- You want leakage-safe feature generation for cross-validation
- You need every feature row to be generated without access to future data relative to that row

## Quickstart

```python
from rtdfeatures import (
    ForwardChainingSplitConfig,
    SimplexKernelLearner,
    fit_transform_oof,
)

oof = fit_transform_oof(
    df=df,
    learner=SimplexKernelLearner(max_lag="6h"),
    split_config=ForwardChainingSplitConfig(
        n_folds=3,
        min_train_size=240,
        validation_size=80,
        gap=0,
    ),
    input_col="upstream_signal",
    target_col="downstream_signal",
    time_col="time",
    numeric_cols=["upstream_signal", "flowrate"],
    category_cols=["operating_mode"],
)
oof_features = oof.features
```

## Split configuration

`ForwardChainingSplitConfig` controls the fold boundaries:

| Field | Description |
|---|---|
| `n_folds` | Number of forward-chaining folds |
| `min_train_size` | Minimum rows in the training window |
| `validation_size` | Rows in each validation window |
| `gap` (optional) | Gap between train and validation windows (default 0) |
| `max_train_size` (optional) | Cap on training window size |

## OOF with candidate comparison

```python
from rtdfeatures import KernelCandidate, KernelCandidateSet

candidate_set = KernelCandidateSet(
    candidate_set_id="oof-demo",
    input_col="upstream_signal",
    target_col="downstream_signal",
    time_col="time",
    candidates=(
        KernelCandidate(candidate_id="simplex_6h", family="simplex",
                        candidate_type="empirical_learner",
                        min_lag="0m", max_lag="6h",
                        learner_parameters={"seed": 42}),
    ),
)

oof = fit_transform_oof(
    df=df, candidate_set=candidate_set,
    split_config=ForwardChainingSplitConfig(n_folds=3, min_train_size=240, validation_size=80),
    input_col="upstream_signal", target_col="downstream_signal", time_col="time",
    numeric_cols=["upstream_signal"],
)
```

## Result structure

`OutOfFoldKernelFeatureResult` contains:

- `features` — the stitched Polars DataFrame with OOF-generated features
- `fold_results` — per-fold fitting results
- `fold_reports` — per-fold transform reports
- `combined_transform_report` — aggregate report
- `feature_evidence_report` — evidence for OOF-generated features
- `split_summary` — fold boundary metadata
- `warnings` — any fold-level warnings

## Important notes

- Fold boundaries are strictly chronological (forward-chaining, not shuffled)
- Validation rows never appear in the fold's training rows
- Each held-out row only receives features from a kernel fitted without that row
- OOF is feature generation only — it does not train or evaluate downstream models
- Warmup rows within folds still produce `null` values

## See also

- [Generating features](generating-features.md) — non-OOF feature generation
- [08_api_design.md](../08_api_design.md) — normative OOF API contracts
