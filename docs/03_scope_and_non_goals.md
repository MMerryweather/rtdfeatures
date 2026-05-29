# Scope And Non-Goals

This document is normative for the stable package boundary.

## Stable Scope

`rtdfeatures` owns:

- Define constrained kernels
- Learn kernels from regular-grid process time series
- Compare kernels against package baselines
- Quantify kernel uncertainty
- Generate kernel-based features
- Define serialisable feature-plan contracts (execution surfaces are post-`v1.0`)
- Report feature evidence and diagnostics
- Compose kernels using kernel algebra
- Define feature-plan compatibility contracts (validation reports are post-`v1.0`)
- Return Polars DataFrames from public user-facing data APIs

## Non-Goals

`rtdfeatures` does not own:

- Final predictive modelling
- Historian connectors
- Real-time feature serving
- Forecasting workflows
- Control workflows
- Digital-twin workflows
- Full topology or genealogy modelling
- Graph traversal
- Flowsheet graph schema ownership
- Regime detection
- Motif mining
- TDA workflows
- SINDy workflows
- Conformal prediction
- MLOps orchestration
- Databricks, SageMaker, or MLflow replacement
- Process-specific API templates
- Hard-coded pyrometallurgy, flotation, or crushing feature names
- Causal discovery execution
- Pass-through wrappers around Tigramite methods

## Assumptions

- Users already have cleaned process tables
- Inputs are on a regular time grid
- Timestamps are aligned into a single table
- Input data is sorted by `time_col`, unless the caller opts into sorting
- Numeric and categorical columns are already standardised enough for modelling
- If `dt` is supplied, it must match the observed regular grid
- CPU execution is the design target

## Inclusion Test For New Features

Add something only if it directly helps:

- Define a kernel
- Learn a kernel
- Validate a kernel
- Compare kernels
- Compose kernels
- Generate features from kernels
- Validate a kernel feature plan
- Emit evidence and provenance for kernel-generated features
- Keep the package generic and domain-neutral
