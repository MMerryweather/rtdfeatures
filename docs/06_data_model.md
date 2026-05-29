# Data Model

## Table Standard

Public inputs and outputs are `polars.DataFrame`.

```python
import polars as pl
```

## Input Contract

- One row per timestep
- One `time_col`
- Regular time grid
- Single aligned table
- Numeric columns already cleaned enough for modelling
- Categorical columns already standardised enough for modelling

## Prepared Modelling Table Contract

Upstream orchestration is expected to handle:

- historian extraction
- resampling
- joins
- unit cleaning
- bad-value handling
- lab/event alignment

`rtdfeatures` validates only the contracts required for kernel learning and
feature generation.

## Ordering Contract

- Default: unsorted input raises a clear error
- Opt-in: `order_by_time=True` sorts by `time_col` and continues
- The package does not silently reorder input

## Time Grid Contract

- If `dt` is omitted, infer it from `time_col` when the grid is regular
- If `dt` is supplied, validate that it matches the observed grid
- If the grid is irregular, raise a clear error
- `max_lag` and `min_lag` are converted to integer lag steps using `dt`

Accepted duration-like strings:

- `"5m"`
- `"30m"`
- `"2h"`
- `"1d"`

## Naming Principles

Prefer generic, tidy-style names:

- `time_col`
- `input_col`
- `target_col`
- `numeric_cols`
- `category_cols`
- `weight_col`

Avoid domain-locked names in the core API.

## `weight_col` Semantics

`weight_col` may represent mass flow, volumetric flow, throughput, sample weight, or another contribution weight. The builder uses it only as a weighting signal; it does not solve plant-wide material balance.

## Column Role

Column roles are optional metadata used by validation and leakage checks.

Candidate role labels:

- `input_tag`
- `target`
- `lab_label`
- `event`
- `setpoint`
- `controller_output`
- `downstream_measurement`
- `future_known`
- `forbidden`

Role metadata supports guardrails; it does not create a full feature-store
ownership boundary in this package.

## Observation Semantics

Observation semantics are optional metadata describing timestamp interpretation
and sample basis for a column.

Example label vocabulary:

- `timestamp_type`: `instant`, `interval_start`, `interval_end`, `interval_midpoint`
- `sample_basis`: `online`, `grab`, `composite`, `event`, `derived`

These labels inform warnings/validation only. They do not imply automatic
resampling or time realignment.

## TargetHorizon

`TargetHorizon` is a future-facing validation concept for documenting how
targets relate to prediction horizons. It is metadata only in this boundary.

No prediction-horizon modeling logic is implied inside `rtdfeatures` unless
added by a later versioned plan.

## Weighting Basis

When weighting metadata is provided, supported basis labels are:

- `time`
- `wet_mass`
- `dry_mass`
- `volume`
- `energy`
- `unknown`

These labels keep the package domain-neutral while supporting broader process
contexts.
