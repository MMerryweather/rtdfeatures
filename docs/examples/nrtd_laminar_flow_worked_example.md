# nRTD Laminar-Flow Worked Example

Generated from `test_data/benchmarks/nrtd/hsa_000_laminar_flow_signals.parquet`.

## Introductory Time Series

![Laminar flow intro signals](generated/nrtd_laminar_intro_timeseries.png)

## Data Load

- rows: `200`
- columns: `['time', 'input_signal', 'target_signal']`
- inferred regular-grid step: `0.502513000` seconds

## Learner Setup

Fitted with public learners only on `input_signal -> target_signal` using `time`:
`SimplexKernelLearner`, `GammaKernelLearner`, `ExponentialKernelLearner`, `FixedDelayKernelLearner`.

## Fit Diagnostics And Baselines

| learner | validation_loss | no_lag | best_single_lag | mean_lag_s | warning_codes |
|---|---:|---:|---:|---:|---|
| `simplex` | 0.083533 | 0.083008 | 0.083008 | 19.830 | WEAK_NO_LAG_IMPROVEMENT,LARGE_VALIDATION_GAP,DIFFUSE_KERNEL |
| `gamma` | 0.083530 | 0.083008 | 0.083008 | 9.701 | WEAK_NO_LAG_IMPROVEMENT |
| `exponential` | 0.083720 | 0.083008 | 0.083008 | 8.522 | WEAK_NO_LAG_IMPROVEMENT |
| `fixed_delay` | 0.083008 | 0.083008 | 0.083008 | 26.633 | WEAK_NO_LAG_IMPROVEMENT,LARGE_VALIDATION_GAP |

## Fit Quality Plots

![Observed vs fitted response](generated/nrtd_laminar_observed_vs_fit.png)

![Recommended kernel profile](generated/nrtd_laminar_kernel_profile.png)

## Recommended Kernel For Feature Generation

- `recommended_kernel`: `fixed_delay`
- `recommendation_status`: `recommended`
- `recommendation_reason`: `lowest validation_loss among the fitted public learners`
- Fit RMSE: `0.003943`
- Fit MAE: `0.002691`
- Observed/predicted correlation: `0.9796`
- Fit evidence interpretation: correlation above 0.7 and low absolute error support a useful lag fit for feature generation.

## Generated Feature Preview

```text
shape: (8, 8)
┌────────────┬────────────┬────────────┬───────────┬───────────┬───────────┬───────────┬───────────┐
│ time       ┆ learned_nu ┆ learned_nu ┆ learned_n ┆ learned_a ┆ learned_a ┆ learned_a ┆ learned_a │
│ ---        ┆ m_input_si ┆ m_input_si ┆ um_input_ ┆ ge_mean   ┆ ge_p50    ┆ ge_p90    ┆ ge_tail_g │
│ datetime[μ ┆ gnal_wmean ┆ gnal_wstd  ┆ signal_ws ┆ ---       ┆ ---       ┆ ---       ┆ t_thresho │
│ s]         ┆ ---        ┆ ---        ┆ um        ┆ f64       ┆ f64       ┆ f64       ┆ ld        │
│            ┆ f64        ┆ f64        ┆ ---       ┆           ┆           ┆           ┆ ---       │
│            ┆            ┆            ┆ f64       ┆           ┆           ┆           ┆ f64       │
╞════════════╪════════════╪════════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════╡
│ T0 ┆ 0.999436   ┆ 0.0        ┆ 0.999436  ┆ 26.633189 ┆ 26.633189 ┆ 26.633189 ┆ 0.0       │
│ 00:01:36.4 ┆            ┆            ┆           ┆           ┆           ┆           ┆           │
│ 82496      ┆            ┆            ┆           ┆           ┆           ┆           ┆           │
│ T0 ┆ 0.999436   ┆ 0.0        ┆ 0.999436  ┆ 26.633189 ┆ 26.633189 ┆ 26.633189 ┆ 0.0       │
│ 00:01:36.9 ┆            ┆            ┆           ┆           ┆           ┆           ┆           │
│ 85009      ┆            ┆            ┆           ┆           ┆           ┆           ┆           │
│ T0 ┆ 0.999436   ┆ 0.0        ┆ 0.999436  ┆ 26.633189 ┆ 26.633189 ┆ 26.633189 ┆ 0.0       │
│ 00:01:37.4 ┆            ┆            ┆           ┆           ┆           ┆           ┆           │
│ 87522      ┆            ┆            ┆           ┆           ┆           ┆           ┆           │
│ T0 ┆ 0.999436   ┆ 0.0        ┆ 0.999436  ┆ 26.633189 ┆ 26.633189 ┆ 26.633189 ┆ 0.0       │
│ 00:01:37.9 ┆            ┆            ┆           ┆           ┆           ┆           ┆           │
│ 90035      ┆            ┆            ┆           ┆           ┆           ┆           ┆           │
│ T0 ┆ 0.999436   ┆ 0.0        ┆ 0.999436  ┆ 26.633189 ┆ 26.633189 ┆ 26.633189 ┆ 0.0       │
│ 00:01:38.4 ┆            ┆            ┆           ┆           ┆           ┆           ┆           │
│ 92548      ┆            ┆            ┆           ┆           ┆           ┆           ┆           │
│ T0 ┆ 0.999436   ┆ 0.0        ┆ 0.999436  ┆ 26.633189 ┆ 26.633189 ┆ 26.633189 ┆ 0.0       │
│ 00:01:38.9 ┆            ┆            ┆           ┆           ┆           ┆           ┆           │
│ 95061      ┆            ┆            ┆           ┆           ┆           ┆           ┆           │
│ T0 ┆ 0.999436   ┆ 0.0        ┆ 0.999436  ┆ 26.633189 ┆ 26.633189 ┆ 26.633189 ┆ 0.0       │
│ 00:01:39.4 ┆            ┆            ┆           ┆           ┆           ┆           ┆           │
│ 97574      ┆            ┆            ┆           ┆           ┆           ┆           ┆           │
│ T0 ┆ 0.999436   ┆ 0.0        ┆ 0.999436  ┆ 26.633189 ┆ 26.633189 ┆ 26.633189 ┆ 0.0       │
│ 00:01:40.0 ┆            ┆            ┆           ┆           ┆           ┆           ┆           │
│ 00087      ┆            ┆            ┆           ┆           ┆           ┆           ┆           │
└────────────┴────────────┴────────────┴───────────┴───────────┴───────────┴───────────┴───────────┘
```

## Boundary: nRTD Fixture Scope

This repository currently supports end-to-end learning from nRTD fixtures only for
`laminar_flow` because it has a trusted input/target signal-pair fixture.

`adler`, `cholette`, and `dispersion` remain reference-only benchmark context and
must not be treated as learned-feature workflows until trusted signal-pair fixtures
are added.