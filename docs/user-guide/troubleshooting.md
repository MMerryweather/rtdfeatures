# Troubleshooting

## Common errors

| Error / Symptom | Likely Cause | Corrective Action |
|---|---|---|
| `Missing required columns: ...` | DataFrame doesn't have the expected columns | Check column names — they are case-sensitive. Ensure `input_col` and `target_col` exist in the DataFrame. |
| `max_lag (N steps) must be >= min_lag` | Lag window is reversed or misconfigured | Set `min_lag <= max_lag`; both are in time units (e.g. `"10m"`, `"20m"`). |
| `Insufficient rows for the specified` | Too few rows or lag window too wide given warmup requirements | Use more historical data or reduce `max_lag`. Minimum rows needed is roughly `max_lag_steps + 2`. |
| `order_by_time=True` error | Unsorted data without opt-in for auto-sort | Pass `order_by_time=True` when calling `.fit()` or pre-sort the DataFrame by time. |
| Irregular time grid detected | Time deltas between consecutive rows are non-uniform | Resample the time series to a regular grid before fitting. |
| `CUDA initialization` warning | PyTorch detecting environment without CUDA | Ignore — the package runs on CPU. Silence with `CUDA_VISIBLE_DEVICES=""`. |
| `RecoverableFoldError` | OOF fold training failed for one fold | Check fold configuration, data sufficiency per fold, and that each fold has enough variance. |
| Flat or constant target signal | Input signal has near-zero variance | Kernel learning requires signal variance. Check target data or use a different input/target pair. |
| `Generated feature name collision` | Duplicate kernel names or conflicting category columns | Check that kernel `name` values are unique. Verify `category_cols` don't overlap with each other. |
| Candidate selection fails — `succeeded_losses` is empty | No candidate kernel fits the data, or constraints are too tight | Relax candidate constraints (wider `max_lag`, more bootstrap samples) or check data quality. |
| Feature counts don't match expectation | FeatureRegistry vs generated feature table mismatch | Use `transform_result()` instead of `transform()`. Check `feature_registry.specs` length matches columns. |
| `polars` version compatibility | Polars API changed between releases | Pin polars to `>=0.20` per `pyproject.toml`. If using a newer version, check for renamed methods. |
| OOF `candidate_set` mismatch | `candidate_set.input_col` doesn't match the `input_col` argument | Ensure the candidate set's columns match the user-provided column names exactly. |
| `make_parametric_learned_kernel` with `shape_alpha < 1.0` at zero lag | Gamma kernel with shape < 1 diverges at zero | Use a minimum lag that excludes zero, or use shape_alpha >= 1.0. |
| `loss_tolerance_fraction` must be non-negative | Selected candidates tolerance is negative | Ensure candidate selection tolerance is >= 0. Common value: `0.05` (5%). |
| Kernel weights do not sum to one | Invalid kernel configuration or float precision on extreme weights | Re-check kernel construction. Known tolerance: `KERNEL_WEIGHT_SUM_TOLERANCE`. |
| `Unsupported generated feature naming pattern` | Feature name doesn't match expected parsed format | Switch to using `FeatureRegistry` for explicit feature spec instead of name-based parsing. |
| Kernel learner fit succeeds but transform produces all-null rows | All rows are within the warmup window | Check that the DataFrame has more rows than `max_lag_steps`. Warmup rows (before the first full lag window) produce null features. |

## Diagnostic warnings

| Warning | Meaning | Action |
|---|---|---|
| `Fit diagnostics indicate weak identifiability` | The kernel estimate is not uniquely determined by the data | Use regularisation or compare against baselines before interpreting. |
| `Baseline no_lag outperforms learned kernel` | A zero-lag shift explains the relationship better | The relationship may be instantaneous; consider a different input or reduce `min_lag`. |
| `Categorical interaction has sparse levels` | Some category levels have very few samples | Consider collapsing rare levels or using a different categorical encoding. |

## When to open an issue

Open a GitHub issue if:

- An error message lacks corrective guidance (we want every error to tell you what to do).
- A known failure mode is not documented in this table.
- You find a workaround that feels like it should be a proper fix.
- A regression from a previous version — include the version you upgraded from and to.

Before opening, search existing issues to avoid duplicates. Include:

- `rtdfeatures` version (`import rtdfeatures; print(rtdfeatures.__version__)`)
- Python version
- A minimal reproducible example (synthetic data preferred)
- Full traceback
