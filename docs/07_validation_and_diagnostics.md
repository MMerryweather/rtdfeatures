# Validation And Diagnostics

Diagnostics are part of the public contract. They are not optional polish.

## Result Objects

- `KernelFitResult`
- `FitDiagnostics`
- `IdentifiabilityReport`
- `BaselineComparison`
- `TransformReport`
- `SharedPairFitResult`
- `SharedKernelFitResult`
- `KernelCandidate`
- `KernelCandidateSet`
- `KernelFamilyFitResult`
- `KernelComparisonResult`
- `KernelSelectionResult`
- `BootstrapResult`
- `BootstrapWeightSample`
- `BootstrapParameterSample`
- `BootstrapLagSummarySample`
- `KernelBootstrapSummary`
- `ParameterUncertaintySummary`
- `WeightUncertaintySummary`
- `FeatureEvidence`
- `FeatureEvidenceReport`

## `KernelFitResult`

Contains:

- `kernel`
- `fit_diagnostics`
- `identifiability_report`
- `baseline_comparison`

`v0.5` parametric learners return the same `KernelFitResult` for successful
fits. Parametric-only metadata is additive and must not remove or rename
existing fields.

Parametric metadata access path:

- `fit.fit_provenance["parametric_family"]`
- `fit.fit_provenance["parametric_parameters"]`
- `fit.fit_provenance["parametric_initial_parameters"]`
- `fit.fit_provenance["parametric_conversion_status"]`
- `fit.fit_provenance["parametric_conversion_message"]`

These are additive provenance fields on `KernelFitResult`, not `FitDiagnostics`
fields.

## `SharedPairFitResult` (`v0.2`)

Contains one shared-learning pair outcome:

- `pair_id`
- `input_col`
- `target_col`
- `fit_result` (`KernelFitResult | None`)
- `error` (`str | None`)
- `succeeded` property

If a pair fails, `fit_result` is `None` and `error` is populated.

## `SharedKernelFitResult` (`v0.2`)

Contains:

- `pairs`: ordered tuple of `SharedPairFitResult`

Accessors:

- `pair_ids()` for stable ordered identifiers
- `get_pair(pair_id)` for pair-level outcome inspection
- `get_pair_result(pair_id)` for successful per-pair `KernelFitResult`
- `summary()` for all per-pair kernels and diagnostics, including failed pairs

## Candidate And Comparison Result Objects (`v0.7`)

`v0.7` adds descriptor and comparison contracts for kernel-only candidate
evaluation. These objects do not perform final predictive model selection.

### `KernelCandidate`

Serializable descriptor for one candidate configuration with fields:

- `candidate_id`
- `family`
- `candidate_type` (`fixed_kernel`, `empirical_learner`, `parametric_learner`, `baseline`)
- `min_lag`
- `max_lag`
- `fixed_parameters`
- `learner_parameters`
- `interpretation_hint`
- `metadata`

Contract:

- descriptor only: no live learner/kernel Python objects
- parameter/metadata payloads must be JSON-serializable
- fixed-kernel candidates require `fixed_parameters`
- empirical/parametric learner candidates require `learner_parameters`
- baseline candidates do not carry learner or fixed-kernel parameters

### `KernelCandidateSet`

Represents one input/target/time comparison scope with fields:

- `candidate_set_id`
- `input_col`
- `target_col`
- `time_col`
- `candidates`
- `baseline_names`
- `selection_metric`
- `metadata`

Contract:

- `candidates` must be non-empty
- candidate IDs must be unique
- `input_col`, `target_col`, and `time_col` are required non-empty metadata
- baseline names, when provided, are unique
- `selection_metric` is fail-closed and currently supports only `validation_loss`
- each `baseline_names` value must match at least one baseline candidate `family`

### `KernelFamilyFitResult`

Represents one candidate outcome with fields:

- `candidate`
- `fit_result` (`KernelFitResult | None`)
- `succeeded`
- `error`
- `is_parametric`
- `is_empirical`
- `is_baseline`
- `n_parameters`
- `validation_loss` (`float | None`)
- `train_loss` (`float | None`)
- `warning_codes` (`tuple[str, ...]`)
- `evaluated_fixed_kernel` (`Kernel | None`)

For fixed-kernel/baseline rows, `fit_result` may be `None`; failures must still
be explicit via `succeeded=False` and a non-empty `error`.

Conservative selection gate contract:

- learner candidates (`fit_result` present) must pass baseline and
  identifiability reliability gates from `KernelFitResult`.
- fixed-kernel candidates (`fit_result=None`) are selection-eligible only when
  explicit fixed evidence is present:
  `evaluated_fixed_kernel`, `fixed_baseline_comparison`, and
  `evaluation_provenance`.
- baseline candidates are never selected.
- selection context comparability fails closed: if successful non-baseline
  finite-loss candidates mix present and missing evaluation-context signatures,
  no recommendation is returned with explicit warnings.

### `KernelComparisonResult`

Represents full candidate-set outcomes with fields:

- `candidate_set`
- `family_results`
- `comparison_table`
- `warnings`
- `selection_summary`

Failed candidates are first-class rows in `family_results`.

### `KernelSelectionResult`

Optional kernel-only selection summary with fields:

- `selected_candidate_id`
- `selected_kernel`
- `selected_fit_result`
- `selection_reason`
- `selection_warnings`
- `all_candidates`

Selection is optional. When no recommendation is made, selected fields may be
`None`.

## Feature Plan Validation And Manifest Contracts (`v1.5`/`v1.6`, provisional)

Feature-plan diagnostics extend plan execution checks; they do not replace
learner-fit diagnostics.

Version scope:

- `FeaturePlanManifest` is planned for `v1.5` and is not part of the shipped
  `v1.0` contract.
- `FeaturePlanValidationReport` is planned for `v1.6` and is not part of the
  shipped `v1.0` contract.
- Field lists below are provisional and may change before those versions ship.

### `FeaturePlanValidationReport`

Contains:

- `valid`
- `errors`
- `warnings`
- `feature_count`
- `missing_columns`
- `duplicate_feature_names`
- `invalid_kernel_refs`
- `column_role_warnings`
- `observation_semantics_warnings`
- `warmup_risk_summary`

### `FeaturePlanManifest`

Contains:

- `plan_id`
- `plan_hash`
- `created_at`
- `package_versions`
- `input_schema_hash`
- `kernel_registry_hash`
- `feature_count`
- `output_schema`
- `warnings`

## Validation Is Not MLOps

`rtdfeatures` emits validation artifacts and manifests.

External systems (for example Databricks, SageMaker, MLflow, or internal
tooling) store and compare those artifacts.

## Leakage Checks Within Package Scope

`rtdfeatures` enforces only checks inside package scope:

- no future lag usage
- learned kernels fit using declared train/validation boundaries (OOF where requested)
- target/forbidden columns blocked as feature sources when column roles are provided
- warning when observation semantics are missing for lab/event/derived columns
- warning when downstream measurement roles are disallowed by config

These checks reduce leakage risk but do not prove complete leakage absence.

## `FitDiagnostics`

Contains:

- `train_loss`
- `validation_loss`
- `input_variance`
- `target_variance`
- `kernel_weight_sum`
- `mean_lag`
- `p50_lag`
- `p90_lag`
- `tail_mass`
- `boundary_mass_fraction`

`v0.5` keeps `FitDiagnostics` schema stable; parametric metadata is stored in
`KernelFitResult.fit_provenance` as additive fields.

## `IdentifiabilityReport`

Contains warnings such as:

- Input is too flat
- Target signal is too flat or too noisy
- Validation loss is much worse than training loss
- Kernel piles mass at the minimum or maximum lag boundary
- Kernel is too diffuse to interpret confidently
- `best_single_lag` beats the learned kernel

`v0.3` additive fields:

- `warning_codes`: stable identifiers aligned to each warning message.
- `warning_severity_by_code`: deterministic severity mapping for emitted codes.

Current stable warning code map:

- `INPUT_TOO_FLAT` -> "Input is too flat." (`high`)
- `TARGET_TOO_FLAT` -> "Target signal is too flat." (`high`)
- `WEAK_NO_LAG_IMPROVEMENT` -> "Target signal appears noisy or weakly explained." (`medium`)
- `LARGE_VALIDATION_GAP` -> "Validation loss is much worse than training loss." (`high`)
- `BOUNDARY_PILED_KERNEL` -> "Kernel piles mass at the lag boundary." (`medium`)
- `DIFFUSE_KERNEL` -> "Kernel is too diffuse to interpret confidently." (`medium`)
- `BEST_SINGLE_LAG_BEATS_LEARNED` -> "best_single_lag beats the learned kernel." (`medium`)
- `UNIFORM_BASELINE_BEATS_LEARNED` -> "uniform baseline beats the learned kernel." (`medium`)
- `EXPONENTIAL_BASELINE_BEATS_LEARNED` -> "exponential baseline beats the learned kernel." (`medium`)

## Bootstrap Uncertainty Contract (`v0.8`)

Bootstrap scope and boundary:

- default bootstrap strategy is blocked bootstrap over valid training lag-window rows
- bootstrap samples lag-window rows, not raw dataframe rows
- validation lag-window rows remain fixed by default
- bootstrap outputs uncertainty indicators, not causal proof
- bootstrap outputs are not proof of physical RTD truth
- failed iterations are retained and reported

Default interval quantiles:

- `DEFAULT_BOOTSTRAP_INTERVAL_QUANTILES = (0.025, 0.975)`

### `BootstrapResult`

Contains:

- `n_bootstrap`
- `n_succeeded`
- `n_failed`
- `failures`
- `weight_samples`
- `parameter_samples`
- `lag_summary_samples`
- `family_selection_counts`
- `warnings`
- `bootstrap_config`

`failures` retention contract:

- each failed iteration is preserved as one failure record in `failures`
- failures are never silently dropped

### `BootstrapWeightSample`

One lag-weight record from one bootstrap iteration:

- `bootstrap_id`
- `candidate_id`
- `lag_step`
- `lag_time`
- `weight`

### `BootstrapParameterSample`

One parameter record from one bootstrap iteration:

- `bootstrap_id`
- `candidate_id`
- `parameter_name`
- `parameter_value`

Missing parameter-sample contract:

- missing parameters are represented as `null` (`None`) `parameter_value`
- rows are retained instead of dropped when provenance is incomplete

### `BootstrapLagSummarySample`

One lag-summary record from one bootstrap iteration:

- `bootstrap_id`
- `candidate_id`
- `mean_lag`
- `p50_lag`
- `p90_lag`
- `tail_mass`

### `KernelBootstrapSummary`

Aggregate interval summary:

- `mean_lag_interval`
- `p50_lag_interval`
- `p90_lag_interval`
- `tail_mass_interval`
- `weight_interval_by_lag`
- `parameter_interval_by_name`
- `stability_score`

### `ParameterUncertaintySummary`

- `parameter_name`
- `estimate`
- `lower`
- `upper`
- `bootstrap_std`
- `n_samples`

### `WeightUncertaintySummary`

- `lag_step`
- `lag_time`
- `weight_estimate`
- `lower`
- `upper`
- `bootstrap_std`

### Bootstrap Warning Codes

- `BOOTSTRAP_TOO_FEW_SUCCESSES`
- `BOOTSTRAP_WEIGHT_UNSTABLE`
- `BOOTSTRAP_PARAMETER_UNSTABLE`
- `BOOTSTRAP_PARAMETER_PROVENANCE_MISSING`
- `BOOTSTRAP_LAG_SUMMARY_UNSTABLE`
- `BOOTSTRAP_FAMILY_UNSTABLE`
- `BOOTSTRAP_INTERVAL_TOUCHES_BOUNDARY`
- `BOOTSTRAP_VALIDATION_WINDOW_CHANGED`
- `BOOTSTRAP_CONTEXT_MISMATCH`
- `BOOTSTRAP_BLOCK_LENGTH_INVALID`

## Feature Evidence Contract (`v0.9`)

`FeatureEvidence` is a per-feature evidence descriptor with fields:

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

`FeatureEvidenceReport` is an aggregate summary with fields:

- `feature_evidence`
- `feature_count`
- `kernel_count`
- `source_columns`
- `warning_summary`
- `evidence_summary_by_kernel`
- `evidence_summary_by_feature_family`

Interpretation and completeness are separate dimensions:

- `interpretation` answers what the kernel appears to mean:
  `material_memory`, `process_response`, `statistical_pattern`, `unknown`
- `evidence_completeness` answers how much support exists:
  `kernel_only`, `fit_evidence`, `comparison_evidence`,
  `bootstrap_evidence`, `full_evidence`

Conservative contract rules:

- feature evidence is descriptive metadata, not downstream model-feature
  selection
- feature evidence does not change `transform()` outputs
- missing optional evidence is represented as `None` for:
  `fit_result_id`, `candidate_id`, `baseline_summary`,
  `identifiability_warnings`, `bootstrap_summary`
- invalid labels for `interpretation` or `evidence_completeness` raise
  `ValueError`
- `FeatureEvidenceReport.feature_count` must equal
  `len(feature_evidence)`; mismatches raise `ValueError`

## Interpretation And Evidence Separation

Interpretation classes:

- `material_memory`
- `process_response`
- `statistical_pattern`
- `unknown`

Evidence completeness classes:

- `kernel_only`
- `fit_evidence`
- `comparison_evidence`
- `bootstrap_evidence`
- `full_evidence`

## `v0.1` Default Diagnostic Heuristics

These defaults are intentionally conservative. They are warning heuristics, not
formal statistical tests.

Compute learner diagnostics on the same valid windows used for fitting after
the learner's scaling step. Baseline-loss comparisons use validation loss for
the learner's configured loss function.

`v0.5` rule: parametric diagnostics are computed on the same post-scaling valid
windows and blocked validation split used by simplex learners.

Named defaults:

- `flat_variance_threshold = 1e-8`
- `validation_gap_ratio = 2.0`
- `baseline_improvement_margin = 0.05`
- `boundary_mass_threshold = 0.35`
- `diffuse_entropy_fraction = 0.85`
- `diffuse_max_weight_threshold = 0.20`
- `tail_mass_fraction_of_lag_window = 0.75`

Warnings:

- Input is too flat when `input_variance < flat_variance_threshold`.
- Target is too flat when `target_variance < flat_variance_threshold`.
- Target is noisy or weakly explained when the learned validation loss improves
  on `no_lag` by less than `baseline_improvement_margin`:

  ```text
  (no_lag_loss - validation_loss) / max(no_lag_loss, flat_variance_threshold)
    < baseline_improvement_margin
  ```

- Validation loss is much worse than training loss when
  `validation_loss / max(train_loss, flat_variance_threshold) >
  validation_gap_ratio`.
- Kernel mass is boundary-piled when either the minimum-lag weight or the
  maximum-lag weight is at least `boundary_mass_threshold`.
- Kernel is diffuse when its normalized entropy is at least
  `diffuse_entropy_fraction` and its largest single lag weight is no greater
  than `diffuse_max_weight_threshold`.
- `best_single_lag` beats the learned kernel when its validation loss is at
  least `baseline_improvement_margin` lower than the learned validation loss:

  ```text
  (validation_loss - best_single_lag_loss)
    / max(validation_loss, flat_variance_threshold)
    >= baseline_improvement_margin
  ```

Normalized kernel entropy is:

```text
entropy = -sum_k weight_k * log(weight_k)
normalized_entropy = entropy / log(number_of_admissible_lag_steps)
```

Terms with `weight_k = 0` contribute `0`.

The default threshold for `tail_mass` is:

```text
min_lag + tail_mass_fraction_of_lag_window * (max_lag - min_lag)
```

Implementations may expose these as advanced configuration later, but `v0.1`
should first ship stable named defaults with tests.

## `BaselineComparison`

In `v0.2`, contains:

- `no_lag`
- `best_single_lag`
- `uniform`
- `exponential`
- The learner's validation loss for each
- The configured primary ranking metric

`uniform` and `exponential` are baseline comparators only; they are not
parametric learners.

`v0.5` note: `ExponentialKernelLearner` is a learner family. The existing
`exponential` baseline remains a fixed comparator and is not this learner.

Primary ranking metric:

- Validation loss using the learner's configured loss function
- Lower is better

Failed parametric parameter fits:

- raise clear fit-time errors for invalid parameters or failed
  parametric-to-discrete conversion
- may emit explicit warnings in `IdentifiabilityReport` for weak but valid fits
- never silently fall back to `SimplexKernelLearner` or a baseline method

Gamma zero-lag constraint:

- when the admissible lag grid includes zero lag (`min_lag=0`), gamma fits
  require `shape_alpha > 1.0`
- providing `init_shape_alpha <= 1.0` in that case raises `ValueError` with a
  clear message indicating the bound is strictly greater than `1.0`
- a zero-only lag grid (`min_lag=0`, `max_lag=0`) raises `ValueError` during
  fit setup before optimization

## Parametric Diagnostics Interpretation Guide (`v0.5`)

For parametric learners, keep the same interpretation order:

1. Check baseline comparison first.
2. Check identifiability warnings second.
3. Check transform integrity last.

Additional `v0.5` guidance:

- Treat `parametric_conversion_status != "ok"` as a fit failure signal; do not
  proceed with feature generation from that fit.
- If a parametric learner has weak diagnostics or is beaten by
  `best_single_lag` by margin, prefer simplex learning for a less
  assumption-heavy kernel estimate.
- Use comparison helpers to review simplex and parametric fits side by side:
  `learner_diagnostic_comparison_table` and
  `learner_diagnostic_warning_table`.

Diagnostics and helpers support kernel-trust decisions only; they do not select
a downstream predictive model.

`v0.3` additive field:

- `summary_by_baseline`: per-baseline diagnostic summary with baseline loss,
  learned loss, fractional delta vs learned, and a deterministic
  `beats_learned_by_margin` boolean.
- `delta_fraction_vs_learned` uses a single sign convention:
  `(learned_validation_loss - baseline_validation_loss) / learned_validation_loss`.
  Positive means the baseline validated better than the learned kernel
  (lower validation loss).

## Additional Fit Summaries (`v0.3`, additive)

`KernelFitResult` may include:

- `kernel_shape_summary`:
  - `normalized_entropy`
  - `max_weight`
  - `min_weight`
  - `concentration_hhi`
  - `effective_lag_count`
- `fit_data_coverage_summary`:
  - `total_rows`
  - `valid_windows`
  - `train_windows`
  - `validation_windows`
  - `retained_row_fraction`
  - `retained_window_fraction`

## `TransformReport`

Contains:

- `row_count`
- `output_row_count`
- `warmup_rows`
- `feature_names`
- `missing_rows_by_feature`
- `missing_rows_by_kernel` (additive `v0.3` field; per-kernel total count of
  missing feature cells across all generated features for that kernel)
- `zero_denominator_rows_by_feature`
- `zero_denominator_rows_by_kernel` (additive `v0.3` field; counts
  kernel-level zero-denominator feature cells across generated features, not
  unique input row ids)
  - Example: one numeric source column yields three features
    (`wmean/wstd/wsum`), so each zero-denominator row contributes `3` to this
    kernel-level count.
- `warmup_unusable_summary` (additive `v0.3` field; aggregate row-level
  usability summary with keys:
  `input_rows`, `warmup_rows`, `rows_after_warmup`,
  `rows_all_features_usable`, `rows_with_any_unusable_feature`)
- `collision_naming_summary` (additive `v0.3` field; always a summary map used
  to interpret generated feature names across kernels, including
  `has_name_collision` as a collision flag)

`transform()` returns only the feature table. Use `diagnose_transform()` for the report, and `last_transform_report` as the last cached helper value.

`v0.3` report-extension rule:

- Existing `TransformReport` and `FitDiagnostics` fields remain available.
- Any richer diagnostics added in `v0.3` must be additive fields on existing
  objects, or gated behind an explicit new versioned report object.
- Existing field names and semantics are stable unless an explicit versioned
  contract is introduced.

## Interpretation Guidance

Diagnostics should help the user decide whether the result is best treated as:

- A plausible RTD kernel
- A plausible response kernel
- A weak kernel not worth trusting

Practical interpretation flow:

1. Check baseline comparison first. If `best_single_lag` materially beats the
   learned kernel, treat the learned kernel as weak.
2. Check identifiability warnings second. Boundary-piled or diffuse warnings
   reduce confidence even when loss metrics are acceptable.
3. Check transform integrity last. High missing or zero-denominator counts mean
   generated features may be unusable for downstream modeling.

Diagnostics remain decision support for kernel trustworthiness and feature
quality. They do not rank or prescribe final predictive models.

## Reporting Presentation Helpers (`v0.3`, additive)

Use lightweight helpers in `rtdfeatures.reporting` for compact summaries.

Baseline comparison helpers:

- `baseline_comparison_table(baseline) -> polars.DataFrame`
- `baseline_comparison_compact_dict(baseline) -> dict`
- `baseline_comparison_compact_text(baseline) -> str`

`baseline_comparison_table` output schema:

- `baseline` (`str`)
- `validation_loss` (`float`)
- `learned_validation_loss` (`float`)
- `delta_fraction_vs_learned` (`float`)
- `beats_learned` (`bool`)
- `is_learned` (`bool`)

For table rows, `delta_fraction_vs_learned > 0` means the baseline row has
lower validation loss than the learned row.

Warning summary helpers:

- `warning_summary_table(report) -> polars.DataFrame`
- `warning_summary_compact_dict(report) -> dict`
- `warning_summary_compact_text(report) -> str`

`warning_summary_table` output schema:

- `warning_index` (`int`)
- `warning_code` (`str`)
- `warning_severity` (`str`)
- `warning_message` (`str`)

Helpers consume public result/report objects only and are presentation-only.
They do not mutate result objects and do not prescribe downstream model
choices.

## Out-Of-Fold Diagnostics Contract (`v0.95`)

Public objects:

- `OutOfFoldKernelFeatureResult`
- `OutOfFoldSplitSummary`

Split behavior contract:

- folds are used for kernel fitting and feature generation only
- forward-chaining validation windows are strictly after training windows
- validation rows never appear in the fold's training rows
- optional `gap` excludes rows between training and validation windows
- future rows are not used for earlier validation rows

Split metadata is reproducible from deterministic boundary generation and is
returned through `OutOfFoldSplitSummary.fold_boundaries`.
