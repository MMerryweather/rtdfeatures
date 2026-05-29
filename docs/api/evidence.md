# Evidence and Diagnostics API

## Diagnostic result objects

### FitDiagnostics

Fields: `train_loss`, `validation_loss`, `input_variance`, `target_variance`, `kernel_weight_sum`, `mean_lag`, `p50_lag`, `p90_lag`, `tail_mass`, `boundary_mass_fraction`.

### IdentifiabilityReport

Fields: `warnings` (list of message strings), `warning_codes` (stable identifiers), `warning_severity_by_code` (deterministic severity mapping).

### BaselineComparison

Baseline loss comparison. Methods: `no_lag`, `best_single_lag`, `uniform`, `exponential`. Field: `summary_by_baseline`.

## Evidence objects

### FeatureEvidence

Per-feature provenance descriptor:

- `feature_name`, `source_col`, `feature_family`
- `kernel_name`, `kernel_family`, `kernel_summary`
- `fit_result_id`, `candidate_id`, `baseline_summary`
- `identifiability_warnings`, `bootstrap_summary`
- `interpretation`, `evidence_completeness`
- `metadata`

### FeatureEvidenceReport

Aggregate evidence summary: `feature_evidence`, `feature_count`, `kernel_count`, `source_columns`, `warning_summary`, `evidence_summary_by_kernel`, `evidence_summary_by_feature_family`.

## Label constants

```python
from rtdfeatures.diagnostics import FEATURE_INTERPRETATION_LABELS, FEATURE_EVIDENCE_COMPLETENESS_LABELS
```

## Bootstrap objects

Available in the current release:

- `BootstrapResult`, `BootstrapWeightSample`, `BootstrapParameterSample`, `BootstrapLagSummarySample`
- `KernelBootstrapSummary`, `ParameterUncertaintySummary`, `WeightUncertaintySummary`
- `BOOTSTRAP_WARNING_CODES`, `DEFAULT_BOOTSTRAP_INTERVAL_QUANTILES`
- Schema helpers: `bootstrap_weight_samples_schema()`, `bootstrap_parameter_samples_schema()`, etc.

## Reporting helpers

```python
from rtdfeatures.reporting import (
    baseline_comparison_table,
    baseline_comparison_compact_dict,
    baseline_comparison_compact_text,
    warning_summary_table,
    warning_summary_compact_dict,
    warning_summary_compact_text,
    learner_diagnostic_comparison_table,
    learner_diagnostic_warning_table,
)
```

## Candidate and comparison objects

Available in the current release:

- `KernelCandidate`, `KernelCandidateSet`
- `KernelFamilyFitResult`, `KernelComparisonResult`, `KernelSelectionResult`
- `fit_kernel_candidates()`, `select_kernel_candidate()`
- `kernel_comparison_table()`, `kernel_comparison_compact_dict()`, `kernel_comparison_compact_text()`

## Out-of-fold objects

Available in the current release:

- `ForwardChainingSplitConfig`, `ForwardChainingFoldSplit`
- `generate_forward_chaining_splits()`
- `fit_transform_oof()`
- `OutOfFoldKernelFeatureResult`, `OutOfFoldSplitSummary`
