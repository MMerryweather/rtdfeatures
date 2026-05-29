# Feature Evidence

Every generated feature column can carry structured metadata that records its provenance. The `FeatureRegistry` returned by `transform_result()` provides per-column specs. For richer evidence that includes fit diagnostics, use `diagnose_feature_evidence()` on the builder.

## Building evidence from a transform

TransformResult is the preferred auditable output. It keeps the feature table, transform diagnostics, and feature registry together.

Generated feature arrays and `FeatureSpec` metadata share one source of truth.
Specs are created at feature-generation time; registry construction must not
parse feature names to recover metadata.

The recommended auditable workflow starts with `transform_result()`:

```python
from rtdfeatures import KernelFeatureBuilder, feature_evidence_table

builder = KernelFeatureBuilder(
    kernels={"learned": fit.kernel},
    time_col="time",
    numeric_cols=["temperature", "pressure"],
)

result = builder.transform_result(df)
evidence = builder.diagnose_feature_evidence(
    feature_registry=result.feature_registry,
    fit_result_by_kernel={"learned": fit},
)
print(feature_evidence_table(evidence))
```

This returns a `FeatureEvidenceReport` with evidence for every generated feature.

## Direct evidence construction

For advanced use cases, you can construct evidence from individual
`FeatureSpec` entries (available from `result.feature_registry`).
Use `result.feature_registry.names()` for ordered names or
`result.feature_registry.to_frame()` for a tabular metadata view.

## Interpretation labels

| Label | Meaning |
|---|---|
| `material_memory` | Physically plausible material residence interpretation |
| `process_response` | Delayed predictive relationship (safe default) |
| `statistical_pattern` | Useful lagged association only |
| `unknown` | No interpretation assigned |

## Evidence completeness

| Label | Meaning |
|---|---|
| `kernel_only` | Kernel shape only |
| `fit_evidence` | Includes fit diagnostics |
| `comparison_evidence` | Includes baseline comparison |
| `bootstrap_evidence` | Includes bootstrap uncertainty |
| `full_evidence` | All available evidence |

## Compact summaries

```python
from rtdfeatures import feature_evidence_compact_dict, feature_evidence_compact_text

d = feature_evidence_compact_dict(evidence_report)
print(feature_evidence_compact_text(evidence_report))
```

## See also

- [Interpretation boundary](../concepts/interpretation-boundary.md) — guidelines for choosing interpretation labels
- [Comparing kernels](comparing-kernels.md) — generating baseline comparison evidence
- [07_validation_and_diagnostics.md](../07_validation_and_diagnostics.md) — normative evidence contract
