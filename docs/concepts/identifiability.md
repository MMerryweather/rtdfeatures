# Identifiability

Not every fitted kernel is worth trusting. Identifiability is the degree to which the data supports a unique, stable kernel estimate.

## When a kernel is identifiable

- The input signal has sufficient variation
- The target signal has sufficient signal-to-noise ratio
- The learned kernel converges to a clear, stable shape
- Validation loss is close to training loss (no overfitting)
- The kernel beats simple baselines (`no_lag`, `best_single_lag`) by a meaningful margin
- Bootstrap uncertainty intervals are narrow relative to the lag window

## Identifiability warnings

The `IdentifiabilityReport` attached to every `KernelFitResult` checks for these conditions:

| Warning Code | Meaning | Severity |
|---|---|---|
| `INPUT_TOO_FLAT` | Input signal has near-zero variance | high |
| `TARGET_TOO_FLAT` | Target signal has near-zero variance | high |
| `WEAK_NO_LAG_IMPROVEMENT` | Learned kernel barely beats `no_lag` | medium |
| `LARGE_VALIDATION_GAP` | Validation loss is much worse than training loss | high |
| `BOUNDARY_PILED_KERNEL` | Kernel piles mass at min or max lag boundary | medium |
| `DIFFUSE_KERNEL` | Kernel is too spread out to interpret confidently | medium |
| `BEST_SINGLE_LAG_BEATS_LEARNED` | A single fixed lag outperforms the learned kernel | medium |
| `UNIFORM_BASELINE_BEATS_LEARNED` | Uniform weighting outperforms the learned kernel | medium |
| `EXPONENTIAL_BASELINE_BEATS_LEARNED` | Exponential decay baseline outperforms the learned kernel | medium |

## Default diagnostic thresholds

These conservative heuristics trigger warnings:

| Threshold | Default | Triggers |
|---|---|---|
| `flat_variance_threshold` | 1e-8 | Input or target too flat |
| `validation_gap_ratio` | 2.0 | Validation loss / training loss exceeds this |
| `baseline_improvement_margin` | 0.05 | Learned kernel must beat baselines by at least this fraction |
| `boundary_mass_threshold` | 0.35 | Boundary mass fraction triggers warning |
| `diffuse_entropy_fraction` | 0.85 | Normalized entropy above this triggers diffuse warning |
| `diffuse_max_weight_threshold` | 0.20 | Max weight below this contributes to diffuse warning |

## Bootstrap uncertainty

When available (v0.8+), bootstrap intervals quantify weight stability:

- Narrow intervals across lags → more identifiable
- Wide intervals or boundary-touching intervals → weak identifiability
- `BOOTSTRAP_WEIGHT_UNSTABLE` and `BOOTSTRAP_LAG_SUMMARY_UNSTABLE` warnings indicate poor identifiability

## Practical guidance

1. **Check baseline comparison first.** If `best_single_lag` beats the learned kernel materially, treat it as weak.
2. **Check identifiability warnings second.** Boundary-piled or diffuse kernels reduce confidence even when loss metrics look acceptable.
3. **Check transform integrity last.** High null counts or zero-denominator features are unusable regardless of kernel quality.

> A kernel that passes all checks is plausible — not proven. Diagnostics are decision support, not formal statistical tests.

## See also

- [Interpretation boundary](interpretation-boundary.md) — labelling kernels responsibly based on evidence
- [07_validation_and_diagnostics.md](../07_validation_and_diagnostics.md) — normative reference for all diagnostic contracts and default thresholds
