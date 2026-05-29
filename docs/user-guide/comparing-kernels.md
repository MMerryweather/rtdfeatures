# Comparing Kernels

Every `KernelFitResult` includes a `BaselineComparison` with at least two baselines:

| Baseline | Description |
|---|---|
| `no_lag` | Predict current target with current input only |
| `best_single_lag` | Use the single best lag step as the predictor |
| `uniform` | Uniform weighting over the full lag window |
| `exponential` | Exponential decay weighting |

If the learned kernel does not meaningfully beat these, it is weak.

## Interpreting baseline comparison

```python
fit = learner.fit(df, input_col="in", target_col="out", time_col="t")

print(fit.baseline_comparison.summary_by_baseline)
# {
#     "best_single_lag": {
#         "delta_fraction_vs_learned": -0.15,
#         "beats_learned_by_margin": True,
#         ...
#     }
# }
```

**Positive** `delta_fraction_vs_learned` means the baseline validated better (lower loss) than the learned kernel.

## Candidate comparison

Compare multiple kernel families side by side:

```python
from rtdfeatures import (
    KernelCandidate, KernelCandidateSet,
    fit_kernel_candidates, kernel_comparison_table,
)

candidate_set = KernelCandidateSet(
    candidate_set_id="my-comparison",
    input_col="feed",
    target_col="product",
    time_col="time",
    candidates=(
        KernelCandidate(candidate_id="gamma", family="gamma",
                        candidate_type="parametric_learner",
                        min_lag="0m", max_lag="6h"),
        KernelCandidate(candidate_id="simplex", family="simplex",
                        candidate_type="empirical_learner",
                        min_lag="0m", max_lag="6h",
                        learner_parameters={"seed": 42}),
    ),
)

result = fit_kernel_candidates(candidate_set, df)
print(kernel_comparison_table(result))
```

The comparison table shows validation loss, train loss, success status, and warning codes per candidate. This is kernel-quality comparison only — not predictive model selection.

### Including any parametric family as a fixed candidate

Any parametric kernel family — including `delayed_exponential`, `lognormal`, and `erlang` — can be included as a `fixed_kernel` candidate in comparisons:

```python
from rtdfeatures import (
    KernelCandidate, KernelCandidateSet,
    fit_kernel_candidates, kernel_comparison_table,
)

candidate_set = KernelCandidateSet(
    candidate_set_id="compare-fixed-parametric",
    input_col="feed",
    target_col="product",
    time_col="time",
    candidates=(
        KernelCandidate(
            candidate_id="gamma_learner",
            family="gamma",
            candidate_type="parametric_learner",
            min_lag="0m", max_lag="6h",
            learner_parameters={"max_epochs": 400},
        ),
        KernelCandidate(
            candidate_id="delayed_exp_fixed",
            family="delayed_exponential",
            candidate_type="fixed_kernel",
            min_lag="0m", max_lag="6h",
            fixed_parameters={"delay": 2.0, "rate_lambda": 0.5},
        ),
        KernelCandidate(
            candidate_id="lognormal_fixed",
            family="lognormal",
            candidate_type="fixed_kernel",
            min_lag="0m", max_lag="6h",
            fixed_parameters={"log_mu": 1.0, "log_sigma": 0.5},
        ),
        KernelCandidate(
            candidate_id="erlang_fixed",
            family="erlang",
            candidate_type="fixed_kernel",
            min_lag="0m", max_lag="6h",
            fixed_parameters={"shape_k": 3, "rate_beta": 0.7},
        ),
    ),
)

result = fit_kernel_candidates(candidate_set, df)
print(kernel_comparison_table(result))
```

This lets you compare hand-specified parametric shapes alongside fitted
learner candidates when you want explicit fixed-parameter controls in the same
evaluation.

## Visualising comparisons

```python
from rtdfeatures import baseline_comparison_table
print(baseline_comparison_table(fit.baseline_comparison))
```

This returns a Polars DataFrame with baselines, validation losses, and a `beats_learned` flag.

## See also

- [Fitting kernels](fitting-kernels.md) — how to fit individual learners
- [Evidence](feature-evidence.md) — attaching comparison evidence to features
- [08_api_design.md](../08_api_design.md) — candidate API contracts
