# Architecture

## Module structure

```
src/rtdfeatures/
  __init__.py          # public API exports
  kernels/             # kernel classes
    __init__.py
    base.py            # Kernel, LearnedKernel base classes
    fixed.py           # FixedDelayKernel, UniformKernel
    parametric.py      # GammaKernel, ExponentialKernel
  learners/            # learner classes
    __init__.py
    simplex.py         # SimplexKernelLearner, SharedSimplexKernelLearner
    gamma.py           # GammaKernelLearner
    exponential.py     # ExponentialKernelLearner
    shared.py          # shared learner utilities
  features/            # feature generation
    __init__.py
    builder.py         # KernelFeatureBuilder
    registry.py        # FeatureRegistry, FeatureSpec, TransformResult
    evidence.py        # build_feature_evidence, feature_evidence_table, etc.
    age.py             # age-related feature helpers
    categorical.py     # categorical feature helpers
    numeric.py         # numeric feature helpers
  diagnostics/         # result objects and validation
    __init__.py
    fit.py             # KernelFitResult, FitDiagnostics, IdentifiabilityReport
    transform.py       # TransformReport
    warnings.py        # shared warning codes
  candidates/          # candidate kernel comparison
    __init__.py
    contracts.py       # KernelCandidate, KernelCandidateSet
    fitting.py         # fit_kernel_candidates
    selection.py       # select_kernel_candidate
  bootstrap/           # blocked bootstrap uncertainty
    __init__.py
    contracts.py       # BlockedBootstrapConfig, BootstrapIndexSplit
    sampling.py        # bootstrap_kernel_fit, generate_blocked_bootstrap_splits
    summaries.py       # bootstrap summary tables
  oof/                 # out-of-fold feature generation
    __init__.py
    splits.py          # ForwardChainingSplitConfig, generate_forward_chaining_splits
    generation.py      # fit_transform_oof
    reports.py         # OOF report objects
  reporting.py         # compact dict/text/table reporting helpers
  synthetic.py         # deterministic synthetic datasets for testing and examples
  utils.py             # shared utilities (time handling, scaling, validation)
  integrations/        # optional adapters (e.g. Tigramite)
```

## Dependency direction

```
kernels/  ←  learners/  ←  features/
   ↑                ↑
diagnostics/     (parametric helpers in kernels/)
   ↑
  candidates/, bootstrap/, oof/
   ↑
  reporting.py, synthetic.py
```

- `kernels/` has no internal dependencies outside `utils.py`
- `diagnostics/` depends on `kernels/`
- `learners/` depends on `kernels/`, `diagnostics/`, `utils.py`
- `features/` depends on `kernels/`, `diagnostics/`
- `candidates/`, `bootstrap/`, `oof/` depend on the above packages
- `reporting.py` and `synthetic.py` consume public result objects

## Design principles

1. **Kernel objects are data, not models.** They store weights and lag metadata. Fit provenance lives in `KernelFitResult`, not on the kernel itself.
2. **Result objects are frozen dataclasses.** They carry diagnostics, provenance, and comparison outcomes without exposing implementation internals.
3. **Polars is the data interface.** All public methods accept and return `polars.DataFrame`.
4. **Fail closed.** Invalid inputs, irregular grids, failed fits, and missing labels raise errors rather than silently producing incorrect results.
5. **Additive evolution.** New fields on result objects are additive. Existing field names and semantics are stable.

## See also

- [09_package_architecture.md](../09_package_architecture.md) — normative architecture reference
