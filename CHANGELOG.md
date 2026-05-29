# Changelog

## 1.0.0

### Added

- SimplexKernelLearner: empirical constrained kernel learning over a simplex of lag weights (causal, non-negative, sum-to-one, bounded lag).
- SharedSimplexKernelLearner: multi-pair shared execution over positional input/target lists with shared lag bound.
- GammaKernelLearner and ExponentialKernelLearner: parametric constrained kernel learners.
- KernelCandidates framework: fit, compare, and select across multiple kernel families with information criteria and cross-validation.
- Bootstrap uncertainty: blocked bootstrap for kernel weights, lag summaries, and parameter intervals with compact text/dict reporting.
- Feature evidence: structured metadata per generated column recording source signal, kernel, interpretation label, lag window, and completeness.
- Out-of-fold (OOF) feature generation: leakage-aware forward-chaining fold split with per-fold kernel fitting and deterministic stitched output.
- Baseline comparisons: `no_lag`, `best_single_lag`, `uniform`, and `exponential` baselines.
- Diagnostic objects: FitDiagnostics, IdentifiabilityReport, BaselineComparison, TransformReport, KernelShapeSummary, FitDataCoverageSummary.
- Reporting helpers: compact dict/text summaries and comparison tables for baselines, warnings, and learner diagnostics.
- KernelFeatureBuilder: deterministic weighted-feature generation from fitted kernels with full transform diagnostics.
- Kernel families: FixedDelayKernel, UniformKernel, LearnedKernel, GammaKernel, ExponentialKernel, DelayedExponentialKernel, ErlangKernel, LogNormalKernel.
- Public result types: KernelFitResult, KernelFamilyFitResult, SharedPairFitResult, SharedKernelFitResult, KernelCandidateSet, KernelComparisonResult, KernelSelectionResult, BootstrapResult, FeatureEvidenceReport, OutOfFoldKernelFeatureResult.

### Changed

- Package metadata updated to `1.0.0` and `Development Status :: 5 - Production/Stable` for the final stable release.
- Diagnostics enriched with additive fields across result objects.
- Documentation expanded with API examples, interpretation guidance, and process-engineering workflow documentation.

### Fixed

- Deterministic fold-level status isolation in OOF generation: per-fold failures produce null features in failed rows without corrupting other folds.
- Schema-union correctness for OOF output across different selected kernels per fold.
