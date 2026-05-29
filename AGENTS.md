# AGENTS.md

## Project

`rtdfeatures` learns constrained causal kernels from regular-grid process time series and converts them into Polars feature tables plus diagnostics. It does not produce final predictive models.

Keep the product wedge narrow:

```text
learn the lag
validate the kernel
generate the features
stop there
```

## Read First

- Boundary and non-goals: [docs/01_product_charter.md](docs/01_product_charter.md), [docs/03_scope_and_non_goals.md](docs/03_scope_and_non_goals.md)
- Terminology: [docs/02_core_concepts.md](docs/02_core_concepts.md)
- Kernel learning: [docs/04_kernel_learning_design.md](docs/04_kernel_learning_design.md)
- Feature/data contract: [docs/05_feature_generation_design.md](docs/05_feature_generation_design.md), [docs/06_data_model.md](docs/06_data_model.md)
- Diagnostics: [docs/07_validation_and_diagnostics.md](docs/07_validation_and_diagnostics.md)
- Public API: [docs/08_api_design.md](docs/08_api_design.md)
- Build order: [docs/12_development_roadmap.md](docs/12_development_roadmap.md)

## Hard Guardrails

- Keep the package generic across process industries.
- Use `kernel` as the generic term; use `RTD kernel` only when physically justified.
- Preserve constrained learned kernels: causal, non-negative, sum-to-one, bounded lag.
- Public data interface is `polars.DataFrame` in and `polars.DataFrame` out.
- Infer `dt` only from a regular time grid.
- `SimplexKernelLearner` is 1 input to 1 target in `v0.1`.
- `transform()` returns only `[time_col + generated feature cols]`.
- Unsorted input raises by default; `order_by_time=True` is opt-in.
- `v0.1` baselines are `no_lag` and `best_single_lag` only.
- Diagnostics are part of the public contract.

## Non-Goals

- prediction library
- forecasting framework
- historian connector
- real-time feature service
- control package
- digital twin or simulator
- plant-topology genealogy engine
- attention/transformer/GNN causal-discovery project
- domain-specific API

## Defaults

- Use a modern `src/` layout and PEP 621 `pyproject.toml`.
- Target Python `>=3.10`.
- Keep dependencies light.
- Prefer clear, tidy-style names over compact names.
- Keep functions small and public APIs typed/docstringed.
- Do not silently handle irregular data, divide-by-zero cases, or leakage risks.

## Core Objects In `v0.1`

- `Kernel`, `LearnedKernel`, `FixedDelayKernel`, `UniformKernel`
- `SimplexKernelLearner`
- `KernelFeatureBuilder`
- `KernelFitResult`, `FitDiagnostics`, `IdentifiabilityReport`, `BaselineComparison`, `TransformReport`

## Quality Bar

Every meaningful change should:

1. Support kernel learning, kernel validation, or kernel-based feature generation.
2. Keep the package domain-neutral.
3. Preserve Polars-first APIs and constrained-kernel semantics.
4. Add or update tests.
5. Update docs when user-facing behaviour changes.
