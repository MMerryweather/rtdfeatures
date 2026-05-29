# rtdfeatures

[![CI](https://github.com/<org-or-user>/rtdfeatures/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/<org-or-user>/rtdfeatures/actions/workflows/ci.yml)

`rtdfeatures` learns constrained lag kernels from regular-grid process time series and turns them into auditable lag-aware features for downstream modelling.

```mermaid
flowchart LR
    A[Regular process time series] --> B[Fit kernel]
    B --> C[Validate against baselines]
    C --> D[Generate weighted features]
    D --> E[Polars feature table]
```

## Why this exists

Process industries have lagged relationships between variables — material transport, mixing, recycle loops, and cascade dynamics mean that a change in one variable affects another only after some time, and the influence is often distributed across multiple past observations. Standard ML pipelines treat each time step independently or use arbitrary lag windows that waste signal or leak information.

`rtdfeatures` learns this lag structure directly from data using physically plausible constraints: causal (no future leakage), non-negative influence, sum-to-one weighting, and bounded lag windows. The result is a compact, interpretable kernel that can be validated against baselines and then used to generate auditable lag-aware features for any downstream model.

It does **not** train final predictive models, perform plant-wide causal discovery, or serve features in real time.

## Install

```bash
pip install rtdfeatures
```

Requires Python **>=3.10**. The package depends on `numpy`, `polars`, and `torch`. CPU operation is the default and expected path — no GPU is required.

## Quickstart

Executable example (`docs-test:readme-primary`):

```python
from rtdfeatures import KernelFeatureBuilder, SimplexKernelLearner
from rtdfeatures.synthetic import make_single_delay_dataset

dataset = make_single_delay_dataset(n_rows=120, dt=60.0, seed=7)
df = dataset.data

learner = SimplexKernelLearner(max_lag="20m")
fit = learner.fit(
    df,
    input_col="input_signal",
    target_col="target_signal",
    time_col="time",
)

builder = KernelFeatureBuilder(
    kernels={"learned": fit.kernel},
    time_col="time",
    numeric_cols=["input_signal"],
)

result = builder.transform_result(df)
features = result.features
report = result.report
registry = result.feature_registry

assert "time" in features.columns
assert "learned_num_input_signal_wmean" in features.columns
assert report.row_count == df.height
```

TransformResult is the preferred auditable output. It keeps the feature table, transform diagnostics, and feature registry together. The primary API `transform_result()` returns a `TransformResult` with `.features` (a Polars DataFrame), `.report` (a `TransformReport` with diagnostics), and `.feature_registry` (structured metadata per column). The simpler `builder.transform(df)` path returns just the feature table.

### Shared multi-pair learning

Executable example (`docs-test:readme-shared`):

```python
from rtdfeatures import KernelFeatureBuilder
from rtdfeatures.learners import SharedSimplexKernelLearner
from rtdfeatures.synthetic import make_multi_pair_dataset

dataset = make_multi_pair_dataset(n_rows=160, dt=60.0, seed=11)
df = dataset.data

shared = SharedSimplexKernelLearner(max_lag="40m", min_lag="10m", loss="huber")
shared_fit = shared.fit(
    df,
    input_cols=["input_signal_a", "input_signal_b"],
    target_cols=["target_signal_a", "target_signal_b"],
    time_col="time",
)

kernels = shared_fit.to_kernels()
builder = KernelFeatureBuilder(
    kernels=kernels,
    time_col="time",
    numeric_cols=["input_signal_a", "input_signal_b"],
)
features = builder.transform(df)

assert features.height == df.height
```

## Core concepts

**Kernel** — the generic object representing a weighted lag distribution. Kernels are causal, non-negative, sum-to-one, and defined over a bounded lag window. This is always the correct term unless a specific physical interpretation is justified.

**RTD-like kernel** — a kernel interpreted as a Residence Time Distribution. This physical interpretation requires independent evidence that the relationship is driven by material or tracer propagation (e.g. known vessel geometry, tracer tests, process knowledge). Do not claim RTD without supporting evidence.

**Response kernel** — a kernel interpreted as a delayed-influence relationship where the physical RTD interpretation is not justified. This is the safe default interpretation for most process-data relationships. The kernel object is identical — only the interpretation label differs.

**Feature evidence** — structured metadata attached to each generated feature column, recording its source column, kernel, interpretation label, lag window, and completeness status. Feature evidence makes every generated column auditable and independently interpretable.

Generated features include weighted lagged aggregations of numeric signals, categorical contribution scores, and age features (time since the kernel-weighted window). See the [data model](docs/06_data_model.md) for the full output schema.

## What it produces

- **Constrained kernels** — `LearnedKernel`, `FixedDelayKernel`, `GammaKernel`, and others, each carrying its lag weights, support, and fit metadata.
- **Feature tables** — Polars DataFrames with deterministic, auditable columns ready for downstream models.
- **Diagnostics** — fit diagnostics, baseline comparisons (`no_lag`, `best_single_lag`), identifiability reports, and transform reports.
- **Feature evidence** — per-column provenance records linking every feature back to its source, kernel, and interpretation.

## Examples gallery

- [nRTD laminar flow worked example](docs/examples/nrtd_laminar_flow_worked_example.md) — flagship extracted benchmark learning example.
- [Plant-first scenario gallery](docs/examples/plant_first_gallery.md) — scenario-first synthetic gallery with fit evidence and feature previews.
- [Parametric vs empirical fit gallery](docs/examples/parametric_empirical_fit_gallery.md) — compares parametric kernels (Gamma, Exponential) against empirical (simplex) fits on synthetic plug-flow and tanks-in-series scenarios. Run with `python examples/parametric_empirical_baseline_fits.py`.

## Limitations

- Input data must have a **regular time grid**. Irregular or missing timestamps raise by default — there is no imputation.
- The `SimplexKernelLearner` fits **one input signal to one target signal**. The `SharedSimplexKernelLearner` extends this to multiple pairs with a shared lag bound.
- **Final predictive modelling is out of scope**. This package produces features and diagnostics — it does not train, evaluate, or deploy models.
- **Not a causal discovery tool**. Learned kernels capture lagged statistical relationships under the constraints; they do not prove physical causation.
- **No online or streaming support**. All operations assume a complete batch of historical data.
- **No plant-topology or genealogy modelling**. The package learns pairwise relationships, not full process graphs.
- Warmup rows (before the maximum lag is satisfied) produce `null` features. The row count is preserved.

## Citation / scientific context

If you use `rtdfeatures` in published work, please cite the repository. The package builds on ideas from constrained kernel learning, residence time distribution analysis, and feature engineering for irregular-spaced process data. See the [cross-field research summary](docs/14_cross_field_research_summary.md) for background.

## Development status

This is the stable **v1.0.0** release. The package is in production use. Changes are documented in [release notes](docs/RELEASE_NOTES.md). CI gates: `pytest -m "not external_data"`.

Read the [documentation hub](docs/index.md) for guides, examples, and API reference.
