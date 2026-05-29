# Examples And Use Cases

## Domain Scope

The package is intended for process domains where downstream behaviour depends on upstream state after a delay:

- Mineral processing
- Hydrometallurgy
- Pyrometallurgy
- Chemical processing
- Oil and gas
- Water and wastewater
- Bioprocessing
- Food processing

## What The Package Produces

The package does not solve the downstream problem directly. It produces lag-aware features such as:

- `learned_num_temperature_wmean`
- `learned_num_flowrate_wsum`
- `learned_cat_operating_mode_startup_frac`
- `learned_cat_feed_class_entropy`
- `learned_age_mean`
- `learned_age_p90`

## Use Cases

### Generic Chemical Process

Fit a kernel from inlet concentration to outlet concentration, then generate lag-aware concentration, temperature, and flow features for a downstream model.

### Pyrometallurgical Furnace

Fit a kernel from burden composition to tap chemistry and generate age and weighted-history features for downstream modelling.

### Grinding Circuit

Fit a response kernel from feed PSD to power, then generate weighted PSD and operating-mode features.

### Wastewater Reactor

Fit an RTD-style kernel from influent to effluent and generate lag-aware concentration and flow features.

## Priority Example Entry Points

- Flagship extracted benchmark worked example: [docs/examples/nrtd_laminar_flow_worked_example.md](examples/nrtd_laminar_flow_worked_example.md)
- Scenario-first synthetic gallery: [docs/examples/plant_first_gallery.md](examples/plant_first_gallery.md)

## Positioning Reminder

Examples may be domain-specific. The core API and package identity remain domain-neutral.

## Synthetic Helper Example Boundary (`v0.3`)

The package-level module `rtdfeatures.synthetic` provides deterministic helpers
for public learner and builder workflows.

Executable example (`docs-test:examples-single-delay`):

```python
from rtdfeatures.synthetic import make_single_delay_dataset

out = make_single_delay_dataset(n_rows=120, dt=1.0, seed=7)

assert out.data.height == 120
assert "input_signal->target_signal" in out.true_kernels
assert out.scenario["name"] == "single_delay"
```

Return shape used by all documented helpers:

- `out.data`: regular-grid `polars.DataFrame`
- `out.true_kernels`: ground-truth kernel metadata for validation
- `out.scenario`: scenario metadata including generation parameters

Keep benchmark usage distinct from synthetic helper usage; for external layers
and nRTD reference handling see
[benchmarks/nrtd_benchmark_layers.md](benchmarks/nrtd_benchmark_layers.md).

nRTD experimental learned RTDs are references, not synthetic helper ground truth.

Flagship worked example:
`docs/examples/nrtd_laminar_flow_worked_example.md`

## Shared Learning Example (`v0.2`)

Use shared learning when you have multiple aligned input/target pairs and want
per-pair kernels with a single public workflow.

Illustrative example (non-executable in docs tests; requires user-provided `df`):

```python
shared = SharedSimplexKernelLearner(max_lag="6h", min_lag="10m", loss="huber")
fit = shared.fit(
    df,
    input_cols=["feed_a", "feed_b"],
    target_cols=["prod_a", "prod_b"],
    time_col="timestamp",
)
summary = fit.summary()
kernels = fit.to_kernels()

builder = KernelFeatureBuilder(
    kernels=kernels,
    time_col="timestamp",
    numeric_cols=["feed_a", "feed_b"],
)
features = builder.transform(df)
```

## Parametric Learner API Examples (`v0.5`)

Use parametric learners when you have a justified kernel-family assumption.
These remain constrained kernel learners and produce the same `KernelFitResult`
surface as simplex learning.

Illustrative gamma example (non-executable in docs tests; requires user-provided `df`):

```python
from rtdfeatures import GammaKernelLearner, KernelFeatureBuilder

gamma = GammaKernelLearner(max_lag="6h", min_lag="10m", loss="huber")
fit = gamma.fit(
    df,
    input_col="feed_signal",
    target_col="product_signal",
    time_col="timestamp",
)

builder = KernelFeatureBuilder(
    kernels={"gamma": fit.kernel},
    time_col="timestamp",
    numeric_cols=["feed_signal"],
)
features = builder.transform(df)
```

Illustrative exponential example (non-executable in docs tests; requires user-provided `df`):

```python
from rtdfeatures import ExponentialKernelLearner, KernelFeatureBuilder

exp = ExponentialKernelLearner(max_lag="6h", min_lag="10m", loss="huber")
fit = exp.fit(
    df,
    input_col="feed_signal",
    target_col="product_signal",
    time_col="timestamp",
)

builder = KernelFeatureBuilder(
    kernels={"exp": fit.kernel},
    time_col="timestamp",
    numeric_cols=["feed_signal"],
)
features = builder.transform(df)
```

Executable example (`docs-test:examples-v05-minimal`):

```python
import rtdfeatures

assert hasattr(rtdfeatures, "GammaKernelLearner")
assert hasattr(rtdfeatures, "ExponentialKernelLearner")
```

## Interpretation Guide: Simplex vs Parametric (`v0.5`)

- `SimplexKernelLearner`: safer default when shape assumptions are uncertain.
- `GammaKernelLearner` / `ExponentialKernelLearner`: use when the assumed
  family shape is physically or operationally justified.
- Compare diagnostics and baselines before selecting a kernel for feature
  generation; this is a kernel-quality decision, not predictive-model
  selection.

## Feature-Evidence Workflows (`v0.9`)

These examples document SME-facing feature-evidence interpretation. They are
workflow guidance only and do not extend the package API.

### 1) Mass-Weighted Chemistry Features

For concentration-style measurements, combine kernel-weighted composition
signals with mass-flow context so delayed chemistry reflects both quality and
throughput conditions over the same lag window.

Use this to inspect whether generated features preserve expected process trends
under changing load, not to claim final model readiness.

### 2) Categorical Source-Fraction Features

When feeds come from multiple categorical sources (for example, route, blend,
or campaign class), use kernel-weighted source fractions to quantify delayed
source contribution at each timestamp.

Use fraction and entropy features as traceable process descriptors for
downstream modelling inputs, not as causal-discovery outputs.

### 3) Empirical vs Parametric Kernel Evidence

Compare feature evidence produced from an empirical simplex kernel against
feature evidence produced from a justified parametric kernel family.

Keep selection grounded in diagnostics, baseline comparisons, and process
plausibility. This is a kernel-validation decision for feature engineering, not
a predictive-model selection API.

### 4) Raw-Kernel-Only Evidence

In constrained operating windows, teams may review only kernel-shape and
baseline evidence before using generated features in separate modelling
workflows.

This package supports that workflow by keeping kernel fit diagnostics and
feature generation diagnostics explicit, without encoding organization-specific
approval logic in the API.

## Out-Of-Fold Feature Generation Workflows (`v0.95`)

These examples document leakage-safe feature generation workflows for teams
that separate kernel/feature engineering from downstream predictive modelling.
`v0.95` includes package-level OOF APIs for this workflow.

### Why OOF Matters For Leakage Control

If a kernel is fit on all rows and then used to generate features on the same
rows, each feature row can embed information learned from future operating
periods relative to that row. That creates train/test contamination in
downstream modelling workflows.

Out-of-fold (OOF) generation reduces this risk by fitting kernels on
chronologically earlier rows and generating features for held-out future rows.
Each held-out row only receives features from kernels fit without that row.

### OOF Feature Generation Example

Illustrative example (non-executable in docs tests; requires user-provided `df`):

```python
from rtdfeatures import ForwardChainingSplitConfig, SimplexKernelLearner, fit_transform_oof

oof = fit_transform_oof(
    df=df,
    learner=SimplexKernelLearner(max_lag="6h"),
    split_config=ForwardChainingSplitConfig(
        n_folds=3,
        min_train_size=240,
        validation_size=80,
        gap=0,
    ),
    input_col="upstream_signal",
    target_col="downstream_signal",
    time_col="time",
    numeric_cols=["upstream_signal", "flowrate"],
    category_cols=["operating_mode"],
)
oof_features = oof.features
```

### OOF Candidate Comparison Example

Compare OOF feature sets from two kernel candidates to support kernel-selection
decisions before downstream modelling. This compares feature evidence quality,
not final predictive model metrics.

Illustrative example (non-executable in docs tests; requires user-provided `df`):

```python
from rtdfeatures import (
    ForwardChainingSplitConfig,
    KernelCandidate,
    KernelCandidateSet,
    fit_transform_oof,
)

candidate_set = KernelCandidateSet(
    candidate_set_id="oof-candidate-example",
    input_col="upstream_signal",
    target_col="downstream_signal",
    time_col="time",
    candidates=(
        KernelCandidate(
            candidate_id="fixed_30m",
            family="fixed_delay",
            candidate_type="fixed_kernel",
            min_lag="0m",
            max_lag="6h",
            fixed_parameters={"delay_steps": 30},
        ),
        KernelCandidate(
            candidate_id="simplex_6h",
            family="simplex",
            candidate_type="empirical_learner",
            min_lag="0m",
            max_lag="6h",
            learner_parameters={"max_epochs": 50, "seed": 11, "learning_rate": 0.05},
        ),
    ),
)

oof = fit_transform_oof(
    df=df,
    candidate_set=candidate_set,
    split_config=ForwardChainingSplitConfig(
        n_folds=3,
        min_train_size=240,
        validation_size=80,
        gap=0,
    ),
    input_col="upstream_signal",
    target_col="downstream_signal",
    time_col="time",
    numeric_cols=["upstream_signal", "flowrate"],
)
oof_candidate_features = oof.features
```

### Leakage-Safe Usage Notes

- Keep fold boundaries chronological; do not shuffle process time rows.
- Fit each kernel only on rows available before the score window.
- Generate score-window features with the fold-specific kernel only.
- Use package OOF APIs for fold-safe generation, and keep downstream model
  evaluation logic in separate orchestration layers.
- Use diagnostics and process plausibility for kernel-quality checks; do not
  interpret OOF feature generation itself as causal discovery.
