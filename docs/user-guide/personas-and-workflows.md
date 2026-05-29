# Personas and Workflows

This document maps common user roles to their workflows, entry-point docs, example
code, and failure modes. It is the reference for the v1.0 release checklist to
ensure every persona has at least one documented path into the package.

## Traceability Matrix

| Persona | Primary Task | Required Docs | Required Examples | Required API |
|---|---|---|---|---|
| Process engineer | Interpret lag shape | concepts + gallery | plug-flow, tanks, recycle | kernel summary + evidence |
| Data scientist | Generate features | quickstart + user guide | quickstart, OOF | learner + builder |
| MLOps engineer | Pipeline use | API stability + performance | OOF smoke | transform result |
| Contributor | Extend package | contributing + architecture | local dev | submodule boundaries |
| Research user | Reproduce / cite | citation + reproducibility | deterministic synthetic | seeds + version |

---

## Process Engineer

**Goal:** understand whether a measured lag between two process variables matches
physical intuition, and if so, interpret the shape.

### Happy path

1. Load a regular-grid process dataset into a `polars.DataFrame`.
2. Fit a `SimplexKernelLearner` to the input-target pair.
3. Compare the learned kernel against baselines (`no_lag`, `best_single_lag`).
4. Inspect `FitDiagnostics` and `KernelShapeSummary` for lag peak, spread, and
   tail behaviour.
5. If the shape matches process knowledge (e.g. a delayed peak from a known
   vessel), generate features with `KernelFeatureBuilder` and label the kernel
   interpretation in `FeatureEvidence`.

```python
from rtdfeatures import SimplexKernelLearner
from rtdfeatures.synthetic import make_single_delay_dataset

dataset = make_single_delay_dataset(n_rows=240, dt=60.0, seed=7, delay_steps=6)
df = dataset.data

learner = SimplexKernelLearner(max_lag="20m")
fit = learner.fit(df, input_col="input_signal", target_col="target_signal", time_col="time")

print(fit.diagnostics)
print(fit.kernel.weights)
```

### Failure mode

The learned kernel has a smooth unimodal shape that looks physically plausible,
but the real process has a multimodal or bypass path. The simplex learner
smooths over two distinct lag populations, producing a single broad peak that
overestimates the lag spread and misses the bypass route. The user does not
notice because `FitDiagnostics` shows no warning — the fit is internally
consistent even when the model is wrong.

**Mitigation:** Always inspect the kernel weights directly, compare against
multiple baselines, and cross-check against process knowledge before labelling a
kernel as an RTD. Use `identifiability_report` to check for multi-modality or
flat regions.

---

## Industrial Data Scientist

**Goal:** generate leakage-aware lag features for a downstream regression or
classification model.

### Happy path

1. Load a regular-grid dataset. Split chronologically for validation.
2. Fit `SimplexKernelLearner` on the training partition.
3. Build features with `KernelFeatureBuilder.transform()` on both train and
   test partitions.
4. Use the resulting `polars.DataFrame` columns in an external model (XGBoost,
   linear model, etc.).
5. Run `diagnose_transform()` to validate row count, warmup, and null coverage.
6. Use `fit_transform_oof()` for out-of-fold feature generation to prevent
   leakage.

```python
from rtdfeatures import KernelFeatureBuilder, SimplexKernelLearner
from rtdfeatures.synthetic import make_single_delay_dataset

dataset = make_single_delay_dataset(n_rows=240, dt=60.0, seed=7)
df = dataset.data

train = df[:200]
test = df[200:]

learner = SimplexKernelLearner(max_lag="20m")
fit = learner.fit(train, input_col="input_signal", target_col="target_signal", time_col="time")

builder = KernelFeatureBuilder(kernels={"learned": fit.kernel}, time_col="time", numeric_cols=["input_signal"])
train_features = builder.transform(train)
test_features = builder.transform(test)
```

### Failure mode

The user calls `transform()` on the full dataset (train + test combined) before
splitting. The kernel was fitted on all rows, so the "test" features contain
information from future observations — test leakage. The downstream model
achieves unrealistic validation scores that do not hold in production.

**Mitigation:** Always fit kernels on training data only, then transform train
and test separately. For cross-validation, use `fit_transform_oof()` with
`ForwardChainingSplitConfig` to generate leakage-free folds automatically.

---

## MLOps / Data Engineer

**Goal:** integrate feature generation into reproducible, versioned batch
pipelines.

### Happy path

1. Pin `rtdfeatures` to a specific version in `requirements.txt`.
2. Fit kernel and build `KernelFeatureBuilder` with a fixed `seed`.
3. Serialize the `KernelFeatureBuilder` or kernel objects for pipeline reuse.
4. Call `transform()` in a batch job. Output is a `polars.DataFrame` with a
   stable schema.
5. Attach `FeatureEvidence` to the feature store registry so downstream
   consumers know provenance (source column, kernel, interpretation label, lag
   window).
6. Monitor `TransformReport` for row count and null-fraction regressions.

```python
from rtdfeatures import KernelFeatureBuilder, SimplexKernelLearner
from rtdfeatures.synthetic import make_single_delay_dataset

dataset = make_single_delay_dataset(n_rows=120, dt=60.0, seed=7)
df = dataset.data

learner = SimplexKernelLearner(max_lag="20m")
fit = learner.fit(df, input_col="input_signal", target_col="target_signal", time_col="time")

builder = KernelFeatureBuilder(kernels={"learned": fit.kernel}, time_col="time", numeric_cols=["input_signal"])
builder.transform(df)

# cache last transform report
last = builder.last_transform_report
```

### Failure mode

After a library upgrade, `transform()` output schema changes (new column name,
dropped column, changed dtype). The downstream pipeline silently consumes the
new schema and produces corrupted feature tables at runtime.

**Mitigation:** Pin `rtdfeatures` versions in production. Add schema assertions
in the pipeline: check column names, dtypes, and row count against a known
baseline after every `transform()` call. Run `pytest -m "not external_data"`
in CI when updating the dependency.

---

## Contributor

**Goal:** add a new kernel learner or diagnostic without breaking existing
contracts.

### Happy path

1. Read [architecture](../09_package_architecture.md) and
   [API design](../08_api_design.md) to understand module boundaries.
2. Add the learner class in `src/rtdfeatures/learners/` following the
   existing pattern (`SimplexKernelLearner`, `GammaKernelLearner`).
3. Add corresponding kernel class in `src/rtdfeatures/kernels/`.
4. Export both from `src/rtdfeatures/__init__.py` only if they are
   stable and user-facing.
5. Add tests under `tests/` that run with `pytest -m "not external_data"`.
6. Run `ruff check src/` and `mypy src/` before opening a PR.

```bash
pip install -e ".[dev]"
pytest -m "not external_data"
ruff check src/
mypy src/
```

### Failure mode

The new learner class is added to `src/rtdfeatures/learners.py` and
automatically imported in `src/rtdfeatures/__init__.py` before its API is
stable. Downstream users depend on it, and a subsequent renaming breaks their
pipelines. The `__init__.py` accumulates exports with no deprecation path.

**Mitigation:** Keep new code in a private module initially. Export from
`__init__.py` only after the API has stabilized and been tested. Use a
deprecation warning when renaming or removing exported names. The root
namespace is the contract — fewer exports is better.

---

## Academic / Research User

**Goal:** reproduce experiments and cite the package in published work.

### Happy path

1. Pin `rtdfeatures == x.y.z` in the experiment environment.
2. Use deterministic synthetic datasets (`seed` parameter) for reproducible
   examples.
3. Fit kernels and generate features with known seeds.
4. Report version, parameters, and dataset seed in the methods section.
5. Cite via the repository DOI or `CITATION.cff`.

```python
from rtdfeatures import SimplexKernelLearner
from rtdfeatures.synthetic import make_single_delay_dataset

# deterministic seed for reproducibility
dataset = make_single_delay_dataset(n_rows=240, dt=60.0, seed=42)
df = dataset.data

learner = SimplexKernelLearner(max_lag="20m", loss="huber", seed=7)
fit = learner.fit(df, input_col="input_signal", target_col="target_signal", time_col="time")

# kernel weights are deterministic given seed
print(fit.kernel.weights)
```

### Failure mode

The user fits a kernel on real plant data and publishes the resulting lag shape
as an "RTD" without acknowledging that the learned kernel is only a
statistically constrained lag distribution — there is no physical evidence of
tracer propagation or vessel geometry. A subsequent study shows the lag is
driven by control logic rather than material transport, undermining the
publication.

**Mitigation:** Always label kernels as "response kernel" unless independent
physical evidence supports an RTD interpretation. The package provides
`FeatureEvidence` with an `interpretation` field — use it explicitly. Report the
distinction in the methods section.
