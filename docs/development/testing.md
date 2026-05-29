# Testing Guide

## Test framework

Tests use **pytest**. No additional test runner is required.

## Required marker: `external_data`

Tests that download or require external benchmark archives **must** be marked:

```python
import pytest

@pytest.mark.external_data
def test_benchmark_comparison() -> None:
    ...
```

These tests are excluded from CI runs by default. Run them locally with:

```bash
pytest -m external_data
```

## Test categories

| Category | Description | Marker |
|---|---|---|
| Unit tests | Test a single function or class in isolation | (none) |
| Integration readiness guards | Marked `xfail` — expected to pass once integration is wired | `pytest.mark.xfail` |
| Benchmark comparison tests | Compare against archived benchmark outputs | `external_data` |
| Snapshot tests | Pin root namespace and schema shape | (none) |

## How to run specific groups

```bash
# All tests except external data (CI equivalent)
pytest -m "not external_data"

# Only external data tests
pytest -m external_data

# Tests matching a keyword
pytest -k "feature_registry"

# A specific test file
pytest tests/test_feature_registry_contract.py -v
```

## Coverage gate

Coverage is enforced in CI with a minimum threshold of 88% line coverage.

Run the local coverage gate with:

```bash
pytest -m "not external_data" \
  --cov=src/rtdfeatures \
  --cov-report=term-missing \
  --cov-report=xml \
  --cov-fail-under=88
```

Coverage configuration is defined in `pyproject.toml` under `[tool.coverage.run]`
and `[tool.coverage.report]`. Keep exclusions minimal and prefer adding tests
over broad excludes. Current project coverage is just below 90%; raise this
threshold back to 90% as additional behavior tests are added.

## How to add tests

### For new kernel types

1. Create or extend a test file in `tests/`, named with the relevant scope reference (e.g. `test_kernels.py`).
2. Import `Kernel` and your subclass from `rtdfeatures.kernels`.
3. Test that `validate()` passes for valid weights and raises for invalid ones.
4. Test `summary()` returns expected keys.
5. Test `to_learned()` round-trips weights correctly.

Example:

```python
from rtdfeatures.kernels import GammaKernel

def test_gamma_kernel_summary_includes_parameters() -> None:
    kernel = GammaKernel(shape_alpha=2.0, rate_beta=0.5, max_lag_steps=10, dt=1.0)
    summary = kernel.summary()
    assert summary["parametric_family"] == "gamma"
    assert "parametric_parameters" in summary
```

### For new learners

1. Create or extend a test file in `tests/`, named with the relevant scope reference.
2. Generate synthetic data with `rtdfeatures.synthetic`.
3. Fit the learner and verify the returned kernel has the expected shape.
4. Test failure modes (mismatched columns, zero-variance signals, etc.).

### For new feature builders

1. Create or extend a test file in `tests/`.
2. Fit a kernel on synthetic data.
3. Build features and verify column counts, names, and types.
4. Test `transform_with_report()` returns the correct diagnostics.

### For new diagnostics

1. Create or extend a test file in `tests/`.
2. Generate known-good and known-bad fits.
3. Verify diagnostic outputs flag the right cases.

## Snapshot test policy

- The root namespace snapshot (`tests/test_root_namespace_snapshot.py`) locks the public API surface. When adding a new public export, update the expected set.
- Schema stability snapshots are not yet required. If added, pin the DataFrame schema (column names and dtypes) for each transform output.

## Fixtures and synthetic data

Shared fixtures live in `tests/conftest.py`. Synthetic dataset generators live in `rtdfeatures.synthetic`:

```python
from rtdfeatures.synthetic import make_single_delay_dataset

dataset = make_single_delay_dataset(n_rows=120, dt=60.0, seed=7)
df = dataset.data  # polars.DataFrame
```

## What constitutes a meaningful test

Per project quality bar, a meaningful test:

1. Tests a user-facing behaviour, not an implementation detail.
2. Covers at least one happy path and one failure mode.
3. Does not depend on external network resources (unless marked `external_data`).
4. Runs in under 5 seconds (unless marked `external_data`).
5. Does not produce side effects (no file I/O, no network calls).
