# release-gate Release Gate: Benchmark And Simulation Harness

This checklist captures the `Work Package 8` release-gate expectations from
Benchmark and simulation harness specification.

## Scope Boundary

- The simulation harness under `tests/simulation_harness/` is test-only.
- Harness generators are fixtures for validation and regression tests, not
  public product APIs.
- `rtdfeatures` remains focused on learning constrained lag kernels, validating
  kernels, and generating feature tables.

Out of scope for this gate:

- public plant simulation APIs
- historian connectors
- forecasting or final predictive modelling
- plant topology/genealogy engines as product APIs

## External nRTD Context (Not Package Identity)

- nRTD-derived data is external benchmark context.
- Layer 1 repo-owned synthetic fixtures remain authoritative for required
  release checks.
- nRTD analytical fixtures and learned experimental references are optional,
  supplementary evidence only.
- in the current readiness readiness tests, strict recovery thresholds for tank,
  flotation, and optional nRTD mean-lag checks are tracked as explicit
  non-blocking readiness guards for known `SimplexKernelLearner` limitations
  in `v0.1`; they are aspirational and not release blockers for this plan

Reference usage boundaries:

- [nRTD Benchmark Layers And Usage Boundaries](./nrtd_benchmark_layers.md)

## Documented Conversion Exception

- release-gate permits one explicit fallback for `HSA_000/cholette` expected-array
  length mismatch during benchmark extraction:
  deterministic trim-to-min (trim both arrays from index 0 to shared min
  length).
- This exception is valid only when the benchmark `manifest.json` records the
  assumption event with original lengths, trimmed length, and deterministic
  rule details.
- Non-`cholette` expected-array length mismatches remain hard-fail extraction
  errors.
- Predicted-array irregular-grid or non-negative-normalization issues are
  allowed only as documented skip events for predicted-derived outputs; retain
  predicted raw arrays where possible and record skip reasons in
  `manifest.json`.
- All other irregular-grid, missing-array, non-finite-value, or mismatched
  length conditions remain hard-fail extraction errors.

## Citation And License Preservation

For extracted nRTD fixtures in `test_data/benchmarks/nrtd/`:

- keep `manifest.json` with Zenodo DOI/record URL and MIT license metadata
- keep `NOTICE.md` with citation and license notice
- preserve source archive identifiers and checksums for tracked fixture files
- keep tracked core fixture files committed:
  - `hsa_000_adler_kernel_reference.parquet`
  - `hsa_000_cholette_kernel_reference.parquet`
  - `hsa_000_dispersion_kernel_reference.parquet`
  - `hsa_000_laminar_flow_kernel_reference.parquet`
  - `hsa_000_laminar_flow_signals.parquet`
- release-gate release-gate tests are expected to fail if the tracked benchmark
  directory or any required tracked fixture file is removed.

## Fixture Size Guardrail

- Tracked extracted nRTD benchmark fixtures should remain under `5 MB` total.
- If this threshold is exceeded, record explicit justification in the plan
  before merge.

## Optional Benchmark Test Separation

- Optional benchmark tests must use `@pytest.mark.external_data`.
- Required unit/integration checks must not depend on optional external
  benchmark files.
- Default CI and local quick checks should exclude optional benchmark tests:
  `pytest -m "not external_data"`.
- If optional external benchmark fixtures are missing, tests must skip with a
  clear reason.

## Local Validation Commands

Run these from the repository root:

```bash
# Required default gate (fast path)
./.venv/bin/python -m pytest -m "not external_data"

# Optional benchmark checks (when fixtures exist)
./.venv/bin/python -m pytest -m "external_data"

# Lint
./.venv/bin/python -m ruff check src tests

# Type checks
./.venv/bin/python -m mypy src tests

# Confirm fixture footprint stays below 5 MB
du -sb test_data/benchmarks/nrtd
```

## Quality Command Policy For This Plan

- Full-repo `ruff`/`mypy` are still useful visibility checks, but pre-existing
  baseline debt outside release-gate benchmark/harness scope is non-blocking for this
  plan.
- release-gate release-gate blocking quality checks should focus on touched files and
  required benchmark/readiness tests in this checklist.
