# Product Charter

`rtdfeatures` learns constrained causal kernels from regular-grid process time series and converts them into Polars feature tables plus diagnostics. It does not produce final predictive models.

## Product Loop

```text
process time-series
-> learn kernel
-> validate against baselines
-> generate features
-> export feature table
```

## Interoperability Loop (Post-v1.0 Seam)

```text
kernel / feature specs
-> validate feature plan
-> generate kernel-weighted features
-> emit feature table + evidence artifacts
```

## Why It Exists

In process systems, downstream behaviour often depends on upstream state only after a delay and over a spread of time. Most teams approximate this with fixed lags or rolling windows. `rtdfeatures` exists to replace those ad hoc features with constrained, inspectable kernels and consistent feature generation.

## Stable Product Boundary

`rtdfeatures` is the kernel engine for process-memory features.

It owns:

- kernel objects
- kernel learners
- kernel comparison
- bootstrap uncertainty
- kernel feature generation
- feature evidence
- serialisable kernel and feature specs

It does not own:

- graph traversal
- process graph schema ownership
- regime detection
- motif mining
- final modelling
- MLOps orchestration
- process-specific templates

## Post-v1.0 Extension Principle

External planners may compile graph, regime, event, or SME hypotheses into `rtdfeatures` feature plans. `rtdfeatures` executes those plans, validates kernel contracts, and emits feature and evidence artifacts.

## Abstraction Principle

Keep abstractions generic across process domains:

```text
unit operation
stream
inventory
state variable
event
measurement
control action
material transfer
energy transfer
quality transfer
residence / response kernel
```

Avoid domain-specific API concepts unless a high burden of proof is met.

## Primary Users

- Process engineers
- Data scientists building downstream models
- Industrial analytics teams working with process time series

## Terminology Rule

- `kernel` is the generic object
- `RTD kernel` is a physical interpretation when material or tracer propagation is justified
- `response kernel` is a non-physical delayed-influence interpretation

The package name stays `rtdfeatures`, but the package itself is broader than strict RTD modelling.

## Stable Contracts

- Public data interface remains Polars DataFrame in and Polars DataFrame out.
- Learned kernels remain constrained: causal, non-negative, sum-to-one, bounded lag.
- Diagnostics remain part of the public package contract, not optional polish.

## Current-Scope Delivery Notes

This charter re-baselines wording for the stable `v1.0` boundary and post-`v1.0` seam. It does not promise implementation beyond the currently documented package scope.
