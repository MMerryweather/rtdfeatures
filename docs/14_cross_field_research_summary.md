# Cross-Field Research Summary

## Purpose

Collect research inspiration without turning all ideas into API.

This document triages concepts from other fields that are mathematically similar to process-industry feature engineering. It preserves promising ideas while keeping `rtdfeatures` focused on kernel-based delayed-influence features.

## Decision Status

| Concept | Decision | Product role |
|---|---|---|
| graph-kernel feature planning | promote | strongest structural fit to process plants |
| regime-conditioned kernels | promote | plant behaviour is non-stationary |
| motif/fingerprint features | promote | shape-based instability features outside core |
| event features | promote | operations and batch/semi-continuous context |
| conformal prediction | modelling layer | outside `rtdfeatures` |
| state estimation / innovation features | promote cautiously | adjacent process state package/module |
| hydrology storage concepts | reframe | higher-level extensions built from kernels |
| TDA | park for EDA | outside production feature pipeline |
| SINDy | park for EDA | hypothesis and residual exploration only |
| generic tsfresh-style features | challenger only | weak physical interpretation |
| Tigramite / causal discovery | consume outputs only | optional adapter, no execution wrapper |

## Abstraction Principle

```
Do not create special feature frameworks for pyrometallurgy, flotation,
crushing, leaching, or any other subdomain unless the abstraction genuinely
breaks.
```

Pyrometallurgy examples are allowed as examples because they stress energy memory, inventory, events, and longer residence times. They must not become hard-coded API concepts.

## Concepts from Research

### Graph-Kernel Feature Planning

**Promote.** A process plant is naturally a graph of unit operations connected by streams carrying material, energy, and information. Graph-driven feature planning is the strongest structural fit for process plants. The compiler should turn a graph into `rtdfeatures` feature plans without GNN overhead.

### Regime-Conditioned Kernels

**Promote.** Plant behaviour is non-stationary. A single kernel often fails across operating regimes (stable, constrained, transitioning, startup, shutdown). Regime-conditioned kernels fit separate memory shapes per regime.

### Motif/Fingerprint Features

**Promote.** Rolling means and kernel-weighted averages miss shape-based instability. Matrix profiles and motif mining capture repeated subsequences, discords, and pattern recurrence. Implement outside `rtdfeatures`.

### Event Features

**Promote.** Operations context (taps, trips, changeovers, maintenance) is not naturally continuous. Time-since-event, event burst, and post-event phase features are generic process concepts.

### Conformal Prediction

**Modelling layer, outside `rtdfeatures`.** Valuable for high-consequence process decisions but belongs in a prediction/playbook layer, not in the kernel engine.

### State Estimation / Innovation Features

**Promote cautiously.** Latent process state (fill level, hardness, thermal inventory) and innovation residuals (observed minus expected) are powerful but belong in an adjacent process-state package.

### TDA

**Park for EDA.** Topological data analysis is promising for state discovery but not ready for production feature pipelines.

### SINDy

**Park for EDA.** Sparse dynamics discovery is useful for hypothesis generation and residual exploration but risky as production truth.

### tsfresh-Style Features

**Challenger only.** Generic statistical features lack physical interpretation. Use as challenger feature banks tagged with `physical_interpretation = weak`.

### Tigramite

**Consume outputs only.** Causal discovery runs outside `rtdfeatures`. An optional adapter converts Tigramite graph/value/p-value payloads into candidate lag evidence for constrained kernel fitting. Tigramite statistics are never interpreted as kernel weights.

## Package Boundary

Research concepts should not become `rtdfeatures` API concepts unless they reduce to kernel/spec/plan/evidence primitives. The pattern is:

- new kernel primitive => consider `rtdfeatures`
- new graph schema or graph traversal => `processfeaturecompiler`
- new feature family extending kernel features => `rtdfeatures` extension or compiler
- new prediction or uncertainty workflow => external modelling layer
- new EDA workflow => parked for exploration, not production
