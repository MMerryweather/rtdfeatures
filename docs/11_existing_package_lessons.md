# Existing Package Lessons

## Goal

Take useful ideas from adjacent packages without inheriting their unnecessary assumptions or architecture.

## From `rtdpy`

Useful lessons:

- Clear RTD object model
- Analytical kernel shapes
- Mean, variance, and step-response summaries
- Straightforward documentation
- Tests against known RTD behaviour

Avoid:

- Using it as the structural base for this package

## From `bio-rtd`

Useful lessons:

- Stream and component thinking
- Process-friendly vocabulary
- Concentration and flow-profile ideas

Avoid:

- Carrying over bioprocess-specific assumptions into a generic package

## From `impulseest`

Useful lessons:

- Impulse-response estimation as a baseline
- Regularised FIR fitting ideas

Avoid:

- Presenting unconstrained impulse responses as physically meaningful RTDs

## From `TCDF`

Useful lessons:

- Temporal models can infer lag structure
- Causal validation matters

Avoid:

- Starting with attention-heavy or CNN-based causal discovery in `v0.1`

## From Tigramite-Style Outputs (Optional Adapter Boundary)

Useful lessons:

- Graph-discovery outputs can provide candidate link structure and lag ranges
- Link-value and p-value payloads are useful as screening evidence before
  constrained kernel fitting

Avoid:

- Treating Tigramite test statistics as constrained kernel weights
- Pulling Tigramite causal discovery execution into `rtdfeatures`
- Making Tigramite a required dependency for core package use
- Treating unsupported graph marks as valid link evidence without warnings

## Cross-Field Research Triage

| Concept | Decision | Where |
| --- | --- | --- |
| graph-kernel features | promote | external compiler plus `rtdfeatures` specs |
| regimes | promote later | process compiler |
| motifs | promote later | process compiler |
| conformal prediction | modelling layer | outside `rtdfeatures` |
| TDA | EDA only | outside `rtdfeatures` |
| SINDy | EDA only | outside `rtdfeatures` |
| tsfresh-style features | challenger only | outside `rtdfeatures` |
| Tigramite | consume outputs only | optional adapter |

Do not let research concepts become `rtdfeatures` API concepts unless they
reduce to kernel/spec/plan/evidence primitives.

## General Lesson

`rtdfeatures` should borrow ideas, not identity. The package should keep its own kernel-first, process-engineering, domain-neutral design.

## Additional Caution

Do not expand the package into a generic forecasting, attention-model, or plant-topology framework before the core loop is solid:

```text
learn the lag
validate the kernel
generate the features
stop there
```
