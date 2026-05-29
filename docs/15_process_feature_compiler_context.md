# Process Feature Compiler Context

## Product Positioning

The strongest product positioning is:

> a process-feature compiler for industrial ML.

It should compile process structure, delayed influence, material and energy memory, operating context, events, recycles, and source and quality history into feature plans, feature tables, validation reports, and evidence artifacts that plug into existing Databricks, SageMaker, MLflow, or internal MLOps workflows.

It should not become a replacement for MLOps.

## Package-Boundary Decisions

```
rtdfeatures:
  delayed influence features

processfeaturecompiler:
  graph-aware, context-aware feature planning and validation

external MLOps:
  training, tracking, registry, deployment
```

`rtdfeatures` must remain the kernel engine. It answers: *What past material or process history matters now, and how should it be weighted?* It does not answer questions about graph structure, regimes, motifs, models, or setpoints.

`processfeaturecompiler` is positioned **outside `rtdfeatures`**. Dependency direction is:

```
rtdfeatures <- processfeaturecompiler <- external MLOps
```

`processfeaturecompiler` may depend on `rtdfeatures`. `rtdfeatures` must not depend on `processfeaturecompiler`.

## Candidate Compiler Modules

```
processfeaturecompiler.graph
processfeaturecompiler.validation
processfeaturecompiler.evidence
processfeaturecompiler.regimes
processfeaturecompiler.motifs
processfeaturecompiler.events
processfeaturecompiler.state
processfeaturecompiler.eda
```

This avoids premature package fragmentation. Later, stable modules may split into separate packages such as `flowsheetfeatures`, `processfeatures`, or `processmodels`.

## Compiler Responsibilities

- turn generic process graphs into `rtdfeatures` feature plans
- validate feature plans before execution
- create machine-generated evidence artifacts
- support regime, motif, event, and state feature workflows outside `rtdfeatures`
- keep broad EDA workflows outside production feature APIs

## Compiler Non-Responsibilities

- kernel fitting internals
- kernel feature execution
- final predictive modelling
- conformal prediction
- replacing Databricks, SageMaker, MLflow, or internal MLOps

## Abstraction Principle

Do not create separate pyrometallurgy-specific, flotation-specific, or crushing-specific package concepts unless there is a high burden of proof. The reusable abstraction is: process graph, unit operation, stream, inventory, state variable, event, measurement, control action, material transfer, energy transfer, quality transfer, and residence/response kernel.

Pyrometallurgy, flotation, crushing, leaching, and other process areas are different applications of this abstraction.
