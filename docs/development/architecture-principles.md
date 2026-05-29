# Architecture principles

## Product centre

`rtdfeatures` is not a generic time-series package.
The package learns or defines causal lag kernels and generates auditable lag-weighted features.

## Core mental model

Golden path: fit or define kernel -> generate `TransformResult` -> inspect features/report/registry.
`Kernel`, `KernelFitResult`, `TransformResult`, `FeatureRegistry`, and `FeatureSpec` are core concepts.
Learned kernels do not prove causality.
RTD interpretation requires independent evidence.

## Stable boundaries

Public API stays small and deliberate.
Optional integrations live under `rtdfeatures.integrations` and are not root-exported by default.
Optional integrations are thin adapters around the core API. They adapt external ecosystem conventions without owning core feature-generation semantics.

## Internal design rules

Internal code prefers one source of truth over duplicate construction or parsing.
Parametric kernel families are registered in one place. Family metadata, parameter validation, and weight generation should not be duplicated across switches, summaries, and constructors.
Explicit parametric kernels share internal initialisation and summary helpers. Public constructors remain separate and readable; internal validation, lag-grid construction, weight generation, and summary extension are centralised.
Registry metadata must not be reconstructed from feature-name parsing when metadata is available at generation time.
Repeated learner result assembly must be centralised.
Learner fit-result assembly is centralised. Learner classes are orchestration layers: prepare data, optimise parameters or weights, then delegate baseline evaluation and result assembly to private helpers.
Learner classes do not inherit from each other merely to share helpers. Shared optimisation and validation mechanics live in private functions. Public learner classes remain thin orchestration layers around preparation, optimisation, baseline evaluation, and result assembly.
Identifiability warnings are shared learner diagnostics, not simplex-specific behaviour. Warning policy and report construction live in a private helper module so empirical and parametric learners use one interpretation path.
Feature-generation internals use an accumulator to keep arrays, metadata, missingness, and per-kernel feature lists together. Parallel dictionaries should not be manually updated in multiple places.
Feature-family logic is isolated by numeric, categorical, and age blocks. Builder orchestration should wire blocks together rather than mixing computation, metadata, and reporting in one method.

## Duplication policy

Prefer shared helpers when multiple learners or builders construct the same contract objects.
Avoid duplicate parsing or object assembly paths for the same runtime concept.

## Special-case policy

Special cases must be named and isolated behind explicit helpers.

## Result-object policy

Result objects should carry structured metadata directly instead of requiring downstream reconstruction.

## Documentation policy

Runtime comments describe current design, not work-package history.

## Refactor safety rules

Refactors must preserve constrained-kernel semantics and the golden path contract.
