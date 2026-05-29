# Interpretation Boundary

A learned kernel is not automatically an RTD. This document defines the labels and guidelines for responsible kernel interpretation.

## Interpretation labels

Every generated feature carries an `interpretation` label in its `FeatureEvidence`. Choose from:

| Label | Meaning | When to use |
|---|---|---|
| `material_memory` | Physically plausible material or tracer residence interpretation | Independent evidence of material transport (tracer tests, vessel geometry, process knowledge) |
| `process_response` | Delayed predictive relationship, not necessarily material residence | Safe default for most process-data relationships |
| `statistical_pattern` | Useful lagged association only | Weak diagnostics, known non-causal coupling, or purely data-driven exploration |
| `unknown` | No interpretation assigned | Default when evidence is incomplete |

## Evidence completeness labels

Separate from interpretation, evidence completeness records how much support exists:

| Label | Meaning |
|---|---|
| `kernel_only` | Kernel shape only, no fit diagnostics |
| `fit_evidence` | Includes fit diagnostics |
| `comparison_evidence` | Includes baseline comparison |
| `bootstrap_evidence` | Includes bootstrap uncertainty |
| `full_evidence` | All of the above |

## Guidelines

### Do claim RTD (material_memory) when

- Tracer injection confirms the lag distribution
- Vessel geometry and flow regime are known and match the kernel shape
- Input and target share a material path (e.g. inlet → outlet concentration)
- Diagnostics are strong and baselines are convincingly beaten

### Do not claim RTD when

- The relationship could be driven by control logic, recycle, or measurement lag
- Only statistical correlation supports the lag shape
- Diagnostics show boundary-piled mass, diffuseness, or identifiability warnings
- The kernel does not materially beat `best_single_lag`

### Default interpretation

Use `process_response` unless material-transport evidence is independently available. This is the conservative choice.

## Examples

### Physical RTD-like case

Tracer injected at a reactor inlet appears at the outlet with a delayed, spread peak. Fitting a kernel recovers a shape consistent with known vessel geometry and flow. **Label:** `material_memory`.

### Response-only case

Mill power responds to feed hardness after a delay. The lag is physically plausible (grinding takes time) but there is no material residence — it is a mechanical response. **Label:** `process_response`.

### Weak identifiability case

The learned kernel beats `no_lag` but shows boundary-piled mass or a large train/validation gap. **Label:** `statistical_pattern` or `unknown`; do not use for critical decisions without further investigation.

### Misleading low-loss case

A parametric family (e.g. Gamma) fits well numerically but the true lag shape is multimodal (bypass stream). The neat family assumption hides the process reality. Prefer an empirical (simplex) fit and label conservatively.

## Key statements

> Feature evidence is not final model evidence.

> A learned response kernel is not automatically an RTD.

> Kernel diagnostics warn about trustworthiness — they do not rank downstream models.

## See also

- [Kernels and RTDs](kernels-and-rtds.md) — what kernels and RTDs are
- [Identifiability](identifiability.md) — when a kernel is worth trusting
- [Feature evidence](../user-guide/feature-evidence.md) — how interpretation labels are attached
- [02_core_concepts.md](../02_core_concepts.md) — normative terminology reference
