# Documentation Delivery Decision (`v0.3`)

## Status

Accepted.

## Decision

Do not ship a documentation site for `v0.3`. Keep docs as repository markdown.

## Why

- A hosted docs service is not required for the current product wedge.
- Markdown docs already cover contract, architecture, diagnostics, and examples.
- Adding a docs-site stack now increases maintenance and dependency overhead
  without improving package correctness.

## How We Keep Docs Usable Without A Site

- `README.md` remains the top-level entrypoint with direct links to core docs.
- Executable examples are explicitly labeled with `docs-test:` markers.
- Non-executable examples are explicitly marked as illustrative.
- Tests validate runnable README/docs examples and practical internal links.

## Consequences

- No docs build or deployment pipeline is required for `v0.3`.
- Docs quality is enforced through repository tests instead of hosted previews.
- A minimal static site can be revisited later if markdown navigation becomes a
  clear usability blocker.
