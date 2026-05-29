# Contributing

## Local environment setup, tests, and checks

See [docs/development/local-dev.md](docs/development/local-dev.md) for setup instructions, running tests, lint, type checks, building, and running examples.

## Public API rules

- Keep the root `rtdfeatures` namespace small. Only expose user-facing classes and functions.
- Add internal implementation to submodules (e.g. `rtdfeatures.kernels`, `rtdfeatures.learners`, `rtdfeatures.features`).
- Re-export public symbols in `rtdfeatures/__init__.py`.
- Do not import private submodule paths into user code.

## Docs rules

- User-facing behaviour changes must include doc updates.
- Docstrings should explain *what* and *why*, not *how*.
- New features should be reflected in the examples gallery or quickstart if applicable.

## Where new functionality belongs

| What                         | Where                          |
|------------------------------|--------------------------------|
| Kernel objects               | `src/rtdfeatures/kernels/`     |
| Learners                     | `src/rtdfeatures/learners/`    |
| Feature generation           | `src/rtdfeatures/features/`    |
| Diagnostics                  | `src/rtdfeatures/diagnostics/` |
| Synthetic data / test utils  | `src/rtdfeatures/synthetic.py` |
| Public re-exports            | `src/rtdfeatures/__init__.py`  |
| Examples                     | `examples/`                    |
| Docs                         | `docs/`                        |

## Release boundaries

This package does **not** cross into:

- prediction / forecasting
- historian or real-time connectors
- control-system or digital-twin simulation
- plant-topology genealogy engines
- attention / transformer / GNN causal discovery
- domain-specific (metallurgy-only, mining-only) APIs

Keep the product wedge narrow: learn the lag, validate the kernel, generate the features.

## How to regenerate examples

```bash
# From the repo root with the examples extra installed:
python examples/parametric_empirical_baseline_fits.py
```

To add a new example, create a `.py` file in `examples/` and update any gallery docs that reference it.

## PR checklist

Before submitting a pull request:

- [ ] Does the change affect the public API?
- [ ] Are docs updated for user-facing changes?
- [ ] Are examples updated (or a new example added) if the change introduces new functionality?
- [ ] Are tests added or updated?
- [ ] Have you run `ruff check .` and `mypy src tests`?
- [ ] Have you run `pytest -m "not external_data"` and confirmed all tests pass?
