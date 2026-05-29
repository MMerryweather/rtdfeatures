# Local Development Setup

## Prerequisites

- Python **>=3.10**
- `git`
- (Recommended) a virtual environment tool: `venv`, `conda`, or `virtualenv`

## Clone and install

```bash
git clone https://github.com/<org-or-user>/rtdfeatures
cd rtdfeatures
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev,test,examples]"
```

## Dev dependencies

Install the `dev` extra to get `mypy`, `ruff`, and `pytest`:

```bash
pip install -e ".[dev]"
```

Extras:

| Extra       | Includes                               |
|-------------|----------------------------------------|
| `dev`       | mypy, ruff, pytest, pyarrow            |
| `test`      | pytest, pyarrow                        |
| `examples`  | matplotlib, seaborn                    |
| `benchmark` | pyarrow                                |

## Run tests

```bash
pytest -m "not external_data"
```

Tests that require external benchmark files are marked `external_data` and excluded from default CI runs.

## Run lint and type checks

```bash
ruff check .
mypy src tests
```

## Build and verify package

```bash
python -m build
twine check dist/*
```

## Run examples

```bash
python examples/01_quickstart_simplex.py
python examples/02_parametric_vs_empirical.py
# ... see examples/ directory for all available examples
```

## Run benchmarks

```bash
python benchmarks/smoke_transform_performance.py
```
