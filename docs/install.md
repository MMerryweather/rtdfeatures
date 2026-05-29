# Installation

## Python version

Requires **Python >= 3.10**.

## From PyPI

```bash
pip install rtdfeatures
pip install rtdfeatures[sklearn]
pip install rtdfeatures[examples]
pip install rtdfeatures[benchmark]
```

## From source

```bash
git clone https://github.com/<org-or-user>/rtdfeatures.git
cd rtdfeatures
pip install -e .
```

## Dependencies

Core dependencies (installed automatically):

- `numpy >= 1.24`
- `polars >= 0.20`
- `torch >= 2.0`

`torch` is a core dependency because learned kernels use PyTorch for constrained optimisation. CPU operation is the default and expected path — no GPU is required.

## Optional extras

| Extra | Packages | Use |
|---|---|---|
| `dev` | pytest, ruff, mypy, pyarrow | Development and testing |
| `test` | pytest, pyarrow | Running tests |
| `examples` | matplotlib, seaborn | Running example scripts |
| `benchmark` | pyarrow | Benchmark-layer operations |
| `sklearn` | scikit-learn>=1.3, pandas>=2.0 | Enables `KernelFeatureTransformer` |

```bash
pip install "rtdfeatures[dev,examples]"
```

## Verification

```python
import rtdfeatures
print(rtdfeatures.__version__)
```
