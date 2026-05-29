# Categorical Genealogy

Categorical features track how the fraction of each category level changes over the kernel-weighted window. This is useful when the source of a stream changes over time (blend composition, feed source, operating mode).

## Fraction features

For each category level `L` at time `t`:

```
frac_t(L) = sum_k w_k * I(category_{t-k} = L) / sum_k w_k
```

Without a `weight_col`, this is the kernel-weighted fraction of observations in category `L`. With a `weight_col`, it becomes the kernel-weighted mass fraction.

## Entropy feature

Categorical entropy measures how evenly distributed the categories are over the lag window:

```
entropy_t = - sum_L p_t(L) * log(p_t(L))
```

High entropy means the window spans a diverse mix of categories. Low entropy means one category dominates.

## Example

Given a column `feed_source` with levels `{"ore_A", "ore_B", "blend"}`:

```python
builder = KernelFeatureBuilder(
    kernels={"learned": fit.kernel},
    time_col="time",
    category_cols=["feed_source"],
)
features = builder.transform(df)
# Generates:
#   learned_cat_feed_source_ore_A_frac
#   learned_cat_feed_source_ore_B_frac
#   learned_cat_feed_source_blend_frac
#   learned_cat_feed_source_entropy
```

## With weight column

When `weight_col="mass_flow"` is provided, the fraction becomes mass-weighted:

```
frac_t(L) = sum_k w_k * m_{t-k} * I(category_{t-k} = L) / sum_k w_k * m_{t-k}
```

This gives a throughput-aware view of source contribution.

## Interpretation

- Monotonically changing fraction features can indicate a blend transition propagating through the process.
- Entropy dips during periods of single-source operation and rises during blending.
- Compare fraction features against known operational changes to validate kernel lag estimates.

## See also

- [Generating features](generating-features.md) — general feature generation workflow
- [Feature evidence](feature-evidence.md) — attaching interpretation labels to categorical features
