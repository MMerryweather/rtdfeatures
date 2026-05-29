#!/usr/bin/env python3
"""Minimal sklearn adapter example: synthetic data → KernelFeatureTransformer → features."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


from rtdfeatures.integrations.sklearn import KernelFeatureTransformer
from rtdfeatures.kernels import FixedDelayKernel
from rtdfeatures.synthetic import make_single_delay_dataset


def main() -> None:
    ds = make_single_delay_dataset(n_rows=50, dt=1.0, seed=7)
    df = ds.data

    kernel = FixedDelayKernel(delay_steps=6, max_lag_steps=10, dt=1.0)
    transformer = KernelFeatureTransformer(
        time_col="time",
        kernels={"feature": kernel},
        numeric_cols=["input_signal"],
        include_time_col=False,
    )

    result = transformer.fit_transform(df)
    print(f"Output shape: {result.shape}")
    print(f"Feature names: {list(result.columns)}")
    print("08_sklearn_adapter.py completed.")


if __name__ == "__main__":
    main()
