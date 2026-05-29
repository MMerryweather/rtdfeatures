#!/usr/bin/env python3
"""Compare parametric vs empirical fits on a gamma-like delay dataset."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from rtdfeatures import (
    DelayedExponentialKernel,
    ExponentialKernelLearner,
    GammaKernelLearner,
    SimplexKernelLearner,
)
from rtdfeatures.reporting import learner_diagnostic_comparison_table
from rtdfeatures.synthetic import make_gamma_kernel_dataset

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def main() -> None:
    ds = make_gamma_kernel_dataset(
        n_rows=500, dt=60.0, seed=53, min_lag_steps=1, max_lag_steps=12,
        shape_alpha=3.0, rate_beta=0.06, noise_std=0.03,
    )
    df = ds.data
    print(f"Data: {df.shape[0]} rows")
    true_meta = ds.true_kernels["input_signal->target_signal"]
    print(f"True kernel metadata: mean_lag={true_meta['mean_lag']:.1f}s")

    min_lag, max_lag = 1, 12

    simplex_learner = SimplexKernelLearner(
        min_lag=min_lag, max_lag=max_lag, seed=101, max_epochs=400
    )
    simplex_result = simplex_learner.fit(
        df, input_col="input_signal", target_col="target_signal", time_col="time",
    )
    gamma_learner = GammaKernelLearner(
        min_lag=min_lag, max_lag=max_lag, seed=102, max_epochs=400
    )
    gamma_result = gamma_learner.fit(
        df, input_col="input_signal", target_col="target_signal", time_col="time",
    )
    exp_learner = ExponentialKernelLearner(
        min_lag=min_lag, max_lag=max_lag, seed=103, max_epochs=400
    )
    exp_result = exp_learner.fit(
        df, input_col="input_signal", target_col="target_signal", time_col="time",
    )

    # Fixed parametric kernel (no learner) — DelayedExponentialKernel has no
    # dedicated learner in V1 but can still be constructed directly.
    fixed_delayed_exp = DelayedExponentialKernel(
        delay=30.0,
        rate_lambda=0.08,
        min_lag_steps=min_lag,
        max_lag_steps=max_lag,
        dt=60.0,
        name="delayed_exp_fixed",
    )

    results = {
        "Empirical (simplex)": simplex_result,
        "Parametric (gamma)": gamma_result,
        "Parametric (exponential)": exp_result,
    }

    print("\nFixed kernel (not learned):")
    print(f"  DelayedExponentialKernel: mean_lag={fixed_delayed_exp.mean_lag():.1f}s, "
          f"weights={[round(w, 3) for w in fixed_delayed_exp.weights]}")

    print("\nKernel comparison:")
    for name, r in results.items():
        k = r.kernel
        d = r.fit_diagnostics
        print(f"  {name}: loss={d.validation_loss:.5f}, mean_lag={d.mean_lag:.1f}s, "
              f"weights={[round(w, 3) for w in k.weights]}")

    comparison = learner_diagnostic_comparison_table(
        fit_results_by_family={
            "simplex": simplex_result,
            "gamma": gamma_result,
            "exponential": exp_result,
        }
    )
    print(f"\nDiagnostic comparison:\n{comparison}")

    if _HAS_MPL:
        fig, ax = plt.subplots(figsize=(8, 4))
        for label, r in results.items():
            k = r.kernel
            ax.plot(list(k.lag_steps), list(k.weights), marker="o", label=label)
        ax.set_xlabel("Lag step")
        ax.set_ylabel("Kernel weight")
        ax.set_title("Parametric vs Empirical Kernel Fit")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = "/tmp/parametric_vs_empirical.png"
        fig.savefig(out_path, dpi=100)
        print(f"\nPlot saved to {out_path}")
        plt.close(fig)
    else:
        print("\n(matplotlib not available, skipping plot)")

    print("\nDone — 02_parametric_vs_empirical.py completed.")


if __name__ == "__main__":
    main()
