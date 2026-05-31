"""Generate the nRTD laminar-flow worked example markdown with embedded PNG plots."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypedDict

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from rtdfeatures import (
    ExponentialKernelLearner,
    GammaKernelLearner,
    KernelFeatureBuilder,
    SimplexKernelLearner,
)
from rtdfeatures.diagnostics.fit import KernelFitResult
from rtdfeatures.learners import FixedDelayKernelLearner

INPUT_PATH = Path("test_data/benchmarks/nrtd/hsa_000_laminar_flow_signals.parquet")
OUTPUT_DIR = Path("docs/examples/generated")
OUTPUT_MD = Path("docs/examples/nrtd_laminar_flow_worked_example.md")
PLOT_LINK_PREFIX = "generated"
SIGNAL_PNG = OUTPUT_DIR / "nrtd_laminar_intro_timeseries.png"
FIT_PNG = OUTPUT_DIR / "nrtd_laminar_observed_vs_fit.png"
KERNEL_PNG = OUTPUT_DIR / "nrtd_laminar_kernel_profile.png"


@dataclass(frozen=True)
class Outcome:
    name: str
    result: KernelFitResult


class _CommonLearnerKwargs(TypedDict):
    min_lag: int
    max_lag: int
    loss: str
    validation_fraction: float


class _SupportsFit(Protocol):
    def fit(
        self,
        df: pl.DataFrame,
        *,
        input_col: str,
        target_col: str,
        time_col: str,
        order_by_time: bool = False,
    ) -> KernelFitResult: ...


def _fit(df: pl.DataFrame) -> tuple[Outcome, ...]:
    common: _CommonLearnerKwargs = {
        "min_lag": 0,
        "max_lag": 100,
        "loss": "huber",
        "validation_fraction": 0.2,
    }
    learners: tuple[tuple[str, _SupportsFit], ...] = (
        (
            "simplex",
            SimplexKernelLearner(
                seed=901, max_epochs=420, smoothness_penalty=0.0004, **common
            ),
        ),
        ("gamma", GammaKernelLearner(seed=902, max_epochs=420, **common)),
        ("exponential", ExponentialKernelLearner(seed=903, max_epochs=420, **common)),
        ("fixed_delay", FixedDelayKernelLearner(seed=904, **common)),
    )
    out: list[Outcome] = []
    for name, learner in learners:
        result = learner.fit(
            df, input_col="input_signal", target_col="target_signal", time_col="time"
        )
        out.append(Outcome(name=name, result=result))
    return tuple(out)


def _predict(
    input_signal: np.ndarray, lag_steps: tuple[int, ...], weights: tuple[float, ...]
) -> np.ndarray:
    pred = np.full(input_signal.shape[0], np.nan, dtype=np.float64)
    for idx in range(input_signal.shape[0]):
        total = 0.0
        valid = True
        for lag, weight in zip(lag_steps, weights):
            src = idx - lag
            if src < 0:
                valid = False
                break
            total += float(weight) * float(input_signal[src])
        if valid:
            pred[idx] = total
    return pred


def _write_plots(df: pl.DataFrame, best: Outcome) -> tuple[float, float, float]:
    time_axis = np.arange(df.height)
    input_signal = np.array(df["input_signal"].to_list(), dtype=np.float64)
    target_signal = np.array(df["target_signal"].to_list(), dtype=np.float64)

    kernel = best.result.kernel
    pred = _predict(
        input_signal,
        tuple(int(x) for x in kernel.lag_steps),
        tuple(float(x) for x in kernel.weights),
    )
    mask = np.isfinite(pred)
    rmse = (
        float(np.sqrt(np.mean((target_signal[mask] - pred[mask]) ** 2)))
        if np.any(mask)
        else float("nan")
    )
    corr = (
        float(np.corrcoef(target_signal[mask], pred[mask])[0, 1])
        if np.any(mask)
        else float("nan")
    )
    mae = float(np.mean(np.abs(target_signal[mask] - pred[mask]))) if np.any(mask) else float("nan")

    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=140)
    ax.plot(time_axis, input_signal, label="input_signal", color="#2563eb", linewidth=1.7)
    ax.plot(time_axis, target_signal, label="target_signal", color="#dc2626", linewidth=1.7)
    ax.set_title("Laminar-flow benchmark signals")
    ax.set_xlabel("row index")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(SIGNAL_PNG, format="png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=140)
    ax.plot(
        time_axis[mask],
        target_signal[mask],
        label="observed target",
        color="#111827",
        linewidth=1.7,
    )
    ax.plot(
        time_axis[mask],
        pred[mask],
        label=f"{best.name} fitted response",
        color="#16a34a",
        linewidth=1.7,
    )
    ax.set_title("Observed vs fitted response")
    ax.set_xlabel("row index")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIT_PNG, format="png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=140)
    ax.plot(kernel.lag_steps, kernel.weights, color="#7c3aed", linewidth=2.0)
    ax.fill_between(kernel.lag_steps, kernel.weights, 0.0, color="#7c3aed", alpha=0.15)
    ax.set_title(f"Recommended kernel profile ({best.name})")
    ax.set_xlabel("lag steps")
    ax.set_ylabel("weight")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(KERNEL_PNG, format="png")
    plt.close(fig)

    return rmse, mae, corr


def _render_markdown(
    df: pl.DataFrame,
    outcomes: tuple[Outcome, ...],
    best: Outcome,
    rmse: float,
    mae: float,
    corr: float,
) -> str:
    dt = float((df["time"][1] - df["time"][0]).total_seconds()) if df.height > 1 else float("nan")

    rows = [
        "| learner | validation_loss | no_lag | best_single_lag | mean_lag_s | warning_codes |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for out in outcomes:
        diag = out.result.fit_diagnostics
        base = out.result.baseline_comparison
        warns = ",".join(out.result.identifiability_report.warning_codes) or "none"
        rows.append(
            f"| `{out.name}` | {diag.validation_loss:.6f} | {base.no_lag_validation_loss:.6f} | "
            f"{base.best_single_lag_validation_loss:.6f} | {diag.mean_lag:.3f} | {warns} |"
        )

    builder = KernelFeatureBuilder(
        kernels={"learned": best.result.kernel},
        time_col="time",
        numeric_cols=["input_signal"],
    )
    preview = builder.transform(df).tail(8)

    evidence = [
        f"- `recommended_kernel`: `{best.name}`",
        "- `recommendation_status`: `recommended`",
        "- `recommendation_reason`: `lowest validation_loss among the fitted public learners`",
        f"- Fit RMSE: `{rmse:.6f}`",
        f"- Fit MAE: `{mae:.6f}`",
        f"- Observed/predicted correlation: `{corr:.4f}`",
        "- Fit evidence interpretation: correlation above 0.7 and low absolute error "
        "support a useful lag fit for feature generation.",
    ]

    parts = [
        "# nRTD Laminar-Flow Worked Example",
        "",
        "Generated from `test_data/benchmarks/nrtd/hsa_000_laminar_flow_signals.parquet`.",
        "",
        "## Introductory Time Series",
        "",
        f"![Laminar flow intro signals]({PLOT_LINK_PREFIX}/{SIGNAL_PNG.name})",
        "",
        "## Data Load",
        "",
        f"- rows: `{df.height}`",
        f"- columns: `{list(df.columns)}`",
        f"- inferred regular-grid step: `{dt:.9f}` seconds",
        "",
        "## Learner Setup",
        "",
        "Fitted with public learners only on `input_signal -> target_signal` using `time`:",
        "`SimplexKernelLearner`, `GammaKernelLearner`, `ExponentialKernelLearner`, "
        "`FixedDelayKernelLearner`.",
        "",
        "## Fit Diagnostics And Baselines",
        "",
        *rows,
        "",
        "## Fit Quality Plots",
        "",
        f"![Observed vs fitted response]({PLOT_LINK_PREFIX}/{FIT_PNG.name})",
        "",
        f"![Recommended kernel profile]({PLOT_LINK_PREFIX}/{KERNEL_PNG.name})",
        "",
        "## Recommended Kernel For Feature Generation",
        "",
        *evidence,
        "",
        "## Generated Feature Preview",
        "",
        "```text",
        str(preview),
        "```",
        "",
        "## Boundary: nRTD Fixture Scope",
        "",
        "This repository currently supports end-to-end learning from nRTD fixtures only for",
        "`laminar_flow` because it has a trusted input/target signal-pair fixture.",
        "",
        "`adler`, `cholette`, and `dispersion` remain reference-only benchmark context and",
        "must not be treated as learned-feature workflows until trusted signal-pair fixtures",
        "are added.",
    ]
    return "\n".join(parts)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pl.read_parquet(INPUT_PATH).select("time", "input_signal", "target_signal")
    if raw.schema.get("time") in (pl.Float32, pl.Float64):
        dt_seconds = float(raw["time"][1] - raw["time"][0]) if raw.height > 1 else 1.0
        dt_micros = max(1, int(round(dt_seconds * 1_000_000)))
        df = (
            raw.with_row_index("_row")
            .with_columns(
                (
                    pl.datetime(1970, 1, 1)
                    + pl.col("_row") * pl.duration(microseconds=dt_micros)
                ).alias("time")
            )
            .drop("_row")
        )
    else:
        df = raw
    outcomes = _fit(df)
    best = min(outcomes, key=lambda o: float(o.result.fit_diagnostics.validation_loss))
    rmse, mae, corr = _write_plots(df, best)
    markdown = _render_markdown(df, outcomes, best, rmse, mae, corr)
    OUTPUT_MD.write_text(markdown)


if __name__ == "__main__":
    main()
