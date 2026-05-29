"""Generate static PNG examples for RTD-style kernel fit evidence.

The script writes one PNG per physical scenario instead of one crowded canvas.
It compares synthetic truth, empirical simplex, parametric gamma, and parametric
exponential fits, then writes a separate genealogy/categorical weighting PNG.

Run from the repository root:

    python examples/parametric_empirical_baseline_fits.py
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from html import escape
from pathlib import Path

# This documentation example is small and CPU-only. Disable CUDA before importing
# rtdfeatures/torch so stale local NVIDIA drivers do not emit irrelevant warnings.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import polars as pl
from matplotlib import pyplot as plt

from rtdfeatures import ExponentialKernelLearner, GammaKernelLearner, SimplexKernelLearner
from rtdfeatures.kernels import Kernel
from rtdfeatures.synthetic import (
    KernelMetadata,
    SyntheticDataset,
    make_delayed_exponential_kernel_dataset,
    make_gamma_kernel_dataset,
    make_single_delay_dataset,
)

OUTPUT_DIR = Path("docs/examples/generated")
OUTPUT_MD = Path("docs/examples/parametric_empirical_fit_gallery.md")
PLOT_LINK_PREFIX = "generated"

SERIES_STYLES = {
    "truth": {"stroke": "#111827", "dash": "", "width": "3.2", "marker": "6"},
    "empirical simplex": {"stroke": "#2563eb", "dash": "", "width": "2.7", "marker": "4"},
    "parametric gamma": {"stroke": "#16a34a", "dash": "7 5", "width": "2.5", "marker": "4"},
    "parametric exponential": {"stroke": "#dc2626", "dash": "3 4", "width": "2.5", "marker": "4"},
}

CATEGORY_COLOURS = {
    "Pit A": "#2563eb",
    "Pit B": "#16a34a",
    "Pit C": "#f97316",
    "Unknown": "#94a3b8",
}


@dataclass(frozen=True)
class FitScenario:
    """A physical scenario and its expected kernel shape."""

    name: str
    slug: str
    physical_condition: str
    dataset_factory: Callable[[], SyntheticDataset]
    min_lag: int
    max_lag: int
    expected_shape: str
    interpretation_note: str


@dataclass(frozen=True)
class FittedKernel:
    """A display-friendly fitted kernel result."""

    label: str
    family: str
    weights: tuple[float, ...]
    lag_steps: tuple[int, ...]
    validation_loss: float | None
    warning_codes: tuple[str, ...]
    mean_lag: float
    p50_lag: float
    p90_lag: float


@dataclass(frozen=True)
class ScenarioResult:
    """Gallery-ready scenario output."""

    scenario: FitScenario
    true_kernel: FittedKernel
    fitted_kernels: tuple[FittedKernel, ...]
    plot_path: Path


def _regular_time(n_rows: int, dt: float) -> list[datetime]:
    start = datetime(2020, 1, 1)
    return [start + timedelta(seconds=i * dt) for i in range(n_rows)]


def _normalise(weights: list[float]) -> list[float]:
    total = float(sum(weights))
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    return [float(weight / total) for weight in weights]


def _kernel_metadata(lag_steps: list[int], weights: list[float], *, dt: float) -> KernelMetadata:
    normalised = _normalise(weights)

    def percentile(q: float) -> float:
        cumulative = 0.0
        for step, weight in zip(lag_steps, normalised):
            cumulative += weight
            if cumulative >= q:
                return float(step * dt)
        return float(lag_steps[-1] * dt)

    return {
        "lag_steps": [int(step) for step in lag_steps],
        "weights": [float(weight) for weight in normalised],
        "dt": float(dt),
        "min_lag": int(lag_steps[0]),
        "max_lag": int(lag_steps[-1]),
        "mean_lag": float(sum(step * weight for step, weight in zip(lag_steps, normalised)) * dt),
        "p50_lag": percentile(0.5),
        "p90_lag": percentile(0.9),
    }


def _apply_kernel(signal: np.ndarray, lag_steps: list[int], weights: list[float]) -> np.ndarray:
    out = np.zeros(signal.shape[0], dtype=np.float64)
    for row_idx in range(signal.shape[0]):
        value = 0.0
        for lag, weight in zip(lag_steps, weights):
            source_idx = row_idx - lag
            if source_idx >= 0:
                value += weight * float(signal[source_idx])
        out[row_idx] = value
    return out


def _custom_dataset(
    *,
    name: str,
    n_rows: int,
    dt: float,
    seed: int,
    lag_steps: list[int],
    weights: list[float],
    noise_std: float,
) -> SyntheticDataset:
    metadata = _kernel_metadata(lag_steps, weights, dt=dt)
    rng = np.random.default_rng(seed)
    impulse_like = rng.normal(0.0, 1.0, size=n_rows)
    slow_wave = np.sin(np.linspace(0.0, 10.0 * np.pi, n_rows, dtype=np.float64))
    medium_wave = 0.5 * np.sin(np.linspace(0.0, 32.0 * np.pi, n_rows, dtype=np.float64))
    x = (0.58 * impulse_like) + (0.28 * slow_wave) + (0.14 * medium_wave)
    y = _apply_kernel(x, metadata["lag_steps"], metadata["weights"])
    if noise_std > 0.0:
        y = y + rng.normal(0.0, noise_std, size=n_rows)
    return SyntheticDataset(
        data=pl.DataFrame({
            "time": _regular_time(n_rows, dt),
            "input_signal": x,
            "target_signal": y,
        }),
        true_kernels={"input_signal->target_signal": metadata},
        scenario={
            "name": name,
            "seed": seed,
            "n_rows": n_rows,
            "dt": dt,
            "params": {"lag_steps": lag_steps, "weights": weights, "noise_std": noise_std},
        },
    )


def _multimodal_parallel_dataset() -> SyntheticDataset:
    return _custom_dataset(
        name="multimodal_parallel_paths",
        n_rows=620,
        dt=60.0,
        seed=81,
        lag_steps=[2, 3, 9, 10, 11],
        weights=[0.24, 0.26, 0.16, 0.24, 0.10],
        noise_std=0.025,
    )


def _bypass_plus_recycle_dataset() -> SyntheticDataset:
    return _custom_dataset(
        name="bypass_plus_recycle_tail",
        n_rows=620,
        dt=60.0,
        seed=83,
        lag_steps=[1, 5, 6, 7, 8, 11, 14, 17],
        weights=[0.32, 0.10, 0.12, 0.12, 0.10, 0.10, 0.08, 0.06],
        noise_std=0.03,
    )


def _true_kernel_from_metadata(metadata: KernelMetadata) -> FittedKernel:
    lag_steps = tuple(int(step) for step in metadata["lag_steps"])
    weights = tuple(float(weight) for weight in metadata["weights"])
    return FittedKernel(
        label="truth",
        family=str(metadata.get("parametric_family", "synthetic")),
        weights=weights,
        lag_steps=lag_steps,
        validation_loss=None,
        warning_codes=(),
        mean_lag=float(metadata["mean_lag"]),
        p50_lag=float(metadata["p50_lag"]),
        p90_lag=float(metadata["p90_lag"]),
    )


def _fitted_kernel_from_result(label: str, family: str, fit_result: object) -> FittedKernel:
    kernel: Kernel = fit_result.kernel  # type: ignore[attr-defined]
    diagnostics = fit_result.fit_diagnostics  # type: ignore[attr-defined]
    identifiability = fit_result.identifiability_report  # type: ignore[attr-defined]
    return FittedKernel(
        label=label,
        family=family,
        weights=tuple(float(weight) for weight in kernel.weights),
        lag_steps=tuple(int(step) for step in kernel.lag_steps),
        validation_loss=float(diagnostics.validation_loss),
        warning_codes=tuple(str(code) for code in identifiability.warning_codes),
        mean_lag=float(diagnostics.mean_lag),
        p50_lag=float(diagnostics.p50_lag),
        p90_lag=float(diagnostics.p90_lag),
    )


def _fit_scenario(scenario: FitScenario) -> ScenarioResult:
    dataset = scenario.dataset_factory()
    df = dataset.data
    true_metadata = dataset.true_kernels["input_signal->target_signal"]

    simplex = SimplexKernelLearner(
        min_lag=scenario.min_lag,
        max_lag=scenario.max_lag,
        seed=101,
        loss="huber",
        max_epochs=420,
        smoothness_penalty=0.0004,
    ).fit(df, input_col="input_signal", target_col="target_signal", time_col="time")
    gamma = GammaKernelLearner(
        min_lag=scenario.min_lag,
        max_lag=scenario.max_lag,
        seed=102,
        loss="huber",
        max_epochs=420,
    ).fit(df, input_col="input_signal", target_col="target_signal", time_col="time")
    exponential = ExponentialKernelLearner(
        min_lag=scenario.min_lag,
        max_lag=scenario.max_lag,
        seed=103,
        loss="huber",
        max_epochs=420,
    ).fit(df, input_col="input_signal", target_col="target_signal", time_col="time")

    return ScenarioResult(
        scenario=scenario,
        true_kernel=_true_kernel_from_metadata(true_metadata),
        fitted_kernels=(
            _fitted_kernel_from_result("empirical simplex", "empirical", simplex),
            _fitted_kernel_from_result("parametric gamma", "parametric", gamma),
            _fitted_kernel_from_result("parametric exponential", "parametric", exponential),
        ),
        plot_path=OUTPUT_DIR / f"fit_{scenario.slug}.png",
    )


def _write_fit_png(result: ScenarioResult) -> None:
    kernels = _all_kernels(result)
    lag_union = sorted({int(step) for kernel in kernels for step in kernel.lag_steps})
    fig, ax = plt.subplots(figsize=(10.5, 5.5), dpi=140)
    for kernel in kernels:
        ys = [_weight_at(kernel, lag_step) for lag_step in lag_union]
        style = SERIES_STYLES.get(kernel.label, {"stroke": "#334155"})
        ax.plot(
            lag_union,
            ys,
            label=kernel.label,
            linewidth=2.2 if kernel.label == "truth" else 1.9,
            color=style["stroke"],
        )
    ax.set_title(result.scenario.name)
    ax.set_xlabel("lag step")
    ax.set_ylabel("weight")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper right", fontsize=9)
    ax.text(
        0.01,
        0.98,
        f"Physical condition: {result.scenario.physical_condition}\n"
        f"Expected shape: {result.scenario.expected_shape}\n"
        f"Interpretation: {result.scenario.interpretation_note}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.6,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cbd5e1"},
    )
    fig.tight_layout()
    fig.savefig(result.plot_path, format="png")
    plt.close(fig)


def _write_genealogy_png(path: Path) -> None:
    categories = ["Pit A", "Pit B", "Pit C"]
    time_axis = np.arange(0, 12)
    fractions = np.vstack(
        [
            np.clip(0.65 - 0.04 * time_axis, 0.15, 0.8),
            np.clip(0.2 + 0.03 * time_axis, 0.1, 0.55),
            np.clip(0.15 + 0.01 * np.sin(time_axis), 0.05, 0.2),
        ]
    )
    col_sums = fractions.sum(axis=0)
    fractions = fractions / col_sums

    fig, ax = plt.subplots(figsize=(10.5, 5), dpi=140)
    ax.stackplot(
        time_axis,
        fractions,
        labels=categories,
        colors=[CATEGORY_COLOURS[c] for c in categories],
        alpha=0.9,
    )
    ax.set_title("Genealogy and categorical weighting")
    ax.set_xlabel("output timestamp index")
    ax.set_ylabel("fraction")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, format="png")
    plt.close(fig)


def _all_kernels(result: ScenarioResult) -> tuple[FittedKernel, ...]:
    return (result.true_kernel,) + result.fitted_kernels


def _weight_at(kernel: FittedKernel, lag_step: int) -> float:
    for step, weight in zip(kernel.lag_steps, kernel.weights):
        if step == lag_step:
            return float(weight)
    return 0.0


def _fmt_loss(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4g}"


def _points_text(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def _nice_y_max(max_weight: float) -> float:
    if max_weight <= 0.0:
        return 1.0
    return float(min(1.0, max_weight * 1.18))


def _base_style() -> str:
    return """
<style>
text{font-family:Inter,Segoe UI,Arial,sans-serif;fill:#111827}
.title{font-size:28px;font-weight:760;letter-spacing:-0.02em}
.subtitle{font-size:14px;fill:#475569}
.card{fill:white;stroke:#e5e7eb;stroke-width:1.2;filter:url(#shadow)}
.section-title{font-size:13px;font-weight:760;fill:#334155;text-transform:uppercase;letter-spacing:0.06em}
.axis{stroke:#334155;stroke-width:1.2}
.grid{stroke:#e2e8f0;stroke-width:1}
.grid-light{stroke:#f1f5f9;stroke-width:1}
.tick{font-size:11px;fill:#64748b}
.axis-label{font-size:12px;font-weight:650;fill:#475569}
.legend-label{font-size:12px;fill:#334155}
.metric-label{font-size:11px;fill:#64748b;text-transform:uppercase;letter-spacing:0.04em}
.metric-value{font-size:15px;font-weight:760;fill:#111827}
.note{font-size:12px;fill:#475569}
.note-bold{font-size:12px;font-weight:760;fill:#334155}
.table-head{font-size:11px;font-weight:760;fill:#64748b;text-transform:uppercase;letter-spacing:0.04em}
.table-text{font-size:12px;fill:#334155}
.table-mono{font-size:12px;fill:#111827;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
</style>
<defs>
  <filter id="shadow" x="-5%" y="-5%" width="110%" height="120%">
    <feDropShadow dx="0" dy="4" stdDeviation="6" flood-color="#0f172a" flood-opacity="0.08"/>
  </filter>
</defs>
"""


def _metric(x: float, y: float, label: str, value: str) -> str:
    return (
        f'<text class="metric-label" x="{x:.1f}" y="{y:.1f}">{escape(label)}</text>'
        f'<text class="metric-value" x="{x:.1f}" y="{y + 20:.1f}">{escape(value)}</text>'
    )


def _series_svg(
    *,
    kernel: FittedKernel,
    min_lag: int,
    max_lag: int,
    x_for_lag: Callable[[int], float],
    y_for_weight: Callable[[float], float],
) -> str:
    style = SERIES_STYLES[kernel.label]
    points = [
        (x_for_lag(lag_step), y_for_weight(_weight_at(kernel, lag_step)))
        for lag_step in range(min_lag, max_lag + 1)
    ]
    dash = f' stroke-dasharray="{style["dash"]}"' if style["dash"] else ""
    marker_radius = float(style["marker"])
    chunks = [
        (
            f'<polyline fill="none" stroke="{style["stroke"]}" stroke-width="{style["width"]}"'
            f'{dash} stroke-linejoin="round" stroke-linecap="round" '
            f'points="{_points_text(points)}" />'
        )
    ]
    for x, y in points:
        chunks.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{marker_radius:.1f}" '
            f'fill="white" stroke="{style["stroke"]}" stroke-width="1.7" />'
        )
    return "\n".join(chunks)


def _render_fit_svg(result: ScenarioResult) -> str:
    width = 1180
    height = 720
    chart_left = 86
    chart_right = 815
    chart_top = 170
    chart_bottom = 530
    all_kernels = _all_kernels(result)
    min_lag = min(min(kernel.lag_steps) for kernel in all_kernels)
    max_lag = max(max(kernel.lag_steps) for kernel in all_kernels)
    y_max = _nice_y_max(max(max(kernel.weights) for kernel in all_kernels))

    def x_for_lag(lag_step: int) -> float:
        if max_lag == min_lag:
            return (chart_left + chart_right) / 2.0
        fraction = (lag_step - min_lag) / (max_lag - min_lag)
        return chart_left + fraction * (chart_right - chart_left)

    def y_for_weight(weight: float) -> float:
        return chart_bottom - (weight / y_max) * (chart_bottom - chart_top)

    best_fit = min(result.fitted_kernels, key=lambda kernel: kernel.validation_loss or float("inf"))
    warning_count = sum(len(kernel.warning_codes) for kernel in result.fitted_kernels)

    chunks = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="#f8fafc" />',
        _base_style(),
        f'<text class="title" x="42" y="52">{escape(result.scenario.name)}</text>',
        f'<text class="subtitle" x="42" y="82">{escape(result.scenario.physical_condition)}</text>',
        f'<text class="subtitle" x="42" y="106">Expected shape: '
        f'{escape(result.scenario.expected_shape)}</text>',
        '<rect class="card" x="36" y="132" width="830" height="450" rx="18" />',
        '<rect class="card" x="892" y="132" width="248" height="450" rx="18" />',
        '<text class="section-title" x="64" y="162">Kernel shape</text>',
        '<text class="section-title" x="920" y="162">Evidence summary</text>',
    ]

    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y_tick = chart_bottom - tick * (chart_bottom - chart_top)
        chunks.extend(
            [
                f'<line class="grid" x1="{chart_left}" y1="{y_tick:.1f}" '
                f'x2="{chart_right}" y2="{y_tick:.1f}" />',
                f'<text class="tick" x="{chart_left - 10}" y="{y_tick + 4:.1f}" '
                f'text-anchor="end">{tick * y_max:.2f}</text>',
            ]
        )
    x_ticks = sorted({round(min_lag + idx * (max_lag - min_lag) / 6) for idx in range(7)})
    for lag_step in x_ticks:
        x_tick = x_for_lag(int(lag_step))
        chunks.extend(
            [
                f'<line class="grid-light" x1="{x_tick:.1f}" y1="{chart_top}" '
                f'x2="{x_tick:.1f}" y2="{chart_bottom}" />',
                f'<text class="tick" x="{x_tick:.1f}" y="{chart_bottom + 22}" '
                f'text-anchor="middle">{lag_step}</text>',
            ]
        )
    chunks.extend(
        [
            f'<line class="axis" x1="{chart_left}" y1="{chart_bottom}" '
            f'x2="{chart_right}" y2="{chart_bottom}" />',
            f'<line class="axis" x1="{chart_left}" y1="{chart_top}" '
            f'x2="{chart_left}" y2="{chart_bottom}" />',
            f'<text class="axis-label" '
            f'x="{(chart_left + chart_right) / 2:.1f}" '
            f'y="{chart_bottom + 46}" '
            f'text-anchor="middle">Lag step</text>',
            (
                f'<text class="axis-label" x="42" '
                f'y="{(chart_top + chart_bottom) / 2:.1f}" '
                f'text-anchor="middle" '
                f'transform="rotate(-90 42 '
                f'{(chart_top + chart_bottom) / 2:.1f})">Kernel weight</text>'
            ),
        ]
    )

    for kernel in all_kernels:
        chunks.append(
            _series_svg(
                kernel=kernel,
                min_lag=min_lag,
                max_lag=max_lag,
                x_for_lag=x_for_lag,
                y_for_weight=y_for_weight,
            )
        )

    legend_y = 192
    for idx, kernel in enumerate(all_kernels):
        style = SERIES_STYLES[kernel.label]
        y = legend_y + idx * 34
        dash = f' stroke-dasharray="{style["dash"]}"' if style["dash"] else ""
        chunks.extend(
            [
                f'<line x1="920" y1="{y}" x2="958" y2="{y}" '
                f'stroke="{style["stroke"]}" '
                f'stroke-width="{style["width"]}"{dash} '
                f'stroke-linecap="round" />',
                f'<circle cx="939" cy="{y}" r="4" fill="white" '
                f'stroke="{style["stroke"]}" stroke-width="1.7" />',
                f'<text class="legend-label" x="970" y="{y + 4}">{escape(kernel.label)}</text>',
            ]
        )

    chunks.extend(
        [
            _metric(920, 350, "Best fitted loss", best_fit.label),
            _metric(920, 408, "Truth mean lag", f"{result.true_kernel.mean_lag:.0f}s"),
            _metric(1032, 408, "Warnings", str(warning_count)),
            '<text class="note-bold" x="920" y="488">Interpretation boundary</text>',
            f'<text class="note" x="920" '
            f'y="512">{escape(result.scenario.interpretation_note[:42])}</text>',
            f'<text class="note" x="920" '
            f'y="532">{escape(result.scenario.interpretation_note[42:84])}</text>',
        ]
    )

    table_y = 620
    chunks.extend(
        [
            '<text class="section-title" x="42" y="610">Fit diagnostics</text>',
            '<text class="table-head" x="42" y="642">Kernel</text>',
            '<text class="table-head" x="304" y="642">Loss</text>',
            '<text class="table-head" x="418" y="642">Mean</text>',
            '<text class="table-head" x="528" y="642">P90</text>',
            '<text class="table-head" x="638" y="642">Warnings</text>',
        ]
    )
    for idx, kernel in enumerate(all_kernels):
        y = table_y + 50 + idx * 24
        chunks.extend(
            [
                f'<text class="table-text" x="42" y="{y}">{escape(kernel.label)}</text>',
                f'<text class="table-mono" x="304" '
                f'y="{y}">{escape(_fmt_loss(kernel.validation_loss))}</text>',
                f'<text class="table-mono" x="418" y="{y}">{kernel.mean_lag:.0f}s</text>',
                f'<text class="table-mono" x="528" y="{y}">{kernel.p90_lag:.0f}s</text>',
                f'<text class="table-mono" x="638" y="{y}">{len(kernel.warning_codes)}</text>',
            ]
        )
    chunks.append("</svg>")
    return "\n".join(chunks)


def _render_genealogy_svg() -> str:
    width = 1180
    height = 760
    kernel_lags = list(range(0, 10))
    kernel_weights = _normalise([0.00, 0.04, 0.10, 0.20, 0.26, 0.18, 0.10, 0.07, 0.035, 0.015])
    source_sequence = [
        "Pit A", "Pit A", "Pit B", "Pit B", "Pit C",
        "Pit A", "Pit C", "Pit B", "Pit C", "Pit A",
    ]
    output_times = [18, 19, 20, 21, 22, 23]
    categories = ["Pit A", "Pit B", "Pit C"]

    fractions: list[dict[str, float]] = []
    for out_t in output_times:
        row = {category: 0.0 for category in categories}
        for lag, weight in zip(kernel_lags, kernel_weights):
            source_idx = (out_t - lag) % len(source_sequence)
            row[source_sequence[source_idx]] += weight
        fractions.append(row)

    chunks = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="#f8fafc" />',
        _base_style(),
        '<text class="title" x="42" y="52">Genealogy and categorical weighting</text>',
        (
            '<text class="subtitle" x="42" y="82">A kernel converts previous input '
            'slices into weighted source fractions at each output time.</text>'
        ),
        '<rect class="card" x="36" y="120" width="1108" height="250" rx="18" />',
        '<rect class="card" x="36" y="400" width="1108" height="310" rx="18" />',
        (
            '<text class="section-title" x="64" y="154">Input time slices and '
            'active kernel window</text>'
        ),
    ]

    cell_w = 82
    cell_h = 52
    start_x = 92
    start_y = 192
    for idx, source in enumerate(source_sequence):
        x = start_x + idx * cell_w
        chunks.extend(
            [
                (
                    f'<rect x="{x}" y="{start_y}" width="{cell_w - 8}" '
                    f'height="{cell_h}" rx="10" '
                    f'fill="{CATEGORY_COLOURS[source]}" opacity="0.88" />'
                ),
                (
                    f'<text x="{x + (cell_w - 8) / 2:.1f}" y="{start_y + 31}" '
                    f'text-anchor="middle" fill="white" font-size="13" '
                    f'font-weight="750">{escape(source)}</text>'
                ),
                (
                    f'<text class="tick" x="{x + (cell_w - 8) / 2:.1f}" '
                    f'y="{start_y + 76}" '
                    f'text-anchor="middle">t-{9 - idx}</text>'
                ),
            ]
        )
    chunks.append(
        '<text class="note" x="92" y="314">For one output timestamp, each prior slice '
        'receives kernel weight by lag. Same source labels are summed.</text>'
    )

    mini_x = 910
    mini_y = 172
    max_w = max(kernel_weights)
    chunks.append(f'<text class="section-title" x="{mini_x}" y="154">Kernel weights</text>')
    for idx, weight in enumerate(kernel_weights):
        bar_h = 130 * weight / max_w
        x = mini_x + idx * 20
        chunks.extend(
            [
                (
                    f'<rect x="{x}" y="{mini_y + 130 - bar_h:.1f}" '
                    f'width="13" height="{bar_h:.1f}" fill="#334155" rx="3" />'
                ),
                f'<text class="tick" x="{x + 6.5}" y="{mini_y + 150}" '
                f'text-anchor="middle">{idx}</text>',
            ]
        )

    chart_x = 94
    chart_y = 472
    chart_w = 620
    chart_h = 170
    bar_w = 64
    chunks.extend(
        [
            '<text class="section-title" x="64" y="434">Weighted categorical '
            'output features</text>',
            f'<line class="axis" x1="{chart_x}" y1="{chart_y + chart_h}" '
            f'x2="{chart_x + chart_w}" y2="{chart_y + chart_h}" />',
            f'<line class="axis" x1="{chart_x}" y1="{chart_y}" '
            f'x2="{chart_x}" y2="{chart_y + chart_h}" />',
            (
                f'<text class="axis-label" x="{chart_x - 46}" '
                f'y="{chart_y + chart_h / 2}" text-anchor="middle" '
                f'transform="rotate(-90 {chart_x - 46} '
                f'{chart_y + chart_h / 2})">Weighted fraction</text>'
            ),
        ]
    )
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = chart_y + chart_h - tick * chart_h
        chunks.extend(
            [
                f'<line class="grid" x1="{chart_x}" y1="{y:.1f}" '
                f'x2="{chart_x + chart_w}" y2="{y:.1f}" />',
                f'<text class="tick" x="{chart_x - 10}" y="{y + 4:.1f}" '
                f'text-anchor="end">{tick:.2f}</text>',
            ]
        )
    for out_idx, out_t in enumerate(output_times):
        x = chart_x + 42 + out_idx * 92
        y_cursor = chart_y + chart_h
        for category in categories:
            height_px = fractions[out_idx][category] * chart_h
            y_cursor -= height_px
            chunks.append(
                f'<rect x="{x}" y="{y_cursor:.1f}" width="{bar_w}" '
                f'height="{height_px:.1f}" '
                f'fill="{CATEGORY_COLOURS[category]}" rx="2" />'
            )
        chunks.append(
            f'<text class="tick" x="{x + bar_w / 2}" '
            f'y="{chart_y + chart_h + 24}" '
            f'text-anchor="middle">t={out_t}</text>'
        )

    legend_x = 780
    legend_y = 472
    chunks.append(
        f'<text class="section-title" x="{legend_x}" '
        f'y="434">Generated categorical features</text>'
    )
    for idx, category in enumerate(categories):
        y = legend_y + idx * 34
        chunks.extend(
            [
                (
                    f'<rect x="{legend_x}" y="{y - 14}" width="18" height="18" '
                    f'rx="4" fill="{CATEGORY_COLOURS[category]}" />'
                ),
                f'<text class="legend-label" x="{legend_x + 30}" '
                f'y="{y}">{escape(category)} fraction</text>',
            ]
        )
    chunks.extend(
        [
            f'<text class="note-bold" x="{legend_x}" y="610">Feature meaning</text>',
            f'<text class="note" x="{legend_x}" '
            f'y="635">Each bar sums to 1.0 because source labels</text>',
            f'<text class="note" x="{legend_x}" '
            f'y="655">inherit the kernel weights for earlier input slices.</text>',
            f'<text class="note" x="{legend_x}" '
            f'y="675">This is categorical genealogy, not final model selection.</text>',
        ]
    )
    chunks.append("</svg>")
    return "\n".join(chunks)


def _fit_rows(result: ScenarioResult) -> list[str]:
    rows = [
        "| kernel | family | validation loss | mean lag | p50 lag | p90 lag | warnings |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for kernel in _all_kernels(result):
        warning_text = ", ".join(kernel.warning_codes) if kernel.warning_codes else "none"
        rows.append(
            "| "
            + " | ".join(
                [
                    kernel.label,
                    kernel.family,
                    _fmt_loss(kernel.validation_loss),
                    f"{kernel.mean_lag:.1f}",
                    f"{kernel.p50_lag:.1f}",
                    f"{kernel.p90_lag:.1f}",
                    warning_text,
                ]
            )
            + " |"
        )
    return rows


def _render_markdown(results: tuple[ScenarioResult, ...], genealogy_plot: Path) -> str:
    parts = [
        "# Parametric vs Empirical Fit Gallery",
        "",
        "This gallery replaces the old single crowded canvas with one static SVG per example.",
        "Each fit example compares synthetic truth, empirical simplex, parametric gamma, "
        "and parametric exponential fits.",
        "The genealogy example separately shows categorical/source weighting from kernel mass.",
        "",
        "## Fit Examples",
        "",
    ]
    for result in results:
        plot_name = result.plot_path.name
        parts.extend(
            [
                f"### {result.scenario.name}",
                "",
                f"![{result.scenario.name}]({PLOT_LINK_PREFIX}/{plot_name})",
                "",
                f"- Physical condition: {result.scenario.physical_condition}",
                f"- Expected shape: {result.scenario.expected_shape}",
                f"- Interpretation: {result.scenario.interpretation_note}",
                "",
                *_fit_rows(result),
                "",
            ]
        )
    parts.extend(
        [
            "## Genealogy and Categorical Weighting",
            "",
            f"![Genealogy and categorical weighting]({PLOT_LINK_PREFIX}/{genealogy_plot.name})",
            "",
            "This plot demonstrates how a kernel turns historical input slices "
            "into categorical source-fraction features.",
            "For each output timestamp, earlier source labels inherit the lag weights, "
            "and labels are summed by category.",
            "",
            "## Interpretation Boundary",
            "",
            "- The empirical simplex fit is the flexible learned shape. "
            "It is useful when the physical family is uncertain.",
            "- Parametric fits are lower-dimensional and easier to explain "
            "when the physical family is plausible.",
            "- Multimodal fits are deliberately included because real process paths "
            "may contain bypasses, parallel paths, recycle, or mixed operating modes.",
            "- Do not call a response kernel an RTD unless material movement, tracer, "
            "topology, or SME evidence supports that interpretation.",
            "",
        ]
    )
    return "\n".join(parts)


def _scenarios() -> tuple[FitScenario, ...]:
    return (
        FitScenario(
            name="Plug-flow / near-fixed transport delay",
            slug="plug_flow",
            physical_condition="Conveyor-like transfer or narrow residence-time path.",
            dataset_factory=lambda: make_single_delay_dataset(
                n_rows=420, dt=60.0, seed=7, delay_steps=6, noise_std=0.02
            ),
            min_lag=1,
            max_lag=10,
            expected_shape="narrow spike around one lag step",
            interpretation_note=(
                "A simplex or fixed-delay shape is often more honest "
                "than a diffuse smooth tail."
            ),
        ),
        FitScenario(
            name="Tanks-in-series / mixed residence",
            slug="tanks_in_series",
            physical_condition="Well-mixed stages with a unimodal residence-time distribution.",
            dataset_factory=lambda: make_gamma_kernel_dataset(
                n_rows=520, dt=60.0, seed=53,
                min_lag_steps=1, max_lag_steps=12,
                shape_alpha=3.0, rate_beta=0.06, noise_std=0.03
            ),
            min_lag=1,
            max_lag=12,
            expected_shape="smooth unimodal peak with a right tail",
            interpretation_note=(
                "Gamma should be competitive when a staged-mixing "
                "assumption is credible."
            ),
        ),
        FitScenario(
            name="Crushing loop / recirculating tail surrogate",
            slug="crushing_loop_tail",
            physical_condition="Recycle-like delayed response with a long declining tail.",
            dataset_factory=lambda: make_delayed_exponential_kernel_dataset(
                n_rows=520, dt=60.0, seed=67,
                min_lag_steps=1, max_lag_steps=14,
                delay=180.0, rate_lambda=0.025, noise_std=0.03
            ),
            min_lag=1,
            max_lag=14,
            expected_shape="delayed onset followed by a decaying tail",
            interpretation_note=(
                "Simple exponential can capture tailing "
                "but may miss dead-time onset."
            ),
        ),
        FitScenario(
            name="Multimodal / parallel path surrogate",
            slug="multimodal_parallel_paths",
            physical_condition=(
                "Two dominant residence paths, such as bypass plus slower mixed path."
            ),
            dataset_factory=_multimodal_parallel_dataset,
            min_lag=1,
            max_lag=13,
            expected_shape="two separated peaks",
            interpretation_note=(
                "A flexible empirical kernel should expose modes that "
                "a single parametric family smooths away."
            ),
        ),
        FitScenario(
            name="Bypass plus recycle tail",
            slug="bypass_plus_recycle_tail",
            physical_condition="Immediate bypass mass plus a later recycle/tailing component.",
            dataset_factory=_bypass_plus_recycle_dataset,
            min_lag=1,
            max_lag=18,
            expected_shape="early spike plus broad late tail",
            interpretation_note=(
                "This is the kind of shape where evidence beats "
                "assuming one neat family."
            ),
        ),
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = tuple(_fit_scenario(scenario) for scenario in _scenarios())
    for result in results:
        _write_fit_png(result)
        print(f"Wrote {result.plot_path}")

    genealogy_plot = OUTPUT_DIR / "genealogy_categorical_weighting.png"
    _write_genealogy_png(genealogy_plot)
    print(f"Wrote {genealogy_plot}")

    markdown = _render_markdown(results, genealogy_plot)
    OUTPUT_MD.write_text(markdown, encoding="utf-8")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
