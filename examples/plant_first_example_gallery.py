#!/usr/bin/env python3
"""Generate plant-first scenario gallery artifacts."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import polars as pl
from examples._support.plant_first_scenarios import (
    ScenarioFixture,
    core_scenario_fixtures,
    make_mini_flowsheet_dataset,
)

from rtdfeatures import (
    ExponentialKernelLearner,
    GammaKernelLearner,
    KernelFeatureBuilder,
    SimplexKernelLearner,
)
from rtdfeatures.diagnostics import KernelFitResult
from rtdfeatures.kernels.base import Kernel
from rtdfeatures.learners import (
    DelayedExponentialKernelLearner,
    ErlangKernelLearner,
    FixedDelayKernelLearner,
    LogNormalKernelLearner,
    UniformKernelLearner,
)

OUTPUT_DIR = Path("docs/examples/generated")
OUTPUT_GALLERY = Path("docs/examples/plant_first_gallery.md")
PLOT_LINK_PREFIX = "generated"

SCENARIO_FILE_BY_NAME = {
    "conveyor": "fit_conveyor.png",
    "cstr": "fit_cstr.png",
    "flotation_banks": "fit_flotation_banks.png",
    "tanks_in_series": "fit_tanks_in_series.png",
    "closed_loop_crushing": "fit_closed_loop_crushing.png",
    "bounded_hold_up_tank": "fit_bounded_holdup_tank.png",
    "mini_flowsheet": "fit_mini_flowsheet.png",
}

SCENARIO_STORY = {
    "conveyor": "Near plug-flow conveyor with narrow transport delay.",
    "cstr": "Single well-mixed vessel with first-order tailing.",
    "flotation_banks": "Right-skewed spread from staged flotation residence-time effects.",
    "tanks_in_series": "Staged mixing with unimodal spread.",
    "closed_loop_crushing": "Dead-time onset plus recycle/tailing.",
    "bounded_hold_up_tank": "Bounded blended hold-up window with finite support.",
    "mini_flowsheet": "Composite synthetic plant story for feed to cleaner response.",
}

LEARNER_BY_NAME = {
    "SimplexKernelLearner": lambda seed, cfg: SimplexKernelLearner(
        seed=seed,
        max_epochs=420,
        smoothness_penalty=0.0004,
        **cfg,
    ),
    "FixedDelayKernelLearner": lambda seed, cfg: FixedDelayKernelLearner(seed=seed, **cfg),
    "ExponentialKernelLearner": lambda seed, cfg: ExponentialKernelLearner(
        seed=seed, max_epochs=420, **cfg
    ),
    "GammaKernelLearner": lambda seed, cfg: GammaKernelLearner(seed=seed, max_epochs=420, **cfg),
    "ErlangKernelLearner": lambda seed, cfg: ErlangKernelLearner(seed=seed, max_epochs=420, **cfg),
    "LogNormalKernelLearner": lambda seed, cfg: LogNormalKernelLearner(
        seed=seed, max_epochs=420, **cfg
    ),
    "DelayedExponentialKernelLearner": lambda seed, cfg: DelayedExponentialKernelLearner(
        seed=seed, max_epochs=420, **cfg
    ),
    "UniformKernelLearner": lambda seed, cfg: UniformKernelLearner(seed=seed, **cfg),
}

BASELINE_MARGIN = 0.05

MINI_FEW_WEIGHTS_THRESHOLD = 2
MINI_UNIFORM_ENTROPY_THRESHOLD = 0.98
MINI_UNIFORM_MAX_WEIGHT_THRESHOLD = 0.18
MINI_UNIFORM_COEFF_VAR_THRESHOLD = 0.08


class _KernelLearnerLike(Protocol):
    def fit(
        self, df: pl.DataFrame, *, input_col: str, target_col: str, time_col: str
    ) -> KernelFitResult: ...


@dataclass(frozen=True)
class LearnerOutcome:
    learner_name: str
    fit_result: KernelFitResult
    recommendation_status: str
    recommendation_reason: str


@dataclass(frozen=True)
class ScenarioReport:
    scenario_name: str
    title: str
    source_desc: str
    outcomes: tuple[LearnerOutcome, ...]
    recommended_kernel: str
    recommendation_status: str
    recommendation_reason: str
    feature_preview: pl.DataFrame
    feature_preview_columns: tuple[str, ...]
    signal_plot_name: str
    kernel_plot_name: str
    fit_plot_name: str
    fit_quality_gate_passed: bool
    fit_quality_gate_reason: str
    fit_quality_selected_kernel: str
    fit_quality_validation_loss: float
    fit_quality_no_lag: float
    fit_quality_best_single_lag: float
    fit_quality_warning_codes: tuple[str, ...]
    fit_quality_mean_lag: float
    fit_quality_p50_lag: float
    fit_quality_p90_lag: float
    positive_kernels: tuple[str, ...] = ()
    comparison_kernels: tuple[str, ...] = ()
    mini_transition_row: int | None = None
    mini_unit_lag_table: str | None = None


def _fit(
    df: pl.DataFrame,
    *,
    input_col: str,
    target_col: str,
    learner_names: tuple[str, ...],
    seed_offset: int,
) -> tuple[LearnerOutcome, ...]:
    common = {"min_lag": 0, "max_lag": 60, "loss": "huber", "validation_fraction": 0.2}
    outcomes: list[LearnerOutcome] = []
    for idx, learner_name in enumerate(learner_names):
        learner = cast(
            _KernelLearnerLike,
            LEARNER_BY_NAME[learner_name](900 + seed_offset + idx, common),
        )
        result = learner.fit(df, input_col=input_col, target_col=target_col, time_col="time")
        outcomes.append(
            LearnerOutcome(
                learner_name=learner_name,
                fit_result=result,
                recommendation_status="reference_only",
                recommendation_reason="comparison fit",
            )
        )
    return tuple(outcomes)


def _fit_quality_gate_for_positive(
    outcomes: tuple[LearnerOutcome, ...], *, positive_kernel_names: tuple[str, ...]
) -> tuple[bool, str, LearnerOutcome]:
    positive_outcomes = [out for out in outcomes if out.learner_name in positive_kernel_names]
    if not positive_outcomes:
        selected = min(
            outcomes,
            key=lambda out: float(out.fit_result.fit_diagnostics.validation_loss),
        )
        return False, "no configured positive kernel fit for scenario", selected

    selected = min(
        positive_outcomes, key=lambda out: float(out.fit_result.fit_diagnostics.validation_loss)
    )
    result = selected.fit_result
    diagnostics = result.fit_diagnostics
    baseline = result.baseline_comparison
    warnings = set(result.identifiability_report.warning_codes)
    reasons: list[str] = []
    try:
        result.kernel.validate()
    except Exception:
        reasons.append("invalid kernel constraints")
    if not math.isfinite(float(diagnostics.validation_loss)):
        reasons.append("non-finite validation loss")
    if "BEST_SINGLE_LAG_BEATS_LEARNED" in warnings:
        reasons.append("best_single_lag beats selected positive kernel")
    if math.isfinite(float(baseline.best_single_lag_validation_loss)):
        best_delta = (diagnostics.validation_loss - baseline.best_single_lag_validation_loss) / max(
            abs(diagnostics.validation_loss), 1e-12
        )
        if best_delta >= BASELINE_MARGIN:
            reasons.append("best_single_lag beats selected positive kernel by margin")
    if math.isfinite(float(baseline.no_lag_validation_loss)):
        no_lag_delta = (diagnostics.validation_loss - baseline.no_lag_validation_loss) / max(
            abs(diagnostics.validation_loss), 1e-12
        )
        if no_lag_delta >= BASELINE_MARGIN:
            reasons.append("no_lag beats selected positive kernel by margin")
    if reasons:
        return False, "; ".join(reasons), selected
    return True, "selected positive kernel passes deterministic fit checks", selected


def _shape_contradiction_for_mini(result: KernelFitResult) -> bool:
    """Detect kernel shapes that contradict mini-flowsheet's expected non-uniform memory.

    Thresholds detect approximately uniform-looking kernels:
    - Very few lags (≤2) are too narrow for the composite mini-flowsheet response
    - High normalized entropy (≥0.98) with low max weight (≤0.18) indicates near-uniform
    - Low coefficient of variation (≤0.08) among weights indicates near-flat distribution
    """
    shape = result.kernel_shape_summary
    weights = [float(w) for w in result.kernel.weights]
    if len(weights) <= MINI_FEW_WEIGHTS_THRESHOLD:
        return True
    mean_w = sum(weights) / float(len(weights))
    variance = sum((w - mean_w) ** 2 for w in weights) / float(len(weights))
    coeff_var = math.sqrt(variance) / max(mean_w, 1e-12)
    if (
        shape is not None
        and shape.normalized_entropy >= MINI_UNIFORM_ENTROPY_THRESHOLD
        and shape.max_weight <= MINI_UNIFORM_MAX_WEIGHT_THRESHOLD
    ):
        return True
    return coeff_var <= MINI_UNIFORM_COEFF_VAR_THRESHOLD


def _kernel_lag_stats(kernel: Kernel) -> tuple[float, float, float]:
    return kernel.mean_lag(), kernel.percentile(0.5), kernel.percentile(0.9)


def _mini_recommendation(outcomes: tuple[LearnerOutcome, ...]) -> tuple[LearnerOutcome, ...]:
    reviewed: list[LearnerOutcome] = []
    for outcome in outcomes:
        result = outcome.fit_result
        diag = result.fit_diagnostics
        baseline = result.baseline_comparison
        report = result.identifiability_report

        reasons: list[str] = []
        try:
            result.kernel.validate()
        except Exception:
            reasons.append("invalid kernel constraints")
        if not math.isfinite(float(diag.validation_loss)):
            reasons.append("non-finite validation loss")
        if any(
            report.warning_severity_by_code.get(code) == "high"
            for code in report.warning_codes
        ):
            reasons.append("high-severity identifiability warning")
        if "BEST_SINGLE_LAG_BEATS_LEARNED" in report.warning_codes:
            reasons.append("best_single_lag baseline beats learned kernel")

        if math.isfinite(float(baseline.no_lag_validation_loss)):
            no_lag_delta = (diag.validation_loss - baseline.no_lag_validation_loss) / max(
                abs(diag.validation_loss), 1e-12
            )
            if no_lag_delta >= BASELINE_MARGIN:
                reasons.append("no_lag baseline beats learned kernel")
        if math.isfinite(float(baseline.best_single_lag_validation_loss)):
            bsl_delta = (diag.validation_loss - baseline.best_single_lag_validation_loss) / max(
                abs(diag.validation_loss), 1e-12
            )
            if bsl_delta >= BASELINE_MARGIN:
                reasons.append("best_single_lag baseline beats learned kernel by margin")
        if _shape_contradiction_for_mini(result):
            reasons.append("shape check contradicts non-uniform mini-flowsheet memory")

        if reasons:
            status = "not_recommended"
            reason = "; ".join(reasons)
        else:
            status = "recommended"
            reason = "passes deterministic fit evidence and shape checks"
        reviewed.append(
            LearnerOutcome(
                learner_name=outcome.learner_name,
                fit_result=outcome.fit_result,
                recommendation_status=status,
                recommendation_reason=reason,
            )
        )
    return tuple(reviewed)


def _choose_recommended(outcomes: tuple[LearnerOutcome, ...]) -> LearnerOutcome:
    recommended = [
        outcome for outcome in outcomes if outcome.recommendation_status == "recommended"
    ]
    if recommended:
        return min(
            recommended,
            key=lambda out: float(out.fit_result.fit_diagnostics.validation_loss),
        )
    return min(outcomes, key=lambda out: float(out.fit_result.fit_diagnostics.validation_loss))


def _build_plots(
    *,
    scenario_name: str,
    df: pl.DataFrame,
    input_col: str,
    target_col: str,
    recommended: LearnerOutcome,
    preview: pl.DataFrame,
) -> tuple[str, str, str]:
    base_name = SCENARIO_FILE_BY_NAME[scenario_name].removesuffix(".png")
    signal_file = f"{base_name}_signals.png"
    kernel_file = f"{base_name}_kernel.png"
    fit_file = f"{base_name}_fit.png"
    signal_path = OUTPUT_DIR / signal_file
    kernel_path = OUTPUT_DIR / kernel_file
    fit_path = OUTPUT_DIR / fit_file
    n_rows = min(df.height, 240)
    signal_df = df.head(n_rows)
    xs = list(range(n_rows))
    input_vals = [float(v) for v in signal_df[input_col].to_list()]
    target_vals = [float(v) for v in signal_df[target_col].to_list()]
    kernel = recommended.fit_result.kernel
    lag_seconds = [float(step * kernel.dt) for step in kernel.lag_steps]
    weights = [float(w) for w in kernel.weights]

    pred_vals: list[float] = []
    for idx in range(n_rows):
        acc = 0.0
        valid = True
        for step, weight in zip(kernel.lag_steps, kernel.weights):
            source_idx = idx - int(step)
            if source_idx < 0:
                valid = False
                break
            acc += float(weight) * input_vals[source_idx]
        pred_vals.append(acc if valid else float("nan"))

    fit_x = [i for i, val in enumerate(pred_vals) if math.isfinite(val)]
    fit_y = [target_vals[i] for i in fit_x]
    fit_pred = [pred_vals[i] for i in fit_x]

    fig, ax_signal = plt.subplots(figsize=(11, 4.2), dpi=140)
    ax_signal.plot(xs, input_vals, color="#2563eb", linewidth=1.8, label=input_col)
    ax_signal.plot(xs, target_vals, color="#dc2626", linewidth=1.8, label=target_col)
    ax_signal.set_title("Input and target signals")
    ax_signal.set_xlabel("sample index")
    ax_signal.grid(alpha=0.2)
    ax_signal.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(signal_path, format="png")
    plt.close(fig)

    fig, ax_kernel = plt.subplots(figsize=(11, 4.2), dpi=140)
    ax_kernel.plot(lag_seconds, weights, color="#7c3aed", linewidth=2.2)
    ax_kernel.fill_between(lag_seconds, weights, 0.0, color="#7c3aed", alpha=0.15)
    ax_kernel.set_title("Learned kernel weights")
    ax_kernel.set_xlabel("lag (seconds)")
    ax_kernel.set_ylabel("weight")
    ax_kernel.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(kernel_path, format="png")
    plt.close(fig)

    fig, ax_fit = plt.subplots(figsize=(11, 4.2), dpi=140)
    ax_fit.plot(fit_x, fit_y, color="#111827", linewidth=1.8, label="observed target")
    ax_fit.plot(fit_x, fit_pred, color="#16a34a", linewidth=1.8, label="kernel fit prediction")
    ax_fit.set_title("Observed vs kernel-fit prediction")
    ax_fit.set_xlabel("sample index")
    ax_fit.grid(alpha=0.2)
    ax_fit.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(fit_path, format="png")
    plt.close(fig)
    return signal_file, kernel_file, fit_file


def _markdown_table(outcomes: tuple[LearnerOutcome, ...]) -> str:
    lines = [
        "| learner | validation_loss | no_lag | best_single_lag | warning_codes "
        "| recommendation_status | recommendation_reason |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for outcome in outcomes:
        result = outcome.fit_result
        diag = result.fit_diagnostics
        base = result.baseline_comparison
        warnings = ",".join(result.identifiability_report.warning_codes) or "none"
        lines.append(
            f"| `{outcome.learner_name}` | {diag.validation_loss:.6f} | "
            f"{base.no_lag_validation_loss:.6f} | {base.best_single_lag_validation_loss:.6f} | "
            f"{warnings} | `{outcome.recommendation_status}` | {outcome.recommendation_reason} |"
        )
    return "\n".join(lines)


def _scenario_title(name: str) -> str:
    return name.replace("_", " ").title()


def _mini_unit_lag_table(df: pl.DataFrame) -> str:
    links = (
        ("Crushing", "feed_mass", "crusher_output_mass"),
        ("Ball mill", "crusher_output_mass", "ball_mill_product_mass"),
        ("Cyclone overflow", "ball_mill_product_mass", "cyclone_overflow_mass"),
        ("Flotation bank 1", "cyclone_overflow_mass", "flotation_bank_1_mass"),
        ("Flotation bank 2", "flotation_bank_1_mass", "flotation_bank_2_mass"),
        ("Flotation bank 3", "flotation_bank_2_mass", "flotation_bank_3_mass"),
        ("Cleaner", "flotation_bank_3_mass", "cleaner_product_mass"),
    )
    lines = [
        "| unit_link | learner | validation_loss | no_lag | best_single_lag | "
        "mean_lag_seconds | p50_lag_seconds | p90_lag_seconds | warning_codes | "
        "recommendation_status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    learner_names = (
        "FixedDelayKernelLearner",
        "SimplexKernelLearner",
        "GammaKernelLearner",
        "ErlangKernelLearner",
        "LogNormalKernelLearner",
        "DelayedExponentialKernelLearner",
        "ExponentialKernelLearner",
        "UniformKernelLearner",
    )
    common = {"min_lag": 0, "max_lag": 60, "loss": "huber", "validation_fraction": 0.2}
    for idx, (name, input_col, target_col) in enumerate(links):
        pair_df = df.select("time", input_col, target_col)
        raw_outcomes: list[LearnerOutcome] = []
        for learner_idx, learner_name in enumerate(learner_names):
            learner = cast(
                _KernelLearnerLike,
                LEARNER_BY_NAME[learner_name](
                    1300 + (idx * 20) + learner_idx,
                    common,
                ),
            )
            result = learner.fit(
                pair_df,
                input_col=input_col,
                target_col=target_col,
                time_col="time",
            )
            raw_outcomes.append(
                LearnerOutcome(
                    learner_name=learner_name,
                    fit_result=result,
                    recommendation_status="reference_only",
                    recommendation_reason="comparison fit",
                )
            )
        outcomes = _mini_recommendation(tuple(raw_outcomes))
        selected = _choose_recommended(outcomes)
        result = selected.fit_result
        diag = result.fit_diagnostics
        base = result.baseline_comparison
        warnings = ",".join(result.identifiability_report.warning_codes) or "none"
        mean_lag, p50_lag, p90_lag = _kernel_lag_stats(result.kernel)
        status = selected.recommendation_status
        lines.append(
            f"| `{input_col} -> {target_col}` ({name}) | `{selected.learner_name}` | "
            f"{diag.validation_loss:.6f} | {base.no_lag_validation_loss:.6f} | "
            f"{base.best_single_lag_validation_loss:.6f} | {mean_lag:.2f} | {p50_lag:.2f} | "
            f"{p90_lag:.2f} | {warnings} | `{status}` |"
        )
    return "\n".join(lines)


def _build_core_report(idx: int, fixture: ScenarioFixture) -> ScenarioReport:
    dataset = fixture.dataset_factory()
    df = dataset.data.select("time", "input_signal", "target_signal")
    learner_names = fixture.positive_kernels + fixture.comparison_kernels
    outcomes = _fit(
        df,
        input_col="input_signal",
        target_col="target_signal",
        learner_names=learner_names,
        seed_offset=(idx * 20),
    )
    recommended = _choose_recommended(outcomes)
    gate_passed, gate_reason, gate_selected = _fit_quality_gate_for_positive(
        outcomes, positive_kernel_names=fixture.positive_kernels
    )
    builder = KernelFeatureBuilder(
        kernels={"recommended": recommended.fit_result.kernel},
        time_col="time",
        numeric_cols=["input_signal"],
    )
    preview = builder.transform(df).tail(8)
    signal_plot_name, kernel_plot_name, fit_plot_name = _build_plots(
        scenario_name=fixture.name,
        df=df,
        input_col="input_signal",
        target_col="target_signal",
        recommended=recommended,
        preview=preview,
    )
    mean_lag, p50_lag, p90_lag = _kernel_lag_stats(gate_selected.fit_result.kernel)
    return ScenarioReport(
        scenario_name=fixture.name,
        title=_scenario_title(fixture.name),
        source_desc="Deterministic synthetic fixture from scenario layer.",
        outcomes=outcomes,
        recommended_kernel=recommended.learner_name,
        recommendation_status="recommended" if gate_passed else "not_recommended",
        recommendation_reason=(
            "lowest validation_loss among fitted learners for this scenario"
            if gate_passed
            else f"fit quality gate failed: {gate_reason}"
        ),
        feature_preview=preview,
        feature_preview_columns=tuple(preview.columns),
        signal_plot_name=signal_plot_name,
        kernel_plot_name=kernel_plot_name,
        fit_plot_name=fit_plot_name,
        fit_quality_gate_passed=gate_passed,
        fit_quality_gate_reason=gate_reason,
        fit_quality_selected_kernel=gate_selected.learner_name,
        fit_quality_validation_loss=float(gate_selected.fit_result.fit_diagnostics.validation_loss),
        fit_quality_no_lag=float(gate_selected.fit_result.baseline_comparison.no_lag_validation_loss),
        fit_quality_best_single_lag=float(
            gate_selected.fit_result.baseline_comparison.best_single_lag_validation_loss
        ),
        fit_quality_warning_codes=tuple(gate_selected.fit_result.identifiability_report.warning_codes),
        fit_quality_mean_lag=mean_lag,
        fit_quality_p50_lag=p50_lag,
        fit_quality_p90_lag=p90_lag,
        positive_kernels=fixture.positive_kernels,
        comparison_kernels=fixture.comparison_kernels,
    )


def _build_mini_report() -> ScenarioReport:
    dataset = make_mini_flowsheet_dataset()
    df = dataset.data
    learners = (
        "SimplexKernelLearner",
        "GammaKernelLearner",
        "LogNormalKernelLearner",
        "DelayedExponentialKernelLearner",
        "ExponentialKernelLearner",
    )
    raw_outcomes = _fit(
        df,
        input_col="feed_copper_grade",
        target_col="cleaner_copper_grade",
        learner_names=learners,
        seed_offset=300,
    )
    outcomes = _mini_recommendation(raw_outcomes)
    recommended = _choose_recommended(outcomes)

    builder = KernelFeatureBuilder(
        kernels={"recommended": recommended.fit_result.kernel},
        time_col="time",
        numeric_cols=["feed_copper_grade"],
        category_cols=["ore_type"],
        weight_col="feed_mass",
    )
    features = builder.transform(df)
    preview_cols = [
        "time",
        "recommended_num_feed_copper_grade_wmean",
        "recommended_num_feed_copper_grade_wstd",
        "recommended_num_feed_copper_grade_wsum",
        "recommended_cat_ore_type_A_frac",
        "recommended_cat_ore_type_B_frac",
        "recommended_cat_ore_type_entropy",
    ]
    preview = features.select([col for col in preview_cols if col in features.columns]).tail(10)
    signal_plot_name, kernel_plot_name, fit_plot_name = _build_plots(
        scenario_name="mini_flowsheet",
        df=df,
        input_col="feed_copper_grade",
        target_col="cleaner_copper_grade",
        recommended=recommended,
        preview=preview,
    )
    mean_lag, p50_lag, p90_lag = _kernel_lag_stats(recommended.fit_result.kernel)
    unit_lag_table = _mini_unit_lag_table(df)
    return ScenarioReport(
        scenario_name="mini_flowsheet",
        title="Mini Flowsheet",
        source_desc="Deterministic synthetic fixture from mini-flowsheet layer.",
        outcomes=outcomes,
        recommended_kernel=recommended.learner_name,
        recommendation_status=recommended.recommendation_status,
        recommendation_reason=recommended.recommendation_reason,
        feature_preview=preview,
        feature_preview_columns=tuple(preview.columns),
        signal_plot_name=signal_plot_name,
        kernel_plot_name=kernel_plot_name,
        fit_plot_name=fit_plot_name,
        fit_quality_gate_passed=recommended.recommendation_status == "recommended",
        fit_quality_gate_reason=recommended.recommendation_reason,
        fit_quality_selected_kernel=recommended.learner_name,
        fit_quality_validation_loss=float(recommended.fit_result.fit_diagnostics.validation_loss),
        fit_quality_no_lag=float(recommended.fit_result.baseline_comparison.no_lag_validation_loss),
        fit_quality_best_single_lag=float(
            recommended.fit_result.baseline_comparison.best_single_lag_validation_loss
        ),
        fit_quality_warning_codes=tuple(recommended.fit_result.identifiability_report.warning_codes),
        fit_quality_mean_lag=mean_lag,
        fit_quality_p50_lag=p50_lag,
        fit_quality_p90_lag=p90_lag,
        mini_transition_row=int(dataset.scenario["params"]["ore_transition_row"]),
        mini_unit_lag_table=unit_lag_table,
    )


def _write_gallery(reports: tuple[ScenarioReport, ...]) -> None:
    coverage_table = _coverage_table_markdown(reports)
    lines = [
        "# Plant-First Scenario Gallery",
        "",
        "Generated from deterministic fixture sources using current public learners.",
        "",
        coverage_table,
        "",
    ]
    for report in reports:
        lines.extend(
            [
                f"## {report.title}",
                "",
                f"- Scenario key: `{report.scenario_name}`",
                f"- Source: {report.source_desc}",
                f"- Plant story: {SCENARIO_STORY[report.scenario_name]}",
                f"- Fitted learners: `{', '.join(out.learner_name for out in report.outcomes)}`",
                f"- `recommended_kernel`: `{report.recommended_kernel}`",
                f"- `recommendation_status`: `{report.recommendation_status}`",
                f"- `recommendation_reason`: `{report.recommendation_reason}`",
                f"- `fit_quality_gate_passed`: `{report.fit_quality_gate_passed}`",
                f"- `fit_quality_gate_reason`: `{report.fit_quality_gate_reason}`",
                f"- `fit_quality_selected_kernel`: `{report.fit_quality_selected_kernel}`",
                f"- `fit_quality_selected_validation_loss`: "
                f"`{report.fit_quality_validation_loss:.6f}`",
                f"- `fit_quality_selected_no_lag`: `{report.fit_quality_no_lag:.6f}`",
                f"- `fit_quality_selected_best_single_lag`: "
                f"`{report.fit_quality_best_single_lag:.6f}`",
                f"- `fit_quality_selected_warning_codes`: "
                f"`{','.join(report.fit_quality_warning_codes) or 'none'}`",
                f"- `fit_quality_selected_mean_lag_seconds`: `{report.fit_quality_mean_lag:.2f}`",
                f"- `fit_quality_selected_p50_lag_seconds`: `{report.fit_quality_p50_lag:.2f}`",
                f"- `fit_quality_selected_p90_lag_seconds`: `{report.fit_quality_p90_lag:.2f}`",
            ]
        )
        if report.scenario_name == "mini_flowsheet":
            lines.extend(
                [
                    "- API mapping: `input_col=\"feed_copper_grade\"`, "
                    "`target_col=\"cleaner_copper_grade\"`, "
                    "`category_cols=[\"ore_type\"]`, `weight_col=\"feed_mass\"`",
                    f"- Ore campaign transition row: `{report.mini_transition_row}`",
                    "",
                    "```mermaid",
                    "flowchart LR",
                    "    FEED[Feed grade + feed mass + ore type] --> CRUSH[Crushing]",
                    "    CRUSH --> MILL[Ball mill]",
                    "    MILL --> CYCLONE[Hydrocyclone classification]",
                    "    CYCLONE --> FLOAT[Flotation banks]",
                    "    FLOAT --> CLEAN[Cleaner]",
                    "    CYCLONE -->|Underflow recycle| MILL",
                    "    CLEAN --> OUT[Cleaner copper grade]",
                    "```",
                    "",
                    "- Unit-level lag fits (best public learner per link):",
                    "",
                    report.mini_unit_lag_table or "",
                    "",
                    "- Future work boundary: regime-conditioned or ore-conditioned kernels may "
                    "be useful later, but are out of scope for this documentation plan.",
                ]
            )
        lines.extend(
            [
                "",
                _markdown_table(report.outcomes),
                "",
                "Feature preview:",
                "```text",
                str(report.feature_preview),
                "```",
                "",
                f"![{report.title} signals]({PLOT_LINK_PREFIX}/{report.signal_plot_name})",
                f"![{report.title} kernel]({PLOT_LINK_PREFIX}/{report.kernel_plot_name})",
                f"![{report.title} observed-vs-fit]({PLOT_LINK_PREFIX}/{report.fit_plot_name})",
                "",
            ]
        )
    OUTPUT_GALLERY.write_text("\n".join(lines))


def _coverage_table_markdown(reports: tuple[ScenarioReport, ...]) -> str:
    kernel_families = (
        "FixedDelayKernelLearner",
        "ExponentialKernelLearner",
        "GammaKernelLearner",
        "ErlangKernelLearner",
        "LogNormalKernelLearner",
        "DelayedExponentialKernelLearner",
        "UniformKernelLearner",
    )
    lines = [
        "## Coverage Table",
        "",
        "| kernel_family | scenario | validation_loss | recommendation_status |",
        "|---|---|---:|---|",
    ]
    for family in kernel_families:
        best_match: tuple[str, float, str] | None = None
        for report in reports:
            for outcome in report.outcomes:
                if outcome.learner_name != family:
                    continue
                loss = float(outcome.fit_result.fit_diagnostics.validation_loss)
                if outcome.learner_name in report.positive_kernels:
                    stat = "recommended" if report.fit_quality_gate_passed else "not_recommended"
                else:
                    stat = "comparison"
                item = (report.scenario_name, loss, stat)
                if best_match is None or loss < best_match[1]:
                    best_match = item
        if best_match is None:
            lines.append(f"| `{family}` | `missing` | nan | `reference_only` |")
        else:
            scenario_name, loss, status = best_match
            lines.append(f"| `{family}` | `{scenario_name}` | {loss:.6f} | `{status}` |")
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    core_reports = tuple(
        _build_core_report(idx, fixture)
        for idx, fixture in enumerate(core_scenario_fixtures())
    )
    mini_report = _build_mini_report()
    all_reports = core_reports + (mini_report,)
    _write_gallery(all_reports)


if __name__ == "__main__":
    main()
