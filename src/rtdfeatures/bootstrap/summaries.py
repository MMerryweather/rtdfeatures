"""Bootstrap summary/interval table builders."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import polars as pl

from rtdfeatures.diagnostics import (
    DEFAULT_BOOTSTRAP_INTERVAL_QUANTILES,
    BootstrapParameterSample,
    BootstrapResult,
    BootstrapWeightSample,
    KernelBootstrapSummary,
    KernelFamilyFitResult,
    ParameterUncertaintySummary,
    WeightUncertaintySummary,
    bootstrap_lag_summary_samples_schema,
    bootstrap_parameter_samples_schema,
    bootstrap_weight_samples_schema,
)


def build_kernel_bootstrap_summary(
    bootstrap_result: BootstrapResult,
    *,
    candidate_id: str | None = None,
    interval_quantiles: tuple[float, float] = DEFAULT_BOOTSTRAP_INTERVAL_QUANTILES,
    min_successes_for_warning: int = 20,
    family_stability_threshold: float = 0.8,
    lag_bounds: tuple[float, float] | None = None,
) -> tuple[KernelBootstrapSummary, tuple[str, ...]]:
    lower_q, upper_q = interval_quantiles
    if not (0.0 <= lower_q < upper_q <= 1.0):
        raise ValueError("interval_quantiles must satisfy 0.0 <= lower < upper <= 1.0.")
    if min_successes_for_warning <= 0:
        raise ValueError("min_successes_for_warning must be a positive integer.")
    if not (0.0 < family_stability_threshold <= 1.0):
        raise ValueError("family_stability_threshold must be in (0.0, 1.0].")

    weight_samples = [
        sample
        for sample in bootstrap_result.weight_samples
        if candidate_id is None or sample.candidate_id == candidate_id
    ]
    parameter_samples = [
        sample
        for sample in bootstrap_result.parameter_samples
        if candidate_id is None or sample.candidate_id == candidate_id
    ]
    lag_summary_samples = [
        sample
        for sample in bootstrap_result.lag_summary_samples
        if candidate_id is None or sample.candidate_id == candidate_id
    ]

    lag_by_name: dict[str, list[float]] = {
        "mean_lag": [sample.mean_lag for sample in lag_summary_samples],
        "p50_lag": [sample.p50_lag for sample in lag_summary_samples],
        "p90_lag": [sample.p90_lag for sample in lag_summary_samples],
        "tail_mass": [sample.tail_mass for sample in lag_summary_samples],
    }

    weight_interval_by_lag = _build_weight_uncertainty_summaries(
        weight_samples=weight_samples, lower_q=lower_q, upper_q=upper_q,
    )
    parameter_interval_by_name = _build_parameter_uncertainty_summaries(
        parameter_samples=parameter_samples, lower_q=lower_q, upper_q=upper_q,
    )

    summary = KernelBootstrapSummary(
        mean_lag_interval=_interval(lag_by_name["mean_lag"], lower_q, upper_q),
        p50_lag_interval=_interval(lag_by_name["p50_lag"], lower_q, upper_q),
        p90_lag_interval=_interval(lag_by_name["p90_lag"], lower_q, upper_q),
        tail_mass_interval=_interval(lag_by_name["tail_mass"], lower_q, upper_q),
        weight_interval_by_lag=weight_interval_by_lag,
        parameter_interval_by_name=parameter_interval_by_name,
        stability_score=_stability_score(bootstrap_result),
    )

    warnings: list[str] = []
    if bootstrap_result.n_succeeded < min_successes_for_warning:
        warnings.append("BOOTSTRAP_TOO_FEW_SUCCESSES")
    total_selected = sum(bootstrap_result.family_selection_counts.values())
    if total_selected > 0 and len(bootstrap_result.family_selection_counts) > 1:
        top = max(bootstrap_result.family_selection_counts.values())
        if (top / total_selected) < family_stability_threshold:
            warnings.append("BOOTSTRAP_FAMILY_UNSTABLE")
    if _interval_touches_boundary(summary, lag_bounds=lag_bounds):
        warnings.append("BOOTSTRAP_INTERVAL_TOUCHES_BOUNDARY")
    return summary, tuple(warnings)


def bootstrap_weight_samples_table(
    bootstrap_result: BootstrapResult,
    *,
    candidate_id: str | None = None,
) -> pl.DataFrame:
    rows = [
        {
            "bootstrap_id": int(sample.bootstrap_id),
            "candidate_id": str(sample.candidate_id),
            "lag_step": int(sample.lag_step),
            "lag_time": float(sample.lag_time),
            "weight": float(sample.weight),
        }
        for sample in bootstrap_result.weight_samples
        if candidate_id is None or sample.candidate_id == candidate_id
    ]
    rows.sort(key=lambda row: (row["candidate_id"], row["bootstrap_id"], row["lag_step"]))
    return pl.DataFrame(rows, schema=bootstrap_weight_samples_schema()).select(
        ["bootstrap_id", "candidate_id", "lag_step", "lag_time", "weight"]
    )


def bootstrap_parameter_samples_table(
    bootstrap_result: BootstrapResult,
    *,
    candidate_id: str | None = None,
) -> pl.DataFrame:
    rows = [
        {
            "bootstrap_id": int(sample.bootstrap_id),
            "candidate_id": str(sample.candidate_id),
            "parameter_name": str(sample.parameter_name),
            "parameter_value": (
                float(sample.parameter_value) if sample.parameter_value is not None else None
            ),
        }
        for sample in bootstrap_result.parameter_samples
        if candidate_id is None or sample.candidate_id == candidate_id
    ]
    rows.sort(key=lambda row: (row["candidate_id"], row["bootstrap_id"], row["parameter_name"]))
    return pl.DataFrame(rows, schema=bootstrap_parameter_samples_schema()).select(
        ["bootstrap_id", "candidate_id", "parameter_name", "parameter_value"]
    )


def bootstrap_lag_summary_samples_table(
    bootstrap_result: BootstrapResult,
    *,
    candidate_id: str | None = None,
) -> pl.DataFrame:
    rows = [
        {
            "bootstrap_id": int(sample.bootstrap_id),
            "candidate_id": str(sample.candidate_id),
            "mean_lag": float(sample.mean_lag),
            "p50_lag": float(sample.p50_lag),
            "p90_lag": float(sample.p90_lag),
            "tail_mass": float(sample.tail_mass),
        }
        for sample in bootstrap_result.lag_summary_samples
        if candidate_id is None or sample.candidate_id == candidate_id
    ]
    rows.sort(key=lambda row: (row["candidate_id"], row["bootstrap_id"]))
    return pl.DataFrame(rows, schema=bootstrap_lag_summary_samples_schema()).select(
        ["bootstrap_id", "candidate_id", "mean_lag", "p50_lag", "p90_lag", "tail_mass"]
    )


def bootstrap_weight_interval_table(
    bootstrap_result: BootstrapResult,
    *,
    candidate_id: str | None = None,
    interval_quantiles: tuple[float, float] = DEFAULT_BOOTSTRAP_INTERVAL_QUANTILES,
) -> pl.DataFrame:
    lower_q, upper_q = interval_quantiles
    rows: list[dict[str, Any]] = []
    for candidate in _ordered_candidate_ids(bootstrap_result, candidate_id=candidate_id):
        summary, _ = build_kernel_bootstrap_summary(
            bootstrap_result, candidate_id=candidate, interval_quantiles=(lower_q, upper_q),
        )
        grouped_counts = _weight_counts_by_lag(bootstrap_result, candidate)
        for item in summary.weight_interval_by_lag:
            rows.append({
                "candidate_id": candidate,
                "lag_step": int(item.lag_step),
                "lag_time": float(item.lag_time),
                "weight_estimate": float(item.weight_estimate),
                "lower": float(item.lower),
                "upper": float(item.upper),
                "bootstrap_std": float(item.bootstrap_std),
                "n_samples": int(grouped_counts.get((item.lag_step, item.lag_time), 0)),
            })
    rows.sort(key=lambda row: (row["candidate_id"], row["lag_step"]))
    return pl.DataFrame(rows, schema={
        "candidate_id": pl.String, "lag_step": pl.Int64, "lag_time": pl.Float64,
        "weight_estimate": pl.Float64, "lower": pl.Float64, "upper": pl.Float64,
        "bootstrap_std": pl.Float64, "n_samples": pl.Int64,
    }).select([
        "candidate_id", "lag_step", "lag_time", "weight_estimate",
        "lower", "upper", "bootstrap_std", "n_samples",
    ])


def bootstrap_parameter_interval_table(
    bootstrap_result: BootstrapResult,
    *,
    candidate_id: str | None = None,
    interval_quantiles: tuple[float, float] = DEFAULT_BOOTSTRAP_INTERVAL_QUANTILES,
) -> pl.DataFrame:
    lower_q, upper_q = interval_quantiles
    rows: list[dict[str, Any]] = []
    for candidate in _ordered_candidate_ids(bootstrap_result, candidate_id=candidate_id):
        summary, _ = build_kernel_bootstrap_summary(
            bootstrap_result, candidate_id=candidate, interval_quantiles=(lower_q, upper_q),
        )
        for item in summary.parameter_interval_by_name:
            rows.append({
                "candidate_id": candidate,
                "parameter_name": str(item.parameter_name),
                "estimate": float(item.estimate),
                "lower": float(item.lower),
                "upper": float(item.upper),
                "bootstrap_std": float(item.bootstrap_std),
                "n_samples": int(item.n_samples),
            })
    rows.sort(key=lambda row: (row["candidate_id"], row["parameter_name"]))
    return pl.DataFrame(rows, schema={
        "candidate_id": pl.String, "parameter_name": pl.String, "estimate": pl.Float64,
        "lower": pl.Float64, "upper": pl.Float64,
        "bootstrap_std": pl.Float64, "n_samples": pl.Int64,
    }).select([
        "candidate_id", "parameter_name", "estimate",
        "lower", "upper", "bootstrap_std", "n_samples",
    ])


def bootstrap_lag_interval_table(
    bootstrap_result: BootstrapResult,
    *,
    candidate_id: str | None = None,
    interval_quantiles: tuple[float, float] = DEFAULT_BOOTSTRAP_INTERVAL_QUANTILES,
) -> pl.DataFrame:
    lower_q, upper_q = interval_quantiles
    rows: list[dict[str, Any]] = []
    for candidate in _ordered_candidate_ids(bootstrap_result, candidate_id=candidate_id):
        samples = [
            sample for sample in bootstrap_result.lag_summary_samples
            if sample.candidate_id == candidate
        ]
        by_metric = {
            "mean_lag": [float(sample.mean_lag) for sample in samples],
            "p50_lag": [float(sample.p50_lag) for sample in samples],
            "p90_lag": [float(sample.p90_lag) for sample in samples],
            "tail_mass": [float(sample.tail_mass) for sample in samples],
        }
        summary, _ = build_kernel_bootstrap_summary(
            bootstrap_result, candidate_id=candidate, interval_quantiles=(lower_q, upper_q),
        )
        bounds = {
            "mean_lag": summary.mean_lag_interval,
            "p50_lag": summary.p50_lag_interval,
            "p90_lag": summary.p90_lag_interval,
            "tail_mass": summary.tail_mass_interval,
        }
        for metric in ("mean_lag", "p50_lag", "p90_lag", "tail_mass"):
            values = by_metric[metric]
            lower, upper = bounds[metric]
            rows.append({
                "candidate_id": candidate,
                "metric": metric,
                "estimate": float(np.mean(values)) if values else float("nan"),
                "lower": float(lower),
                "upper": float(upper),
                "bootstrap_std": float(np.std(values, ddof=0)) if values else float("nan"),
                "n_samples": len(values),
            })
    rows.sort(key=lambda row: (row["candidate_id"], row["metric"]))
    return pl.DataFrame(rows, schema={
        "candidate_id": pl.String, "metric": pl.String, "estimate": pl.Float64,
        "lower": pl.Float64, "upper": pl.Float64,
        "bootstrap_std": pl.Float64, "n_samples": pl.Int64,
    }).select([
        "candidate_id", "metric", "estimate",
        "lower", "upper", "bootstrap_std", "n_samples",
    ])


def bootstrap_summary_compact_dict(
    bootstrap_result: BootstrapResult,
    *,
    candidate_id: str | None = None,
    interval_quantiles: tuple[float, float] = DEFAULT_BOOTSTRAP_INTERVAL_QUANTILES,
) -> dict[str, Any]:
    lag_intervals = bootstrap_lag_interval_table(
        bootstrap_result, candidate_id=candidate_id, interval_quantiles=interval_quantiles,
    )
    return {
        "n_bootstrap": int(bootstrap_result.n_bootstrap),
        "n_succeeded": int(bootstrap_result.n_succeeded),
        "n_failed": int(bootstrap_result.n_failed),
        "candidate_ids": tuple(_ordered_candidate_ids(bootstrap_result, candidate_id=candidate_id)),
        "interval_quantiles": tuple(float(q) for q in interval_quantiles),
        "family_selection_counts": dict(sorted(bootstrap_result.family_selection_counts.items())),
        "warning_codes": tuple(bootstrap_result.warnings),
        "lag_metrics": lag_intervals.to_dicts(),
    }


def bootstrap_summary_compact_text(
    bootstrap_result: BootstrapResult,
    *,
    candidate_id: str | None = None,
    interval_quantiles: tuple[float, float] = DEFAULT_BOOTSTRAP_INTERVAL_QUANTILES,
) -> str:
    summary = bootstrap_summary_compact_dict(
        bootstrap_result, candidate_id=candidate_id, interval_quantiles=interval_quantiles,
    )
    candidate_text = ",".join(summary["candidate_ids"]) if summary["candidate_ids"] else "none"
    return (
        "bootstrap summary: "
        f"n_bootstrap={summary['n_bootstrap']}, "
        f"n_succeeded={summary['n_succeeded']}, "
        f"n_failed={summary['n_failed']}, "
        f"candidates={candidate_text}, "
        f"warnings={len(summary['warning_codes'])}"
    )


def _ordered_candidate_ids(
    bootstrap_result: BootstrapResult, *, candidate_id: str | None = None,
) -> tuple[str, ...]:
    ids = {
        str(sample.candidate_id) for sample in bootstrap_result.weight_samples
    } | {
        str(sample.candidate_id) for sample in bootstrap_result.parameter_samples
    } | {
        str(sample.candidate_id) for sample in bootstrap_result.lag_summary_samples
    }
    if candidate_id is not None:
        return (candidate_id,) if candidate_id in ids else ()
    return tuple(sorted(ids))


def _weight_counts_by_lag(
    bootstrap_result: BootstrapResult, candidate_id: str
) -> dict[tuple[int, float], int]:
    counts: dict[tuple[int, float], int] = {}
    for sample in bootstrap_result.weight_samples:
        if sample.candidate_id != candidate_id:
            continue
        key = (int(sample.lag_step), float(sample.lag_time))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _select_bootstrap_candidate(
    *,
    succeeded_losses: list[tuple[KernelFamilyFitResult, float]],
    loss_tolerance_fraction: float,
) -> KernelFamilyFitResult:
    if not succeeded_losses:
        raise ValueError("succeeded_losses must contain at least one candidate.")
    if loss_tolerance_fraction < 0.0:
        raise ValueError("loss_tolerance_fraction must be non-negative.")
    best_loss = min(loss for _result, loss in succeeded_losses)
    tolerance_pool = [
        (result, loss) for result, loss in succeeded_losses
        if _loss_delta_fraction(best_loss, loss) <= loss_tolerance_fraction
    ]
    if not tolerance_pool:
        tolerance_pool = list(succeeded_losses)
    tolerance_pool_with_updated_loss = [
        (replace(result, validation_loss=float(loss)), float(loss))
        for result, loss in tolerance_pool
    ]
    selected_result, _selected_loss = min(
        tolerance_pool_with_updated_loss,
        key=lambda item: (
            _bootstrap_simplicity_rank(item[0]), item[1], item[0].candidate.candidate_id
        ),
    )
    return selected_result


def _bootstrap_simplicity_rank(result: KernelFamilyFitResult) -> int:
    if result.is_baseline:
        return 4
    if result.candidate.candidate_type == "fixed_kernel":
        return 1
    if result.candidate.candidate_type == "empirical_learner":
        return 2
    if result.candidate.candidate_type == "parametric_learner":
        return 3
    return 5


def _loss_delta_fraction(loss_a: float, loss_b: float) -> float:
    return abs(loss_b - loss_a) / max(abs(loss_a), 1e-12)


def _build_weight_uncertainty_summaries(
    *, weight_samples: list[BootstrapWeightSample], lower_q: float, upper_q: float,
) -> tuple[WeightUncertaintySummary, ...]:
    grouped: dict[tuple[int, float], list[float]] = {}
    for sample in weight_samples:
        key = (int(sample.lag_step), float(sample.lag_time))
        grouped.setdefault(key, []).append(float(sample.weight))
    rows: list[WeightUncertaintySummary] = []
    for lag_step, lag_time in sorted(grouped):
        values = grouped[(lag_step, lag_time)]
        lower, upper = _interval(values, lower_q, upper_q)
        rows.append(WeightUncertaintySummary(
            lag_step=lag_step, lag_time=lag_time,
            weight_estimate=float(np.mean(values)) if values else float("nan"),
            lower=lower, upper=upper,
            bootstrap_std=float(np.std(values, ddof=0)) if values else float("nan"),
        ))
    return tuple(rows)


def _build_parameter_uncertainty_summaries(
    *, parameter_samples: list[BootstrapParameterSample], lower_q: float, upper_q: float,
) -> tuple[ParameterUncertaintySummary, ...]:
    grouped: dict[str, list[float]] = {}
    for sample in parameter_samples:
        if sample.parameter_value is None:
            continue
        grouped.setdefault(sample.parameter_name, []).append(float(sample.parameter_value))
    rows: list[ParameterUncertaintySummary] = []
    for parameter_name in sorted(grouped):
        values = grouped[parameter_name]
        lower, upper = _interval(values, lower_q, upper_q)
        rows.append(ParameterUncertaintySummary(
            parameter_name=parameter_name,
            estimate=float(np.mean(values)) if values else float("nan"),
            lower=lower, upper=upper,
            bootstrap_std=float(np.std(values, ddof=0)) if values else float("nan"),
            n_samples=len(values),
        ))
    return tuple(rows)


def _interval(values: Sequence[float], lower_q: float, upper_q: float) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    arr = np.asarray(values, dtype=np.float64)
    return (
        float(np.quantile(arr, lower_q, method="linear")),
        float(np.quantile(arr, upper_q, method="linear")),
    )


def _stability_score(bootstrap_result: BootstrapResult) -> float | None:
    if bootstrap_result.n_bootstrap <= 0:
        return None
    score = bootstrap_result.n_succeeded / bootstrap_result.n_bootstrap
    return float(max(0.0, min(1.0, score)))


def _interval_touches_boundary(
    summary: KernelBootstrapSummary, *, lag_bounds: tuple[float, float] | None,
) -> bool:
    if lag_bounds is None:
        return False
    min_lag, max_lag = lag_bounds
    lag_intervals = (summary.mean_lag_interval, summary.p50_lag_interval, summary.p90_lag_interval)
    for lower, upper in lag_intervals:
        if math.isnan(lower) or math.isnan(upper):
            continue
        if lower <= min_lag or upper >= max_lag:
            return True
    return False
