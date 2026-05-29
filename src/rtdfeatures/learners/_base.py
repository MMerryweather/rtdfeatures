"""Shared learner fit preparation and result-assembly utilities (internal)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np
import polars as pl

from rtdfeatures.diagnostics import (
    BaselineComparison,
    FitDataCoverageSummary,
    FitDiagnostics,
    KernelShapeSummary,
)
from rtdfeatures.kernels import Kernel
from rtdfeatures.utils import lag_to_steps, resolve_and_validate_dt, validate_or_sort_time


@dataclass(frozen=True)
class _ScalingStats:
    center: float
    scale: float


@dataclass(frozen=True)
class PreparedFitData:
    ordered: pl.DataFrame
    dt_seconds: float
    min_lag_steps: int
    max_lag_steps: int
    input_values: np.ndarray
    target_values: np.ndarray
    design_matrix: np.ndarray
    response_vector: np.ndarray
    valid_indices: np.ndarray
    x_train: np.ndarray
    y_train: np.ndarray
    x_valid: np.ndarray
    y_valid: np.ndarray
    x_train_scaled: np.ndarray
    y_train_scaled: np.ndarray
    x_valid_scaled: np.ndarray
    y_valid_scaled: np.ndarray
    no_lag_valid_scaled: np.ndarray
    train_windows: int
    validation_windows: int
    total_valid_windows: int


@dataclass(frozen=True)
class LearnerConfig:
    max_lag: int | str | timedelta
    min_lag: int | str | timedelta
    dt: str | timedelta | None
    loss: str
    validation_fraction: float
    huber_delta: float


@dataclass(frozen=True)
class _ResolvedFitGrid:
    ordered: pl.DataFrame
    dt_seconds: float
    min_lag_steps: int
    max_lag_steps: int
    input_values: np.ndarray
    target_values: np.ndarray


@dataclass(frozen=True)
class _PreparedWindowData:
    design_matrix: np.ndarray
    response_vector: np.ndarray
    valid_indices: np.ndarray


@dataclass(frozen=True)
class _PreparedSplitData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_valid: np.ndarray
    y_valid: np.ndarray
    x_train_scaled: np.ndarray
    y_train_scaled: np.ndarray
    x_valid_scaled: np.ndarray
    y_valid_scaled: np.ndarray
    no_lag_valid_scaled: np.ndarray


def validate_learner_init(
    *,
    loss: str,
    smoothness_penalty: float,
    validation_fraction: float,
    learning_rate: float,
    max_epochs: int,
    huber_delta: float,
) -> None:
    if loss not in {"huber", "mse"}:
        raise ValueError("loss must be either 'huber' or 'mse'.")
    if smoothness_penalty < 0.0:
        raise ValueError("smoothness_penalty must be non-negative.")
    if validation_fraction <= 0.0 or validation_fraction >= 0.5:
        raise ValueError("validation_fraction must be in (0.0, 0.5).")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be strictly positive.")
    if max_epochs <= 0:
        raise ValueError("max_epochs must be a positive integer.")
    if huber_delta <= 0.0:
        raise ValueError("huber_delta must be strictly positive.")


def validate_fit_columns(
    df: pl.DataFrame,
    *,
    time_col: str,
    input_col: str,
    target_col: str,
) -> None:
    if input_col == target_col:
        raise ValueError("input_col and target_col must be different columns.")
    required_columns = {time_col, input_col, target_col}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Ensure these columns exist in the DataFrame (available: {sorted(df.columns)})."
        )


def _build_lagged_windows(
    *,
    input_values: np.ndarray,
    target_values: np.ndarray,
    min_lag_steps: int,
    max_lag_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lag_steps = list(range(min_lag_steps, max_lag_steps + 1))
    rows: list[np.ndarray] = []
    targets: list[float] = []
    indices: list[int] = []
    for idx in range(max_lag_steps, len(input_values)):
        row = np.array([input_values[idx - lag] for lag in lag_steps], dtype=np.float64)
        target_value = float(target_values[idx])
        if not np.isfinite(target_value) or not np.isfinite(row).all():
            continue
        rows.append(row)
        targets.append(target_value)
        indices.append(idx)

    if not rows:
        raise ValueError(
            "No valid lag windows remain after excluding missing values. "
            "Check that input and target columns have sufficient non-null finite values "
            "or consider a shorter lag window."
        )
    x = np.vstack(rows)
    y_arr = np.asarray(targets, dtype=np.float64)
    valid_indices = np.asarray(indices, dtype=np.int64)
    return x, y_arr, valid_indices


def _robust_scaling_stats(values: np.ndarray) -> _ScalingStats:
    center = float(np.median(values))
    q25 = float(np.percentile(values, 25.0))
    q75 = float(np.percentile(values, 75.0))
    scale = q75 - q25
    if scale <= 1e-12:
        std = float(np.std(values, ddof=0))
        scale = std if std > 1e-12 else 1.0
    return _ScalingStats(center=center, scale=scale)


def _resolve_fit_grid(
    df: pl.DataFrame,
    *,
    input_col: str,
    target_col: str,
    time_col: str,
    order_by_time: bool,
    config: LearnerConfig,
) -> _ResolvedFitGrid:
    validate_fit_columns(df, time_col=time_col, input_col=input_col, target_col=target_col)
    ordered = validate_or_sort_time(df, time_col=time_col, order_by_time=order_by_time)
    resolved_dt = resolve_and_validate_dt(ordered, time_col=time_col, dt=config.dt)
    dt_seconds = resolved_dt.total_seconds()
    min_lag_steps = lag_to_steps(config.min_lag, dt=resolved_dt, param_name="min_lag")
    max_lag_steps = lag_to_steps(config.max_lag, dt=resolved_dt, param_name="max_lag")
    if max_lag_steps < min_lag_steps:
        raise ValueError(
            f"max_lag ({max_lag_steps} steps) must be >= min_lag ({min_lag_steps} steps). "
            "Check the lag window configuration "
            f"(min_lag={config.min_lag!r}, max_lag={config.max_lag!r})."
        )
    input_values = ordered.get_column(input_col).cast(pl.Float64).to_numpy()
    target_values = ordered.get_column(target_col).cast(pl.Float64).to_numpy()
    return _ResolvedFitGrid(
        ordered=ordered,
        dt_seconds=dt_seconds,
        min_lag_steps=min_lag_steps,
        max_lag_steps=max_lag_steps,
        input_values=input_values,
        target_values=target_values,
    )


def _assemble_fit_windows(grid: _ResolvedFitGrid) -> _PreparedWindowData:
    design_matrix, response_vector, valid_indices = _build_lagged_windows(
        input_values=grid.input_values,
        target_values=grid.target_values,
        min_lag_steps=grid.min_lag_steps,
        max_lag_steps=grid.max_lag_steps,
    )
    if design_matrix.shape[0] < 8:
        raise ValueError(
            f"Not enough valid lag windows after missing-value filtering: "
            f"only {design_matrix.shape[0]} remain (minimum 8 required). "
            f"Try a shorter lag window or provide more data."
        )
    return _PreparedWindowData(
        design_matrix=design_matrix,
        response_vector=response_vector,
        valid_indices=valid_indices,
    )


def _split_train_validation(
    grid: _ResolvedFitGrid, windows: _PreparedWindowData, *, validation_fraction: float
) -> _PreparedSplitData:
    train_end = int(math.floor(windows.design_matrix.shape[0] * (1.0 - validation_fraction)))
    train_end = max(1, min(windows.design_matrix.shape[0] - 1, train_end))
    x_train = windows.design_matrix[:train_end]
    y_train = windows.response_vector[:train_end]
    x_valid = windows.design_matrix[train_end:]
    y_valid = windows.response_vector[train_end:]
    valid_idx_valid = windows.valid_indices[train_end:]

    x_stats = _robust_scaling_stats(x_train)
    y_stats = _robust_scaling_stats(y_train)
    x_train_scaled = (x_train - x_stats.center) / x_stats.scale
    y_train_scaled = (y_train - y_stats.center) / y_stats.scale
    x_valid_scaled = (x_valid - x_stats.center) / x_stats.scale
    y_valid_scaled = (y_valid - y_stats.center) / y_stats.scale
    no_lag_valid_scaled = (grid.input_values[valid_idx_valid] - x_stats.center) / x_stats.scale
    return _PreparedSplitData(
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        x_train_scaled=x_train_scaled,
        y_train_scaled=y_train_scaled,
        x_valid_scaled=x_valid_scaled,
        y_valid_scaled=y_valid_scaled,
        no_lag_valid_scaled=no_lag_valid_scaled,
    )


def prepare_fit_data(
    df: pl.DataFrame,
    *,
    input_col: str,
    target_col: str,
    time_col: str,
    order_by_time: bool,
    config: LearnerConfig,
) -> PreparedFitData:
    grid = _resolve_fit_grid(
        df,
        input_col=input_col,
        target_col=target_col,
        time_col=time_col,
        order_by_time=order_by_time,
        config=config,
    )
    windows = _assemble_fit_windows(grid)
    split = _split_train_validation(
        grid,
        windows,
        validation_fraction=config.validation_fraction,
    )

    return PreparedFitData(
        ordered=grid.ordered,
        dt_seconds=grid.dt_seconds,
        min_lag_steps=grid.min_lag_steps,
        max_lag_steps=grid.max_lag_steps,
        input_values=grid.input_values,
        target_values=grid.target_values,
        design_matrix=windows.design_matrix,
        response_vector=windows.response_vector,
        valid_indices=windows.valid_indices,
        x_train=split.x_train,
        y_train=split.y_train,
        x_valid=split.x_valid,
        y_valid=split.y_valid,
        x_train_scaled=split.x_train_scaled,
        y_train_scaled=split.y_train_scaled,
        x_valid_scaled=split.x_valid_scaled,
        y_valid_scaled=split.y_valid_scaled,
        no_lag_valid_scaled=split.no_lag_valid_scaled,
        train_windows=split.x_train.shape[0],
        validation_windows=split.x_valid.shape[0],
        total_valid_windows=windows.design_matrix.shape[0],
    )


# ── math helpers ─────────────────────────────────────────────────────────


def _inverse_softplus(value: float) -> float:
    if value <= 0.0 or not math.isfinite(value):
        raise ValueError("inverse_softplus input must be finite and strictly positive.")
    return float(np.log(np.expm1(value)))


# ── loss helpers ──────────────────────────────────────────────────────────


def _numpy_loss(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    loss: str,
    huber_delta: float,
) -> float:
    prediction_arr = np.asarray(prediction, dtype=np.float64)
    target_arr = np.asarray(target, dtype=np.float64)
    if loss == "mse":
        return float(np.mean((prediction_arr - target_arr) ** 2))
    residual = np.abs(prediction_arr - target_arr)
    quadratic = np.minimum(residual, huber_delta)
    linear = residual - quadratic
    return float(np.mean(0.5 * quadratic**2 + huber_delta * linear))


def _numpy_loss_on_finite_pairs(
    *,
    prediction: np.ndarray,
    target: np.ndarray,
    loss: str,
    huber_delta: float,
) -> float:
    prediction_arr = np.asarray(prediction, dtype=np.float64)
    target_arr = np.asarray(target, dtype=np.float64)
    finite_mask = np.isfinite(prediction_arr) & np.isfinite(target_arr)
    if not np.any(finite_mask):
        return float("inf")
    return _numpy_loss(
        prediction_arr[finite_mask],
        target_arr[finite_mask],
        loss=loss,
        huber_delta=huber_delta,
    )


# ── baseline helpers ──────────────────────────────────────────────────────


def exponential_alpha_grid() -> np.ndarray:
    """Return the alpha grid used for exponential baseline search."""
    return np.linspace(0.0, 4.0, 33, dtype=np.float64)


def best_single_lag_validation_loss(
    *,
    x_valid_scaled: np.ndarray,
    y_valid_scaled: np.ndarray,
    loss: str,
    huber_delta: float,
) -> float:
    best_loss = float("inf")
    for lag_idx in range(x_valid_scaled.shape[1]):
        valid_pred = x_valid_scaled[:, lag_idx]
        valid_loss = _numpy_loss(valid_pred, y_valid_scaled, loss=loss, huber_delta=huber_delta)
        if valid_loss < best_loss:
            best_loss = valid_loss
    return best_loss


def uniform_validation_loss(
    *,
    x_valid_scaled: np.ndarray,
    y_valid_scaled: np.ndarray,
    loss: str,
    huber_delta: float,
) -> float:
    uniform_pred = np.mean(x_valid_scaled, axis=1)
    return _numpy_loss(uniform_pred, y_valid_scaled, loss=loss, huber_delta=huber_delta)


def exponential_validation_loss(
    *,
    x_valid_scaled: np.ndarray,
    y_valid_scaled: np.ndarray,
    loss: str,
    huber_delta: float,
) -> tuple[float, float]:
    lag_offsets = np.arange(x_valid_scaled.shape[1], dtype=np.float64)
    best_loss = float("inf")
    best_alpha = 0.0
    for alpha in exponential_alpha_grid():
        weights = np.exp(-alpha * lag_offsets)
        weights /= np.sum(weights)
        pred = x_valid_scaled @ weights
        valid_loss = _numpy_loss(pred, y_valid_scaled, loss=loss, huber_delta=huber_delta)
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_alpha = float(alpha)
    return best_loss, best_alpha


def build_baseline_summary(
    *,
    learned_validation_loss: float,
    no_lag_validation_loss: float,
    best_single_lag_validation_loss: float,
    uniform_validation_loss: float | None,
    exponential_validation_loss: float | None,
    flat_floor: float,
    baseline_improvement_margin: float,
) -> dict[str, dict[str, float | bool]]:
    baselines: dict[str, float | None] = {
        "no_lag": no_lag_validation_loss,
        "best_single_lag": best_single_lag_validation_loss,
        "uniform": uniform_validation_loss,
        "exponential": exponential_validation_loss,
    }
    summary: dict[str, dict[str, float | bool]] = {}
    for name, baseline_loss in baselines.items():
        if baseline_loss is None or not np.isfinite(baseline_loss):
            continue
        delta_fraction = (learned_validation_loss - baseline_loss) / max(
            learned_validation_loss, flat_floor
        )
        summary[name] = {
            "baseline_validation_loss": float(baseline_loss),
            "learned_validation_loss": float(learned_validation_loss),
            "delta_fraction_vs_learned": float(delta_fraction),
            "beats_learned_by_margin": bool(
                delta_fraction >= baseline_improvement_margin
            ),
        }
    return summary


# ── kernel shape ──────────────────────────────────────────────────────────


def build_kernel_shape_summary(learned_weights: np.ndarray) -> KernelShapeSummary:
    safe_weights = learned_weights[learned_weights > 0.0]
    entropy = float(-np.sum(safe_weights * np.log(safe_weights)))
    normalized_entropy = (
        entropy / math.log(learned_weights.size) if learned_weights.size > 1 else 0.0
    )
    hhi = float(np.sum(learned_weights**2))
    effective_lag_count = float(1.0 / hhi) if hhi > 0.0 else float(learned_weights.size)
    return KernelShapeSummary(
        normalized_entropy=float(normalized_entropy),
        max_weight=float(np.max(learned_weights)),
        min_weight=float(np.min(learned_weights)),
        concentration_hhi=hhi,
        effective_lag_count=effective_lag_count,
    )


# ── high-level result-assembly builders ───────────────────────────────────


def build_common_fit_diagnostics(
    *,
    train_loss: float,
    validation_loss: float,
    x_train_scaled: np.ndarray,
    y_train_scaled: np.ndarray,
    kernel: Kernel,
    min_lag_steps: int,
    max_lag_steps: int,
    dt_seconds: float,
    tail_mass_fraction_of_lag_window: float = 0.75,
) -> FitDiagnostics:
    weights_arr = np.asarray(kernel.weights, dtype=np.float64)
    lag_window_tail_threshold = (
        min_lag_steps + tail_mass_fraction_of_lag_window * (max_lag_steps - min_lag_steps)
    ) * dt_seconds
    return FitDiagnostics(
        train_loss=train_loss,
        validation_loss=validation_loss,
        input_variance=float(np.var(x_train_scaled, ddof=0)),
        target_variance=float(np.var(y_train_scaled, ddof=0)),
        kernel_weight_sum=float(sum(kernel.weights)),
        mean_lag=kernel.mean_lag(),
        p50_lag=kernel.percentile(0.5),
        p90_lag=kernel.percentile(0.9),
        tail_mass=kernel.tail_mass(lag_window_tail_threshold),
        boundary_mass_fraction=float(weights_arr[0] + weights_arr[-1]),
    )


def build_common_baseline_comparison(
    *,
    learned_validation_loss: float,
    no_lag_validation_loss: float,
    best_single_lag_validation_loss: float,
    uniform_validation_loss: float | None,
    exponential_validation_loss: float | None,
    flat_floor: float = 1e-8,
    baseline_improvement_margin: float = 0.05,
) -> BaselineComparison:
    return BaselineComparison(
        no_lag_validation_loss=no_lag_validation_loss,
        best_single_lag_validation_loss=best_single_lag_validation_loss,
        learned_validation_loss=learned_validation_loss,
        uniform_validation_loss=uniform_validation_loss,
        exponential_validation_loss=exponential_validation_loss,
        summary_by_baseline=build_baseline_summary(
            learned_validation_loss=learned_validation_loss,
            no_lag_validation_loss=no_lag_validation_loss,
            best_single_lag_validation_loss=best_single_lag_validation_loss,
            uniform_validation_loss=uniform_validation_loss,
            exponential_validation_loss=exponential_validation_loss,
            flat_floor=flat_floor,
            baseline_improvement_margin=baseline_improvement_margin,
        ),
    )


def build_common_fit_data_coverage_summary(
    *,
    total_rows: int,
    max_lag_steps: int,
    valid_windows: int,
    train_windows: int,
    validation_windows: int,
) -> FitDataCoverageSummary:
    possible_windows = max(0, total_rows - max_lag_steps)
    retained_row_fraction = (
        float(valid_windows / total_rows) if total_rows > 0 else 0.0
    )
    retained_window_fraction = (
        float(valid_windows / possible_windows) if possible_windows > 0 else 0.0
    )
    return FitDataCoverageSummary(
        total_rows=total_rows,
        valid_windows=valid_windows,
        train_windows=train_windows,
        validation_windows=validation_windows,
        retained_row_fraction=retained_row_fraction,
        retained_window_fraction=retained_window_fraction,
    )


def build_common_fit_provenance(
    *,
    seed: int | None,
    loss: str,
    huber_delta: float,
    validation_fraction: float,
    smoothness_penalty: float,
    dt_seconds: float,
    train_rows: int,
    validation_rows: int,
    total_valid_windows: int,
    exponential_baseline_alpha_grid: tuple[float, ...],
    exponential_baseline_best_alpha: float,
    extra_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "seed": seed,
        "loss": loss,
        "huber_delta": huber_delta if loss == "huber" else None,
        "validation_fraction": validation_fraction,
        "smoothness_penalty": smoothness_penalty,
        "dt_seconds": dt_seconds,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "total_valid_windows": total_valid_windows,
        "baseline_losses_use_configured_loss": True,
        "exponential_baseline_alpha_grid": exponential_baseline_alpha_grid,
        "exponential_baseline_best_alpha": exponential_baseline_best_alpha,
    }
    if extra_provenance is not None:
        provenance.update(extra_provenance)
    return provenance
