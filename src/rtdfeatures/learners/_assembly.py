"""Shared learner fit assembly utilities (internal)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rtdfeatures.diagnostics import KernelFitResult
from rtdfeatures.kernels import Kernel
from rtdfeatures.learners._base import (
    PreparedFitData,
    _numpy_loss_on_finite_pairs,
    best_single_lag_validation_loss,
    build_common_baseline_comparison,
    build_common_fit_data_coverage_summary,
    build_common_fit_diagnostics,
    build_common_fit_provenance,
    build_kernel_shape_summary,
    exponential_alpha_grid,
    exponential_validation_loss,
    uniform_validation_loss,
)
from rtdfeatures.learners._identifiability import build_identifiability_report


@dataclass(frozen=True)
class BaselineLosses:
    no_lag: float
    best_single_lag: float
    uniform: float | None
    exponential: float | None
    best_exponential_alpha: float


@dataclass(frozen=True)
class FitAssemblyInput:
    prepared: PreparedFitData
    kernel: Kernel
    train_loss: float
    validation_loss: float
    learned_weights: np.ndarray
    seed: int | None
    loss: str
    huber_delta: float
    validation_fraction: float
    smoothness_penalty: float
    flat_floor: float
    baseline_improvement_margin: float
    baseline_losses: BaselineLosses
    extra_provenance: dict[str, Any] | None = None


def evaluate_baselines(
    *,
    prepared: PreparedFitData,
    loss: str,
    huber_delta: float,
) -> BaselineLosses:
    no_lag_valid_loss = _numpy_loss_on_finite_pairs(
        prediction=prepared.no_lag_valid_scaled,
        target=prepared.y_valid_scaled,
        loss=loss,
        huber_delta=huber_delta,
    )
    best_single_lag_valid_loss = best_single_lag_validation_loss(
        x_valid_scaled=prepared.x_valid_scaled,
        y_valid_scaled=prepared.y_valid_scaled,
        loss=loss,
        huber_delta=huber_delta,
    )
    uniform_valid_loss = uniform_validation_loss(
        x_valid_scaled=prepared.x_valid_scaled,
        y_valid_scaled=prepared.y_valid_scaled,
        loss=loss,
        huber_delta=huber_delta,
    )
    exponential_valid_loss, best_exponential_alpha = exponential_validation_loss(
        x_valid_scaled=prepared.x_valid_scaled,
        y_valid_scaled=prepared.y_valid_scaled,
        loss=loss,
        huber_delta=huber_delta,
    )
    return BaselineLosses(
        no_lag=no_lag_valid_loss,
        best_single_lag=best_single_lag_valid_loss,
        uniform=uniform_valid_loss,
        exponential=exponential_valid_loss,
        best_exponential_alpha=best_exponential_alpha,
    )


def evaluate_weight_vector_losses(
    *,
    prepared: PreparedFitData,
    weights: np.ndarray,
    loss: str,
    huber_delta: float,
) -> tuple[float, float]:
    weights_arr = np.asarray(weights, dtype=np.float64)
    train_prediction = prepared.x_train_scaled @ weights_arr
    valid_prediction = prepared.x_valid_scaled @ weights_arr
    train_loss = _numpy_loss_on_finite_pairs(
        prediction=train_prediction,
        target=prepared.y_train_scaled,
        loss=loss,
        huber_delta=huber_delta,
    )
    validation_loss = _numpy_loss_on_finite_pairs(
        prediction=valid_prediction,
        target=prepared.y_valid_scaled,
        loss=loss,
        huber_delta=huber_delta,
    )
    return train_loss, validation_loss


def assemble_kernel_fit_result(
    *,
    assembly: FitAssemblyInput,
) -> KernelFitResult:
    fit_diagnostics = build_common_fit_diagnostics(
        train_loss=assembly.train_loss,
        validation_loss=assembly.validation_loss,
        x_train_scaled=assembly.prepared.x_train_scaled,
        y_train_scaled=assembly.prepared.y_train_scaled,
        kernel=assembly.kernel,
        min_lag_steps=assembly.prepared.min_lag_steps,
        max_lag_steps=assembly.prepared.max_lag_steps,
        dt_seconds=assembly.prepared.dt_seconds,
    )
    identifiability_report = build_identifiability_report(
        fit_diagnostics=fit_diagnostics,
        learned_weights=np.asarray(assembly.learned_weights, dtype=np.float64),
        no_lag_validation_loss=assembly.baseline_losses.no_lag,
        best_single_lag_validation_loss=assembly.baseline_losses.best_single_lag,
        uniform_validation_loss=assembly.baseline_losses.uniform,
        exponential_validation_loss=assembly.baseline_losses.exponential,
    )
    baseline_comparison = build_common_baseline_comparison(
        learned_validation_loss=assembly.validation_loss,
        no_lag_validation_loss=assembly.baseline_losses.no_lag,
        best_single_lag_validation_loss=assembly.baseline_losses.best_single_lag,
        uniform_validation_loss=assembly.baseline_losses.uniform,
        exponential_validation_loss=assembly.baseline_losses.exponential,
        flat_floor=assembly.flat_floor,
        baseline_improvement_margin=assembly.baseline_improvement_margin,
    )
    kernel_shape_summary = build_kernel_shape_summary(
        np.asarray(assembly.learned_weights, dtype=np.float64)
    )
    fit_data_coverage_summary = build_common_fit_data_coverage_summary(
        total_rows=assembly.prepared.ordered.height,
        max_lag_steps=assembly.prepared.max_lag_steps,
        valid_windows=assembly.prepared.total_valid_windows,
        train_windows=assembly.prepared.train_windows,
        validation_windows=assembly.prepared.validation_windows,
    )
    provenance = build_common_fit_provenance(
        seed=assembly.seed,
        loss=assembly.loss,
        huber_delta=assembly.huber_delta,
        validation_fraction=assembly.validation_fraction,
        smoothness_penalty=assembly.smoothness_penalty,
        dt_seconds=assembly.prepared.dt_seconds,
        train_rows=assembly.prepared.train_windows,
        validation_rows=assembly.prepared.validation_windows,
        total_valid_windows=assembly.prepared.total_valid_windows,
        exponential_baseline_alpha_grid=tuple(float(v) for v in exponential_alpha_grid()),
        exponential_baseline_best_alpha=assembly.baseline_losses.best_exponential_alpha,
    )
    if assembly.extra_provenance:
        provenance.update(assembly.extra_provenance)

    return KernelFitResult(
        kernel=assembly.kernel,
        fit_diagnostics=fit_diagnostics,
        identifiability_report=identifiability_report,
        baseline_comparison=baseline_comparison,
        kernel_shape_summary=kernel_shape_summary,
        fit_data_coverage_summary=fit_data_coverage_summary,
        fit_provenance=provenance,
    )
