"""Fixed-family kernel learners sharing common fit preparation and assembly."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import polars as pl

from rtdfeatures.diagnostics import KernelFitResult
from rtdfeatures.kernels import FixedDelayKernel, UniformKernel
from rtdfeatures.learners._assembly import (
    FitAssemblyInput,
    assemble_kernel_fit_result,
    evaluate_baselines,
    evaluate_weight_vector_losses,
)
from rtdfeatures.learners._base import (
    LearnerConfig,
    prepare_fit_data,
    validate_learner_init,
)


class FixedDelayKernelLearner:
    """One-input/one-target fixed-delay learner over an admissible lag window."""

    _FLAT_VARIANCE_THRESHOLD = 1e-8
    _BASELINE_IMPROVEMENT_MARGIN = 0.05

    def __init__(
        self,
        *,
        max_lag: int | str | timedelta,
        min_lag: int | str | timedelta = 0,
        dt: str | timedelta | None = None,
        loss: str = "huber",
        smoothness_penalty: float = 0.0,
        seed: int | None = None,
        validation_fraction: float = 0.2,
        learning_rate: float = 0.05,
        max_epochs: int = 800,
        huber_delta: float = 1.0,
    ) -> None:
        validate_learner_init(
            loss=loss,
            smoothness_penalty=smoothness_penalty,
            validation_fraction=validation_fraction,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            huber_delta=huber_delta,
        )
        self.max_lag = max_lag
        self.min_lag = min_lag
        self.dt = dt
        self.loss = loss
        self.smoothness_penalty = smoothness_penalty
        self.seed = seed
        self.validation_fraction = validation_fraction
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.huber_delta = huber_delta

    def fit(
        self,
        df: pl.DataFrame,
        *,
        input_col: str,
        target_col: str,
        time_col: str,
        order_by_time: bool = False,
    ) -> KernelFitResult:
        config = LearnerConfig(
            max_lag=self.max_lag,
            min_lag=self.min_lag,
            dt=self.dt,
            loss=self.loss,
            validation_fraction=self.validation_fraction,
            huber_delta=self.huber_delta,
        )
        prepared = prepare_fit_data(
            df,
            input_col=input_col,
            target_col=target_col,
            time_col=time_col,
            order_by_time=order_by_time,
            config=config,
        )
        n_lags = prepared.max_lag_steps - prepared.min_lag_steps + 1
        lag_steps = tuple(range(prepared.min_lag_steps, prepared.max_lag_steps + 1))

        best_delay_steps = prepared.min_lag_steps
        best_weights = np.zeros(n_lags, dtype=np.float64)
        best_weights[0] = 1.0
        best_train_loss, best_validation_loss = evaluate_weight_vector_losses(
            prepared=prepared,
            weights=best_weights,
            loss=self.loss,
            huber_delta=self.huber_delta,
        )
        for idx, delay_steps in enumerate(lag_steps[1:], start=1):
            candidate = np.zeros(n_lags, dtype=np.float64)
            candidate[idx] = 1.0
            train_loss, validation_loss = evaluate_weight_vector_losses(
                prepared=prepared,
                weights=candidate,
                loss=self.loss,
                huber_delta=self.huber_delta,
            )
            candidate_rank = (validation_loss, train_loss, delay_steps)
            best_rank = (best_validation_loss, best_train_loss, best_delay_steps)
            if candidate_rank < best_rank:
                best_delay_steps = delay_steps
                best_weights = candidate
                best_train_loss = train_loss
                best_validation_loss = validation_loss

        learned_kernel = FixedDelayKernel(
            delay_steps=best_delay_steps,
            max_lag_steps=prepared.max_lag_steps,
            min_lag_steps=prepared.min_lag_steps,
            dt=prepared.dt_seconds,
            name=f"{input_col}->{target_col}",
        )
        baseline_losses = evaluate_baselines(
            prepared=prepared,
            loss=self.loss,
            huber_delta=self.huber_delta,
        )
        return assemble_kernel_fit_result(
            assembly=FitAssemblyInput(
                prepared=prepared,
                kernel=learned_kernel,
                train_loss=best_train_loss,
                validation_loss=best_validation_loss,
                learned_weights=best_weights,
                seed=self.seed,
                loss=self.loss,
                huber_delta=self.huber_delta,
                validation_fraction=self.validation_fraction,
                smoothness_penalty=self.smoothness_penalty,
                flat_floor=self._FLAT_VARIANCE_THRESHOLD,
                baseline_improvement_margin=self._BASELINE_IMPROVEMENT_MARGIN,
                baseline_losses=baseline_losses,
                extra_provenance={
                    "learner_family": "fixed_delay",
                    "fixed_delay_selected_step": best_delay_steps,
                    "fixed_delay_candidate_steps": list(lag_steps),
                    "fixed_delay_tie_break": (
                        "validation_loss_then_train_loss_then_lower_delay"
                    ),
                },
            )
        )


class UniformKernelLearner:
    """One-input/one-target uniform-kernel learner over an admissible lag window."""

    _FLAT_VARIANCE_THRESHOLD = 1e-8
    _BASELINE_IMPROVEMENT_MARGIN = 0.05

    def __init__(
        self,
        *,
        max_lag: int | str | timedelta,
        min_lag: int | str | timedelta = 0,
        dt: str | timedelta | None = None,
        loss: str = "huber",
        smoothness_penalty: float = 0.0,
        seed: int | None = None,
        validation_fraction: float = 0.2,
        learning_rate: float = 0.05,
        max_epochs: int = 800,
        huber_delta: float = 1.0,
    ) -> None:
        validate_learner_init(
            loss=loss,
            smoothness_penalty=smoothness_penalty,
            validation_fraction=validation_fraction,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            huber_delta=huber_delta,
        )
        self.max_lag = max_lag
        self.min_lag = min_lag
        self.dt = dt
        self.loss = loss
        self.smoothness_penalty = smoothness_penalty
        self.seed = seed
        self.validation_fraction = validation_fraction
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.huber_delta = huber_delta

    def fit(
        self,
        df: pl.DataFrame,
        *,
        input_col: str,
        target_col: str,
        time_col: str,
        order_by_time: bool = False,
    ) -> KernelFitResult:
        config = LearnerConfig(
            max_lag=self.max_lag,
            min_lag=self.min_lag,
            dt=self.dt,
            loss=self.loss,
            validation_fraction=self.validation_fraction,
            huber_delta=self.huber_delta,
        )
        prepared = prepare_fit_data(
            df,
            input_col=input_col,
            target_col=target_col,
            time_col=time_col,
            order_by_time=order_by_time,
            config=config,
        )
        n_lags = prepared.max_lag_steps - prepared.min_lag_steps + 1
        learned_weights = np.full(n_lags, 1.0 / n_lags, dtype=np.float64)
        train_loss, validation_loss = evaluate_weight_vector_losses(
            prepared=prepared,
            weights=learned_weights,
            loss=self.loss,
            huber_delta=self.huber_delta,
        )
        learned_kernel = UniformKernel(
            max_lag_steps=prepared.max_lag_steps,
            min_lag_steps=prepared.min_lag_steps,
            dt=prepared.dt_seconds,
            name=f"{input_col}->{target_col}",
        )
        baseline_losses = evaluate_baselines(
            prepared=prepared,
            loss=self.loss,
            huber_delta=self.huber_delta,
        )
        return assemble_kernel_fit_result(
            assembly=FitAssemblyInput(
                prepared=prepared,
                kernel=learned_kernel,
                train_loss=train_loss,
                validation_loss=validation_loss,
                learned_weights=learned_weights,
                seed=self.seed,
                loss=self.loss,
                huber_delta=self.huber_delta,
                validation_fraction=self.validation_fraction,
                smoothness_penalty=self.smoothness_penalty,
                flat_floor=self._FLAT_VARIANCE_THRESHOLD,
                baseline_improvement_margin=self._BASELINE_IMPROVEMENT_MARGIN,
                baseline_losses=baseline_losses,
                extra_provenance={
                    "learner_family": "uniform",
                    "uniform_window_min_lag_steps": prepared.min_lag_steps,
                    "uniform_window_max_lag_steps": prepared.max_lag_steps,
                },
            )
        )
