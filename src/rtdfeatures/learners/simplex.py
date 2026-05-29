"""Simplex empirical kernel learner."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import polars as pl
import torch

from rtdfeatures.diagnostics import KernelFitResult
from rtdfeatures.kernels import LearnedKernel
from rtdfeatures.learners._assembly import (
    FitAssemblyInput,
    assemble_kernel_fit_result,
    evaluate_baselines,
)
from rtdfeatures.learners._base import (
    LearnerConfig,
    _robust_scaling_stats,
    _ScalingStats,
    prepare_fit_data,
    validate_learner_init,
)
from rtdfeatures.learners._base import (
    _build_lagged_windows as _build_lagged_windows_base,
)
from rtdfeatures.learners._optimization import (
    BestLossState,
    make_torch_fit_data,
    set_torch_seed,
    smoothness_term,
    torch_loss,
    torch_loss_value,
    update_best_loss_state,
)


class SimplexKernelLearner:
    """One-input/one-target constrained empirical simplex kernel learner."""

    _FLAT_VARIANCE_THRESHOLD = 1e-8
    _BASELINE_IMPROVEMENT_MARGIN = 0.05
    _TAIL_MASS_FRACTION_OF_LAG_WINDOW = 0.75

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

        set_torch_seed(self.seed)
        torch_fit_data = make_torch_fit_data(prepared)
        train_x = torch_fit_data.train_x
        train_y = torch_fit_data.train_y
        valid_x = torch_fit_data.valid_x
        valid_y = torch_fit_data.valid_y

        n_lags = prepared.max_lag_steps - prepared.min_lag_steps + 1
        theta = torch.nn.Parameter(torch.zeros(n_lags, dtype=torch.float32))
        optimizer = torch.optim.Adam([theta], lr=self.learning_rate)

        best_loss = BestLossState()
        best_weights = None
        for _ in range(self.max_epochs):
            optimizer.zero_grad(set_to_none=True)
            weights = torch.softmax(theta, dim=0)
            train_pred = train_x @ weights
            data_loss = torch_loss(
                train_pred,
                train_y,
                loss=self.loss,
                huber_delta=self.huber_delta,
            )
            total_loss = data_loss + self.smoothness_penalty * smoothness_term(weights)
            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                valid_pred = valid_x @ weights
                valid_loss = torch_loss_value(
                    valid_pred,
                    valid_y,
                    loss=self.loss,
                    huber_delta=self.huber_delta,
                )
                if update_best_loss_state(
                    state=best_loss,
                    validation_loss=valid_loss,
                    train_loss=float(data_loss.item()),
                ):
                    best_weights = weights.detach().cpu().numpy().copy()

        if best_weights is None:
            raise RuntimeError("Optimization failed to produce a valid simplex weight vector.")

        lag_steps = tuple(range(prepared.min_lag_steps, prepared.max_lag_steps + 1))
        learned_kernel = LearnedKernel(
            weights=tuple(float(w) for w in best_weights),
            lag_steps=lag_steps,
            dt=prepared.dt_seconds,
            min_lag_steps=prepared.min_lag_steps,
            max_lag_steps=prepared.max_lag_steps,
            name=f"{input_col}->{target_col}",
        )
        learned_kernel.validate()

        baseline_losses = evaluate_baselines(
            prepared=prepared,
            loss=self.loss,
            huber_delta=self.huber_delta,
        )
        return assemble_kernel_fit_result(
            assembly=FitAssemblyInput(
                prepared=prepared,
                kernel=learned_kernel,
                train_loss=best_loss.train_loss,
                validation_loss=best_loss.validation_loss,
                learned_weights=np.asarray(best_weights, dtype=np.float64),
                seed=self.seed,
                loss=self.loss,
                huber_delta=self.huber_delta,
                validation_fraction=self.validation_fraction,
                smoothness_penalty=self.smoothness_penalty,
                flat_floor=self._FLAT_VARIANCE_THRESHOLD,
                baseline_improvement_margin=self._BASELINE_IMPROVEMENT_MARGIN,
                baseline_losses=baseline_losses,
            )
        )

    @staticmethod
    def _build_lagged_windows(
        *,
        input_values: np.ndarray,
        target_values: np.ndarray,
        min_lag_steps: int,
        max_lag_steps: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _build_lagged_windows_base(
            input_values=input_values,
            target_values=target_values,
            min_lag_steps=min_lag_steps,
            max_lag_steps=max_lag_steps,
        )

    @staticmethod
    def _robust_scaling_stats(values: np.ndarray) -> _ScalingStats:
        return _robust_scaling_stats(values)
