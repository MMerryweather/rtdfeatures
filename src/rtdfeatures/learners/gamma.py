"""Gamma parametric kernel learner."""

from __future__ import annotations

import math
from datetime import timedelta
from typing import Any

import numpy as np
import polars as pl
import torch

from rtdfeatures.diagnostics import (
    KernelFitResult,
)
from rtdfeatures.kernels.parametric import (
    _make_parametric_learned_kernel,
    build_parametric_fit_provenance,
)
from rtdfeatures.learners._assembly import (
    FitAssemblyInput,
    assemble_kernel_fit_result,
    evaluate_baselines,
)
from rtdfeatures.learners._base import (
    LearnerConfig,
    _inverse_softplus,
    prepare_fit_data,
    validate_learner_init,
)
from rtdfeatures.learners._optimization import (
    make_torch_fit_data,
    optimize_parametric_weights,
    set_torch_seed,
)


class GammaKernelLearner:
    """One-input/one-target constrained gamma parametric kernel learner."""

    _FLAT_VARIANCE_THRESHOLD = 1e-8
    _BASELINE_IMPROVEMENT_MARGIN = 0.05
    _MIN_SHAPE_ALPHA = 1e-8
    _MIN_SHAPE_ALPHA_WITH_ZERO_LAG = 1.0 + 1e-8
    _MIN_RATE_BETA = 1e-8
    _LOG_EPS = 1e-12

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
        init_shape_alpha: float = 2.0,
        init_rate_beta: float | None = None,
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
        if not math.isfinite(init_shape_alpha) or init_shape_alpha <= 0.0:
            raise ValueError("init_shape_alpha must be finite and strictly positive.")
        if init_rate_beta is not None and (
            not math.isfinite(init_rate_beta) or init_rate_beta <= 0.0
        ):
            raise ValueError("init_rate_beta must be finite and strictly positive.")
        self.init_shape_alpha = float(init_shape_alpha)
        self.init_rate_beta = init_rate_beta

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

        if prepared.min_lag_steps == 0 and prepared.max_lag_steps == 0:
            raise ValueError(
                "GammaKernelLearner requires at least one strictly positive lag step; "
                "min_lag=0 and max_lag=0 is not supported."
            )

        init_shape_alpha = self._resolve_initial_shape_alpha(
            min_lag_steps=prepared.min_lag_steps
        )
        if prepared.min_lag_steps == 0 and init_shape_alpha <= 1.0:
            raise ValueError(
                "init_shape_alpha must be strictly greater than 1.0 when "
                "lag grid includes zero lag."
            )
        init_rate_beta = self._resolve_initial_rate_beta(
            min_lag_steps=prepared.min_lag_steps,
            max_lag_steps=prepared.max_lag_steps,
            dt_seconds=prepared.dt_seconds,
        )

        set_torch_seed(self.seed)
        torch_fit_data = make_torch_fit_data(prepared)
        train_x = torch_fit_data.train_x
        train_y = torch_fit_data.train_y
        valid_x = torch_fit_data.valid_x
        valid_y = torch_fit_data.valid_y
        lag_steps = tuple(range(prepared.min_lag_steps, prepared.max_lag_steps + 1))
        lag_times_np = np.asarray(lag_steps, dtype=np.float32) * float(prepared.dt_seconds)
        lag_times = torch.as_tensor(lag_times_np, dtype=torch.float32)
        lag_times_safe = torch.clamp(lag_times, min=self._LOG_EPS)

        min_shape_alpha = (
            self._MIN_SHAPE_ALPHA_WITH_ZERO_LAG
            if prepared.min_lag_steps == 0
            else self._MIN_SHAPE_ALPHA
        )
        init_shape_raw = _inverse_softplus(
            max(init_shape_alpha - min_shape_alpha, self._MIN_SHAPE_ALPHA)
        )
        init_rate_raw = _inverse_softplus(
            max(init_rate_beta - self._MIN_RATE_BETA, self._MIN_RATE_BETA)
        )
        raw_shape = torch.nn.Parameter(torch.tensor(float(init_shape_raw), dtype=torch.float32))
        raw_rate = torch.nn.Parameter(torch.tensor(float(init_rate_raw), dtype=torch.float32))
        optimizer = torch.optim.Adam([raw_shape, raw_rate], lr=self.learning_rate)

        def _forward_gamma() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            shape_alpha = torch.nn.functional.softplus(raw_shape) + min_shape_alpha
            rate_beta = torch.nn.functional.softplus(raw_rate) + self._MIN_RATE_BETA
            log_pdf = (
                shape_alpha * torch.log(rate_beta)
                + (shape_alpha - 1.0) * torch.log(lag_times_safe)
                - rate_beta * lag_times
                - torch.lgamma(shape_alpha)
            )
            log_pdf = torch.where(lag_times > 0.0, log_pdf, torch.full_like(log_pdf, -1.0e9))
            weights = torch.softmax(log_pdf, dim=0)
            return weights, {"shape_alpha": shape_alpha, "rate_beta": rate_beta}

        best_loss, best_parameters, _best_weights = optimize_parametric_weights(
            optimizer=optimizer,
            max_epochs=self.max_epochs,
            train_x=train_x,
            train_y=train_y,
            valid_x=valid_x,
            valid_y=valid_y,
            loss=self.loss,
            huber_delta=self.huber_delta,
            smoothness_penalty=self.smoothness_penalty,
            forward=_forward_gamma,
            failure_message="Optimization failed to produce a valid gamma parameter fit.",
        )
        best_shape_alpha = best_parameters["shape_alpha"]
        best_rate_beta = best_parameters["rate_beta"]

        learned_kernel = _make_parametric_learned_kernel(
            family="gamma",
            dt=prepared.dt_seconds,
            min_lag_steps=prepared.min_lag_steps,
            max_lag_steps=prepared.max_lag_steps,
            parameters={"shape_alpha": best_shape_alpha, "rate_beta": best_rate_beta},
            name=f"{input_col}->{target_col}",
        )
        learned_weights = np.asarray(learned_kernel.weights, dtype=np.float64)
        extra_provenance: dict[str, Any] = {
            "parametric_lag_time_grid_seconds": tuple(
                float(step) * prepared.dt_seconds for step in lag_steps
            ),
        }
        extra_provenance.update(
            build_parametric_fit_provenance(
                family="gamma",
                parameters={"shape_alpha": best_shape_alpha, "rate_beta": best_rate_beta},
                initial_parameters={
                    "shape_alpha": init_shape_alpha,
                    "rate_beta": init_rate_beta,
                },
                conversion_status="ok",
                conversion_message=None,
            )
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
                train_loss=best_loss.train_loss,
                validation_loss=best_loss.validation_loss,
                learned_weights=learned_weights,
                seed=self.seed,
                loss=self.loss,
                huber_delta=self.huber_delta,
                validation_fraction=self.validation_fraction,
                smoothness_penalty=self.smoothness_penalty,
                flat_floor=self._FLAT_VARIANCE_THRESHOLD,
                baseline_improvement_margin=self._BASELINE_IMPROVEMENT_MARGIN,
                baseline_losses=baseline_losses,
                extra_provenance=extra_provenance,
            )
        )

    def _resolve_initial_shape_alpha(self, *, min_lag_steps: int) -> float:
        if self.init_shape_alpha is not None:
            return float(self.init_shape_alpha)
        return 2.0

    def _resolve_initial_rate_beta(
        self,
        *,
        min_lag_steps: int,
        max_lag_steps: int,
        dt_seconds: float,
    ) -> float:
        if self.init_rate_beta is not None:
            return float(self.init_rate_beta)
        midpoint_lag_seconds = 0.5 * (min_lag_steps + max_lag_steps) * dt_seconds
        if midpoint_lag_seconds <= 0.0:
            raise ValueError(
                "Cannot derive default init_rate_beta from lag window midpoint <= 0. "
                "Provide init_rate_beta explicitly."
            )
        return float(1.0 / midpoint_lag_seconds)
