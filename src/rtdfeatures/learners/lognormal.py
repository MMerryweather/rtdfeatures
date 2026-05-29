"""Log-normal parametric kernel learner."""

from __future__ import annotations

import math
from datetime import timedelta
from typing import Any

import numpy as np
import polars as pl
import torch

from rtdfeatures.diagnostics import KernelFitResult
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


class LogNormalKernelLearner:
    """One-input/one-target constrained log-normal parametric kernel learner."""

    _FLAT_VARIANCE_THRESHOLD = 1e-8
    _BASELINE_IMPROVEMENT_MARGIN = 0.05
    _MIN_LOG_SIGMA = 1e-8
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
        init_log_mu: float | None = None,
        init_log_sigma: float = 0.5,
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
        if init_log_mu is not None and not math.isfinite(init_log_mu):
            raise ValueError("init_log_mu must be finite when provided.")
        if not math.isfinite(init_log_sigma) or init_log_sigma <= 0.0:
            raise ValueError("init_log_sigma must be finite and strictly positive.")
        self.init_log_mu = init_log_mu
        self.init_log_sigma = float(init_log_sigma)

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
                "LogNormalKernelLearner requires at least one strictly positive lag step; "
                "min_lag=0 and max_lag=0 is not supported."
            )

        init_log_mu = self._resolve_initial_log_mu(
            min_lag_steps=prepared.min_lag_steps,
            max_lag_steps=prepared.max_lag_steps,
            dt_seconds=prepared.dt_seconds,
        )

        set_torch_seed(self.seed)
        torch_fit_data = make_torch_fit_data(prepared)
        lag_steps = tuple(range(prepared.min_lag_steps, prepared.max_lag_steps + 1))
        lag_times_np = np.asarray(lag_steps, dtype=np.float32) * float(prepared.dt_seconds)
        lag_times = torch.as_tensor(lag_times_np, dtype=torch.float32)
        lag_times_safe = torch.clamp(lag_times, min=self._LOG_EPS)

        raw_log_mu = torch.nn.Parameter(torch.tensor(float(init_log_mu), dtype=torch.float32))
        init_log_sigma_raw = _inverse_softplus(
            max(self.init_log_sigma - self._MIN_LOG_SIGMA, self._MIN_LOG_SIGMA)
        )
        raw_log_sigma = torch.nn.Parameter(
            torch.tensor(float(init_log_sigma_raw), dtype=torch.float32)
        )
        optimizer = torch.optim.Adam([raw_log_mu, raw_log_sigma], lr=self.learning_rate)

        def _forward_lognormal() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            log_mu = raw_log_mu
            log_sigma = torch.nn.functional.softplus(raw_log_sigma) + self._MIN_LOG_SIGMA
            centered = (torch.log(lag_times_safe) - log_mu) / log_sigma
            log_pdf = (
                -0.5 * (centered**2)
                - torch.log(lag_times_safe)
                - torch.log(log_sigma)
                - 0.5 * math.log(2.0 * math.pi)
            )
            log_pdf = torch.where(lag_times > 0.0, log_pdf, torch.full_like(log_pdf, -1.0e9))
            weights = torch.softmax(log_pdf, dim=0)
            return weights, {"log_mu": log_mu, "log_sigma": log_sigma}

        best_loss, best_parameters, _best_weights = optimize_parametric_weights(
            optimizer=optimizer,
            max_epochs=self.max_epochs,
            train_x=torch_fit_data.train_x,
            train_y=torch_fit_data.train_y,
            valid_x=torch_fit_data.valid_x,
            valid_y=torch_fit_data.valid_y,
            loss=self.loss,
            huber_delta=self.huber_delta,
            smoothness_penalty=self.smoothness_penalty,
            forward=_forward_lognormal,
            failure_message="Optimization failed to produce a valid log-normal parameter fit.",
        )
        best_log_mu = best_parameters["log_mu"]
        best_log_sigma = best_parameters["log_sigma"]

        learned_kernel = _make_parametric_learned_kernel(
            family="lognormal",
            dt=prepared.dt_seconds,
            min_lag_steps=prepared.min_lag_steps,
            max_lag_steps=prepared.max_lag_steps,
            parameters={"log_mu": best_log_mu, "log_sigma": best_log_sigma},
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
                family="lognormal",
                parameters={"log_mu": best_log_mu, "log_sigma": best_log_sigma},
                initial_parameters={"log_mu": init_log_mu, "log_sigma": self.init_log_sigma},
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

    def _resolve_initial_log_mu(
        self,
        *,
        min_lag_steps: int,
        max_lag_steps: int,
        dt_seconds: float,
    ) -> float:
        if self.init_log_mu is not None:
            return float(self.init_log_mu)
        midpoint_lag_seconds = 0.5 * (min_lag_steps + max_lag_steps) * dt_seconds
        if midpoint_lag_seconds <= 0.0:
            raise ValueError(
                "Cannot derive default init_log_mu from lag window midpoint <= 0. "
                "Provide init_log_mu explicitly."
            )
        return float(math.log(midpoint_lag_seconds))
