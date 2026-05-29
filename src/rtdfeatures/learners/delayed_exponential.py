"""Delayed-exponential parametric kernel learner."""

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


class DelayedExponentialKernelLearner:
    """One-input/one-target constrained delayed-exponential kernel learner."""

    _FLAT_VARIANCE_THRESHOLD = 1e-8
    _BASELINE_IMPROVEMENT_MARGIN = 0.05
    _MIN_RATE_LAMBDA = 1e-8
    _MIN_DELAY = 0.0
    _DELAY_INIT_EPS = 1e-4

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
        init_delay: float | None = None,
        init_rate_lambda: float | None = None,
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
        if init_delay is not None and (not math.isfinite(init_delay) or init_delay < 0.0):
            raise ValueError("init_delay must be finite and non-negative.")
        if init_rate_lambda is not None and (
            not math.isfinite(init_rate_lambda) or init_rate_lambda <= 0.0
        ):
            raise ValueError("init_rate_lambda must be finite and strictly positive.")
        self.init_delay = init_delay
        self.init_rate_lambda = init_rate_lambda

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
        init_delay = self._resolve_initial_delay(
            min_lag_steps=prepared.min_lag_steps,
            max_lag_steps=prepared.max_lag_steps,
            dt_seconds=prepared.dt_seconds,
        )
        init_rate_lambda = self._resolve_initial_rate_lambda(
            min_lag_steps=prepared.min_lag_steps,
            max_lag_steps=prepared.max_lag_steps,
            dt_seconds=prepared.dt_seconds,
        )

        set_torch_seed(self.seed)
        torch_fit_data = make_torch_fit_data(prepared)
        lag_steps = tuple(range(prepared.min_lag_steps, prepared.max_lag_steps + 1))
        lag_times_np = np.asarray(lag_steps, dtype=np.float32) * float(prepared.dt_seconds)
        lag_times = torch.as_tensor(lag_times_np, dtype=torch.float32)
        min_lag_time = float(lag_times_np[0])
        max_lag_time = float(lag_times_np[-1])
        lag_window_width = max_lag_time - min_lag_time

        raw_delay = None
        if lag_window_width > 0.0:
            init_delay_fraction = self._to_delay_fraction(
                init_delay=init_delay,
                min_lag_time=min_lag_time,
                max_lag_time=max_lag_time,
            )
            init_delay_raw = math.log(init_delay_fraction / (1.0 - init_delay_fraction))
            raw_delay = torch.nn.Parameter(torch.tensor(float(init_delay_raw), dtype=torch.float32))
        init_rate_raw = _inverse_softplus(
            max(init_rate_lambda - self._MIN_RATE_LAMBDA, self._MIN_RATE_LAMBDA)
        )
        raw_rate = torch.nn.Parameter(torch.tensor(float(init_rate_raw), dtype=torch.float32))
        parameter_list = [raw_rate] if raw_delay is None else [raw_delay, raw_rate]
        optimizer = torch.optim.Adam(parameter_list, lr=self.learning_rate)

        def _forward_delayed_exponential() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            if raw_delay is None:
                delay = torch.tensor(min_lag_time, dtype=torch.float32)
            else:
                delay = min_lag_time + lag_window_width * torch.sigmoid(raw_delay)
            rate_lambda = torch.nn.functional.softplus(raw_rate) + self._MIN_RATE_LAMBDA
            relu_term = torch.nn.functional.relu(lag_times - delay)
            logits = -rate_lambda * relu_term
            logits = torch.where(lag_times >= delay, logits, torch.full_like(logits, -1.0e9))
            weights = torch.softmax(logits, dim=0)
            return weights, {"delay": delay, "rate_lambda": rate_lambda}

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
            forward=_forward_delayed_exponential,
            failure_message=(
                "Optimization failed to produce a valid delayed-exponential parameter fit."
            ),
        )
        best_delay = best_parameters["delay"]
        best_rate_lambda = best_parameters["rate_lambda"]

        learned_kernel = _make_parametric_learned_kernel(
            family="delayed_exponential",
            dt=prepared.dt_seconds,
            min_lag_steps=prepared.min_lag_steps,
            max_lag_steps=prepared.max_lag_steps,
            parameters={"delay": best_delay, "rate_lambda": best_rate_lambda},
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
                family="delayed_exponential",
                parameters={"delay": best_delay, "rate_lambda": best_rate_lambda},
                initial_parameters={"delay": init_delay, "rate_lambda": init_rate_lambda},
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

    def _resolve_initial_delay(
        self,
        *,
        min_lag_steps: int,
        max_lag_steps: int,
        dt_seconds: float,
    ) -> float:
        if self.init_delay is not None:
            return float(self.init_delay)
        return float(0.5 * (min_lag_steps + max_lag_steps) * dt_seconds)

    def _to_delay_fraction(
        self,
        *,
        init_delay: float,
        min_lag_time: float,
        max_lag_time: float,
    ) -> float:
        lag_window_width = max_lag_time - min_lag_time
        if lag_window_width <= 0.0:
            return 0.5
        fraction = (init_delay - min_lag_time) / lag_window_width
        return float(min(max(fraction, self._DELAY_INIT_EPS), 1.0 - self._DELAY_INIT_EPS))

    def _resolve_initial_rate_lambda(
        self,
        *,
        min_lag_steps: int,
        max_lag_steps: int,
        dt_seconds: float,
    ) -> float:
        if self.init_rate_lambda is not None:
            return float(self.init_rate_lambda)
        midpoint_lag_seconds = 0.5 * (min_lag_steps + max_lag_steps) * dt_seconds
        if midpoint_lag_seconds <= 0.0:
            raise ValueError(
                "Cannot derive default init_rate_lambda from lag window midpoint <= 0. "
                "Provide init_rate_lambda explicitly."
            )
        return float(1.0 / midpoint_lag_seconds)
