"""Shared multi-pair kernel learner."""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from rtdfeatures.diagnostics import SharedKernelFitResult, SharedPairFitResult
from rtdfeatures.learners.simplex import SimplexKernelLearner
from rtdfeatures.utils import resolve_and_validate_dt, validate_or_sort_time


class SharedSimplexKernelLearner:
    """Shared multi-pair learner coordinating independent simplex fits."""

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
        input_cols: list[str] | tuple[str, ...],
        target_cols: list[str] | tuple[str, ...],
        time_col: str,
        pair_names: list[str] | tuple[str, ...] | None = None,
        order_by_time: bool = False,
    ) -> SharedKernelFitResult:
        if len(input_cols) != len(target_cols):
            raise ValueError("input_cols and target_cols must have the same length.")
        if len(input_cols) == 0:
            raise ValueError("At least one input/target pair is required.")
        if pair_names is not None and len(pair_names) != len(input_cols):
            raise ValueError("pair_names must have the same length as input_cols/target_cols.")

        ordered = validate_or_sort_time(df, time_col=time_col, order_by_time=order_by_time)
        resolved_dt = resolve_and_validate_dt(ordered, time_col=time_col, dt=self.dt)

        resolved_pair_ids: list[str] = []
        seen_pair_ids: set[str] = set()
        for idx, (input_col, target_col) in enumerate(zip(input_cols, target_cols)):
            pair_name = pair_names[idx] if pair_names is not None else None
            pair_id = SharedKernelFitResult.make_pair_id(
                input_col, target_col, pair_name=pair_name
            )
            if pair_id in seen_pair_ids:
                raise ValueError(
                    f"Duplicate pair_id detected: '{pair_id}'. Pair ids must be unique."
                )
            seen_pair_ids.add(pair_id)
            resolved_pair_ids.append(pair_id)

        pairs: list[SharedPairFitResult] = []
        for pair_idx, (input_col, target_col) in enumerate(zip(input_cols, target_cols)):
            pair_id = resolved_pair_ids[pair_idx]
            pair_seed = None if self.seed is None else int(self.seed + pair_idx)
            learner = SimplexKernelLearner(
                max_lag=self.max_lag,
                min_lag=self.min_lag,
                dt=resolved_dt,
                loss=self.loss,
                smoothness_penalty=self.smoothness_penalty,
                seed=pair_seed,
                validation_fraction=self.validation_fraction,
                learning_rate=self.learning_rate,
                max_epochs=self.max_epochs,
                huber_delta=self.huber_delta,
            )
            try:
                fit_result = learner.fit(
                    ordered,
                    input_col=input_col,
                    target_col=target_col,
                    time_col=time_col,
                    order_by_time=False,
                )
                pairs.append(
                    SharedPairFitResult(
                        pair_id=pair_id,
                        input_col=input_col,
                        target_col=target_col,
                        fit_result=fit_result,
                    )
                )
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                # Deliberate resilient-batch behavior: keep fitting independent pairs.
                pairs.append(
                    SharedPairFitResult(
                        pair_id=pair_id,
                        input_col=input_col,
                        target_col=target_col,
                        fit_result=None,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )

        return SharedKernelFitResult(pairs=tuple(pairs))
