"""Fixed/deterministic kernel implementations."""

from __future__ import annotations

from dataclasses import dataclass

from rtdfeatures.kernels.base import Kernel


@dataclass(frozen=True, init=False)
class FixedDelayKernel(Kernel):
    """One-hot kernel concentrated at a fixed delay step."""

    def __init__(
        self,
        *,
        delay_steps: int,
        max_lag_steps: int,
        dt: float,
        min_lag_steps: int = 0,
        name: str | None = None,
    ) -> None:
        if delay_steps < min_lag_steps or delay_steps > max_lag_steps:
            raise ValueError("`delay_steps` must be within [min_lag_steps, max_lag_steps].")
        lag_steps = tuple(range(min_lag_steps, max_lag_steps + 1))
        weights = tuple(1.0 if step == delay_steps else 0.0 for step in lag_steps)
        super().__init__(
            weights=weights,
            lag_steps=lag_steps,
            dt=dt,
            min_lag_steps=min_lag_steps,
            max_lag_steps=max_lag_steps,
            name=name,
        )


@dataclass(frozen=True, init=False)
class UniformKernel(Kernel):
    """Uniform kernel over bounded lag steps."""

    def __init__(
        self,
        *,
        max_lag_steps: int,
        dt: float,
        min_lag_steps: int = 0,
        name: str | None = None,
    ) -> None:
        if max_lag_steps < min_lag_steps:
            raise ValueError("`max_lag_steps` must be >= `min_lag_steps`.")
        lag_steps = tuple(range(min_lag_steps, max_lag_steps + 1))
        uniform_weight = 1.0 / len(lag_steps)
        weights = tuple(uniform_weight for _ in lag_steps)
        super().__init__(
            weights=weights,
            lag_steps=lag_steps,
            dt=dt,
            min_lag_steps=min_lag_steps,
            max_lag_steps=max_lag_steps,
            name=name,
        )
