"""Base kernel data structure."""

from __future__ import annotations

from dataclasses import dataclass

KERNEL_WEIGHT_SUM_TOLERANCE = 1e-6


@dataclass(frozen=True)
class Kernel:
    """Constrained causal kernel over bounded lag steps."""

    weights: tuple[float, ...]
    lag_steps: tuple[int, ...]
    dt: float
    min_lag_steps: int
    max_lag_steps: int
    name: str | None = None

    def validate(self) -> None:
        """Validate kernel constraints."""
        if self.dt <= 0:
            raise ValueError("Kernel `dt` must be strictly positive.")
        if len(self.weights) == 0:
            raise ValueError("Kernel must contain at least one weight.")
        if len(self.weights) != len(self.lag_steps):
            raise ValueError("Kernel `weights` and `lag_steps` must have equal length.")
        if self.min_lag_steps < 0:
            raise ValueError("Kernel `min_lag_steps` must be non-negative.")
        if self.max_lag_steps < self.min_lag_steps:
            raise ValueError("Kernel `max_lag_steps` must be >= `min_lag_steps`.")
        if any(step < self.min_lag_steps or step > self.max_lag_steps for step in self.lag_steps):
            raise ValueError("Kernel lag steps must stay within [min_lag_steps, max_lag_steps].")
        if list(self.lag_steps) != sorted(self.lag_steps):
            raise ValueError("Kernel lag steps must be sorted in ascending order.")
        if len(set(self.lag_steps)) != len(self.lag_steps):
            raise ValueError("Kernel lag steps must not contain duplicates.")
        negative_indices = [i for i, w in enumerate(self.weights) if w < 0.0]
        if negative_indices:
            raise ValueError(
                f"Kernel weights must be non-negative; found negative weight(s) at index(es) "
                f"{negative_indices} with values {[self.weights[i] for i in negative_indices]}."
            )
        weight_sum = sum(self.weights)
        if abs(weight_sum - 1.0) > KERNEL_WEIGHT_SUM_TOLERANCE:
            raise ValueError(
                f"Kernel weight sum must be approximately 1.0; got {weight_sum:.10f}. "
                f"Weights must sum to 1 (within {KERNEL_WEIGHT_SUM_TOLERANCE:.0e})."
            )

    def summary(self) -> dict[str, float | int | str | None | dict[str, float | int]]:
        """Return a compact kernel summary."""
        self.validate()
        return {
            "name": self.name,
            "n_lags": len(self.lag_steps),
            "dt": self.dt,
            "min_lag_steps": self.min_lag_steps,
            "max_lag_steps": self.max_lag_steps,
            "mean_lag": self.mean_lag(),
            "p50_lag": self.percentile(0.5),
            "p90_lag": self.percentile(0.9),
            "tail_mass_at_75pct_window": self.tail_mass(
                (self.min_lag_steps + 0.75 * (self.max_lag_steps - self.min_lag_steps)) * self.dt
            ),
        }

    def mean_lag(self) -> float:
        """Return weighted mean lag in time units."""
        self.validate()
        return sum(weight * (step * self.dt) for step, weight in zip(self.lag_steps, self.weights))

    def percentile(self, q: float) -> float:
        """Return weighted lag percentile in time units for q in [0, 1]."""
        self.validate()
        if q < 0.0 or q > 1.0:
            raise ValueError("Kernel percentile `q` must be in [0, 1].")
        cumulative = 0.0
        for step, weight in zip(self.lag_steps, self.weights):
            cumulative += weight
            if cumulative >= q:
                return step * self.dt
        return self.lag_steps[-1] * self.dt

    def tail_mass(self, threshold: float) -> float:
        """Return cumulative mass for lags at or above `threshold` in time units."""
        self.validate()
        return sum(
            weight
            for step, weight in zip(self.lag_steps, self.weights)
            if (step * self.dt) >= threshold
        )


@dataclass(frozen=True)
class LearnedKernel(Kernel):
    """Learned constrained kernel for one input/target pair."""
