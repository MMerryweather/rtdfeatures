"""Age feature computation helpers."""

from __future__ import annotations

from rtdfeatures.kernels import Kernel


def resolve_age_tail_threshold(
    *,
    min_lag_steps: int,
    max_lag_steps: int,
    dt: float,
    configured_threshold: float | None,
) -> float:
    if configured_threshold is not None:
        return float(configured_threshold)
    return float(
        min_lag_steps + 0.75 * (max_lag_steps - min_lag_steps)
    ) * float(dt)


def age_feature_values(*, kernel: Kernel, threshold: float) -> dict[str, float]:
    return {
        "mean": kernel.mean_lag(),
        "p50": kernel.percentile(0.5),
        "p90": kernel.percentile(0.9),
        "tail_gt_threshold": kernel.tail_mass(threshold),
    }
