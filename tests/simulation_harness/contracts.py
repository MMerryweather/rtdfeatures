"""Typed output contracts for simulation harness generators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

import polars as pl


class KernelMetadata(TypedDict):
    lag_steps: list[int]
    weights: list[float]
    dt: float
    min_lag: int
    max_lag: int
    mean_lag: float
    p50_lag: float
    p90_lag: float


class ScenarioMetadata(TypedDict, total=False):
    name: str
    seed: int
    n_rows: int
    dt: float
    params: dict[str, Any]


@dataclass(frozen=True)
class GeneratorOutput:
    data: pl.DataFrame
    true_kernels: dict[str, KernelMetadata]
    genealogy: pl.DataFrame
    scenario: ScenarioMetadata
