"""Internal feature accumulation helpers for feature-generation orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rtdfeatures.features.registry import FeatureSpec


@dataclass(frozen=True)
class FeatureBlock:
    arrays: dict[str, np.ndarray]
    specs: tuple[FeatureSpec, ...]
    zero_denominator_rows: dict[str, int]


@dataclass
class FeatureAccumulator:
    n_rows: int
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    specs: list[FeatureSpec] = field(default_factory=list)
    missing_rows_by_feature: dict[str, int] = field(default_factory=dict)
    missing_fraction_by_feature: dict[str, float] = field(default_factory=dict)
    zero_denominator_rows_by_feature: dict[str, int] = field(default_factory=dict)
    kernel_feature_names: dict[str, list[str]] = field(default_factory=dict)

    def ensure_kernel(self, kernel_name: str) -> None:
        if not kernel_name:
            raise ValueError("kernel names must be non-empty.")
        self.kernel_feature_names.setdefault(kernel_name, [])

    def add(
        self,
        *,
        name: str,
        values: np.ndarray,
        spec: FeatureSpec,
        zero_denominator_rows: int = 0,
    ) -> None:
        if spec.name != name:
            raise ValueError(
                "Feature name mismatch between add() argument and FeatureSpec: "
                f"name={name!r}, spec.name={spec.name!r}."
            )
        if name in self.arrays:
            raise ValueError(
                f"Generated feature name collision: '{name}'. "
                "Check for duplicate kernel names or conflicting column names."
            )
        if len(values) != self.n_rows:
            raise ValueError(
                f"Feature '{name}' length mismatch: got {len(values)}, expected {self.n_rows}."
            )

        self.ensure_kernel(spec.kernel_name)
        self.arrays[name] = values
        self.specs.append(spec)
        self.missing_rows_by_feature[name] = int(np.sum(~np.isfinite(values)))
        self.zero_denominator_rows_by_feature[name] = int(zero_denominator_rows)
        self.kernel_feature_names[spec.kernel_name].append(name)

    def finalize_missing_fractions(self) -> None:
        for feature_name, missing_rows in self.missing_rows_by_feature.items():
            if self.n_rows == 0:
                self.missing_fraction_by_feature[feature_name] = 0.0
            else:
                self.missing_fraction_by_feature[feature_name] = missing_rows / self.n_rows
