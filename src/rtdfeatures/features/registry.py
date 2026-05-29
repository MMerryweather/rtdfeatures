"""Feature specification and registry types.

FeatureSpec holds structured metadata for one generated feature,
FeatureRegistry holds a collection, and TransformResult bundles
the output table with diagnostics and registry for the new API.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import polars as pl

from rtdfeatures.diagnostics import TransformReport


@dataclass(frozen=True)
class FeatureSpec:
    """Structured metadata for one generated feature column.

    ``lag_steps`` is always a copy of the kernel's step schedule and
    is independent of any particular transform invocation.

    ``kernel_summary`` is the kernel's own :meth:`Kernel.summary()` dict
    and may contain parametric-family keys, weight arrays as lists, etc.
    """

    name: str
    kernel_name: str
    source_col: str
    family: str
    metric: str
    category_level: str | None
    lag_steps: tuple[int, ...]
    kernel_summary: dict[str, Any]


@dataclass(frozen=True)
class FeatureRegistry:
    """Ordered, immutable collection of :class:`FeatureSpec` objects.

    Constructed once per transform invocation.  Supports iteration,
    ``len()``, and simple attribute-based filtering.
    """

    specs: tuple[FeatureSpec, ...]

    def __len__(self) -> int:
        return len(self.specs)

    def __iter__(self) -> Iterator[FeatureSpec]:
        return iter(self.specs)

    def filter(
        self,
        *,
        kernel_name: str | None = None,
        source_col: str | None = None,
        family: str | None = None,
        metric: str | None = None,
    ) -> FeatureRegistry:
        """Return a new ``FeatureRegistry`` containing only specs that match
        all supplied criteria.  ``None`` means *any*."""
        filtered = [
            s
            for s in self.specs
            if (kernel_name is None or s.kernel_name == kernel_name)
            and (source_col is None or s.source_col == source_col)
            and (family is None or s.family == family)
            and (metric is None or s.metric == metric)
        ]
        return FeatureRegistry(specs=tuple(filtered))

    def names(self) -> tuple[str, ...]:
        """Return spec names in registry order."""
        return tuple(spec.name for spec in self.specs)

    def to_frame(self) -> pl.DataFrame:
        """Materialize specs as a Polars DataFrame in registry order."""
        schema: list[tuple[str, Any]] = [
            ("name", pl.Utf8),
            ("kernel_name", pl.Utf8),
            ("source_col", pl.Utf8),
            ("family", pl.Utf8),
            ("metric", pl.Utf8),
            ("category_level", pl.Utf8),
            ("lag_steps", pl.List(pl.Int64)),
            ("kernel_summary", pl.Object),
        ]
        columns = [name for name, _dtype in schema]
        rows = [
            {
                "name": spec.name,
                "kernel_name": spec.kernel_name,
                "source_col": spec.source_col,
                "family": spec.family,
                "metric": spec.metric,
                "category_level": spec.category_level,
                "lag_steps": spec.lag_steps,
                "kernel_summary": spec.kernel_summary,
            }
            for spec in self.specs
        ]
        return pl.DataFrame(rows, schema=schema).select(columns)


@dataclass(frozen=True)
class TransformResult:
    """Complete output of one ``KernelFeatureBuilder.transform`` invocation.

    Usage::

        result = builder.transform_result(df)
        result.features        # -> pl.DataFrame (time_col + feature cols)
        result.report          # -> TransformReport
        result.feature_registry  # -> FeatureRegistry
    """

    features: pl.DataFrame
    report: TransformReport
    feature_registry: FeatureRegistry
