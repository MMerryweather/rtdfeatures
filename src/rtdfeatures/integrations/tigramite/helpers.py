"""Graph-to-lag helper utilities for optional Tigramite integration.

Lag sign convention
-------------------
For graph payloads shaped ``graph[target][source][tau]``, a directed mark at
``tau > 0`` is interpreted as:

``source(t - tau) -> target(t)``

and the emitted descriptor stores ``lag_steps=tau`` as a positive integer.
Contemporaneous links (``tau == 0``) are ignored and warned.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from statistics import median
from typing import Any, cast

import polars as pl

from rtdfeatures.integrations.tigramite.payloads import TigramitePayloadError
from rtdfeatures.kernels import Kernel

_SUPPORTED_DIRECTED_MARKS: tuple[str, ...] = ("-->", "->")
_DEFAULT_CANDIDATE_FAMILIES: tuple[str, ...] = (
    "simplex",
    "fixed_delay",
    "gamma",
    "exponential",
    "delayed_exponential",
    "lognormal",
    "erlang",
)
_DEFAULT_OUTSIDE_SUPPORT_WARNING_THRESHOLD = 0.20
TIGRAMITE_NO_LINKS_FOUND = "TIGRAMITE_NO_LINKS_FOUND"
TIGRAMITE_LAG_RANGE_EMPTY = "TIGRAMITE_LAG_RANGE_EMPTY"
TIGRAMITE_CONTEMPORANEOUS_LINK_IGNORED = "TIGRAMITE_CONTEMPORANEOUS_LINK_IGNORED"
TIGRAMITE_GRAPH_MARK_UNSUPPORTED = "TIGRAMITE_GRAPH_MARK_UNSUPPORTED"
TIGRAMITE_VALUES_NOT_KERNEL_WEIGHTS = "TIGRAMITE_VALUES_NOT_KERNEL_WEIGHTS"
TIGRAMITE_KERNEL_SUPPORT_THRESHOLD_BREACH = "TIGRAMITE_KERNEL_SUPPORT_THRESHOLD_BREACH"
TIGRAMITE_PAYLOAD_SHAPE_INVALID = "TIGRAMITE_PAYLOAD_SHAPE_INVALID"
TIGRAMITE_VARIABLE_NAME_MISSING = "TIGRAMITE_VARIABLE_NAME_MISSING"
_TIGRAMITE_LAG_TABLE_SCHEMA: tuple[str, ...] = (
    "source_col",
    "target_col",
    "lag_step",
    "link_value",
    "p_value",
    "graph_mark",
    "source",
    "warning_count",
    "warning_codes",
)


def _warn(code: str, detail: str, *, stacklevel: int) -> None:
    warnings.warn(f"{code}: {detail}", UserWarning, stacklevel=stacklevel)


def _warn_invalid_payload(detail: str) -> None:
    _warn(
        TIGRAMITE_PAYLOAD_SHAPE_INVALID,
        f"Invalid Tigramite payload shape: {detail}",
        stacklevel=3,
    )


def _warn_missing_variable_name(detail: str) -> None:
    _warn(
        TIGRAMITE_VARIABLE_NAME_MISSING,
        f"Missing or invalid variable name: {detail}",
        stacklevel=3,
    )


@dataclass(frozen=True)
class LagCandidateDescriptor:
    """Serializable lagged directed-link descriptor extracted from a graph payload."""

    source_col: str
    target_col: str
    lag_steps: int
    mark: str
    p_value: float | None = None
    value: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_col.strip() or not self.target_col.strip():
            raise ValueError("source_col and target_col must be non-empty strings.")
        if self.lag_steps <= 0:
            raise ValueError("lag_steps must be a positive integer.")
        try:
            json.dumps(self.metadata)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be JSON-serializable.") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_col": self.source_col,
            "target_col": self.target_col,
            "lag_steps": self.lag_steps,
            "mark": self.mark,
            "p_value": self.p_value,
            "value": self.value,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class LagCandidateExtractionResult:
    """Serializable container for extracted lag-candidate descriptors."""

    candidates: tuple[LagCandidateDescriptor, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            json.dumps(self.metadata)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be JSON-serializable.") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class TigramiteLagCandidateResult:
    """Public lag-candidate evidence contract for optional Tigramite adapter."""

    source_col: str
    target_col: str
    lag_steps: tuple[int, ...]
    min_lag_step: int | None
    max_lag_step: int | None
    link_values: tuple[float, ...]
    p_values: tuple[float, ...]
    graph_marks: tuple[str, ...]
    source: str
    warnings: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_col.strip() or not self.target_col.strip():
            raise ValueError("source_col and target_col must be non-empty strings.")
        if any(step <= 0 for step in self.lag_steps):
            raise ValueError("lag_steps must contain only positive integers.")
        lag_count = len(self.lag_steps)
        if self.link_values and len(self.link_values) != lag_count:
            raise ValueError(
                "link_values length must match lag_steps length when link_values are provided."
            )
        if self.p_values and len(self.p_values) != lag_count:
            raise ValueError(
                "p_values length must match lag_steps length when p_values are provided."
            )
        if self.graph_marks and len(self.graph_marks) != lag_count:
            raise ValueError(
                "graph_marks length must match lag_steps length when graph_marks are provided."
            )
        if self.lag_steps and (
            self.min_lag_step is None
            or self.max_lag_step is None
            or self.min_lag_step != min(self.lag_steps)
            or self.max_lag_step != max(self.lag_steps)
        ):
            raise ValueError("min_lag_step/max_lag_step must match lag_steps bounds.")
        if not self.lag_steps and (self.min_lag_step is not None or self.max_lag_step is not None):
            raise ValueError("min_lag_step/max_lag_step must be None when lag_steps is empty.")
        try:
            json.dumps(self.metadata)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be JSON-serializable.") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_col": self.source_col,
            "target_col": self.target_col,
            "lag_steps": list(self.lag_steps),
            "min_lag_step": self.min_lag_step,
            "max_lag_step": self.max_lag_step,
            "link_values": list(self.link_values),
            "p_values": list(self.p_values),
            "graph_marks": list(self.graph_marks),
            "source": self.source,
            "warnings": list(self.warnings),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class KernelTigramiteSupportComparison:
    """Serializable overlap/mass comparison between kernel and Tigramite lag support.

    The outside-support warning threshold is configurable through
    ``outside_support_warning_threshold`` in
    :func:`compare_kernel_to_tigramite_links`.
    """

    kernel_support_lag_steps: tuple[int, ...]
    candidate_support_lag_steps: tuple[int, ...]
    overlap_lag_steps: tuple[int, ...]
    kernel_mass_inside_candidate_support: float
    kernel_mass_outside_candidate_support: float
    outside_support_warning_threshold: float
    outside_support_exceeds_threshold: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.outside_support_warning_threshold < 0.0:
            raise ValueError("outside_support_warning_threshold must be >= 0.0.")
        if abs(
            (self.kernel_mass_inside_candidate_support + self.kernel_mass_outside_candidate_support)
            - 1.0
        ) > 1e-6:
            raise ValueError("Kernel inside/outside support mass must sum to approximately 1.0.")
        try:
            json.dumps(self.metadata)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be JSON-serializable.") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "kernel_support_lag_steps": list(self.kernel_support_lag_steps),
            "candidate_support_lag_steps": list(self.candidate_support_lag_steps),
            "overlap_lag_steps": list(self.overlap_lag_steps),
            "kernel_mass_inside_candidate_support": self.kernel_mass_inside_candidate_support,
            "kernel_mass_outside_candidate_support": self.kernel_mass_outside_candidate_support,
            "outside_support_warning_threshold": self.outside_support_warning_threshold,
            "outside_support_exceeds_threshold": self.outside_support_exceeds_threshold,
            "metadata": self.metadata,
        }

    def to_row(self) -> dict[str, Any]:
        return {
            "kernel_support_count": len(self.kernel_support_lag_steps),
            "candidate_support_count": len(self.candidate_support_lag_steps),
            "overlap_count": len(self.overlap_lag_steps),
            "kernel_mass_inside_candidate_support": self.kernel_mass_inside_candidate_support,
            "kernel_mass_outside_candidate_support": self.kernel_mass_outside_candidate_support,
            "outside_support_warning_threshold": self.outside_support_warning_threshold,
            "outside_support_exceeds_threshold": self.outside_support_exceeds_threshold,
        }


def compare_kernel_to_tigramite_links(
    kernel: Kernel,
    lag_candidates: Sequence[LagCandidateDescriptor] | Sequence[int],
    *,
    outside_support_warning_threshold: float = _DEFAULT_OUTSIDE_SUPPORT_WARNING_THRESHOLD,
) -> KernelTigramiteSupportComparison:
    """Compare kernel lag-mass placement against Tigramite candidate lag support.

    This helper reports support overlap and mass placement only; it does not
    infer causal truth.
    """
    kernel.validate()
    if outside_support_warning_threshold < 0.0:
        raise ValueError("outside_support_warning_threshold must be >= 0.0.")

    candidate_support = _extract_candidate_support_lag_steps(lag_candidates)
    kernel_support = tuple(int(step) for step in kernel.lag_steps)
    candidate_support_set = set(candidate_support)
    overlap = tuple(sorted(set(kernel_support).intersection(candidate_support_set)))

    inside_mass = float(
        sum(
            float(weight)
            for step, weight in zip(kernel.lag_steps, kernel.weights)
            if int(step) in candidate_support_set
        )
    )
    outside_mass = float(1.0 - inside_mass)
    exceeds_threshold = outside_mass > outside_support_warning_threshold
    if exceeds_threshold:
        detail = (
            "Kernel mass outside Tigramite candidate lag support exceeds threshold: "
            f"{outside_mass:.6f} > {outside_support_warning_threshold:.6f}."
        )
        _warn(
            TIGRAMITE_KERNEL_SUPPORT_THRESHOLD_BREACH,
            detail,
            stacklevel=2,
        )

    return KernelTigramiteSupportComparison(
        kernel_support_lag_steps=tuple(kernel_support),
        candidate_support_lag_steps=tuple(candidate_support),
        overlap_lag_steps=overlap,
        kernel_mass_inside_candidate_support=inside_mass,
        kernel_mass_outside_candidate_support=outside_mass,
        outside_support_warning_threshold=outside_support_warning_threshold,
        outside_support_exceeds_threshold=exceeds_threshold,
        metadata={"comparison_kind": "support_mass_only"},
    )


def parent_pairs_from_tigramite_graph(
    graph: object,
    *,
    var_names: Sequence[str] | Mapping[int | str, str],
    allowed_directed_marks: Sequence[str] = _SUPPORTED_DIRECTED_MARKS,
    lag_step_range: tuple[int, int] | None = None,
) -> tuple[tuple[str, str, int, str], ...]:
    """Extract ``(source_col, target_col, lag_steps, mark)`` tuples from graph payloads."""
    graph_payload = _extract_payload_value(graph, key="graph")
    graph_cube = _parse_graph_cube(graph_payload)
    index_to_name = _normalize_var_names(var_names, expected_n=len(graph_cube))
    directed_mark_set = _normalize_allowed_directed_marks(allowed_directed_marks)
    min_lag_steps, max_lag_steps = _normalize_lag_step_range(lag_step_range)

    pairs: list[tuple[str, str, int, str]] = []
    saw_contemporaneous = False
    saw_unsupported_mark = False

    for target_idx, source_rows in enumerate(graph_cube):
        for source_idx, lag_marks in enumerate(source_rows):
            for tau, mark in enumerate(lag_marks):
                normalized_mark = mark.strip()
                if not normalized_mark:
                    continue
                if tau == 0:
                    saw_contemporaneous = True
                    continue
                if min_lag_steps is not None and tau < min_lag_steps:
                    continue
                if max_lag_steps is not None and tau > max_lag_steps:
                    continue
                if normalized_mark not in directed_mark_set:
                    saw_unsupported_mark = True
                    continue
                pairs.append(
                    (
                        index_to_name[source_idx],
                        index_to_name[target_idx],
                        tau,
                        normalized_mark,
                    )
                )

    if saw_contemporaneous:
        _warn(
            TIGRAMITE_CONTEMPORANEOUS_LINK_IGNORED,
            "Contemporaneous links (tau=0) were ignored while extracting lag candidates.",
            stacklevel=2,
        )
    if saw_unsupported_mark:
        _warn(
            TIGRAMITE_GRAPH_MARK_UNSUPPORTED,
            "Unsupported graph mark encountered and ignored while extracting lag candidates.",
            stacklevel=2,
        )
    if not pairs:
        _warn(
            TIGRAMITE_NO_LINKS_FOUND,
            "No lagged directed links were found in graph payload.",
            stacklevel=2,
        )

    return tuple(pairs)


def lag_candidates_from_pcmci_graph(
    graph: object,
    *,
    var_names: Sequence[str] | Mapping[int | str, str],
    p_matrix: object | None = None,
    val_matrix: object | None = None,
    allowed_directed_marks: Sequence[str] = _SUPPORTED_DIRECTED_MARKS,
    lag_step_range: tuple[int, int] | None = None,
) -> LagCandidateExtractionResult:
    """Build serializable lag-candidate descriptors from plain graph/value/p-value payloads."""
    graph_payload = _extract_payload_value(graph, key="graph")
    p_payload = _extract_payload_value(p_matrix, key="p_matrix")
    val_payload = _extract_payload_value(val_matrix, key="val_matrix")
    graph_cube = _parse_graph_cube(graph_payload)
    p_cube = _parse_optional_numeric_cube(p_payload, key="p_matrix")
    val_cube = _parse_optional_numeric_cube(val_payload, key="val_matrix")
    _validate_optional_cube_shape(p_cube, graph_cube=graph_cube, key="p_matrix")
    _validate_optional_cube_shape(val_cube, graph_cube=graph_cube, key="val_matrix")
    if val_cube is not None:
        _warn(
            TIGRAMITE_VALUES_NOT_KERNEL_WEIGHTS,
            "Tigramite val_matrix values are treated as link-evidence scores, not kernel weights.",
            stacklevel=2,
        )

    pairs = parent_pairs_from_tigramite_graph(
        graph_cube,
        var_names=var_names,
        allowed_directed_marks=allowed_directed_marks,
        lag_step_range=lag_step_range,
    )

    index_to_name = _normalize_var_names(var_names, expected_n=len(graph_cube))
    name_to_index = {name: index for index, name in index_to_name.items()}

    candidates: list[LagCandidateDescriptor] = []
    for source_col, target_col, lag_steps, mark in pairs:
        source_idx = name_to_index[source_col]
        target_idx = name_to_index[target_col]
        p_value = p_cube[target_idx][source_idx][lag_steps] if p_cube is not None else None
        value = val_cube[target_idx][source_idx][lag_steps] if val_cube is not None else None
        candidates.append(
            LagCandidateDescriptor(
                source_col=source_col,
                target_col=target_col,
                lag_steps=lag_steps,
                mark=mark,
                p_value=p_value,
                value=value,
                metadata={},
            )
        )

    metadata: dict[str, Any] = {
        "graph_shape": [len(graph_cube), len(graph_cube[0]), len(graph_cube[0][0])],
    }
    if p_cube is not None:
        metadata["p_matrix"] = p_cube
    if val_cube is not None:
        metadata["val_matrix"] = val_cube

    return LagCandidateExtractionResult(candidates=tuple(candidates), metadata=metadata)


def candidate_set_from_tigramite_links(
    graph: object,
    *,
    var_names: Sequence[str] | Mapping[int | str, str],
    input_col: str,
    target_col: str,
    time_col: str = "ts",
    candidate_set_id: str | None = None,
    candidate_families: Sequence[str] | None = None,
    family_defaults: Mapping[str, Mapping[str, Any]] | None = None,
    p_matrix: object | None = None,
    val_matrix: object | None = None,
    allowed_directed_marks: Sequence[str] = _SUPPORTED_DIRECTED_MARKS,
    lag_step_range: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Build a serializable candidate-set descriptor from Tigramite lag evidence.

    This function never constructs live learners/kernels and never performs fitting.
    """
    extracted = lag_candidates_from_pcmci_graph(
        graph,
        var_names=var_names,
        p_matrix=p_matrix,
        val_matrix=val_matrix,
        allowed_directed_marks=allowed_directed_marks,
        lag_step_range=lag_step_range,
    )
    lag_evidence = _select_pair_lag_evidence(
        extracted.candidates,
        input_col=input_col,
        target_col=target_col,
    )
    lag_summary = _summarize_lag_evidence(lag_evidence)
    baseline_names = ["no_lag", "best_single_lag"]
    baseline_candidates = _baseline_candidate_descriptors(
        input_col=input_col,
        target_col=target_col,
        best_single_lag_step=None if lag_summary is None else int(lag_summary["median_lag_steps"]),
    )
    if lag_summary is None:
        _warn(
            TIGRAMITE_LAG_RANGE_EMPTY,
            "No usable lag evidence found for requested input/target pair; "
            "returning fallback candidate descriptors.",
            stacklevel=2,
        )
        candidate_items = baseline_candidates
        candidate_items[0]["metadata"]["generated_reason"] = TIGRAMITE_LAG_RANGE_EMPTY
        candidate_items[1]["metadata"]["generated_reason"] = TIGRAMITE_LAG_RANGE_EMPTY
    else:
        family_list = tuple(candidate_families or _DEFAULT_CANDIDATE_FAMILIES)
        candidate_items = _build_candidate_descriptors(
            family_list=family_list,
            family_defaults=family_defaults or {},
            lag_summary=lag_summary,
            input_col=input_col,
            target_col=target_col,
        )
        candidate_items.extend(baseline_candidates)

    return {
        "candidate_set_id": candidate_set_id or f"tigramite:{input_col}->{target_col}",
        "input_col": input_col,
        "target_col": target_col,
        "time_col": time_col,
        "candidates": candidate_items,
        "baseline_names": baseline_names,
        "selection_metric": "validation_loss",
        "metadata": {
            "source": "tigramite_link_adapter",
            "graph_metadata": extracted.metadata,
            "lag_evidence_count": len(lag_evidence),
            "lag_evidence": [item.to_dict() for item in lag_evidence],
            "lag_evidence_summary": lag_summary,
        },
    }


def _select_pair_lag_evidence(
    candidates: Sequence[LagCandidateDescriptor], *, input_col: str, target_col: str
) -> list[LagCandidateDescriptor]:
    return [
        item
        for item in candidates
        if item.source_col == input_col and item.target_col == target_col
    ]


def _summarize_lag_evidence(
    lag_evidence: Sequence[LagCandidateDescriptor],
) -> dict[str, Any] | None:
    if not lag_evidence:
        return None
    lag_steps = sorted({int(item.lag_steps) for item in lag_evidence if int(item.lag_steps) > 0})
    if not lag_steps:
        _warn(
            TIGRAMITE_LAG_RANGE_EMPTY,
            "Invalid lag evidence encountered: lag steps must be positive integers.",
            stacklevel=3,
        )
        return None
    marks = sorted({item.mark for item in lag_evidence if item.mark.strip()})
    p_values = [float(item.p_value) for item in lag_evidence if item.p_value is not None]
    values = [float(item.value) for item in lag_evidence if item.value is not None]
    summary: dict[str, Any] = {
        "lag_steps": lag_steps,
        "min_lag_steps": lag_steps[0],
        "max_lag_steps": lag_steps[-1],
        "median_lag_steps": int(median(lag_steps)),
        "graph_marks": marks,
    }
    if p_values:
        summary["p_value_min"] = min(p_values)
        summary["p_value_max"] = max(p_values)
    if values:
        summary["value_min"] = min(values)
        summary["value_max"] = max(values)
    return summary


def _build_candidate_descriptors(
    *,
    family_list: Sequence[str],
    family_defaults: Mapping[str, Mapping[str, Any]],
    lag_summary: Mapping[str, Any],
    input_col: str,
    target_col: str,
) -> list[dict[str, Any]]:
    allowed_families = set(_DEFAULT_CANDIDATE_FAMILIES)
    min_lag = int(lag_summary["min_lag_steps"])
    max_lag = int(lag_summary["max_lag_steps"])
    median_lag = int(lag_summary["median_lag_steps"])
    items: list[dict[str, Any]] = []
    for family in family_list:
        if family not in allowed_families:
            warnings.warn(
                f"Unsupported candidate family {family!r} skipped.",
                UserWarning,
                stacklevel=3,
            )
            continue
        defaults = dict(family_defaults.get(family, {}))
        candidate_type = _candidate_type_for_family(family)
        fixed_parameters = _fixed_params_for_family(
            family=family,
            defaults=defaults,
            min_lag=min_lag,
            max_lag=max_lag,
            median_lag=median_lag,
        )
        learner_parameters = _learner_params_for_family(
            family=family,
            candidate_type=candidate_type,
            defaults=defaults,
            max_lag=max_lag,
        )
        items.append(
            {
                "candidate_id": f"tigramite_{input_col}_to_{target_col}_{family}",
                "family": family,
                "candidate_type": candidate_type,
                "min_lag": min_lag,
                "max_lag": max_lag,
                "fixed_parameters": fixed_parameters,
                "learner_parameters": learner_parameters,
                "interpretation_hint": (
                    f"Auto-generated from Tigramite lag evidence for {input_col}->{target_col}"
                ),
                "metadata": {
                    "source_col": input_col,
                    "target_col": target_col,
                    "graph_marks": list(lag_summary.get("graph_marks", [])),
                    "lag_evidence_summary": dict(lag_summary),
                },
            }
        )
    return items


def _baseline_candidate_descriptors(
    *, input_col: str, target_col: str, best_single_lag_step: int | None
) -> list[dict[str, Any]]:
    selected_lag = int(best_single_lag_step) if best_single_lag_step is not None else 1
    return [
        {
            "candidate_id": f"tigramite_{input_col}_to_{target_col}_no_lag",
            "family": "no_lag",
            "candidate_type": "baseline",
            "min_lag": 0,
            "max_lag": 0,
            "fixed_parameters": {},
            "learner_parameters": {},
            "interpretation_hint": (
                f"Baseline descriptor for no-lag comparison on {input_col}->{target_col}"
            ),
            "metadata": {
                "source_col": input_col,
                "target_col": target_col,
            },
        },
        {
            "candidate_id": f"tigramite_{input_col}_to_{target_col}_best_single_lag",
            "family": "best_single_lag",
            "candidate_type": "baseline",
            "min_lag": selected_lag,
            "max_lag": selected_lag,
            "fixed_parameters": {},
            "learner_parameters": {},
            "interpretation_hint": (
                "Baseline descriptor selecting the best single lag under validation_loss for "
                f"{input_col}->{target_col}"
            ),
            "metadata": {
                "source_col": input_col,
                "target_col": target_col,
                "seed_lag_step": selected_lag,
            },
        },
    ]


def _learner_params_for_family(
    *,
    family: str,
    candidate_type: str,
    defaults: Mapping[str, Any],
    max_lag: int,
) -> dict[str, Any]:
    if candidate_type == "fixed_kernel":
        return {}
    params = dict(defaults)
    if candidate_type == "empirical_learner":
        params.setdefault("max_epochs", 100)
        params.setdefault("learning_rate", 0.05)
    elif candidate_type == "parametric_learner":
        # Backward-compatibility shim: accept old non-init names and map them
        # to learner-constructor parameter names used by fit paths.
        if "shape_alpha" in params and "init_shape_alpha" not in params:
            params["init_shape_alpha"] = params.pop("shape_alpha")
        if "rate_beta" in params and "init_rate_beta" not in params:
            params["init_rate_beta"] = params.pop("rate_beta")
        if "rate_lambda" in params and "init_rate_lambda" not in params:
            params["init_rate_lambda"] = params.pop("rate_lambda")
        if family == "gamma":
            params.setdefault("init_shape_alpha", 2.0)
            params.setdefault("init_rate_beta", 1.0 / max(1, max_lag))
        elif family == "exponential":
            params.setdefault("init_rate_lambda", 1.0 / max(1, max_lag))
    return params


def _candidate_type_for_family(family: str) -> str:
    if family == "simplex":
        return "empirical_learner"
    if family in {"gamma", "exponential"}:
        return "parametric_learner"
    return "fixed_kernel"


def _fixed_params_for_family(
    *,
    family: str,
    defaults: dict[str, Any],
    min_lag: int,
    max_lag: int,
    median_lag: int,
) -> dict[str, Any]:
    if family == "fixed_delay":
        defaults.setdefault("delay_steps", median_lag)
    elif family == "delayed_exponential":
        defaults.setdefault("delay", float(median_lag))
        defaults.setdefault("rate_lambda", 1.0 / max(1, max_lag))
    elif family == "lognormal":
        defaults.setdefault("log_mu", 0.0)
        defaults.setdefault("log_sigma", 1.0)
    elif family == "erlang":
        defaults.setdefault("shape_k", 2)
        defaults.setdefault("rate_beta", 1.0 / max(1, max_lag))
    elif family == "gamma":
        defaults.setdefault("shape_alpha", 2.0)
        defaults.setdefault("rate_beta", 1.0 / max(1, max_lag))
    elif family == "exponential":
        defaults.setdefault("rate_lambda", 1.0 / max(1, max_lag))
    if family in {
        "fixed_delay",
        "delayed_exponential",
        "lognormal",
        "erlang",
        "gamma",
        "exponential",
    }:
        return defaults
    return {}


def _extract_candidate_support_lag_steps(
    lag_candidates: Sequence[LagCandidateDescriptor] | Sequence[int],
) -> tuple[int, ...]:
    values: list[int] = []
    for item in lag_candidates:
        lag_step = int(item.lag_steps) if isinstance(item, LagCandidateDescriptor) else int(item)
        if lag_step <= 0:
            _warn(
                TIGRAMITE_LAG_RANGE_EMPTY,
                f"Non-positive lag step {lag_step} ignored in candidate support extraction.",
                stacklevel=3,
            )
            continue
        values.append(lag_step)
    return tuple(sorted(set(values)))


def tigramite_lag_candidate_table(
    result: TigramiteLagCandidateResult,
) -> pl.DataFrame:
    """Expand one lag-candidate result into a stable, tabular row-per-lag form."""
    warning_count = len(result.warnings)
    warning_codes = "|".join(result.warnings)
    rows = [
        {
            "source_col": result.source_col,
            "target_col": result.target_col,
            "lag_step": int(lag_step),
            "link_value": _value_at(result.link_values, idx),
            "p_value": _value_at(result.p_values, idx),
            "graph_mark": _value_at(result.graph_marks, idx, default=""),
            "source": result.source,
            "warning_count": warning_count,
            "warning_codes": warning_codes,
        }
        for idx, lag_step in enumerate(result.lag_steps)
    ]
    if not rows:
        rows = [
            {
                "source_col": result.source_col,
                "target_col": result.target_col,
                "lag_step": None,
                "link_value": None,
                "p_value": None,
                "graph_mark": "",
                "source": result.source,
                "warning_count": warning_count,
                "warning_codes": warning_codes,
            }
        ]
    return pl.DataFrame(rows).select(list(_TIGRAMITE_LAG_TABLE_SCHEMA))


def tigramite_lag_candidate_compact_dict(
    result: TigramiteLagCandidateResult,
) -> dict[str, Any]:
    """Return deterministic compact dict summary for one lag-candidate result."""
    table = tigramite_lag_candidate_table(result).sort(
        by=["source_col", "target_col", "lag_step"],
        descending=[False, False, False],
        nulls_last=True,
    )
    lag_steps = [
        int(step) for step in table.get_column("lag_step").to_list() if isinstance(step, int)
    ]
    return {
        "source_col": result.source_col,
        "target_col": result.target_col,
        "source": result.source,
        "lag_steps": lag_steps,
        "warning_codes": list(result.warnings),
        "rows": table.to_dicts(),
    }


def tigramite_lag_candidate_compact_text(
    result: TigramiteLagCandidateResult,
) -> str:
    """Render deterministic compact one-line summary for logs/release notes."""
    compact = tigramite_lag_candidate_compact_dict(result)
    lag_steps = compact["lag_steps"]
    if lag_steps:
        lag_text = ",".join(str(step) for step in lag_steps)
    else:
        lag_text = "none"
    warning_text = ",".join(compact["warning_codes"]) if compact["warning_codes"] else "none"
    return (
        f"{compact['source_col']}->{compact['target_col']} "
        f"source={compact['source']} lags={lag_text} warnings={warning_text}"
    )


def _value_at(values: Sequence[Any], idx: int, *, default: Any = None) -> Any:
    if idx < len(values):
        return values[idx]
    return default


def _normalize_var_names(
    var_names: Sequence[str] | Mapping[int | str, str], *, expected_n: int
) -> dict[int, str]:
    if isinstance(var_names, Mapping):
        mapping: dict[int, str] = {}
        normalized_inputs: dict[int, object] = {}
        seen_names: dict[str, int] = {}
        for raw_index, raw_name in var_names.items():
            normalized_index: int
            if isinstance(raw_index, int):
                normalized_index = raw_index
            elif isinstance(raw_index, str):
                try:
                    normalized_index = int(raw_index)
                except ValueError as exc:
                    _warn_missing_variable_name(
                        f"mapping key {raw_index!r} is not an integer index."
                    )
                    raise TigramitePayloadError(
                        f"Invalid variable-name mapping key {raw_index!r}; expected integer index."
                    ) from exc
            else:
                _warn_missing_variable_name(
                    f"mapping key {raw_index!r} has unsupported type {type(raw_index).__name__}."
                )
                raise TigramitePayloadError(
                    "Invalid variable-name mapping key type; expected integer index."
                )
            if normalized_index in normalized_inputs:
                _warn_missing_variable_name(
                    f"duplicate mapping keys normalize to index {normalized_index}."
                )
                raise TigramitePayloadError(
                    f"Duplicate variable-name mapping for index {normalized_index}."
                )
            normalized_inputs[normalized_index] = raw_name
        for index in range(expected_n):
            if index not in normalized_inputs:
                _warn_missing_variable_name(f"index {index} is not present in mapping.")
                raise TigramitePayloadError(f"Missing variable-name mapping for index {index}.")
            name = normalized_inputs[index]
            if not isinstance(name, str) or not name.strip():
                _warn_missing_variable_name(f"index {index} has invalid value {name!r}.")
                raise TigramitePayloadError(f"Invalid variable-name mapping for index {index}.")
            normalized_name = name.strip()
            if normalized_name in seen_names:
                prior_index = seen_names[normalized_name]
                raise ValueError(
                    "var_names must be unique; "
                    "duplicate name "
                    f"{normalized_name!r} found at indices {prior_index} and {index}."
                )
            seen_names[normalized_name] = index
            mapping[index] = normalized_name
        return mapping

    if isinstance(var_names, Sequence) and not isinstance(var_names, (str, bytes)):
        if len(var_names) != expected_n:
            _warn_missing_variable_name(
                f"sequence length {len(var_names)} does not match expected {expected_n}."
            )
            raise TigramitePayloadError(
                f"Variable-name sequence length {len(var_names)} "
                f"does not match graph size {expected_n}."
            )
        mapping = {}
        seen_seq_names: dict[str, int] = {}
        for index, name in enumerate(var_names):
            if not isinstance(name, str) or not name.strip():
                _warn_missing_variable_name(f"position {index} has invalid value {name!r}.")
                raise TigramitePayloadError(f"Invalid variable-name at position {index}.")
            normalized_name = name.strip()
            if normalized_name in seen_seq_names:
                prior_index = seen_seq_names[normalized_name]
                raise ValueError(
                    "var_names must be unique; "
                    "duplicate name "
                    f"{normalized_name!r} found at indices {prior_index} and {index}."
                )
            seen_seq_names[normalized_name] = index
            mapping[index] = normalized_name
        return mapping

    _warn_missing_variable_name("var_names is not a sequence or mapping keyed by index.")
    raise TigramitePayloadError("var_names must be a sequence of names or mapping by index.")


def _extract_payload_value(value: object | None, *, key: str) -> object | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        if key not in value:
            _warn_invalid_payload(f"mapping payload is missing required key '{key}'.")
            raise TigramitePayloadError(f"Malformed payload: missing required key '{key}'.")
        return cast(object, value[key])
    return value


def _normalize_allowed_directed_marks(allowed_directed_marks: object) -> set[str]:
    if isinstance(allowed_directed_marks, (str, bytes)):
        raise TypeError(
            "allowed_directed_marks must be a non-string iterable of strings, "
            "not a bare string."
        )
    if not isinstance(allowed_directed_marks, Iterable):
        raise TypeError("allowed_directed_marks must be a non-string iterable of strings.")

    normalized: set[str] = set()
    for mark in allowed_directed_marks:
        if not isinstance(mark, str):
            raise ValueError("allowed_directed_marks must contain only strings.")
        stripped = mark.strip()
        if not stripped:
            raise ValueError("allowed_directed_marks must not contain empty strings.")
        normalized.add(stripped)
    return normalized


def _parse_graph_cube(graph: object) -> list[list[list[str]]]:
    if not isinstance(graph, Sequence) or isinstance(graph, (str, bytes)):
        _warn_invalid_payload("graph must be a 3D sequence.")
        raise TigramitePayloadError("Malformed graph payload: expected 3D sequence.")

    cube: list[list[list[str]]] = []
    for target_idx, source_rows in enumerate(graph):
        if not isinstance(source_rows, Sequence) or isinstance(source_rows, (str, bytes)):
            _warn_invalid_payload(f"graph[{target_idx}] must be a sequence.")
            raise TigramitePayloadError(
                f"Malformed graph payload: graph[{target_idx}] must be a sequence."
            )
        parsed_source_rows: list[list[str]] = []
        for source_idx, lag_marks in enumerate(source_rows):
            if not isinstance(lag_marks, Sequence) or isinstance(lag_marks, (str, bytes)):
                _warn_invalid_payload(f"graph[{target_idx}][{source_idx}] must be a sequence.")
                raise TigramitePayloadError(
                    "Malformed graph payload: "
                    f"graph[{target_idx}][{source_idx}] must be a sequence."
                )
            parsed_lag_marks: list[str] = []
            for tau, mark in enumerate(lag_marks):
                if not isinstance(mark, str):
                    _warn_invalid_payload(
                        f"graph[{target_idx}][{source_idx}][{tau}] must be a string mark."
                    )
                    raise TigramitePayloadError(
                        "Malformed graph payload: "
                        f"graph[{target_idx}][{source_idx}][{tau}] must be a string mark."
                    )
                parsed_lag_marks.append(mark)
            parsed_source_rows.append(parsed_lag_marks)
        cube.append(parsed_source_rows)

    if not cube:
        _warn_invalid_payload("graph must not be empty.")
        raise TigramitePayloadError("Malformed graph payload: graph must not be empty.")

    n = len(cube)
    tau_count = len(cube[0][0]) if cube[0] and cube[0][0] else 0
    if tau_count == 0:
        _warn_invalid_payload("graph must include lag axis with at least one mark.")
        raise TigramitePayloadError(
            "Malformed graph payload: graph must include lag axis with >=1 mark."
        )

    for target_idx, source_rows in enumerate(cube):
        if len(source_rows) != n:
            _warn_invalid_payload(f"graph[{target_idx}] has non-square source axis length.")
            raise TigramitePayloadError(
                "Malformed graph payload: expected square first two axes; "
                f"row {target_idx} has {len(source_rows)} sources, expected {n}."
            )
        for source_idx, lag_marks in enumerate(source_rows):
            if len(lag_marks) != tau_count:
                _warn_invalid_payload(
                    f"graph[{target_idx}][{source_idx}] has inconsistent lag axis length."
                )
                raise TigramitePayloadError(
                    "Malformed graph payload: inconsistent lag axis length at "
                    f"graph[{target_idx}][{source_idx}]."
                )
    return cube


def _parse_optional_numeric_cube(
    value: object | None, *, key: str
) -> list[list[list[float]]] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _warn_invalid_payload(f"'{key}' must be a 3D numeric sequence.")
        raise TigramitePayloadError(f"Malformed payload: '{key}' must be a 3D numeric sequence.")

    cube: list[list[list[float]]] = []
    for i, rows in enumerate(value):
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            _warn_invalid_payload(f"'{key}[{i}]' must be a sequence.")
            raise TigramitePayloadError(f"Malformed payload: '{key}[{i}]' must be a sequence.")
        parsed_rows: list[list[float]] = []
        for j, vals in enumerate(rows):
            if not isinstance(vals, Sequence) or isinstance(vals, (str, bytes)):
                _warn_invalid_payload(f"'{key}[{i}][{j}]' must be a sequence.")
                raise TigramitePayloadError(
                    f"Malformed payload: '{key}[{i}][{j}]' must be a sequence."
                )
            parsed_vals: list[float] = []
            for k, entry in enumerate(vals):
                if not isinstance(entry, (int, float)):
                    _warn_invalid_payload(f"'{key}[{i}][{j}][{k}]' must be numeric.")
                    raise TigramitePayloadError(
                        f"Malformed payload: '{key}[{i}][{j}][{k}]' must be numeric."
                    )
                parsed_vals.append(float(entry))
            parsed_rows.append(parsed_vals)
        cube.append(parsed_rows)

    return cube


def _validate_optional_cube_shape(
    cube: list[list[list[float]]] | None,
    *,
    graph_cube: list[list[list[str]]],
    key: str,
) -> None:
    if cube is None:
        return
    expected_shape = (len(graph_cube), len(graph_cube[0]), len(graph_cube[0][0]))
    shape: tuple[int, int | None, int | None]
    if len(cube) != expected_shape[0]:
        shape = (len(cube), None, None)
        _warn_invalid_payload(
            f"'{key}' shape {shape} does not match graph shape {expected_shape}."
        )
        raise TigramitePayloadError(
            f"Malformed payload: '{key}' shape {shape} does not match graph shape {expected_shape}."
        )
    for i, rows in enumerate(cube):
        if len(rows) != expected_shape[1]:
            shape = (len(cube), len(rows), None)
            _warn_invalid_payload(
                f"'{key}' shape {shape} does not match graph shape {expected_shape}; "
                f"axis-1 length mismatch at index {i}."
            )
            raise TigramitePayloadError(
                f"Malformed payload: '{key}' shape {shape} does not match graph shape "
                f"{expected_shape} (axis-1 mismatch at index {i})."
            )
        for j, vals in enumerate(rows):
            if len(vals) != expected_shape[2]:
                shape = (len(cube), len(rows), len(vals))
                _warn_invalid_payload(
                    f"'{key}' shape {shape} does not match graph shape {expected_shape}; "
                    f"axis-2 length mismatch at index ({i}, {j})."
                )
                raise TigramitePayloadError(
                    f"Malformed payload: '{key}' shape {shape} does not match graph shape "
                    f"{expected_shape} (axis-2 mismatch at index ({i}, {j}))."
                )


def _normalize_lag_step_range(
    lag_step_range: tuple[int, int] | None,
) -> tuple[int | None, int | None]:
    if lag_step_range is None:
        return None, None
    if (
        not isinstance(lag_step_range, Sequence)
        or isinstance(lag_step_range, (str, bytes))
        or len(lag_step_range) != 2
    ):
        raise ValueError("lag_step_range must be a 2-item sequence: (min_lag_step, max_lag_step).")
    try:
        min_lag_steps = int(lag_step_range[0])
        max_lag_steps = int(lag_step_range[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "lag_step_range bounds must be integers: (min_lag_step, max_lag_step)."
        ) from exc
    if min_lag_steps <= 0 or max_lag_steps <= 0:
        raise ValueError("lag_step_range bounds must be positive integers.")
    if min_lag_steps > max_lag_steps:
        raise ValueError("lag_step_range min bound must be <= max bound.")
    return min_lag_steps, max_lag_steps
