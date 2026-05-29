"""Parametric kernel implementations and discrete-weight helpers."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

from rtdfeatures.kernels.base import Kernel, LearnedKernel

ParametricValue = float | int


@dataclass(frozen=True)
class ParametricFamilySpec:
    family: str
    parameter_keys: frozenset[str]
    weight_fn: Callable[..., tuple[float, ...]]
    positive_parameters: frozenset[str] = frozenset()
    nonnegative_parameters: frozenset[str] = frozenset()
    finite_parameters: frozenset[str] = frozenset()
    positive_integer_parameters: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ParametricKernelInit:
    lag_steps: tuple[int, ...]
    weights: tuple[float, ...]
    parameters: dict[str, ParametricValue]


_MIN_PARAM_VALUE = 0.0


def parametric_lag_steps(*, min_lag_steps: int, max_lag_steps: int) -> tuple[int, ...]:
    """Build the admissible lag-step grid for parametric discretisation."""
    if min_lag_steps < 0:
        raise ValueError("min_lag_steps must be non-negative.")
    if max_lag_steps < min_lag_steps:
        raise ValueError("max_lag_steps must be >= min_lag_steps.")
    return tuple(range(min_lag_steps, max_lag_steps + 1))


def discrete_exponential_weights(
    *,
    rate_lambda: float,
    lag_steps: tuple[int, ...],
    dt: float,
) -> tuple[float, ...]:
    """Evaluate and normalise an exponential family on a lag grid."""
    _validate_positive_finite_parameter(name="rate_lambda", value=rate_lambda)
    _validate_dt(dt)
    _validate_lag_steps(lag_steps)
    return _normalize_nonnegative_weights(
        tuple(math.exp(-rate_lambda * step * dt) for step in lag_steps)
    )


def discrete_delayed_exponential_weights(
    *,
    delay: float,
    rate_lambda: float,
    lag_steps: tuple[int, ...],
    dt: float,
) -> tuple[float, ...]:
    """Evaluate and normalise a delayed-exponential family on a lag grid."""
    _validate_non_negative_finite_parameter(name="delay", value=delay)
    _validate_positive_finite_parameter(name="rate_lambda", value=rate_lambda)
    _validate_dt(dt)
    _validate_lag_steps(lag_steps)

    return _normalize_nonnegative_weights(
        tuple(
            0.0
            if (step * dt) < delay
            else math.exp(-rate_lambda * ((step * dt) - delay))
            for step in lag_steps
        )
    )


def discrete_gamma_weights(
    *,
    shape_alpha: float,
    rate_beta: float,
    lag_steps: tuple[int, ...],
    dt: float,
) -> tuple[float, ...]:
    """Evaluate and normalise a gamma family density proxy on a lag grid."""
    _validate_positive_finite_parameter(name="shape_alpha", value=shape_alpha)
    _validate_positive_finite_parameter(name="rate_beta", value=rate_beta)
    _validate_dt(dt)
    _validate_lag_steps(lag_steps)

    unnormalized: list[float] = []
    for step in lag_steps:
        lag_time = step * dt
        if lag_time <= 0.0:
            if shape_alpha < 1.0:
                raise ValueError(
                    "shape_alpha < 1.0 is not supported when lag grid includes zero lag."
                )
            if math.isclose(shape_alpha, 1.0):
                value = rate_beta
            else:
                value = 0.0
        else:
            value = (
                (rate_beta**shape_alpha)
                * (lag_time ** (shape_alpha - 1.0))
                * math.exp(-rate_beta * lag_time)
                / math.gamma(shape_alpha)
            )
        unnormalized.append(value)
    return _normalize_nonnegative_weights(tuple(unnormalized))


def discrete_lognormal_weights(
    *,
    log_mu: float,
    log_sigma: float,
    lag_steps: tuple[int, ...],
    dt: float,
) -> tuple[float, ...]:
    """Evaluate and normalise a log-normal family density proxy on a lag grid."""
    _validate_finite_parameter(name="log_mu", value=log_mu)
    _validate_positive_finite_parameter(name="log_sigma", value=log_sigma)
    _validate_dt(dt)
    _validate_lag_steps(lag_steps)

    sqrt_two_pi = math.sqrt(2.0 * math.pi)
    unnormalized = []
    for step in lag_steps:
        lag_time = step * dt
        if lag_time <= 0.0:
            unnormalized.append(0.0)
            continue
        z_value = (math.log(lag_time) - log_mu) / log_sigma
        value = math.exp(-0.5 * (z_value**2)) / (lag_time * log_sigma * sqrt_two_pi)
        unnormalized.append(value)
    return _normalize_nonnegative_weights(tuple(unnormalized))


def discrete_erlang_weights(
    *,
    shape_k: int,
    rate_beta: float,
    lag_steps: tuple[int, ...],
    dt: float,
) -> tuple[float, ...]:
    """Evaluate and normalise an Erlang family density proxy on a lag grid."""
    _validate_positive_integer_parameter(name="shape_k", value=shape_k)
    _validate_positive_finite_parameter(name="rate_beta", value=rate_beta)
    _validate_dt(dt)
    _validate_lag_steps(lag_steps)

    factorial_term = math.factorial(shape_k - 1)
    unnormalized = []
    for step in lag_steps:
        lag_time = step * dt
        if lag_time <= 0.0:
            value = rate_beta if shape_k == 1 else 0.0
        else:
            value = (
                (rate_beta**shape_k)
                * (lag_time ** (shape_k - 1))
                * math.exp(-rate_beta * lag_time)
                / factorial_term
            )
        unnormalized.append(value)
    return _normalize_nonnegative_weights(tuple(unnormalized))


def _parametric_family_specs() -> dict[str, ParametricFamilySpec]:
    return {
        "gamma": ParametricFamilySpec(
            family="gamma",
            parameter_keys=frozenset({"shape_alpha", "rate_beta"}),
            weight_fn=discrete_gamma_weights,
            positive_parameters=frozenset({"shape_alpha", "rate_beta"}),
        ),
        "exponential": ParametricFamilySpec(
            family="exponential",
            parameter_keys=frozenset({"rate_lambda"}),
            weight_fn=discrete_exponential_weights,
            positive_parameters=frozenset({"rate_lambda"}),
        ),
        "delayed_exponential": ParametricFamilySpec(
            family="delayed_exponential",
            parameter_keys=frozenset({"delay", "rate_lambda"}),
            weight_fn=discrete_delayed_exponential_weights,
            positive_parameters=frozenset({"rate_lambda"}),
            nonnegative_parameters=frozenset({"delay"}),
        ),
        "lognormal": ParametricFamilySpec(
            family="lognormal",
            parameter_keys=frozenset({"log_mu", "log_sigma"}),
            weight_fn=discrete_lognormal_weights,
            finite_parameters=frozenset({"log_mu"}),
            positive_parameters=frozenset({"log_sigma"}),
        ),
        "erlang": ParametricFamilySpec(
            family="erlang",
            parameter_keys=frozenset({"shape_k", "rate_beta"}),
            weight_fn=discrete_erlang_weights,
            positive_integer_parameters=frozenset({"shape_k"}),
            positive_parameters=frozenset({"rate_beta"}),
        ),
    }


def supported_parametric_families() -> tuple[str, ...]:
    return tuple(sorted(_parametric_family_specs()))


def get_parametric_family_spec(family: str) -> ParametricFamilySpec:
    spec = _parametric_family_specs().get(family)
    if spec is None:
        raise ValueError(
            "unsupported parametric family "
            f"'{family}'. Supported families: {supported_parametric_families()}."
        )
    return spec


def _make_parametric_learned_kernel(
    *,
    family: str,
    dt: float,
    min_lag_steps: int,
    max_lag_steps: int,
    parameters: dict[str, ParametricValue],
    name: str | None = None,
) -> LearnedKernel:
    """Build a standard LearnedKernel from constrained parametric settings."""
    lag_steps = parametric_lag_steps(min_lag_steps=min_lag_steps, max_lag_steps=max_lag_steps)
    _validate_dt(dt)
    spec = get_parametric_family_spec(family)
    validated_parameters = _validated_parametric_parameters(
        family=family,
        parameters=parameters,
    )
    weights = spec.weight_fn(
        lag_steps=lag_steps,
        dt=dt,
        **validated_parameters,
    )

    kernel = LearnedKernel(
        weights=weights,
        lag_steps=lag_steps,
        dt=dt,
        min_lag_steps=min_lag_steps,
        max_lag_steps=max_lag_steps,
        name=name,
    )
    kernel.validate()
    return kernel


def build_parametric_kernel_init(
    *,
    family: str,
    parameters: dict[str, ParametricValue],
    max_lag_steps: int,
    dt: float,
    min_lag_steps: int,
) -> ParametricKernelInit:
    """Build validated lag-grid/weight/parameter inputs for explicit kernels."""
    lag_steps = parametric_lag_steps(
        min_lag_steps=min_lag_steps,
        max_lag_steps=max_lag_steps,
    )
    _validate_dt(dt)
    spec = get_parametric_family_spec(family)
    validated_parameters = _validated_parametric_parameters(
        family=family,
        parameters=parameters,
    )
    weights = spec.weight_fn(
        lag_steps=lag_steps,
        dt=dt,
        **validated_parameters,
    )
    return ParametricKernelInit(
        lag_steps=lag_steps,
        weights=weights,
        parameters=validated_parameters,
    )


def build_parametric_fit_provenance(
    *,
    family: str,
    parameters: dict[str, ParametricValue],
    initial_parameters: dict[str, ParametricValue],
    conversion_status: str = "ok",
    conversion_message: str | None = None,
) -> dict[str, Any]:
    """Return additive fit provenance payload for parametric learners."""
    validated_parameters = _validated_parametric_parameters(
        family=family,
        parameters=parameters,
    )
    validated_initial_parameters = _validated_parametric_parameters(
        family=family,
        parameters=initial_parameters,
    )
    if conversion_status not in {"ok", "failed"}:
        raise ValueError("conversion_status must be either 'ok' or 'failed'.")
    return {
        "parametric_family": family,
        "parametric_parameters": validated_parameters,
        "parametric_initial_parameters": validated_initial_parameters,
        "parametric_conversion_status": conversion_status,
        "parametric_conversion_message": conversion_message,
    }


def parametric_summary(
    base_summary: dict[str, Any],
    *,
    family: str,
    parameters: dict[str, ParametricValue],
) -> dict[str, Any]:
    """Return a base summary enriched with validated parametric metadata."""
    summary = dict(base_summary)
    summary["parametric_family"] = family
    summary["parametric_parameters"] = _validated_parametric_parameters(
        family=family,
        parameters=parameters,
    )
    return summary


def summarize_parametric_kernel(
    *,
    kernel: Kernel,
    family: str,
    parameters: dict[str, ParametricValue],
) -> dict[str, float | int | str | None | dict[str, ParametricValue]]:
    """Return kernel summary enriched with parametric family metadata."""
    summary = cast(dict[str, Any], kernel.summary())
    summary = parametric_summary(
        summary,
        family=family,
        parameters=parameters,
    )
    return cast(
        dict[str, float | int | str | None | dict[str, ParametricValue]],
        summary,
    )


def _validate_dt(dt: float) -> None:
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and strictly positive.")


def _validate_lag_steps(lag_steps: tuple[int, ...]) -> None:
    if not lag_steps:
        raise ValueError("lag_steps must be non-empty.")
    if any(step < 0 for step in lag_steps):
        raise ValueError("lag_steps must be non-negative.")
    if any(left >= right for left, right in zip(lag_steps, lag_steps[1:])):
        raise ValueError("lag_steps must be strictly increasing (sorted unique).")


def _validated_parametric_parameters(
    *,
    family: str,
    parameters: dict[str, ParametricValue],
) -> dict[str, ParametricValue]:
    spec = get_parametric_family_spec(family)
    expected = spec.parameter_keys
    provided = set(parameters)
    missing = sorted(expected - provided)
    extra = sorted(provided - expected)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unexpected {extra}")
        raise ValueError(
            f"{family} parameters must match documented fields exactly: "
            f"{', '.join(details)}."
        )
    validated: dict[str, ParametricValue] = {}
    for key in expected:
        value = parameters[key]
        if key in spec.positive_integer_parameters:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value <= 0
                or int(value) != value
            ):
                raise ValueError(f"{key} must be a positive integer.")
            validated[key] = int(value)
            continue
        if key in spec.finite_parameters:
            _validate_finite_parameter(name=key, value=value)
        elif key in spec.nonnegative_parameters:
            _validate_non_negative_finite_parameter(name=key, value=value)
        elif key in spec.positive_parameters:
            _validate_positive_finite_parameter(name=key, value=value)
        validated[key] = float(value)
    return validated


def _validate_positive_finite_parameter(*, name: str, value: float) -> None:
    if not math.isfinite(value) or value <= _MIN_PARAM_VALUE:
        raise ValueError(f"{name} must be finite and strictly positive.")


def _validate_non_negative_finite_parameter(*, name: str, value: float) -> None:
    if not math.isfinite(value) or value < _MIN_PARAM_VALUE:
        raise ValueError(
            f"{name} must be finite and non-negative (not strictly positive)."
        )


def _validate_finite_parameter(*, name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")


def _validate_positive_integer_parameter(*, name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _normalize_nonnegative_weights(weights: tuple[float, ...]) -> tuple[float, ...]:
    if not weights:
        raise ValueError("Parametric conversion requires at least one lag step.")
    if any(not math.isfinite(weight) for weight in weights):
        raise ValueError("Parametric conversion produced non-finite weights.")
    clamped = tuple(max(0.0, float(weight)) for weight in weights)
    weight_sum = float(sum(clamped))
    if weight_sum <= 0.0:
        raise ValueError("Parametric conversion produced zero total mass.")
    return tuple(weight / weight_sum for weight in clamped)


# ---------------------------------------------------------------------------
# Kernel subclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, init=False)
class GammaKernel(Kernel):
    """Gamma-family kernel constructed from explicit parameters."""

    shape_alpha: float = field(init=False)
    rate_beta: float = field(init=False)

    def __init__(
        self,
        *,
        shape_alpha: float,
        rate_beta: float,
        max_lag_steps: int,
        dt: float,
        min_lag_steps: int = 0,
        name: str | None = None,
    ) -> None:
        init = build_parametric_kernel_init(
            family="gamma",
            parameters={
                "shape_alpha": shape_alpha,
                "rate_beta": rate_beta,
            },
            max_lag_steps=max_lag_steps,
            dt=dt,
            min_lag_steps=min_lag_steps,
        )
        super().__init__(
            weights=init.weights,
            lag_steps=init.lag_steps,
            dt=dt,
            min_lag_steps=min_lag_steps,
            max_lag_steps=max_lag_steps,
            name=name,
        )
        object.__setattr__(self, "shape_alpha", float(init.parameters["shape_alpha"]))
        object.__setattr__(self, "rate_beta", float(init.parameters["rate_beta"]))

    def summary(self) -> dict[str, float | int | str | None | dict[str, float | int]]:
        return parametric_summary(
            super().summary(),
            family="gamma",
            parameters={
                "shape_alpha": self.shape_alpha,
                "rate_beta": self.rate_beta,
            },
        )


@dataclass(frozen=True, init=False)
class ExponentialKernel(Kernel):
    """Exponential-family kernel constructed from explicit parameters."""

    rate_lambda: float = field(init=False)

    def __init__(
        self,
        *,
        rate_lambda: float,
        max_lag_steps: int,
        dt: float,
        min_lag_steps: int = 0,
        name: str | None = None,
    ) -> None:
        init = build_parametric_kernel_init(
            family="exponential",
            parameters={"rate_lambda": rate_lambda},
            max_lag_steps=max_lag_steps,
            dt=dt,
            min_lag_steps=min_lag_steps,
        )
        super().__init__(
            weights=init.weights,
            lag_steps=init.lag_steps,
            dt=dt,
            min_lag_steps=min_lag_steps,
            max_lag_steps=max_lag_steps,
            name=name,
        )
        object.__setattr__(self, "rate_lambda", float(init.parameters["rate_lambda"]))

    def summary(self) -> dict[str, float | int | str | None | dict[str, float | int]]:
        return parametric_summary(
            super().summary(),
            family="exponential",
            parameters={"rate_lambda": self.rate_lambda},
        )


@dataclass(frozen=True, init=False)
class DelayedExponentialKernel(Kernel):
    """Delayed-exponential-family kernel constructed from explicit parameters."""

    delay: float = field(init=False)
    rate_lambda: float = field(init=False)

    def __init__(
        self,
        *,
        delay: float,
        rate_lambda: float,
        max_lag_steps: int,
        dt: float,
        min_lag_steps: int = 0,
        name: str | None = None,
    ) -> None:
        init = build_parametric_kernel_init(
            family="delayed_exponential",
            parameters={
                "delay": delay,
                "rate_lambda": rate_lambda,
            },
            max_lag_steps=max_lag_steps,
            dt=dt,
            min_lag_steps=min_lag_steps,
        )
        super().__init__(
            weights=init.weights,
            lag_steps=init.lag_steps,
            dt=dt,
            min_lag_steps=min_lag_steps,
            max_lag_steps=max_lag_steps,
            name=name,
        )
        object.__setattr__(self, "delay", float(init.parameters["delay"]))
        object.__setattr__(self, "rate_lambda", float(init.parameters["rate_lambda"]))

    def summary(self) -> dict[str, float | int | str | None | dict[str, float | int]]:
        return parametric_summary(
            super().summary(),
            family="delayed_exponential",
            parameters={
                "delay": self.delay,
                "rate_lambda": self.rate_lambda,
            },
        )


@dataclass(frozen=True, init=False)
class LogNormalKernel(Kernel):
    """Log-normal-family kernel constructed from explicit parameters."""

    log_mu: float = field(init=False)
    log_sigma: float = field(init=False)

    def __init__(
        self,
        *,
        log_mu: float,
        log_sigma: float,
        max_lag_steps: int,
        dt: float,
        min_lag_steps: int = 0,
        name: str | None = None,
    ) -> None:
        init = build_parametric_kernel_init(
            family="lognormal",
            parameters={
                "log_mu": log_mu,
                "log_sigma": log_sigma,
            },
            max_lag_steps=max_lag_steps,
            dt=dt,
            min_lag_steps=min_lag_steps,
        )
        super().__init__(
            weights=init.weights,
            lag_steps=init.lag_steps,
            dt=dt,
            min_lag_steps=min_lag_steps,
            max_lag_steps=max_lag_steps,
            name=name,
        )
        object.__setattr__(self, "log_mu", float(init.parameters["log_mu"]))
        object.__setattr__(self, "log_sigma", float(init.parameters["log_sigma"]))

    def summary(self) -> dict[str, float | int | str | None | dict[str, float | int]]:
        return parametric_summary(
            super().summary(),
            family="lognormal",
            parameters={
                "log_mu": self.log_mu,
                "log_sigma": self.log_sigma,
            },
        )


@dataclass(frozen=True, init=False)
class ErlangKernel(Kernel):
    """Erlang-family kernel constructed from explicit parameters."""

    shape_k: int = field(init=False)
    rate_beta: float = field(init=False)

    def __init__(
        self,
        *,
        shape_k: int,
        rate_beta: float,
        max_lag_steps: int,
        dt: float,
        min_lag_steps: int = 0,
        name: str | None = None,
    ) -> None:
        _validate_positive_integer_parameter(name="shape_k", value=shape_k)
        init = build_parametric_kernel_init(
            family="erlang",
            parameters={
                "shape_k": shape_k,
                "rate_beta": rate_beta,
            },
            max_lag_steps=max_lag_steps,
            dt=dt,
            min_lag_steps=min_lag_steps,
        )
        super().__init__(
            weights=init.weights,
            lag_steps=init.lag_steps,
            dt=dt,
            min_lag_steps=min_lag_steps,
            max_lag_steps=max_lag_steps,
            name=name,
        )
        object.__setattr__(self, "shape_k", int(init.parameters["shape_k"]))
        object.__setattr__(self, "rate_beta", float(init.parameters["rate_beta"]))

    def summary(self) -> dict[str, float | int | str | None | dict[str, float | int]]:
        return parametric_summary(
            super().summary(),
            family="erlang",
            parameters={
                "shape_k": self.shape_k,
                "rate_beta": self.rate_beta,
            },
        )
