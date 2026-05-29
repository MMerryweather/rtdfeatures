"""Registry-focused tests for V1 parametric kernel families."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from rtdfeatures.kernels import (
    DelayedExponentialKernel,
    ErlangKernel,
    ExponentialKernel,
    GammaKernel,
    LogNormalKernel,
)
from rtdfeatures.kernels.base import Kernel
from rtdfeatures.kernels.parametric import (
    _make_parametric_learned_kernel,
    _parametric_family_specs,
    _validated_parametric_parameters,
    get_parametric_family_spec,
    summarize_parametric_kernel,
    supported_parametric_families,
)


def test_parametric_family_registry_contains_all_v1_families() -> None:
    specs = _parametric_family_specs()
    assert set(specs) == {
        "gamma",
        "exponential",
        "delayed_exponential",
        "lognormal",
        "erlang",
    }


def test_parametric_family_specs_have_expected_parameter_keys() -> None:
    expected_keys = {
        "gamma": frozenset({"shape_alpha", "rate_beta"}),
        "exponential": frozenset({"rate_lambda"}),
        "delayed_exponential": frozenset({"delay", "rate_lambda"}),
        "lognormal": frozenset({"log_mu", "log_sigma"}),
        "erlang": frozenset({"shape_k", "rate_beta"}),
    }
    for family, expected in expected_keys.items():
        assert get_parametric_family_spec(family).parameter_keys == expected


def test_supported_parametric_families_are_registry_derived() -> None:
    assert supported_parametric_families() == tuple(sorted(_parametric_family_specs().keys()))


def test_get_parametric_family_spec_raises_with_supported_names() -> None:
    with pytest.raises(ValueError, match="unsupported parametric family") as exc_info:
        get_parametric_family_spec("weibull")
    message = str(exc_info.value)
    for family in supported_parametric_families():
        assert family in message


def test_each_parametric_family_weight_function_normalizes_weights() -> None:
    lag_steps = (0, 1, 2, 3, 4)
    dt = 1.0
    params = {
        "gamma": {"shape_alpha": 2.0, "rate_beta": 0.5},
        "exponential": {"rate_lambda": 0.7},
        "delayed_exponential": {"delay": 1.0, "rate_lambda": 0.7},
        "lognormal": {"log_mu": 0.0, "log_sigma": 0.6},
        "erlang": {"shape_k": 3, "rate_beta": 0.8},
    }
    for family, family_params in params.items():
        spec = get_parametric_family_spec(family)
        weights = spec.weight_fn(lag_steps=lag_steps, dt=dt, **family_params)
        assert all(weight >= 0.0 for weight in weights)
        assert sum(weights) == pytest.approx(1.0, abs=1e-9)


def test_explicit_parametric_kernel_constructors_preserve_public_signatures() -> None:
    constructor_cases = [
        (
            GammaKernel,
            ("shape_alpha", "rate_beta", "max_lag_steps", "dt", "min_lag_steps", "name"),
            {"min_lag_steps": 0, "name": None},
        ),
        (
            ExponentialKernel,
            ("rate_lambda", "max_lag_steps", "dt", "min_lag_steps", "name"),
            {"min_lag_steps": 0, "name": None},
        ),
        (
            DelayedExponentialKernel,
            ("delay", "rate_lambda", "max_lag_steps", "dt", "min_lag_steps", "name"),
            {"min_lag_steps": 0, "name": None},
        ),
        (
            LogNormalKernel,
            ("log_mu", "log_sigma", "max_lag_steps", "dt", "min_lag_steps", "name"),
            {"min_lag_steps": 0, "name": None},
        ),
        (
            ErlangKernel,
            ("shape_k", "rate_beta", "max_lag_steps", "dt", "min_lag_steps", "name"),
            {"min_lag_steps": 0, "name": None},
        ),
    ]

    for kernel_cls, expected_parameters, expected_defaults in constructor_cases:
        parameters = inspect.signature(kernel_cls).parameters
        assert tuple(parameters) == expected_parameters
        for parameter_name, expected_default in expected_defaults.items():
            assert parameters[parameter_name].default == expected_default


def test_parametric_kernel_init_helper_matches_explicit_kernel_weights() -> None:
    dt = 1.0
    min_lag_steps = 0
    max_lag_steps = 6

    kernel_cases = [
        (
            "gamma",
            {"shape_alpha": 2.0, "rate_beta": 0.4},
            GammaKernel(
                shape_alpha=2.0,
                rate_beta=0.4,
                min_lag_steps=min_lag_steps,
                max_lag_steps=max_lag_steps,
                dt=dt,
            ),
        ),
        (
            "exponential",
            {"rate_lambda": 0.7},
            ExponentialKernel(
                rate_lambda=0.7,
                min_lag_steps=min_lag_steps,
                max_lag_steps=max_lag_steps,
                dt=dt,
            ),
        ),
        (
            "delayed_exponential",
            {"delay": 1.5, "rate_lambda": 0.6},
            DelayedExponentialKernel(
                delay=1.5,
                rate_lambda=0.6,
                min_lag_steps=min_lag_steps,
                max_lag_steps=max_lag_steps,
                dt=dt,
            ),
        ),
        (
            "lognormal",
            {"log_mu": 0.2, "log_sigma": 0.5},
            LogNormalKernel(
                log_mu=0.2,
                log_sigma=0.5,
                min_lag_steps=min_lag_steps,
                max_lag_steps=max_lag_steps,
                dt=dt,
            ),
        ),
        (
            "erlang",
            {"shape_k": 3, "rate_beta": 0.8},
            ErlangKernel(
                shape_k=3,
                rate_beta=0.8,
                min_lag_steps=min_lag_steps,
                max_lag_steps=max_lag_steps,
                dt=dt,
            ),
        ),
    ]

    for family, parameters, explicit_kernel in kernel_cases:
        learned = _make_parametric_learned_kernel(
            family=family,
            dt=dt,
            min_lag_steps=min_lag_steps,
            max_lag_steps=max_lag_steps,
            parameters=parameters,
        )
        assert learned.lag_steps == explicit_kernel.lag_steps
        assert learned.weights == pytest.approx(explicit_kernel.weights, abs=1e-12)


def test_erlang_shape_k_remains_integer_through_learned_parametric_summary() -> None:
    validated = _validated_parametric_parameters(
        family="erlang",
        parameters={"shape_k": 4.0, "rate_beta": 0.8},
    )
    assert isinstance(validated["shape_k"], int)

    kernel = _make_parametric_learned_kernel(
        family="erlang",
        dt=1.0,
        min_lag_steps=0,
        max_lag_steps=5,
        parameters={"shape_k": 4.0, "rate_beta": 0.8},
    )
    summary = summarize_parametric_kernel(
        kernel=kernel,
        family="erlang",
        parameters={"shape_k": 4.0, "rate_beta": 0.8},
    )
    parametric_parameters = summary["parametric_parameters"]
    assert isinstance(parametric_parameters, dict)
    shape_k = parametric_parameters["shape_k"]
    assert isinstance(shape_k, int)


def test_erlang_constructor_rejects_integral_float_shape_k() -> None:
    with pytest.raises(ValueError, match="shape_k must be a positive integer"):
        ErlangKernel(
            shape_k=3.0,  # type: ignore[arg-type]
            rate_beta=0.8,
            min_lag_steps=0,
            max_lag_steps=5,
            dt=1.0,
        )


def test_parametric_summary_preserves_expected_keys() -> None:
    kernel = GammaKernel(
        shape_alpha=2.0,
        rate_beta=0.4,
        min_lag_steps=0,
        max_lag_steps=6,
        dt=1.0,
    )
    summary = kernel.summary()
    expected_keys = set(Kernel.summary(kernel).keys()) | {
        "parametric_family",
        "parametric_parameters",
    }
    assert set(summary.keys()) == expected_keys
    assert summary["parametric_family"] == "gamma"
    assert summary["parametric_parameters"] == {
        "shape_alpha": 2.0,
        "rate_beta": 0.4,
    }


def test_all_explicit_parametric_summaries_include_family_and_parameters() -> None:
    kernel_cases = [
        (
            GammaKernel(
                shape_alpha=2.0, rate_beta=0.5, min_lag_steps=0, max_lag_steps=5, dt=1.0
            ),
            "gamma",
            {"shape_alpha", "rate_beta"},
        ),
        (
            ExponentialKernel(rate_lambda=0.7, min_lag_steps=0, max_lag_steps=5, dt=1.0),
            "exponential",
            {"rate_lambda"},
        ),
        (
            DelayedExponentialKernel(
                delay=1.0, rate_lambda=0.6, min_lag_steps=0, max_lag_steps=5, dt=1.0
            ),
            "delayed_exponential",
            {"delay", "rate_lambda"},
        ),
        (
            LogNormalKernel(
                log_mu=0.2, log_sigma=0.5, min_lag_steps=0, max_lag_steps=5, dt=1.0
            ),
            "lognormal",
            {"log_mu", "log_sigma"},
        ),
        (
            ErlangKernel(shape_k=3, rate_beta=0.8, min_lag_steps=0, max_lag_steps=5, dt=1.0),
            "erlang",
            {"shape_k", "rate_beta"},
        ),
    ]

    for kernel, expected_family, expected_parameter_keys in kernel_cases:
        summary = kernel.summary()
        assert summary["parametric_family"] == expected_family
        params = summary["parametric_parameters"]
        assert isinstance(params, dict)
        assert set(params) == expected_parameter_keys
        if expected_family == "erlang":
            assert isinstance(params["shape_k"], int)


def test_parametric_family_registry_uses_python_310_safe_type_alias() -> None:
    source = Path("src/rtdfeatures/kernels/parametric.py").read_text(encoding="utf-8")
    assert "ParametricValue = float | int" in source
    assert "type ParametricValue =" not in source
