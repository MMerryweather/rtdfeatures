"""legacy milestone tests for parametric kernel object support."""

from __future__ import annotations

import math

import polars as pl
import pytest

import rtdfeatures
from rtdfeatures.features import KernelFeatureBuilder
from rtdfeatures.kernels import (
    DelayedExponentialKernel,
    ErlangKernel,
    ExponentialKernel,
    FixedDelayKernel,
    GammaKernel,
    LogNormalKernel,
)
from rtdfeatures.kernels.parametric import (
    _make_parametric_learned_kernel,
    build_parametric_fit_provenance,
    discrete_delayed_exponential_weights,
    discrete_erlang_weights,
    discrete_exponential_weights,
    discrete_lognormal_weights,
    summarize_parametric_kernel,
)


def test_exponential_discrete_weights_are_non_negative_and_sum_to_one() -> None:
    kernel = _make_parametric_learned_kernel(
        family="exponential",
        dt=1.0,
        min_lag_steps=0,
        max_lag_steps=3,
        parameters={"rate_lambda": math.log(2.0)},
        name="exp",
    )
    assert all(weight >= 0.0 for weight in kernel.weights)
    assert sum(kernel.weights) == pytest.approx(1.0, abs=1e-6)
    assert kernel.lag_steps == (0, 1, 2, 3)


def test_gamma_discrete_weights_peak_near_expected_mode() -> None:
    kernel = _make_parametric_learned_kernel(
        family="gamma",
        dt=1.0,
        min_lag_steps=1,
        max_lag_steps=6,
        parameters={"shape_alpha": 3.0, "rate_beta": 1.0},
        name="gamma",
    )
    mode_step = kernel.lag_steps[kernel.weights.index(max(kernel.weights))]
    assert mode_step == 2
    assert sum(kernel.weights) == pytest.approx(1.0, abs=1e-6)


def test_delayed_exponential_weights_respect_delay_cutoff() -> None:
    weights = discrete_delayed_exponential_weights(
        delay=2.0,
        rate_lambda=1.0,
        lag_steps=(0, 1, 2, 3, 4),
        dt=1.0,
    )
    assert weights[0] == pytest.approx(0.0)
    assert weights[1] == pytest.approx(0.0)
    assert all(weight >= 0.0 for weight in weights)
    assert sum(weights) == pytest.approx(1.0, abs=1e-6)


def test_lognormal_weights_have_positive_skew() -> None:
    weights = discrete_lognormal_weights(
        log_mu=math.log(2.0),
        log_sigma=0.5,
        lag_steps=(1, 2, 3, 4, 5, 6),
        dt=1.0,
    )
    mode_step = (1, 2, 3, 4, 5, 6)[weights.index(max(weights))]
    assert mode_step in {1, 2}
    assert all(weight >= 0.0 for weight in weights)
    assert sum(weights) == pytest.approx(1.0, abs=1e-6)


def test_erlang_weights_peak_near_expected_mode() -> None:
    lag_steps = (0, 1, 2, 3, 4, 5)
    weights = discrete_erlang_weights(
        shape_k=3,
        rate_beta=1.0,
        lag_steps=lag_steps,
        dt=1.0,
    )
    mode_step = lag_steps[weights.index(max(weights))]
    assert mode_step == 2
    assert all(weight >= 0.0 for weight in weights)
    assert sum(weights) == pytest.approx(1.0, abs=1e-6)


def test_parametric_validation_rejects_invalid_parameter_or_lag_inputs() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        _make_parametric_learned_kernel(
            family="exponential",
            dt=1.0,
            min_lag_steps=0,
            max_lag_steps=2,
            parameters={"rate_lambda": 0.0},
        )
    with pytest.raises(ValueError, match=">= min_lag_steps"):
        _make_parametric_learned_kernel(
            family="exponential",
            dt=1.0,
            min_lag_steps=3,
            max_lag_steps=1,
            parameters={"rate_lambda": 1.0},
        )
    with pytest.raises(ValueError, match="strictly positive"):
        discrete_exponential_weights(
            rate_lambda=float("inf"),
            lag_steps=(0, 1, 2),
            dt=1.0,
        )
    with pytest.raises(ValueError, match="dt must be finite and strictly positive"):
        discrete_exponential_weights(
            rate_lambda=1.0,
            lag_steps=(0, 1, 2),
            dt=0.0,
        )
    with pytest.raises(ValueError, match="lag_steps must be non-empty"):
        discrete_exponential_weights(
            rate_lambda=1.0,
            lag_steps=(),
            dt=1.0,
        )
    with pytest.raises(ValueError, match="lag_steps must be strictly increasing"):
        discrete_exponential_weights(
            rate_lambda=1.0,
            lag_steps=(0, 2, 2),
            dt=1.0,
        )
    with pytest.raises(ValueError, match="lag_steps must be non-negative"):
        discrete_exponential_weights(
            rate_lambda=1.0,
            lag_steps=(-1, 0, 1),
            dt=1.0,
        )
    with pytest.raises(ValueError, match="delay must be finite and non-negative"):
        discrete_delayed_exponential_weights(
            delay=-0.1,
            rate_lambda=1.0,
            lag_steps=(0, 1, 2),
            dt=1.0,
        )
    with pytest.raises(ValueError, match="log_sigma must be finite and strictly positive"):
        discrete_lognormal_weights(
            log_mu=0.0,
            log_sigma=0.0,
            lag_steps=(1, 2, 3),
            dt=1.0,
        )
    with pytest.raises(ValueError, match="shape_k must be a positive integer"):
        discrete_erlang_weights(
            shape_k=0,
            rate_beta=1.0,
            lag_steps=(0, 1, 2),
            dt=1.0,
        )
    with pytest.raises(ValueError, match="shape_k must be a positive integer"):
        discrete_erlang_weights(
            shape_k=2.5,  # type: ignore[arg-type]
            rate_beta=1.0,
            lag_steps=(0, 1, 2),
            dt=1.0,
        )


def test_zero_mass_and_non_finite_conversions_raise_value_error() -> None:
    with pytest.raises(ValueError, match="zero total mass"):
        discrete_lognormal_weights(
            log_mu=0.0,
            log_sigma=1.0,
            lag_steps=(0,),
            dt=1.0,
        )
    with pytest.raises(ValueError, match="must be finite"):
        discrete_lognormal_weights(
            log_mu=float("inf"),
            log_sigma=1.0,
            lag_steps=(1, 2),
            dt=1.0,
        )


def test_feature_builder_consumes_parametric_kernel_without_special_handling() -> None:
    kernel = _make_parametric_learned_kernel(
        family="exponential",
        dt=1.0,
        min_lag_steps=0,
        max_lag_steps=2,
        parameters={"rate_lambda": math.log(2.0)},
        name="exp",
    )
    builder = KernelFeatureBuilder(
        kernels={"exp_kernel": kernel},
        time_col="t",
        numeric_cols=["x"],
    )
    df = pl.DataFrame({"t": [0, 1, 2, 3, 4], "x": [1.0, 2.0, 4.0, 8.0, 16.0]})
    features = builder.transform(df)
    assert features.get_column("exp_kernel_num_x_wmean")[2] == pytest.approx(3.0)
    assert features.get_column("exp_kernel_num_x_wsum")[2] == pytest.approx(3.0)


def test_summary_and_provenance_include_parametric_metadata() -> None:
    kernel = _make_parametric_learned_kernel(
        family="gamma",
        dt=1.0,
        min_lag_steps=1,
        max_lag_steps=4,
        parameters={"shape_alpha": 2.0, "rate_beta": 0.5},
        name="gamma",
    )
    summary = summarize_parametric_kernel(
        kernel=kernel,
        family="gamma",
        parameters={"shape_alpha": 2.0, "rate_beta": 0.5},
    )
    assert summary["parametric_family"] == "gamma"
    assert summary["parametric_parameters"] == {"shape_alpha": 2.0, "rate_beta": 0.5}

    provenance = build_parametric_fit_provenance(
        family="gamma",
        parameters={"shape_alpha": 2.0, "rate_beta": 0.5},
        initial_parameters={"shape_alpha": 2.0, "rate_beta": 0.5},
    )
    assert provenance["parametric_conversion_status"] == "ok"

    fixed = FixedDelayKernel(delay_steps=1, max_lag_steps=2, dt=1.0)
    assert "parametric_family" not in fixed.summary()


def test_package_root_does_not_export_low_level_helpers() -> None:
    forbidden = {
        "build_parametric_fit_provenance",
        "discrete_exponential_weights",
        "discrete_gamma_weights",
        "discrete_delayed_exponential_weights",
        "discrete_lognormal_weights",
        "discrete_erlang_weights",
        "make_parametric_learned_kernel",
        "parametric_lag_steps",
        "summarize_parametric_kernel",
    }
    assert forbidden.isdisjoint(set(rtdfeatures.__all__))


@pytest.mark.parametrize(
    ("kernel", "expected_family", "expected_params"),
    [
        (
            ExponentialKernel(rate_lambda=0.8, min_lag_steps=0, max_lag_steps=4, dt=1.0),
            "exponential",
            {"rate_lambda": 0.8},
        ),
        (
            GammaKernel(
                shape_alpha=2.0,
                rate_beta=0.6,
                min_lag_steps=1,
                max_lag_steps=6,
                dt=1.0,
            ),
            "gamma",
            {"shape_alpha": 2.0, "rate_beta": 0.6},
        ),
        (
            DelayedExponentialKernel(
                delay=2.0,
                rate_lambda=0.5,
                min_lag_steps=0,
                max_lag_steps=8,
                dt=1.0,
            ),
            "delayed_exponential",
            {"delay": 2.0, "rate_lambda": 0.5},
        ),
        (
            LogNormalKernel(
                log_mu=0.2,
                log_sigma=0.5,
                min_lag_steps=1,
                max_lag_steps=10,
                dt=1.0,
            ),
            "lognormal",
            {"log_mu": 0.2, "log_sigma": 0.5},
        ),
        (
            ErlangKernel(
                shape_k=3,
                rate_beta=0.7,
                min_lag_steps=0,
                max_lag_steps=7,
                dt=1.0,
            ),
            "erlang",
            {"shape_k": 3, "rate_beta": 0.7},
        ),
    ],
)
def test_direct_kernel_constructors_include_family_metadata_and_validate(
    kernel: object,
    expected_family: str,
    expected_params: dict[str, float | int],
) -> None:
    assert isinstance(kernel, rtdfeatures.Kernel)
    summary = kernel.summary()
    assert summary["parametric_family"] == expected_family
    assert summary["parametric_parameters"] == expected_params
    assert sum(kernel.weights) == pytest.approx(1.0, abs=1e-6)


def test_direct_kernel_constructors_are_exported_from_package_root() -> None:
    required = {
        "GammaKernel",
        "ExponentialKernel",
        "DelayedExponentialKernel",
        "LogNormalKernel",
        "ErlangKernel",
    }
    import rtdfeatures.kernels as _kern
    assert required.issubset(set(dir(_kern)))


@pytest.mark.parametrize(
    "kernel",
    [
        ExponentialKernel(rate_lambda=0.6, min_lag_steps=0, max_lag_steps=2, dt=1.0),
        GammaKernel(shape_alpha=2.0, rate_beta=0.7, min_lag_steps=1, max_lag_steps=3, dt=1.0),
        DelayedExponentialKernel(
            delay=1.0,
            rate_lambda=0.8,
            min_lag_steps=0,
            max_lag_steps=3,
            dt=1.0,
        ),
        LogNormalKernel(log_mu=0.0, log_sigma=0.5, min_lag_steps=1, max_lag_steps=3, dt=1.0),
        ErlangKernel(shape_k=2, rate_beta=1.2, min_lag_steps=0, max_lag_steps=3, dt=1.0),
    ],
)
def test_feature_builder_handles_all_direct_parametric_kernel_families(kernel: object) -> None:
    assert isinstance(kernel, rtdfeatures.Kernel)
    builder = KernelFeatureBuilder(
        kernels={"fam_kernel": kernel},
        time_col="t",
        numeric_cols=["x"],
    )
    df = pl.DataFrame({"t": [0, 1, 2, 3, 4], "x": [1.0, 2.0, 4.0, 8.0, 16.0]})
    features = builder.transform(df)
    assert "fam_kernel_num_x_wmean" in features.columns
    assert "fam_kernel_num_x_wsum" in features.columns


def test_direct_kernel_constructors_reject_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        ExponentialKernel(rate_lambda=0.0, min_lag_steps=0, max_lag_steps=3, dt=1.0)
    with pytest.raises(ValueError, match="strictly positive"):
        GammaKernel(shape_alpha=0.0, rate_beta=1.0, min_lag_steps=0, max_lag_steps=3, dt=1.0)
    with pytest.raises(ValueError, match="non-negative"):
        DelayedExponentialKernel(
            delay=-1.0,
            rate_lambda=1.0,
            min_lag_steps=0,
            max_lag_steps=3,
            dt=1.0,
        )
    with pytest.raises(ValueError, match="strictly positive"):
        LogNormalKernel(log_mu=0.0, log_sigma=0.0, min_lag_steps=1, max_lag_steps=3, dt=1.0)
    with pytest.raises(ValueError, match="positive integer"):
        ErlangKernel(shape_k=0, rate_beta=1.0, min_lag_steps=0, max_lag_steps=3, dt=1.0)


@pytest.mark.parametrize(
    ("family", "parameters", "message"),
    [
        ("exponential", {}, "missing"),
        ("exponential", {"rate_lambda": 1.0, "rate_lamda": 1.0}, "unexpected"),
        ("exponential", {"rate_lambda": -1.0}, "strictly positive"),
        ("exponential", {"rate_lambda": float("nan")}, "strictly positive"),
        ("gamma", {"shape_alpha": 1.0}, "missing"),
        ("gamma", {"shape_alpha": 1.0, "rate_beta": 2.0, "extra": 1.0}, "unexpected"),
        ("gamma", {"shape_alpha": 0.0, "rate_beta": 1.0}, "strictly positive"),
        ("gamma", {"shape_alpha": 1.0, "rate_beta": float("inf")}, "strictly positive"),
        ("delayed_exponential", {"delay": 1.0}, "missing"),
        ("delayed_exponential", {"delay": 1.0, "rate_lambda": 0.5, "extra": 1.0}, "unexpected"),
        ("delayed_exponential", {"delay": -1.0, "rate_lambda": 0.5}, "strictly positive"),
        ("delayed_exponential", {"delay": 1.0, "rate_lambda": -1.0}, "strictly positive"),
        ("lognormal", {"log_mu": 0.0}, "missing"),
        ("lognormal", {"log_mu": 0.0, "log_sigma": 0.5, "extra": 1.0}, "unexpected"),
        ("lognormal", {"log_mu": 0.0, "log_sigma": 0.0}, "strictly positive"),
        ("lognormal", {"log_mu": float("inf"), "log_sigma": 0.5}, "must be finite"),
        ("erlang", {"shape_k": 2}, "missing"),
        ("erlang", {"shape_k": 2, "rate_beta": 1.0, "extra": 1.0}, "unexpected"),
        ("erlang", {"shape_k": 0, "rate_beta": 1.0}, "positive integer"),
        ("erlang", {"shape_k": 2, "rate_beta": 0.0}, "strictly positive"),
    ],
)
def test_parametric_parameter_dicts_are_strict(
    family: str,
    parameters: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_parametric_fit_provenance(
            family=family,
            parameters=parameters,
            initial_parameters=parameters,
        )
    kernel = FixedDelayKernel(delay_steps=1, max_lag_steps=2, dt=1.0)
    with pytest.raises(ValueError, match=message):
        summarize_parametric_kernel(
            kernel=kernel,
            family=family,
            parameters=parameters,
        )


def test_all_v1_parametric_kernels_are_root_exported() -> None:
    required = {
        "GammaKernel",
        "ExponentialKernel",
        "DelayedExponentialKernel",
    }
    assert required.issubset(set(rtdfeatures.__all__))

    specialist_non_root = {
        "LogNormalKernel",
        "ErlangKernel",
    }
    assert specialist_non_root.isdisjoint(set(rtdfeatures.__all__))


def test_make_parametric_learned_kernel_supports_all_v1_families() -> None:
    from rtdfeatures.kernels.parametric import _make_parametric_learned_kernel

    k1 = _make_parametric_learned_kernel(
        family="delayed_exponential", dt=1.0, min_lag_steps=1, max_lag_steps=5,
        parameters={"delay": 1.0, "rate_lambda": 0.5}, name="de",
    )
    assert sum(k1.weights) == pytest.approx(1.0, abs=1e-6)

    k2 = _make_parametric_learned_kernel(
        family="lognormal", dt=1.0, min_lag_steps=1, max_lag_steps=5,
        parameters={"log_mu": 0.5, "log_sigma": 0.5}, name="ln",
    )
    assert sum(k2.weights) == pytest.approx(1.0, abs=1e-6)

    k3 = _make_parametric_learned_kernel(
        family="erlang", dt=1.0, min_lag_steps=0, max_lag_steps=5,
        parameters={"shape_k": 2, "rate_beta": 1.0}, name="er",
    )
    assert sum(k3.weights) == pytest.approx(1.0, abs=1e-6)
