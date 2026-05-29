"""legacy milestone tests for kernel objects and result contracts."""

import pytest

from rtdfeatures.diagnostics import (
    BaselineComparison,
    FitDiagnostics,
    IdentifiabilityReport,
    KernelFitResult,
    SharedKernelFitResult,
    SharedPairFitResult,
)
from rtdfeatures.kernels import FixedDelayKernel, LearnedKernel, UniformKernel


def test_validate_accepts_well_formed_kernel() -> None:
    kernel = LearnedKernel(
        weights=(0.2, 0.3, 0.5),
        lag_steps=(0, 1, 2),
        dt=1.0,
        min_lag_steps=0,
        max_lag_steps=2,
        name="learned",
    )
    kernel.validate()


@pytest.mark.parametrize(
    ("weights", "lag_steps", "dt", "min_lag_steps", "max_lag_steps", "expected_message"),
    [
        ((0.5, -0.1, 0.6), (0, 1, 2), 1.0, 0, 2, "non-negative"),
        ((0.2, 0.2, 0.2), (0, 1, 2), 1.0, 0, 2, "approximately 1.0"),
        ((0.2, 0.3, 0.5), (0, 2, 1), 1.0, 0, 2, "sorted"),
        ((0.2, 0.3, 0.5), (0, 1, 3), 1.0, 0, 2, "within"),
        ((0.2, 0.3, 0.5), (0, 1, 2), 0.0, 0, 2, "strictly positive"),
        ((), (0, 1, 2), 1.0, 0, 2, "at least one weight"),
        ((0.4, 0.6), (0,), 1.0, 0, 2, "must have equal length"),
        ((0.4, 0.6), (0, 1), 1.0, -1, 2, "must be non-negative"),
        ((0.4, 0.6), (0, 1), 1.0, 0, -1, "must be >="),
        ((0.1, 0.2, 0.3, 0.4), (0, 1, 1, 2), 1.0, 0, 2, "duplicates"),
    ],
)
def test_validate_rejects_invalid_kernel_inputs(
    weights: tuple[float, ...],
    lag_steps: tuple[int, ...],
    dt: float,
    min_lag_steps: int,
    max_lag_steps: int,
    expected_message: str,
) -> None:
    kernel = LearnedKernel(
        weights=weights,
        lag_steps=lag_steps,
        dt=dt,
        min_lag_steps=min_lag_steps,
        max_lag_steps=max_lag_steps,
    )
    with pytest.raises(ValueError, match=expected_message):
        kernel.validate()


def test_percentile_rejects_out_of_range_q() -> None:
    kernel = LearnedKernel(
        weights=(0.5, 0.5), lag_steps=(0, 1), dt=1.0, min_lag_steps=0, max_lag_steps=1,
    )
    with pytest.raises(ValueError, match="must be in"):
        kernel.percentile(-0.1)
    with pytest.raises(ValueError, match="must be in"):
        kernel.percentile(1.1)


def test_percentile_returns_last_step_when_q_exceeds_cumulative() -> None:
    kernel = LearnedKernel(
        weights=(0.3, 0.3, 0.4), lag_steps=(0, 1, 2), dt=2.0,
        min_lag_steps=0, max_lag_steps=2,
    )
    assert kernel.percentile(0.95) == pytest.approx(4.0)


def test_fixed_delay_kernel_rejects_out_of_range_delay() -> None:
    with pytest.raises(ValueError, match="delay_steps"):
        FixedDelayKernel(delay_steps=5, max_lag_steps=3, dt=1.0)


def test_uniform_kernel_rejects_max_less_than_min() -> None:
    with pytest.raises(ValueError, match="max_lag_steps"):
        UniformKernel(max_lag_steps=1, min_lag_steps=3, dt=1.0)


def test_percentile_and_tail_mass_are_stable_on_known_kernel() -> None:
    kernel = LearnedKernel(
        weights=(0.1, 0.2, 0.3, 0.4),
        lag_steps=(0, 1, 2, 3),
        dt=2.0,
        min_lag_steps=0,
        max_lag_steps=3,
    )
    assert kernel.percentile(0.5) == pytest.approx(4.0)
    assert kernel.percentile(0.9) == pytest.approx(6.0)
    assert kernel.tail_mass(4.0) == pytest.approx(0.7)


def test_fixed_delay_kernel_has_one_hot_pattern() -> None:
    kernel = FixedDelayKernel(delay_steps=2, min_lag_steps=0, max_lag_steps=4, dt=1.0)
    assert kernel.lag_steps == (0, 1, 2, 3, 4)
    assert kernel.weights == (0.0, 0.0, 1.0, 0.0, 0.0)
    kernel.validate()


def test_uniform_kernel_has_equal_weights() -> None:
    kernel = UniformKernel(min_lag_steps=1, max_lag_steps=3, dt=0.5)
    assert kernel.lag_steps == (1, 2, 3)
    assert kernel.weights == pytest.approx((1 / 3, 1 / 3, 1 / 3))
    kernel.validate()


def test_fit_provenance_lives_on_kernel_fit_result_not_kernel() -> None:
    kernel = LearnedKernel(
        weights=(0.4, 0.6),
        lag_steps=(0, 1),
        dt=1.0,
        min_lag_steps=0,
        max_lag_steps=1,
    )
    fit = KernelFitResult(
        kernel=kernel,
        fit_diagnostics=FitDiagnostics(
            train_loss=0.5,
            validation_loss=0.6,
            input_variance=1.0,
            target_variance=1.0,
            kernel_weight_sum=1.0,
            mean_lag=0.6,
            p50_lag=1.0,
            p90_lag=1.0,
            tail_mass=0.6,
            boundary_mass_fraction=0.4,
        ),
        identifiability_report=IdentifiabilityReport(warnings=(), is_reliable=True),
        baseline_comparison=BaselineComparison(
            no_lag_validation_loss=1.0,
            best_single_lag_validation_loss=0.8,
            learned_validation_loss=0.6,
        ),
        fit_provenance={"seed": 7},
    )
    assert fit.fit_provenance == {"seed": 7}
    assert not hasattr(kernel, "fit_provenance")


def test_kernel_summary_round_trips_through_json() -> None:
    import json

    from rtdfeatures.kernels import FixedDelayKernel

    kernel = FixedDelayKernel(delay_steps=1, max_lag_steps=3, dt=1.0, name="test")
    summary = kernel.summary()
    round_tripped = json.loads(json.dumps(summary, default=str))
    for k in ("name", "n_lags", "dt", "mean_lag", "p50_lag"):
        assert summary[k] == round_tripped[k], f"Mismatch for {k}"


# -
# SharedKernelFitResult contract gaps (fit.py coverage)
# -


def _shared_pair_ok(pair_id: str) -> SharedPairFitResult:
    kernel = LearnedKernel(
        weights=(0.4, 0.6), lag_steps=(0, 1), dt=1.0,
        min_lag_steps=0, max_lag_steps=1,
    )
    fit = KernelFitResult(
        kernel=kernel,
        fit_diagnostics=FitDiagnostics(
            train_loss=0.5, validation_loss=0.6, input_variance=1.0,
            target_variance=1.0, kernel_weight_sum=1.0,
            mean_lag=0.6, p50_lag=1.0, p90_lag=1.0, tail_mass=0.6,
            boundary_mass_fraction=0.4,
        ),
        identifiability_report=IdentifiabilityReport(warnings=(), is_reliable=True),
        baseline_comparison=BaselineComparison(
            no_lag_validation_loss=1.0, best_single_lag_validation_loss=0.8,
            learned_validation_loss=0.6,
        ),
    )
    return SharedPairFitResult(
        pair_id=pair_id, input_col="in", target_col="out",
        fit_result=fit, error=None,
    )


def _shared_pair_failed(pair_id: str) -> SharedPairFitResult:
    return SharedPairFitResult(
        pair_id=pair_id, input_col="in", target_col="out",
        fit_result=None, error="something broke",
    )


def test_shared_fit_result_duplicate_pair_ids_raises() -> None:
    p1 = _shared_pair_ok("same")
    p2 = _shared_pair_ok("same")
    with pytest.raises(ValueError, match="duplicate"):
        SharedKernelFitResult(pairs=(p1, p2))


def test_shared_fit_result_make_pair_id_empty_name_raises() -> None:
    with pytest.raises(ValueError, match="not be empty"):
        SharedKernelFitResult.make_pair_id("in", "out", pair_name="   ")


def test_shared_fit_result_get_pair_unknown_raises() -> None:
    result = SharedKernelFitResult(pairs=(_shared_pair_ok("p1"),))
    with pytest.raises(KeyError, match="Unknown"):
        result.get_pair("nonexistent")


def test_shared_fit_result_get_pair_result_failed_pair_raises() -> None:
    result = SharedKernelFitResult(pairs=(_shared_pair_failed("fail"),))
    with pytest.raises(ValueError, match="did not produce"):
        result.get_pair_result("fail")


def test_shared_fit_result_to_kernels_failed_pair_raises() -> None:
    result = SharedKernelFitResult(pairs=(_shared_pair_failed("fail"),))
    with pytest.raises(ValueError, match="failed"):
        result.to_kernels()


def test_shared_fit_result_to_kernels_empty_name_raises() -> None:
    result = SharedKernelFitResult(pairs=(_shared_pair_ok(""),))
    with pytest.raises(ValueError, match="non-empty"):
        result.to_kernels()


def test_shared_fit_result_to_kernels_name_collision_raises() -> None:
    result = SharedKernelFitResult(pairs=(_shared_pair_ok("p1"), _shared_pair_ok("p2")))
    with pytest.raises(ValueError, match="collision"):
        result.to_kernels(names={"p1": "dup", "p2": "dup"})


def test_shared_fit_result_summary_contains_failed_pair() -> None:
    result = SharedKernelFitResult(pairs=(_shared_pair_failed("x"),))
    summary = result.summary()
    assert summary["x"]["status"] == "failed"
    assert summary["x"]["error"] == "something broke"


def test_shared_fit_result_summary_contains_ok_pair() -> None:
    result = SharedKernelFitResult(pairs=(_shared_pair_ok("y"),))
    summary = result.summary()
    assert summary["y"]["status"] == "ok"
    assert "kernel" in summary["y"]
