"""legacy milestone tests for shared learner result contract."""

from __future__ import annotations

import pytest

from rtdfeatures.diagnostics import (
    BaselineComparison,
    FitDiagnostics,
    IdentifiabilityReport,
    KernelFitResult,
    SharedKernelFitResult,
    SharedPairFitResult,
)
from rtdfeatures.kernels import UniformKernel


def _kernel_fit_result(name: str, validation_loss: float) -> KernelFitResult:
    kernel = UniformKernel(max_lag_steps=3, min_lag_steps=0, dt=60.0, name=name)
    diagnostics = FitDiagnostics(
        train_loss=validation_loss * 0.9,
        validation_loss=validation_loss,
        input_variance=1.0,
        target_variance=1.0,
        kernel_weight_sum=1.0,
        mean_lag=90.0,
        p50_lag=60.0,
        p90_lag=180.0,
        tail_mass=0.25,
        boundary_mass_fraction=0.5,
    )
    identifiability = IdentifiabilityReport(warnings=(), is_reliable=True)
    baselines = BaselineComparison(
        no_lag_validation_loss=2.0,
        best_single_lag_validation_loss=1.5,
        learned_validation_loss=validation_loss,
    )
    return KernelFitResult(
        kernel=kernel,
        fit_diagnostics=diagnostics,
        identifiability_report=identifiability,
        baseline_comparison=baselines,
    )


def test_pair_identifiers_are_deterministic() -> None:
    assert SharedKernelFitResult.make_pair_id("feed_a", "prod_a") == "feed_a->prod_a"
    assert SharedKernelFitResult.make_pair_id("feed_a", "prod_a", pair_name="Pair-1") == "Pair-1"


def test_shared_result_preserves_pair_ordering() -> None:
    pair1 = SharedPairFitResult(
        pair_id="feed_b->prod_b",
        input_col="feed_b",
        target_col="prod_b",
        fit_result=_kernel_fit_result(name="feed_b->prod_b", validation_loss=0.9),
    )
    pair2 = SharedPairFitResult(
        pair_id="feed_a->prod_a",
        input_col="feed_a",
        target_col="prod_a",
        fit_result=_kernel_fit_result(name="feed_a->prod_a", validation_loss=0.7),
    )
    shared = SharedKernelFitResult(pairs=(pair1, pair2))

    assert shared.pair_ids() == ("feed_b->prod_b", "feed_a->prod_a")


def test_per_pair_diagnostics_remain_accessible() -> None:
    fit = _kernel_fit_result(name="feed_a->prod_a", validation_loss=0.42)
    shared = SharedKernelFitResult(
        pairs=(
            SharedPairFitResult(
                pair_id="feed_a->prod_a",
                input_col="feed_a",
                target_col="prod_a",
                fit_result=fit,
            ),
        )
    )

    pair_fit = shared.get_pair_result("feed_a->prod_a")
    assert pair_fit.fit_diagnostics.validation_loss == 0.42

    summary = shared.summary()
    assert summary["feed_a->prod_a"]["status"] == "ok"
    assert summary["feed_a->prod_a"]["validation_loss"] == 0.42


def test_failed_pair_is_represented_clearly() -> None:
    shared = SharedKernelFitResult(
        pairs=(
            SharedPairFitResult(
                pair_id="feed_a->prod_a",
                input_col="feed_a",
                target_col="prod_a",
                fit_result=None,
                error="Not enough valid lag windows after missing-value filtering.",
            ),
        )
    )

    assert shared.get_pair("feed_a->prod_a").succeeded is False
    summary = shared.summary()
    assert summary["feed_a->prod_a"]["status"] == "failed"
    assert "Not enough valid lag windows" in str(summary["feed_a->prod_a"]["error"])


def test_shared_result_rejects_duplicate_pair_ids() -> None:
    pair1 = SharedPairFitResult(
        pair_id="dup",
        input_col="feed_a",
        target_col="prod_a",
        fit_result=_kernel_fit_result(name="dup1", validation_loss=0.7),
    )
    pair2 = SharedPairFitResult(
        pair_id="dup",
        input_col="feed_b",
        target_col="prod_b",
        fit_result=_kernel_fit_result(name="dup2", validation_loss=0.8),
    )
    with pytest.raises(ValueError, match="requires unique pair_id values"):
        SharedKernelFitResult(pairs=(pair1, pair2))
