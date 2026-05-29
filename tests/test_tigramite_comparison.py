"""Phase 5 tests for Tigramite-kernel support/mass comparison helper."""

from __future__ import annotations

import json

import pytest

from rtdfeatures.integrations.tigramite import (
    LagCandidateDescriptor,
    compare_kernel_to_tigramite_links,
)
from rtdfeatures.integrations.tigramite.helpers import (
    TIGRAMITE_KERNEL_SUPPORT_THRESHOLD_BREACH,
)
from rtdfeatures.kernels import Kernel


def _kernel() -> Kernel:
    return Kernel(
        weights=(0.10, 0.20, 0.30, 0.40),
        lag_steps=(1, 2, 3, 4),
        dt=1.0,
        min_lag_steps=1,
        max_lag_steps=4,
        name="phase5",
    )


def _candidates() -> list[LagCandidateDescriptor]:
    return [
        LagCandidateDescriptor(source_col="x", target_col="y", lag_steps=2, mark="->"),
        LagCandidateDescriptor(source_col="x", target_col="y", lag_steps=4, mark="->"),
    ]


def test_phase5_overlap_and_inside_mass() -> None:
    comparison = compare_kernel_to_tigramite_links(_kernel(), _candidates())

    assert comparison.kernel_support_lag_steps == (1, 2, 3, 4)
    assert comparison.candidate_support_lag_steps == (2, 4)
    assert comparison.overlap_lag_steps == (2, 4)
    assert comparison.kernel_mass_inside_candidate_support == pytest.approx(0.60)


def test_phase5_outside_support_warning() -> None:
    with pytest.warns(UserWarning) as caught:
        comparison = compare_kernel_to_tigramite_links(
            _kernel(),
            _candidates(),
            outside_support_warning_threshold=0.20,
        )

    messages = [str(item.message) for item in caught]
    assert any(
        message.startswith(f"{TIGRAMITE_KERNEL_SUPPORT_THRESHOLD_BREACH}:") for message in messages
    )
    assert not any(
        message.startswith("TIGRAMITE_VALUES_NOT_KERNEL_WEIGHTS:")
        for message in messages
    )
    assert comparison.kernel_mass_outside_candidate_support == pytest.approx(0.40)
    assert comparison.outside_support_exceeds_threshold is True


def test_phase5_configurable_threshold() -> None:
    comparison = compare_kernel_to_tigramite_links(
        _kernel(),
        _candidates(),
        outside_support_warning_threshold=0.50,
    )

    assert comparison.outside_support_warning_threshold == pytest.approx(0.50)
    assert comparison.outside_support_exceeds_threshold is False


def test_phase5_no_kernel_mutation() -> None:
    kernel = _kernel()
    before_weights = kernel.weights
    before_lag_steps = kernel.lag_steps

    _ = compare_kernel_to_tigramite_links(kernel, _candidates())

    assert kernel.weights == before_weights
    assert kernel.lag_steps == before_lag_steps


def test_phase5_output_serializable_and_row_convertible() -> None:
    comparison = compare_kernel_to_tigramite_links(_kernel(), [2, 4])

    encoded = json.dumps(comparison.to_dict())
    decoded = json.loads(encoded)
    assert decoded["overlap_lag_steps"] == [2, 4]

    row = comparison.to_row()
    assert row["overlap_count"] == 2
    assert row["kernel_mass_inside_candidate_support"] == pytest.approx(0.60)
