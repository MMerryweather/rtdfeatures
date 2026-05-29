"""Tests for learner optimization helpers and inheritance structure."""

from __future__ import annotations

import inspect

import numpy as np
import polars as pl
import pytest
import torch

from rtdfeatures.learners import ExponentialKernelLearner, GammaKernelLearner, SimplexKernelLearner
from rtdfeatures.learners._base import PreparedFitData
from rtdfeatures.learners._optimization import (
    BestLossState,
    make_torch_fit_data,
    optimize_parametric_weights,
    set_torch_seed,
    smoothness_term,
    torch_loss,
    torch_loss_value,
    update_best_loss_state,
)


def test_gamma_learner_does_not_subclass_simplex_learner() -> None:
    assert not issubclass(GammaKernelLearner, SimplexKernelLearner)


def test_exponential_learner_does_not_subclass_simplex_learner() -> None:
    assert not issubclass(ExponentialKernelLearner, SimplexKernelLearner)


def test_simplex_gamma_exponential_constructor_signatures_unchanged() -> None:
    assert tuple(inspect.signature(SimplexKernelLearner).parameters) == (
        "max_lag",
        "min_lag",
        "dt",
        "loss",
        "smoothness_penalty",
        "seed",
        "validation_fraction",
        "learning_rate",
        "max_epochs",
        "huber_delta",
    )
    assert tuple(inspect.signature(GammaKernelLearner).parameters) == (
        "max_lag",
        "min_lag",
        "dt",
        "loss",
        "smoothness_penalty",
        "seed",
        "validation_fraction",
        "learning_rate",
        "max_epochs",
        "huber_delta",
        "init_shape_alpha",
        "init_rate_beta",
    )
    assert tuple(inspect.signature(ExponentialKernelLearner).parameters) == (
        "max_lag",
        "min_lag",
        "dt",
        "loss",
        "smoothness_penalty",
        "seed",
        "validation_fraction",
        "learning_rate",
        "max_epochs",
        "huber_delta",
        "init_rate_lambda",
    )


@pytest.mark.parametrize(
    "learner_cls",
    [
        SimplexKernelLearner,
        GammaKernelLearner,
        ExponentialKernelLearner,
    ],
)
def test_common_validation_rejects_invalid_loss_for_all_learners(
    learner_cls: type[SimplexKernelLearner | GammaKernelLearner | ExponentialKernelLearner],
) -> None:
    with pytest.raises(ValueError, match="loss must be either 'huber' or 'mse'"):
        learner_cls(max_lag=5, min_lag=1, loss="mae")


@pytest.mark.parametrize("invalid_validation_fraction", [0.0, 0.5, -0.1, 0.9])
def test_common_validation_rejects_invalid_validation_fraction_for_all_learners(
    invalid_validation_fraction: float,
) -> None:
    learner_classes: tuple[
        type[SimplexKernelLearner] | type[GammaKernelLearner] | type[ExponentialKernelLearner],
        ...,
    ] = (SimplexKernelLearner, GammaKernelLearner, ExponentialKernelLearner)
    for learner_cls in learner_classes:
        with pytest.raises(ValueError, match=r"validation_fraction must be in \(0.0, 0.5\)"):
            learner_cls(max_lag=5, min_lag=1, validation_fraction=invalid_validation_fraction)


def test_torch_loss_matches_expected_mse() -> None:
    prediction = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    target = torch.tensor([1.0, 4.0, 1.0], dtype=torch.float32)

    result = torch_loss(prediction, target, loss="mse", huber_delta=1.0)

    assert float(result.item()) == pytest.approx(8.0 / 3.0)


def test_torch_loss_matches_expected_huber() -> None:
    prediction = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    target = torch.tensor([1.0, 4.0, 1.0], dtype=torch.float32)

    result = torch_loss(prediction, target, loss="huber", huber_delta=1.0)

    assert float(result.item()) == pytest.approx(1.0)


def test_torch_loss_value_matches_torch_loss_item() -> None:
    prediction = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    target = torch.tensor([1.0, 4.0, 1.0], dtype=torch.float32)

    value = torch_loss_value(prediction, target, loss="huber", huber_delta=1.0)
    expected = float(torch_loss(prediction, target, loss="huber", huber_delta=1.0).item())

    assert value == pytest.approx(expected)


def test_update_best_loss_state_tracks_lowest_validation_loss() -> None:
    state = BestLossState()
    improved_first = update_best_loss_state(state=state, validation_loss=0.4, train_loss=0.2)
    improved_second = update_best_loss_state(state=state, validation_loss=0.5, train_loss=0.1)
    improved_third = update_best_loss_state(state=state, validation_loss=0.3, train_loss=0.15)

    assert improved_first is True
    assert improved_second is False
    assert improved_third is True
    assert state.validation_loss == pytest.approx(0.3)
    assert state.train_loss == pytest.approx(0.15)


def test_smoothness_term_matches_manual_second_difference_penalty() -> None:
    weights = torch.tensor([0.1, 0.2, 0.4, 0.3], dtype=torch.float32)
    expected = torch.mean((weights[1:] - weights[:-1]) ** 2)

    result = smoothness_term(weights)

    assert float(result.item()) == pytest.approx(float(expected.item()))


def test_set_torch_seed_none_is_noop_and_int_seed_is_deterministic() -> None:
    torch.manual_seed(12345)
    expected = torch.rand(4)
    torch.manual_seed(12345)
    set_torch_seed(None)
    no_op_actual = torch.rand(4)
    assert torch.equal(no_op_actual, expected)

    set_torch_seed(9876)
    seeded_first = torch.rand(5)
    set_torch_seed(9876)
    seeded_second = torch.rand(5)
    assert torch.equal(seeded_first, seeded_second)


def test_make_torch_fit_data_maps_prepared_arrays_to_float32_tensors() -> None:
    prepared = PreparedFitData(
        ordered=pl.DataFrame({"time": [0, 1], "input": [1.0, 2.0], "target": [3.0, 4.0]}),
        dt_seconds=1.0,
        min_lag_steps=1,
        max_lag_steps=2,
        input_values=np.array([1.0, 2.0], dtype=np.float64),
        target_values=np.array([3.0, 4.0], dtype=np.float64),
        design_matrix=np.array([[1.0, 0.5], [2.0, 1.5]], dtype=np.float64),
        response_vector=np.array([0.25, 0.75], dtype=np.float64),
        valid_indices=np.array([0, 1], dtype=np.int64),
        x_train=np.array([[10.0, 20.0]], dtype=np.float64),
        y_train=np.array([30.0], dtype=np.float64),
        x_valid=np.array([[40.0, 50.0]], dtype=np.float64),
        y_valid=np.array([60.0], dtype=np.float64),
        x_train_scaled=np.array([[1.5, -2.5], [3.5, -4.5]], dtype=np.float64),
        y_train_scaled=np.array([0.125, -0.25], dtype=np.float64),
        x_valid_scaled=np.array([[7.25, -8.5]], dtype=np.float64),
        y_valid_scaled=np.array([9.75], dtype=np.float64),
        no_lag_valid_scaled=np.array([1.0], dtype=np.float64),
        train_windows=2,
        validation_windows=1,
        total_valid_windows=3,
    )

    result = make_torch_fit_data(prepared)

    assert result.train_x.dtype == torch.float32
    assert result.train_y.dtype == torch.float32
    assert result.valid_x.dtype == torch.float32
    assert result.valid_y.dtype == torch.float32
    assert tuple(result.train_x.shape) == (2, 2)
    assert tuple(result.train_y.shape) == (2,)
    assert tuple(result.valid_x.shape) == (1, 2)
    assert tuple(result.valid_y.shape) == (1,)
    assert torch.allclose(
        result.train_x,
        torch.tensor([[1.5, -2.5], [3.5, -4.5]], dtype=torch.float32),
    )
    assert torch.allclose(result.train_y, torch.tensor([0.125, -0.25], dtype=torch.float32))
    assert torch.allclose(result.valid_x, torch.tensor([[7.25, -8.5]], dtype=torch.float32))
    assert torch.allclose(result.valid_y, torch.tensor([9.75], dtype=torch.float32))


def test_optimize_parametric_weights_runs_epoch_loop_and_tracks_best_state() -> None:
    train_x = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
    train_y = torch.tensor([2.0, 4.0, 6.0], dtype=torch.float32)
    valid_x = torch.tensor([[1.5], [2.5]], dtype=torch.float32)
    valid_y = torch.tensor([3.0, 5.0], dtype=torch.float32)

    rate_lambda = torch.nn.Parameter(torch.tensor(0.8, dtype=torch.float32))
    optimizer = torch.optim.SGD([rate_lambda], lr=0.1)

    def forward() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        weights = torch.nn.functional.softplus(rate_lambda).reshape(1)
        return weights, {"rate_lambda": rate_lambda}

    best_loss, best_parameters, best_weights = optimize_parametric_weights(
        optimizer=optimizer,
        max_epochs=40,
        train_x=train_x,
        train_y=train_y,
        valid_x=valid_x,
        valid_y=valid_y,
        loss="mse",
        huber_delta=1.0,
        smoothness_penalty=0.0,
        forward=forward,
        failure_message="should not fail in this deterministic test",
    )

    assert best_loss.validation_loss < np.inf
    assert best_loss.train_loss < np.inf
    assert "rate_lambda" in best_parameters
    assert isinstance(best_parameters["rate_lambda"], float)
    assert best_weights.dtype == torch.float32
    assert tuple(best_weights.shape) == (1,)
    assert float(best_weights[0].item()) == pytest.approx(2.0, abs=0.1)
