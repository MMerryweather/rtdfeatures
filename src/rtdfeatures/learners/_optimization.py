"""Torch optimization helpers shared across learner implementations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import inf

import torch

from rtdfeatures.learners._base import PreparedFitData


@dataclass(frozen=True)
class TorchFitData:
    train_x: torch.Tensor
    train_y: torch.Tensor
    valid_x: torch.Tensor
    valid_y: torch.Tensor


@dataclass
class BestLossState:
    validation_loss: float = inf
    train_loss: float = inf


def set_torch_seed(seed: int | None) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)


def make_torch_fit_data(prepared: PreparedFitData) -> TorchFitData:
    return TorchFitData(
        train_x=torch.as_tensor(prepared.x_train_scaled, dtype=torch.float32),
        train_y=torch.as_tensor(prepared.y_train_scaled, dtype=torch.float32),
        valid_x=torch.as_tensor(prepared.x_valid_scaled, dtype=torch.float32),
        valid_y=torch.as_tensor(prepared.y_valid_scaled, dtype=torch.float32),
    )


def torch_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    loss: str,
    huber_delta: float,
) -> torch.Tensor:
    if loss == "mse":
        return torch.mean((prediction - target) ** 2)
    return torch.nn.functional.huber_loss(
        prediction,
        target,
        delta=huber_delta,
        reduction="mean",
    )


def torch_loss_value(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    loss: str,
    huber_delta: float,
) -> float:
    return float(
        torch_loss(
            prediction,
            target,
            loss=loss,
            huber_delta=huber_delta,
        ).item()
    )


def update_best_loss_state(
    *,
    state: BestLossState,
    validation_loss: float,
    train_loss: float,
) -> bool:
    if validation_loss < state.validation_loss:
        state.validation_loss = validation_loss
        state.train_loss = train_loss
        return True
    return False


def smoothness_term(weights: torch.Tensor) -> torch.Tensor:
    if weights.numel() <= 1:
        return torch.zeros((), dtype=weights.dtype, device=weights.device)
    return torch.mean((weights[1:] - weights[:-1]) ** 2)


def optimize_parametric_weights(
    *,
    optimizer: torch.optim.Optimizer,
    max_epochs: int,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    valid_x: torch.Tensor,
    valid_y: torch.Tensor,
    loss: str,
    huber_delta: float,
    smoothness_penalty: float,
    forward: Callable[[], tuple[torch.Tensor, dict[str, torch.Tensor]]],
    failure_message: str,
) -> tuple[BestLossState, dict[str, float], torch.Tensor]:
    best_loss = BestLossState()
    best_weights: torch.Tensor | None = None
    best_parameters: dict[str, float] | None = None
    for _ in range(max_epochs):
        optimizer.zero_grad(set_to_none=True)
        weights, parameters = forward()
        train_pred = train_x @ weights
        data_loss = torch_loss(
            train_pred,
            train_y,
            loss=loss,
            huber_delta=huber_delta,
        )
        total_loss = data_loss + smoothness_penalty * smoothness_term(weights)
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            valid_pred = valid_x @ weights
            valid_loss = torch_loss_value(
                valid_pred,
                valid_y,
                loss=loss,
                huber_delta=huber_delta,
            )
            if update_best_loss_state(
                state=best_loss,
                validation_loss=valid_loss,
                train_loss=float(data_loss.item()),
            ):
                best_weights = weights.detach().clone()
                best_parameters = {
                    name: float(value.item()) for name, value in parameters.items()
                }
    if best_weights is None or best_parameters is None:
        raise RuntimeError(failure_message)
    return best_loss, best_parameters, best_weights
