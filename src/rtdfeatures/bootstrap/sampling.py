"""Bootstrap sampling logic for blocked bootstrap over lag windows."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import polars as pl
import torch

from rtdfeatures.bootstrap.contracts import (
    BlockedBootstrapConfig,
    BootstrapIndexSplit,
    _BootstrapContext,
)
from rtdfeatures.candidates import fit_kernel_candidates
from rtdfeatures.diagnostics import (
    BOOTSTRAP_WARNING_CODES,
    BootstrapLagSummarySample,
    BootstrapParameterSample,
    BootstrapResult,
    BootstrapWeightSample,
    KernelCandidateSet,
    KernelComparisonConfig,
    KernelFamilyFitResult,
)
from rtdfeatures.kernels import Kernel, LearnedKernel
from rtdfeatures.kernels.parametric import _make_parametric_learned_kernel
from rtdfeatures.learners import ExponentialKernelLearner, GammaKernelLearner, SimplexKernelLearner
from rtdfeatures.learners._base import _inverse_softplus
from rtdfeatures.utils import lag_to_steps, resolve_and_validate_dt, validate_or_sort_time


def generate_blocked_bootstrap_splits(
    *,
    train_window_indices: Sequence[int],
    validation_window_indices: Sequence[int],
    config: BlockedBootstrapConfig,
) -> tuple[BootstrapIndexSplit, ...]:
    """Generate deterministic blocked-bootstrap index splits over training windows."""
    train_indices = tuple(int(index) for index in train_window_indices)
    validation_indices = tuple(int(index) for index in validation_window_indices)
    n_train = len(train_indices)
    if n_train == 0:
        raise ValueError("No training lag-window indices were provided for blocked bootstrap.")
    if config.block_length > n_train:
        raise ValueError(
            "block_length must be less than or equal to the number of available "
            f"training windows ({n_train})."
        )

    rng = np.random.default_rng(config.seed)
    starts_upper_bound = n_train - config.block_length + 1

    splits: list[BootstrapIndexSplit] = []
    for bootstrap_id in range(config.n_bootstrap):
        sampled_train = _sample_train_indices(
            rng=rng,
            train_indices=train_indices,
            block_length=config.block_length,
            starts_upper_bound=starts_upper_bound,
        )
        splits.append(
            BootstrapIndexSplit(
                bootstrap_id=bootstrap_id,
                train_window_indices=sampled_train,
                validation_window_indices=validation_indices,
            )
        )
    return tuple(splits)


def bootstrap_kernel_fit(
    df: pl.DataFrame,
    *,
    candidate_set: KernelCandidateSet,
    family_result: KernelFamilyFitResult,
    config: BlockedBootstrapConfig,
    order_by_time: bool = False,
) -> BootstrapResult:
    """Bootstrap one candidate fit/evaluation using shared lag-window context."""
    if family_result.candidate.candidate_id not in {
        candidate.candidate_id for candidate in candidate_set.candidates
    }:
        raise ValueError("family_result candidate must belong to candidate_set.")
    if not family_result.succeeded:
        raise ValueError("family_result must be successful before bootstrapping.")

    ordered = validate_or_sort_time(
        df, time_col=candidate_set.time_col, order_by_time=order_by_time
    )
    context = _build_bootstrap_context(
        ordered=ordered,
        candidate_set=candidate_set,
        family_result=family_result,
    )
    splits = generate_blocked_bootstrap_splits(
        train_window_indices=tuple(range(context.train_size)),
        validation_window_indices=tuple(range(context.train_size, context.total_windows)),
        config=config,
    )

    weight_samples: list[BootstrapWeightSample] = []
    parameter_samples: list[BootstrapParameterSample] = []
    lag_summary_samples: list[BootstrapLagSummarySample] = []
    failures: list[dict[str, Any]] = []

    for split in splits:
        try:
            kernel, validation_loss, params = _fit_bootstrap_kernel(
                context=context,
                family_result=family_result,
                bootstrap_id=split.bootstrap_id,
                train_indices=split.train_window_indices,
            )
            if validation_loss is not None and not math.isfinite(validation_loss):
                raise ValueError("Bootstrap validation loss is non-finite.")
            for lag_step, weight in zip(kernel.lag_steps, kernel.weights):
                weight_samples.append(
                    BootstrapWeightSample(
                        bootstrap_id=split.bootstrap_id,
                        candidate_id=family_result.candidate.candidate_id,
                        lag_step=int(lag_step),
                        lag_time=float(lag_step * kernel.dt),
                        weight=float(weight),
                    )
                )
            lag_window_tail_threshold = (
                kernel.min_lag_steps
                + SimplexKernelLearner._TAIL_MASS_FRACTION_OF_LAG_WINDOW
                * (kernel.max_lag_steps - kernel.min_lag_steps)
            ) * kernel.dt
            lag_summary_samples.append(
                BootstrapLagSummarySample(
                    bootstrap_id=split.bootstrap_id,
                    candidate_id=family_result.candidate.candidate_id,
                    mean_lag=float(kernel.mean_lag()),
                    p50_lag=float(kernel.percentile(0.5)),
                    p90_lag=float(kernel.percentile(0.9)),
                    tail_mass=float(kernel.tail_mass(lag_window_tail_threshold)),
                )
            )
            for param_name, param_value in sorted(params.items()):
                parameter_samples.append(
                    BootstrapParameterSample(
                        bootstrap_id=split.bootstrap_id,
                        candidate_id=family_result.candidate.candidate_id,
                        parameter_name=param_name,
                        parameter_value=float(param_value),
                    )
                )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            # Deliberate resilient-batch behavior: one bootstrap replicate failure is recorded.
            failures.append(
                {
                    "bootstrap_id": split.bootstrap_id,
                    "error_type": type(exc).__name__,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    warnings: list[str] = []
    if len(BOOTSTRAP_WARNING_CODES) == 0:
        raise RuntimeError("Bootstrap warning codes are unexpectedly empty.")
    if family_result.candidate.candidate_type == "parametric_learner":
        fit_provenance = family_result.fit_result.fit_provenance if family_result.fit_result else {}
        parametric_family_present = (
            isinstance(fit_provenance, dict) and fit_provenance.get("parametric_family") is not None
        )
        parametric_values_present = isinstance(fit_provenance, dict) and isinstance(
            fit_provenance.get("parametric_parameters"), dict
        )
        if parametric_family_present and not parametric_values_present:
            warnings.append("BOOTSTRAP_PARAMETER_PROVENANCE_MISSING")

    return BootstrapResult(
        n_bootstrap=config.n_bootstrap,
        n_succeeded=len({sample.bootstrap_id for sample in lag_summary_samples}),
        n_failed=len(failures),
        failures=tuple(failures),
        weight_samples=tuple(weight_samples),
        parameter_samples=tuple(parameter_samples),
        lag_summary_samples=tuple(lag_summary_samples),
        family_selection_counts={family_result.candidate.family: len(lag_summary_samples)},
        warnings=tuple(warnings),
        bootstrap_config={
            "candidate_id": family_result.candidate.candidate_id,
            "candidate_type": family_result.candidate.candidate_type,
            "family": family_result.candidate.family,
            "n_bootstrap": config.n_bootstrap,
            "block_length": config.block_length,
            "seed": config.seed,
            "validation_window_indices": context.validation_window_indices,
            "validation_window_size": len(context.validation_window_indices),
            "validation_fraction": context.validation_fraction,
            "loss": context.loss,
            "huber_delta": context.huber_delta,
        },
    )


def bootstrap_kernel_candidates(
    df: pl.DataFrame,
    *,
    candidate_set: KernelCandidateSet,
    comparison_config: KernelComparisonConfig,
    config: BlockedBootstrapConfig,
    order_by_time: bool = False,
) -> BootstrapResult:
    """Bootstrap candidate-set comparison with shared training samples."""
    ordered = validate_or_sort_time(
        df, time_col=candidate_set.time_col, order_by_time=order_by_time
    )
    comparison_result = fit_kernel_candidates(ordered, candidate_set, order_by_time=False)
    contexts: dict[str, _BootstrapContext] = {}
    for family_result in comparison_result.family_results:
        if not family_result.succeeded:
            continue
        contexts[family_result.candidate.candidate_id] = _build_bootstrap_context(
            ordered=ordered, candidate_set=candidate_set, family_result=family_result
        )
    if not contexts:
        raise ValueError("No successful candidates available for bootstrap_kernel_candidates.")

    reference_context = contexts[sorted(contexts)[0]]
    validation_indices = reference_context.validation_window_indices
    for candidate_id, context in contexts.items():
        if context.validation_window_indices != validation_indices:
            raise ValueError(
                "All successful candidates must share identical validation window indices; "
                f"candidate '{candidate_id}' differs."
            )
    if reference_context.loss != comparison_config.loss:
        raise ValueError("comparison_config.loss does not match candidate evaluation context.")
    if comparison_config.loss == "huber" and not math.isclose(
        reference_context.huber_delta, comparison_config.huber_delta, rel_tol=1e-9, abs_tol=1e-12
    ):
        raise ValueError(
            "comparison_config.huber_delta does not match candidate evaluation context."
        )
    if not math.isclose(
        reference_context.validation_fraction,
        comparison_config.validation_fraction,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "comparison_config.validation_fraction does not match candidate evaluation context."
        )

    splits = generate_blocked_bootstrap_splits(
        train_window_indices=tuple(range(reference_context.train_size)),
        validation_window_indices=tuple(
            range(reference_context.train_size, reference_context.total_windows)
        ),
        config=config,
    )

    weight_samples: list[BootstrapWeightSample] = []
    parameter_samples: list[BootstrapParameterSample] = []
    lag_summary_samples: list[BootstrapLagSummarySample] = []
    failures: list[dict[str, Any]] = []
    family_selection_counts: dict[str, int] = {}
    candidate_selection_counts: dict[str, int] = {}
    bootstrap_failures = 0

    for split in splits:
        succeeded_losses: list[tuple[KernelFamilyFitResult, float]] = []
        for family_result in comparison_result.family_results:
            candidate_id = family_result.candidate.candidate_id
            if not family_result.succeeded:
                failures.append(
                    {
                        "bootstrap_id": split.bootstrap_id,
                        "candidate_id": candidate_id,
                        "error_type": "ComparisonStageFailure",
                        "error": family_result.error or "candidate fit failed in comparison stage",
                    }
                )
                continue
            try:
                context = contexts[candidate_id]
                kernel, validation_loss, params = _fit_bootstrap_kernel(
                    context=context,
                    family_result=family_result,
                    bootstrap_id=split.bootstrap_id,
                    train_indices=split.train_window_indices,
                )
                if validation_loss is None or not math.isfinite(validation_loss):
                    raise ValueError("Bootstrap validation loss is non-finite.")
                succeeded_losses.append((family_result, float(validation_loss)))
                for lag_step, weight in zip(kernel.lag_steps, kernel.weights):
                    weight_samples.append(
                        BootstrapWeightSample(
                            bootstrap_id=split.bootstrap_id,
                            candidate_id=candidate_id,
                            lag_step=int(lag_step),
                            lag_time=float(lag_step * kernel.dt),
                            weight=float(weight),
                        )
                    )
                lag_window_tail_threshold = (
                    kernel.min_lag_steps
                    + SimplexKernelLearner._TAIL_MASS_FRACTION_OF_LAG_WINDOW
                    * (kernel.max_lag_steps - kernel.min_lag_steps)
                ) * kernel.dt
                lag_summary_samples.append(
                    BootstrapLagSummarySample(
                        bootstrap_id=split.bootstrap_id,
                        candidate_id=candidate_id,
                        mean_lag=float(kernel.mean_lag()),
                        p50_lag=float(kernel.percentile(0.5)),
                        p90_lag=float(kernel.percentile(0.9)),
                        tail_mass=float(kernel.tail_mass(lag_window_tail_threshold)),
                    )
                )
                for param_name, param_value in sorted(params.items()):
                    parameter_samples.append(
                        BootstrapParameterSample(
                            bootstrap_id=split.bootstrap_id,
                            candidate_id=candidate_id,
                            parameter_name=param_name,
                            parameter_value=float(param_value),
                        )
                    )
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                # Deliberate resilient-batch behavior: continue other candidates/bootstraps.
                failures.append(
                    {
                        "bootstrap_id": split.bootstrap_id,
                        "candidate_id": candidate_id,
                        "error_type": type(exc).__name__,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        if not succeeded_losses:
            bootstrap_failures += 1
            continue
        selected_result = _select_bootstrap_candidate(
            succeeded_losses=succeeded_losses,
            loss_tolerance_fraction=comparison_config.loss_tolerance_fraction,
        )
        family = selected_result.candidate.family
        candidate_id = selected_result.candidate.candidate_id
        family_selection_counts[family] = family_selection_counts.get(family, 0) + 1
        candidate_selection_counts[candidate_id] = (
            candidate_selection_counts.get(candidate_id, 0) + 1
        )

    total_selected = sum(candidate_selection_counts.values())
    candidate_stability = {
        candidate_id: count / total_selected
        for candidate_id, count in sorted(candidate_selection_counts.items())
        if total_selected > 0
    }
    warnings: list[str] = []
    if total_selected > 0 and len(family_selection_counts) > 1:
        top = max(family_selection_counts.values())
        if top / total_selected < 0.8:
            warnings.append("BOOTSTRAP_FAMILY_UNSTABLE")

    return BootstrapResult(
        n_bootstrap=config.n_bootstrap,
        n_succeeded=total_selected,
        n_failed=bootstrap_failures,
        failures=tuple(failures),
        weight_samples=tuple(weight_samples),
        parameter_samples=tuple(parameter_samples),
        lag_summary_samples=tuple(lag_summary_samples),
        family_selection_counts=family_selection_counts,
        warnings=tuple(warnings),
        bootstrap_config={
            "candidate_set_id": candidate_set.candidate_set_id,
            "n_bootstrap": config.n_bootstrap,
            "block_length": config.block_length,
            "seed": config.seed,
            "validation_window_indices": validation_indices,
            "validation_window_size": len(validation_indices),
            "comparison_config": comparison_config.to_dict(),
            "candidate_selection_counts": candidate_selection_counts,
            "candidate_stability": candidate_stability,
        },
    )


def _build_bootstrap_context(
    *,
    ordered: pl.DataFrame,
    candidate_set: KernelCandidateSet,
    family_result: KernelFamilyFitResult,
) -> _BootstrapContext:
    candidate = family_result.candidate
    fit_prov = family_result.fit_result.fit_provenance if family_result.fit_result else {}
    eval_prov = family_result.evaluation_provenance or {}
    validation_fraction = float(
        (fit_prov or {}).get("validation_fraction", eval_prov.get("validation_fraction", 0.2))
    )
    if validation_fraction <= 0.0 or validation_fraction >= 0.5:
        raise ValueError("validation_fraction must be in (0.0, 0.5).")

    loss = str((fit_prov or {}).get("loss", eval_prov.get("loss", "huber")))
    huber_delta = float(
        (fit_prov or {}).get("huber_delta", eval_prov.get("huber_delta", 1.0) or 1.0)
    )
    resolved_dt = resolve_and_validate_dt(ordered, time_col=candidate_set.time_col, dt=None)
    min_lag = _coerce_lag_value(candidate.min_lag)
    max_lag = _coerce_lag_value(candidate.max_lag)
    min_lag_steps = lag_to_steps(min_lag, dt=resolved_dt, param_name="min_lag")
    max_lag_steps = lag_to_steps(max_lag, dt=resolved_dt, param_name="max_lag")

    input_values = ordered.get_column(candidate_set.input_col).cast(pl.Float64).to_numpy()
    target_values = ordered.get_column(candidate_set.target_col).cast(pl.Float64).to_numpy()
    x, y_arr, _valid_indices = SimplexKernelLearner._build_lagged_windows(
        input_values=input_values,
        target_values=target_values,
        min_lag_steps=min_lag_steps,
        max_lag_steps=max_lag_steps,
    )
    if x.shape[0] < 8:
        raise ValueError(
            f"Not enough valid lag windows after missing-value filtering: "
            f"only {x.shape[0]} remain (minimum 8 required). "
            f"Try a shorter lag window or provide more data."
        )

    train_end = int(math.floor(x.shape[0] * (1.0 - validation_fraction)))
    train_end = max(1, min(x.shape[0] - 1, train_end))
    x_train = x[:train_end]
    y_train = y_arr[:train_end]
    x_valid = x[train_end:]
    y_valid = y_arr[train_end:]

    x_stats = SimplexKernelLearner._robust_scaling_stats(x_train)
    y_stats = SimplexKernelLearner._robust_scaling_stats(y_train)
    x_train_scaled = (x_train - x_stats.center) / x_stats.scale
    y_train_scaled = (y_train - y_stats.center) / y_stats.scale
    x_valid_scaled = (x_valid - x_stats.center) / x_stats.scale
    y_valid_scaled = (y_valid - y_stats.center) / y_stats.scale

    return _BootstrapContext(
        x_train_scaled=x_train_scaled,
        y_train_scaled=y_train_scaled,
        x_valid_scaled=x_valid_scaled,
        y_valid_scaled=y_valid_scaled,
        lag_steps=tuple(range(min_lag_steps, max_lag_steps + 1)),
        dt_seconds=float(resolved_dt.total_seconds()),
        train_size=int(train_end),
        total_windows=int(x.shape[0]),
        validation_window_indices=tuple(range(train_end, int(x.shape[0]))),
        validation_fraction=validation_fraction,
        loss=loss,
        huber_delta=huber_delta,
        learner_parameters=dict(candidate.learner_parameters),
    )


def _fit_bootstrap_kernel(
    *,
    context: _BootstrapContext,
    family_result: KernelFamilyFitResult,
    bootstrap_id: int,
    train_indices: tuple[int, ...],
) -> tuple[Kernel, float | None, dict[str, float]]:
    x_train = context.x_train_scaled[np.asarray(train_indices, dtype=np.int64)]
    y_train = context.y_train_scaled[np.asarray(train_indices, dtype=np.int64)]
    x_valid = context.x_valid_scaled
    y_valid = context.y_valid_scaled
    candidate = family_result.candidate

    if candidate.candidate_type == "fixed_kernel":
        kernel = family_result.evaluated_fixed_kernel
        if kernel is None:
            raise ValueError("fixed-kernel bootstrap requires evaluated_fixed_kernel.")
        valid_loss = _numpy_loss(
            context.loss, context.huber_delta, x_valid @ np.asarray(kernel.weights), y_valid
        )
        return kernel, valid_loss, {}

    learning_rate = float(context.learner_parameters.get("learning_rate", 0.05))
    max_epochs = int(context.learner_parameters.get("max_epochs", 800))
    smoothness_penalty = float(context.learner_parameters.get("smoothness_penalty", 0.0))
    base_seed = context.learner_parameters.get("seed")
    seed = int(base_seed + bootstrap_id) if isinstance(base_seed, int) else bootstrap_id
    if seed is not None:
        torch.manual_seed(seed)

    if candidate.family == "simplex":
        kernel = _fit_simplex_kernel(
            x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid,
            lag_steps=context.lag_steps, dt_seconds=context.dt_seconds,
            loss=context.loss, huber_delta=context.huber_delta,
            learning_rate=learning_rate, max_epochs=max_epochs,
            smoothness_penalty=smoothness_penalty, name=candidate.candidate_id,
        )
        valid_loss = _numpy_loss(
            context.loss, context.huber_delta, x_valid @ np.asarray(kernel.weights), y_valid
        )
        return kernel, valid_loss, {}

    if candidate.family == "exponential":
        kernel, params = _fit_exponential_kernel(
            x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid,
            lag_steps=context.lag_steps, dt_seconds=context.dt_seconds,
            loss=context.loss, huber_delta=context.huber_delta,
            learning_rate=learning_rate, max_epochs=max_epochs,
            smoothness_penalty=smoothness_penalty,
            init_rate_lambda=context.learner_parameters.get("init_rate_lambda"),
            name=candidate.candidate_id,
        )
        valid_loss = _numpy_loss(
            context.loss, context.huber_delta, x_valid @ np.asarray(kernel.weights), y_valid
        )
        return kernel, valid_loss, params

    if candidate.family == "gamma":
        kernel, params = _fit_gamma_kernel(
            x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid,
            lag_steps=context.lag_steps, dt_seconds=context.dt_seconds,
            loss=context.loss, huber_delta=context.huber_delta,
            learning_rate=learning_rate, max_epochs=max_epochs,
            smoothness_penalty=smoothness_penalty,
            init_shape_alpha=float(context.learner_parameters.get("init_shape_alpha", 2.0)),
            init_rate_beta=context.learner_parameters.get("init_rate_beta"),
            name=candidate.candidate_id,
        )
        valid_loss = _numpy_loss(
            context.loss, context.huber_delta, x_valid @ np.asarray(kernel.weights), y_valid
        )
        return kernel, valid_loss, params

    raise ValueError(f"Unsupported learner family for bootstrap: {candidate.family!r}.")


def _torch_loss(
    loss: str, huber_delta: float, prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    if loss == "mse":
        return torch.mean((prediction - target) ** 2)
    return torch.nn.functional.huber_loss(prediction, target, delta=huber_delta, reduction="mean")


def _numpy_loss(
    loss: str, huber_delta: float, prediction: np.ndarray, target: np.ndarray
) -> float:
    pred = np.asarray(prediction, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if loss == "mse":
        return float(np.mean((pred - tgt) ** 2))
    residual = np.abs(pred - tgt)
    quadratic = np.minimum(residual, huber_delta)
    linear = residual - quadratic
    return float(np.mean(0.5 * quadratic**2 + huber_delta * linear))


def _smoothness_term(weights: torch.Tensor) -> torch.Tensor:
    if weights.numel() <= 1:
        return torch.zeros((), dtype=weights.dtype, device=weights.device)
    return torch.mean((weights[1:] - weights[:-1]) ** 2)


def _fit_simplex_kernel(**kwargs: Any) -> Kernel:
    x_train = torch.as_tensor(kwargs["x_train"], dtype=torch.float32)
    y_train = torch.as_tensor(kwargs["y_train"], dtype=torch.float32)
    x_valid = torch.as_tensor(kwargs["x_valid"], dtype=torch.float32)
    y_valid = torch.as_tensor(kwargs["y_valid"], dtype=torch.float32)
    lag_steps = kwargs["lag_steps"]
    n_lags = len(lag_steps)
    theta = torch.nn.Parameter(torch.zeros(n_lags, dtype=torch.float32))
    optimizer = torch.optim.Adam([theta], lr=kwargs["learning_rate"])

    best_valid_loss = float("inf")
    best_weights: np.ndarray | None = None
    for _ in range(kwargs["max_epochs"]):
        optimizer.zero_grad(set_to_none=True)
        weights = torch.softmax(theta, dim=0)
        train_pred = x_train @ weights
        data_loss = _torch_loss(kwargs["loss"], kwargs["huber_delta"], train_pred, y_train)
        total_loss = data_loss + kwargs["smoothness_penalty"] * _smoothness_term(weights)
        total_loss.backward()
        optimizer.step()
        with torch.no_grad():
            valid_pred = x_valid @ weights
            valid_loss = float(
                _torch_loss(kwargs["loss"], kwargs["huber_delta"], valid_pred, y_valid).item()
            )
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_weights = weights.detach().cpu().numpy().copy()

    if best_weights is None:
        raise RuntimeError("Optimization failed to produce simplex bootstrap weights.")
    kernel = LearnedKernel(
        weights=tuple(float(v) for v in best_weights),
        lag_steps=tuple(int(step) for step in lag_steps),
        dt=float(kwargs["dt_seconds"]),
        min_lag_steps=int(lag_steps[0]),
        max_lag_steps=int(lag_steps[-1]),
        name=kwargs["name"],
    )
    kernel.validate()
    return kernel


def _fit_exponential_kernel(**kwargs: Any) -> tuple[Kernel, dict[str, float]]:
    x_train = torch.as_tensor(kwargs["x_train"], dtype=torch.float32)
    y_train = torch.as_tensor(kwargs["y_train"], dtype=torch.float32)
    x_valid = torch.as_tensor(kwargs["x_valid"], dtype=torch.float32)
    y_valid = torch.as_tensor(kwargs["y_valid"], dtype=torch.float32)
    lag_steps = tuple(int(step) for step in kwargs["lag_steps"])
    lag_times = torch.as_tensor(
        np.asarray(lag_steps, dtype=np.float32) * float(kwargs["dt_seconds"])
    )
    min_rate = ExponentialKernelLearner._MIN_RATE_LAMBDA
    init_rate = kwargs["init_rate_lambda"]
    if init_rate is None:
        midpoint = 0.5 * (lag_steps[0] + lag_steps[-1]) * float(kwargs["dt_seconds"])
        if midpoint <= 0.0:
            raise ValueError("Cannot derive default init_rate_lambda from lag midpoint <= 0.")
        init_rate = 1.0 / midpoint
    init_raw = _inverse_softplus(
        max(float(init_rate) - min_rate, min_rate)
    )
    raw_rate = torch.nn.Parameter(torch.tensor(float(init_raw), dtype=torch.float32))
    optimizer = torch.optim.Adam([raw_rate], lr=kwargs["learning_rate"])

    best_valid_loss = float("inf")
    best_rate = None
    for _ in range(kwargs["max_epochs"]):
        optimizer.zero_grad(set_to_none=True)
        rate = torch.nn.functional.softplus(raw_rate) + min_rate
        logits = -rate * lag_times
        weights = torch.softmax(logits, dim=0)
        train_pred = x_train @ weights
        data_loss = _torch_loss(kwargs["loss"], kwargs["huber_delta"], train_pred, y_train)
        total_loss = data_loss + kwargs["smoothness_penalty"] * _smoothness_term(weights)
        total_loss.backward()
        optimizer.step()
        with torch.no_grad():
            valid_pred = x_valid @ weights
            valid_loss = float(
                _torch_loss(kwargs["loss"], kwargs["huber_delta"], valid_pred, y_valid).item()
            )
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_rate = float(rate.item())

    if best_rate is None:
        raise RuntimeError("Optimization failed to produce exponential bootstrap parameters.")
    kernel = _make_parametric_learned_kernel(
        family="exponential",
        dt=float(kwargs["dt_seconds"]),
        min_lag_steps=int(lag_steps[0]),
        max_lag_steps=int(lag_steps[-1]),
        parameters={"rate_lambda": best_rate},
        name=kwargs["name"],
    )
    return kernel, {"rate_lambda": best_rate}


def _fit_gamma_kernel(**kwargs: Any) -> tuple[Kernel, dict[str, float]]:
    x_train = torch.as_tensor(kwargs["x_train"], dtype=torch.float32)
    y_train = torch.as_tensor(kwargs["y_train"], dtype=torch.float32)
    x_valid = torch.as_tensor(kwargs["x_valid"], dtype=torch.float32)
    y_valid = torch.as_tensor(kwargs["y_valid"], dtype=torch.float32)
    lag_steps = tuple(int(step) for step in kwargs["lag_steps"])
    lag_times = torch.as_tensor(
        np.asarray(lag_steps, dtype=np.float32) * float(kwargs["dt_seconds"])
    )
    lag_times_safe = torch.clamp(lag_times, min=GammaKernelLearner._LOG_EPS)

    min_shape_alpha = (
        GammaKernelLearner._MIN_SHAPE_ALPHA_WITH_ZERO_LAG
        if 0 in lag_steps
        else GammaKernelLearner._MIN_SHAPE_ALPHA
    )
    init_shape_alpha = float(kwargs["init_shape_alpha"])
    init_rate_beta = kwargs["init_rate_beta"]
    if init_rate_beta is None:
        midpoint = 0.5 * (lag_steps[0] + lag_steps[-1]) * float(kwargs["dt_seconds"])
        if midpoint <= 0.0:
            raise ValueError("Cannot derive default init_rate_beta from lag midpoint <= 0.")
        init_rate_beta = 1.0 / midpoint

    init_shape_raw = _inverse_softplus(
        max(init_shape_alpha - min_shape_alpha, GammaKernelLearner._MIN_SHAPE_ALPHA)
    )
    init_rate_raw = _inverse_softplus(
        max(float(init_rate_beta) - GammaKernelLearner._MIN_RATE_BETA,
            GammaKernelLearner._MIN_RATE_BETA)
    )
    raw_shape = torch.nn.Parameter(torch.tensor(float(init_shape_raw), dtype=torch.float32))
    raw_rate = torch.nn.Parameter(torch.tensor(float(init_rate_raw), dtype=torch.float32))
    optimizer = torch.optim.Adam([raw_shape, raw_rate], lr=kwargs["learning_rate"])

    best_valid_loss = float("inf")
    best_shape_alpha = None
    best_rate_beta = None
    best_weights = None
    for _ in range(kwargs["max_epochs"]):
        optimizer.zero_grad(set_to_none=True)
        shape_alpha = torch.nn.functional.softplus(raw_shape) + min_shape_alpha
        rate_beta = torch.nn.functional.softplus(raw_rate) + GammaKernelLearner._MIN_RATE_BETA
        log_pdf = (
            shape_alpha * torch.log(rate_beta)
            + (shape_alpha - 1.0) * torch.log(lag_times_safe)
            - rate_beta * lag_times
            - torch.lgamma(shape_alpha)
        )
        log_pdf = torch.where(lag_times > 0.0, log_pdf, torch.full_like(log_pdf, -1.0e9))
        weights = torch.softmax(log_pdf, dim=0)
        train_pred = x_train @ weights
        data_loss = _torch_loss(kwargs["loss"], kwargs["huber_delta"], train_pred, y_train)
        total_loss = data_loss + kwargs["smoothness_penalty"] * _smoothness_term(weights)
        total_loss.backward()
        optimizer.step()
        with torch.no_grad():
            valid_pred = x_valid @ weights
            valid_loss = float(
                _torch_loss(kwargs["loss"], kwargs["huber_delta"], valid_pred, y_valid).item()
            )
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_shape_alpha = float(shape_alpha.item())
                best_rate_beta = float(rate_beta.item())
                best_weights = weights.detach().cpu().numpy().copy()

    if best_shape_alpha is None or best_rate_beta is None or best_weights is None:
        raise RuntimeError("Optimization failed to produce gamma bootstrap parameters.")
    kernel = _make_parametric_learned_kernel(
        family="gamma",
        dt=float(kwargs["dt_seconds"]),
        min_lag_steps=int(lag_steps[0]),
        max_lag_steps=int(lag_steps[-1]),
        parameters={"shape_alpha": best_shape_alpha, "rate_beta": best_rate_beta},
        name=kwargs["name"],
    )
    return kernel, {"shape_alpha": best_shape_alpha, "rate_beta": best_rate_beta}


def _sample_train_indices(
    *,
    rng: np.random.Generator,
    train_indices: tuple[int, ...],
    block_length: int,
    starts_upper_bound: int,
) -> tuple[int, ...]:
    n_train = len(train_indices)
    sampled: list[int] = []
    while len(sampled) < n_train:
        block_start = int(rng.integers(0, starts_upper_bound))
        block_stop = block_start + block_length
        sampled.extend(train_indices[block_start:block_stop])
    return tuple(sampled[:n_train])


def _coerce_lag_value(value: str | int | float) -> str | int:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if float(value).is_integer():
        return int(value)
    raise ValueError(
        "min_lag/max_lag must be an integer number of steps or a duration-like string."
    )


def _bootstrap_simplicity_rank(result: KernelFamilyFitResult) -> int:
    if result.is_baseline:
        return 4
    if result.candidate.candidate_type == "fixed_kernel":
        return 1
    if result.candidate.candidate_type == "empirical_learner":
        return 2
    if result.candidate.candidate_type == "parametric_learner":
        return 3
    return 5


def _loss_delta_fraction(loss_a: float, loss_b: float) -> float:
    return abs(loss_b - loss_a) / max(abs(loss_a), 1e-12)


def _select_bootstrap_candidate(
    *,
    succeeded_losses: list[tuple[KernelFamilyFitResult, float]],
    loss_tolerance_fraction: float,
) -> KernelFamilyFitResult:
    if not succeeded_losses:
        raise ValueError("succeeded_losses must contain at least one candidate.")
    if loss_tolerance_fraction < 0.0:
        raise ValueError("loss_tolerance_fraction must be non-negative.")
    best_loss = min(loss for _result, loss in succeeded_losses)
    tolerance_pool = [
        (result, loss)
        for result, loss in succeeded_losses
        if _loss_delta_fraction(best_loss, loss) <= loss_tolerance_fraction
    ]
    if not tolerance_pool:
        tolerance_pool = list(succeeded_losses)
    tolerance_pool_with_updated_loss = [
        (replace(result, validation_loss=float(loss)), float(loss))
        for result, loss in tolerance_pool
    ]
    selected_result, _selected_loss = min(
        tolerance_pool_with_updated_loss,
        key=lambda item: (
            _bootstrap_simplicity_rank(item[0]),
            item[1],
            item[0].candidate.candidate_id,
        ),
    )
    return selected_result
