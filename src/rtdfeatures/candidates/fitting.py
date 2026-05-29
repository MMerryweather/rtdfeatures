"""Kernel candidate fitting utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from rtdfeatures.candidates.contracts import (
    BaselineComparison,
    KernelCandidate,
    KernelCandidateSet,
    KernelComparisonResult,
    KernelFamilyFitResult,
    KernelFitResult,
)
from rtdfeatures.kernels import (
    DelayedExponentialKernel,
    ErlangKernel,
    ExponentialKernel,
    FixedDelayKernel,
    GammaKernel,
    Kernel,
    LogNormalKernel,
    UniformKernel,
)
from rtdfeatures.learners import ExponentialKernelLearner, GammaKernelLearner, SimplexKernelLearner
from rtdfeatures.utils import lag_to_steps, resolve_and_validate_dt, validate_or_sort_time

DEFAULT_SELECTION_TOLERANCE = 0.02
_COMPARISON_TABLE_COLUMNS: tuple[str, ...] = (
    "candidate_id",
    "family",
    "candidate_type",
    "succeeded",
    "validation_loss",
    "train_loss",
    "mean_lag",
    "p50_lag",
    "p90_lag",
    "tail_mass",
    "warning_count",
    "warning_codes",
    "n_parameters",
    "beats_no_lag",
    "beats_best_single_lag",
    "error",
)


def fit_kernel_candidates(
    df: pl.DataFrame,
    candidate_set: KernelCandidateSet,
    *,
    order_by_time: bool = False,
    loss: str = "huber",
    huber_delta: float = 1.0,
) -> KernelComparisonResult:
    """Fit/evaluate all candidates in a set without aborting on per-candidate failures."""
    ordered = validate_or_sort_time(
        df,
        time_col=candidate_set.time_col,
        order_by_time=order_by_time,
    )
    family_results: list[KernelFamilyFitResult] = []
    for candidate in candidate_set.candidates:
        try:
            family_results.append(
                _fit_one_candidate(
                    ordered, candidate_set, candidate, loss=loss, huber_delta=huber_delta
                )
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            # Deliberate resilient-batch behavior: one candidate failure should not abort the sweep.
            family_results.append(
                KernelFamilyFitResult(
                    candidate=candidate,
                    fit_result=None,
                    succeeded=False,
                    error=f"{type(exc).__name__}: {exc}",
                    is_parametric=(candidate.candidate_type == "parametric_learner"),
                    is_empirical=(candidate.candidate_type == "empirical_learner"),
                    is_baseline=(candidate.candidate_type == "baseline"),
                    n_parameters=None,
                )
            )

    comparison_rows: list[dict[str, Any]] = []
    for result in family_results:
        comparison_rows.append(_comparison_row_for_result(result))
    comparison_table = _stable_comparison_table(comparison_rows)
    warnings = tuple(
        f"{result.candidate.candidate_id}: {result.error}"
        for result in family_results
        if not result.succeeded and result.error is not None
    )
    return KernelComparisonResult(
        candidate_set=candidate_set,
        family_results=tuple(family_results),
        comparison_table=comparison_table,
        warnings=warnings,
        selection_summary={},
    )


def _fit_one_candidate(
    df: pl.DataFrame,
    candidate_set: KernelCandidateSet,
    candidate: KernelCandidate,
    *,
    loss: str = "huber",
    huber_delta: float = 1.0,
) -> KernelFamilyFitResult:
    if candidate.candidate_type in {"empirical_learner", "parametric_learner"}:
        fit_result = _fit_learner_candidate(df, candidate_set, candidate)
        is_parametric = candidate.candidate_type == "parametric_learner"
        is_empirical = candidate.candidate_type == "empirical_learner"
        return KernelFamilyFitResult(
            candidate=candidate,
            fit_result=fit_result,
            succeeded=True,
            error=None,
            is_parametric=is_parametric,
            is_empirical=is_empirical,
            is_baseline=False,
            n_parameters=_count_candidate_parameters(candidate, include_lag_bounds=False),
            validation_loss=float(fit_result.fit_diagnostics.validation_loss),
            train_loss=float(fit_result.fit_diagnostics.train_loss),
            warning_codes=tuple(fit_result.identifiability_report.warning_codes),
        )
    if candidate.candidate_type == "fixed_kernel":
        evaluated = _evaluate_fixed_kernel_candidate(df, candidate_set, candidate)
        return KernelFamilyFitResult(
            candidate=candidate,
            fit_result=None,
            succeeded=True,
            error=None,
            is_parametric=False,
            is_empirical=False,
            is_baseline=False,
            n_parameters=_count_candidate_parameters(candidate, include_lag_bounds=True),
            validation_loss=float(evaluated["validation_loss"]),
            evaluated_fixed_kernel=evaluated["kernel"],
            fixed_baseline_comparison=evaluated["baseline_comparison"],
            evaluation_provenance=evaluated["evaluation_provenance"],
        )
    if candidate.candidate_type == "baseline":
        validation_loss = _evaluate_baseline_candidate(
            df, candidate_set, candidate, loss=loss, huber_delta=huber_delta
        )
        return KernelFamilyFitResult(
            candidate=candidate,
            fit_result=None,
            succeeded=True,
            error=None,
            is_parametric=False,
            is_empirical=False,
            is_baseline=True,
            n_parameters=0,
            validation_loss=float(validation_loss),
        )
    raise ValueError(f"Unsupported candidate_type: {candidate.candidate_type!r}.")


def _fit_learner_candidate(
    df: pl.DataFrame,
    candidate_set: KernelCandidateSet,
    candidate: KernelCandidate,
) -> KernelFitResult:
    learner_params = dict(candidate.learner_parameters)
    min_lag = _coerce_lag_value(candidate.min_lag, param_name="min_lag")
    max_lag = _coerce_lag_value(candidate.max_lag, param_name="max_lag")
    learner: SimplexKernelLearner | GammaKernelLearner | ExponentialKernelLearner
    if candidate.family == "simplex":
        learner = SimplexKernelLearner(
            min_lag=min_lag, max_lag=max_lag, **learner_params
        )
    elif candidate.family == "gamma":
        learner = GammaKernelLearner(
            min_lag=min_lag, max_lag=max_lag, **learner_params
        )
    elif candidate.family == "exponential":
        learner = ExponentialKernelLearner(
            min_lag=min_lag, max_lag=max_lag, **learner_params
        )
    else:
        raise ValueError(
            f"Unsupported learner family for candidate '{candidate.candidate_id}': "
            f"{candidate.family!r}."
        )
    return learner.fit(
        df,
        input_col=candidate_set.input_col,
        target_col=candidate_set.target_col,
        time_col=candidate_set.time_col,
        order_by_time=False,
    )


def _evaluate_fixed_kernel_candidate(
    df: pl.DataFrame,
    candidate_set: KernelCandidateSet,
    candidate: KernelCandidate,
) -> dict[str, Any]:
    eval_params = dict(candidate.fixed_parameters)
    allowed_eval_keys = {"loss", "huber_delta", "validation_fraction"}
    allowed_kernel_keys = _allowed_fixed_kernel_parameter_keys(candidate.family)
    allowed_keys = allowed_eval_keys | allowed_kernel_keys
    unknown_keys = sorted(key for key in eval_params if key not in allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"Unsupported fixed_parameters keys for fixed-kernel candidate "
            f"'{candidate.candidate_id}': {unknown_keys}. "
            f"Allowed keys: {sorted(allowed_keys)}."
        )
    loss_name = str(eval_params.pop("loss", "huber"))
    huber_delta = float(eval_params.pop("huber_delta", 1.0))
    validation_fraction = float(eval_params.pop("validation_fraction", 0.2))
    evaluator = _WindowedKernelEvaluator(
        loss=loss_name,
        huber_delta=huber_delta,
        validation_fraction=validation_fraction,
    )
    kernel = _build_fixed_kernel(df, candidate_set.time_col, candidate, eval_params)
    validation_loss = evaluator.evaluate_kernel(
        df=df,
        time_col=candidate_set.time_col,
        input_col=candidate_set.input_col,
        target_col=candidate_set.target_col,
        kernel=kernel,
    )
    no_lag_loss = evaluator.evaluate_baseline(
        df=df,
        time_col=candidate_set.time_col,
        input_col=candidate_set.input_col,
        target_col=candidate_set.target_col,
        min_lag=candidate.min_lag,
        max_lag=candidate.max_lag,
        baseline_name="no_lag",
    )
    best_single_lag_loss = evaluator.evaluate_baseline(
        df=df,
        time_col=candidate_set.time_col,
        input_col=candidate_set.input_col,
        target_col=candidate_set.target_col,
        min_lag=candidate.min_lag,
        max_lag=candidate.max_lag,
        baseline_name="best_single_lag",
    )
    return {
        "kernel": kernel,
        "validation_loss": float(validation_loss),
        "baseline_comparison": BaselineComparison(
            no_lag_validation_loss=float(no_lag_loss),
            best_single_lag_validation_loss=float(best_single_lag_loss),
            learned_validation_loss=float(validation_loss),
        ),
        "evaluation_provenance": {
            "loss": loss_name,
            "huber_delta": huber_delta if loss_name == "huber" else None,
            "validation_fraction": validation_fraction,
            "dt_seconds": float(kernel.dt),
            "total_valid_windows": int(evaluator.total_valid_windows),
            "validation_windows": int(evaluator.validation_windows),
        },
    }


def _evaluate_baseline_candidate(
    df: pl.DataFrame,
    candidate_set: KernelCandidateSet,
    candidate: KernelCandidate,
    *,
    loss: str = "huber",
    huber_delta: float = 1.0,
) -> float:
    evaluator = _WindowedKernelEvaluator(
        loss=loss,
        huber_delta=huber_delta,
        validation_fraction=0.2,
    )
    return evaluator.evaluate_baseline(
        df=df,
        time_col=candidate_set.time_col,
        input_col=candidate_set.input_col,
        target_col=candidate_set.target_col,
        min_lag=candidate.min_lag,
        max_lag=candidate.max_lag,
        baseline_name=candidate.family,
    )


def _build_fixed_kernel(
    df: pl.DataFrame,
    time_col: str,
    candidate: KernelCandidate,
    fixed_parameters: dict[str, Any],
) -> Kernel:
    if "dt" in fixed_parameters:
        dt_like = fixed_parameters["dt"]
    else:
        dt_like = None
    resolved_dt = resolve_and_validate_dt(df, time_col=time_col, dt=dt_like)
    dt_seconds = float(resolved_dt.total_seconds())
    min_lag_steps = lag_to_steps(
        _coerce_lag_value(candidate.min_lag, param_name="min_lag"),
        dt=resolved_dt,
        param_name="min_lag",
    )
    max_lag_steps = lag_to_steps(
        _coerce_lag_value(candidate.max_lag, param_name="max_lag"),
        dt=resolved_dt,
        param_name="max_lag",
    )

    if candidate.family == "fixed_delay":
        if "delay_steps" not in fixed_parameters:
            raise ValueError(
                "fixed_delay candidates require explicit fixed_parameters['delay_steps']."
            )
        delay_steps = int(fixed_parameters["delay_steps"])
        return FixedDelayKernel(
            delay_steps=delay_steps,
            min_lag_steps=min_lag_steps,
            max_lag_steps=max_lag_steps,
            dt=dt_seconds,
            name=candidate.candidate_id,
        )
    if candidate.family == "uniform":
        return UniformKernel(
            min_lag_steps=min_lag_steps,
            max_lag_steps=max_lag_steps,
            dt=dt_seconds,
            name=candidate.candidate_id,
        )
    if candidate.family == "gamma":
        return GammaKernel(
            shape_alpha=float(fixed_parameters["shape_alpha"]),
            rate_beta=float(fixed_parameters["rate_beta"]),
            min_lag_steps=min_lag_steps,
            max_lag_steps=max_lag_steps,
            dt=dt_seconds,
            name=candidate.candidate_id,
        )
    if candidate.family == "exponential":
        return ExponentialKernel(
            rate_lambda=float(fixed_parameters["rate_lambda"]),
            min_lag_steps=min_lag_steps,
            max_lag_steps=max_lag_steps,
            dt=dt_seconds,
            name=candidate.candidate_id,
        )
    if candidate.family == "delayed_exponential":
        return DelayedExponentialKernel(
            delay=float(fixed_parameters["delay"]),
            rate_lambda=float(fixed_parameters["rate_lambda"]),
            min_lag_steps=min_lag_steps,
            max_lag_steps=max_lag_steps,
            dt=dt_seconds,
            name=candidate.candidate_id,
        )
    if candidate.family == "lognormal":
        return LogNormalKernel(
            log_mu=float(fixed_parameters["log_mu"]),
            log_sigma=float(fixed_parameters["log_sigma"]),
            min_lag_steps=min_lag_steps,
            max_lag_steps=max_lag_steps,
            dt=dt_seconds,
            name=candidate.candidate_id,
        )
    if candidate.family == "erlang":
        return ErlangKernel(
            shape_k=int(fixed_parameters["shape_k"]),
            rate_beta=float(fixed_parameters["rate_beta"]),
            min_lag_steps=min_lag_steps,
            max_lag_steps=max_lag_steps,
            dt=dt_seconds,
            name=candidate.candidate_id,
        )
    raise ValueError(
        f"Unsupported fixed kernel family for candidate '{candidate.candidate_id}': "
        f"{candidate.family!r}."
    )


def _allowed_fixed_kernel_parameter_keys(family: str) -> set[str]:
    if family == "fixed_delay":
        return {"dt", "delay_steps"}
    if family == "uniform":
        return {"dt"}
    if family == "gamma":
        return {"dt", "shape_alpha", "rate_beta"}
    if family == "exponential":
        return {"dt", "rate_lambda"}
    if family == "delayed_exponential":
        return {"dt", "delay", "rate_lambda"}
    if family == "lognormal":
        return {"dt", "log_mu", "log_sigma"}
    if family == "erlang":
        return {"dt", "shape_k", "rate_beta"}
    return set()


def _count_candidate_parameters(candidate: KernelCandidate, *, include_lag_bounds: bool) -> int:
    count = len(candidate.fixed_parameters) + len(candidate.learner_parameters)
    if include_lag_bounds:
        count += 2
    return count


class _WindowedKernelEvaluator:
    def __init__(self, *, loss: str, huber_delta: float, validation_fraction: float) -> None:
        if loss not in {"huber", "mse"}:
            raise ValueError("loss must be either 'huber' or 'mse'.")
        if huber_delta <= 0.0:
            raise ValueError("huber_delta must be strictly positive.")
        if validation_fraction <= 0.0 or validation_fraction >= 0.5:
            raise ValueError("validation_fraction must be in (0.0, 0.5).")
        self.loss = loss
        self.huber_delta = huber_delta
        self.validation_fraction = validation_fraction
        self.total_valid_windows = 0
        self.validation_windows = 0

    def evaluate_kernel(
        self,
        *,
        df: pl.DataFrame,
        time_col: str,
        input_col: str,
        target_col: str,
        kernel: Kernel,
    ) -> float:
        windows = self._scaled_validation_windows(
            df=df,
            time_col=time_col,
            input_col=input_col,
            target_col=target_col,
            min_lag_steps=kernel.min_lag_steps,
            max_lag_steps=kernel.max_lag_steps,
        )
        prediction = windows.x_valid_scaled @ np.asarray(kernel.weights, dtype=np.float64)
        return self._numpy_loss(prediction, windows.y_valid_scaled)

    def evaluate_baseline(
        self,
        *,
        df: pl.DataFrame,
        time_col: str,
        input_col: str,
        target_col: str,
        min_lag: str | int | float,
        max_lag: str | int | float,
        baseline_name: str,
    ) -> float:
        resolved_dt = resolve_and_validate_dt(df, time_col=time_col, dt=None)
        min_lag_steps = lag_to_steps(
            _coerce_lag_value(min_lag, param_name="min_lag"),
            dt=resolved_dt,
            param_name="min_lag",
        )
        max_lag_steps = lag_to_steps(
            _coerce_lag_value(max_lag, param_name="max_lag"),
            dt=resolved_dt,
            param_name="max_lag",
        )
        windows = self._scaled_validation_windows(
            df=df,
            time_col=time_col,
            input_col=input_col,
            target_col=target_col,
            min_lag_steps=min_lag_steps,
            max_lag_steps=max_lag_steps,
        )

        if baseline_name == "no_lag":
            prediction = windows.no_lag_valid_scaled
        elif baseline_name == "best_single_lag":
            best_loss = float("inf")
            best_prediction: np.ndarray | None = None
            for lag_idx in range(windows.x_valid_scaled.shape[1]):
                lag_prediction = windows.x_valid_scaled[:, lag_idx]
                lag_loss = self._numpy_loss(lag_prediction, windows.y_valid_scaled)
                if lag_loss < best_loss:
                    best_loss = lag_loss
                    best_prediction = lag_prediction
            if best_prediction is None:
                raise ValueError("best_single_lag baseline could not resolve any lag predictions.")
            prediction = best_prediction
        else:
            raise ValueError(f"Unsupported baseline family: {baseline_name!r}.")
        return self._numpy_loss(prediction, windows.y_valid_scaled)

    def _scaled_validation_windows(
        self,
        *,
        df: pl.DataFrame,
        time_col: str,
        input_col: str,
        target_col: str,
        min_lag_steps: int,
        max_lag_steps: int,
    ) -> _ValidationWindows:
        if max_lag_steps < min_lag_steps:
            raise ValueError(
                f"max_lag ({max_lag_steps} steps) must be >= min_lag ({min_lag_steps} steps). "
                f"Check the candidate's lag window configuration (min_lag, max_lag)."
            )
        input_values = df.get_column(input_col).cast(pl.Float64).to_numpy()
        target_values = df.get_column(target_col).cast(pl.Float64).to_numpy()
        x, y_arr, valid_indices = SimplexKernelLearner._build_lagged_windows(
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

        train_end = int(np.floor(x.shape[0] * (1.0 - self.validation_fraction)))
        train_end = max(1, min(x.shape[0] - 1, train_end))
        x_train = x[:train_end]
        x_valid = x[train_end:]
        y_train = y_arr[:train_end]
        y_valid = y_arr[train_end:]
        valid_idx_valid = valid_indices[train_end:]
        valid_idx_train = valid_indices[:train_end]

        x_stats = SimplexKernelLearner._robust_scaling_stats(x_train)
        y_stats = SimplexKernelLearner._robust_scaling_stats(y_train)
        x_valid_scaled = (x_valid - x_stats.center) / x_stats.scale
        y_valid_scaled = (y_valid - y_stats.center) / y_stats.scale
        lag_steps = list(range(min_lag_steps, max_lag_steps + 1))
        if 0 in lag_steps:
            no_lag_column = lag_steps.index(0)
            no_lag_valid_scaled = x_valid_scaled[:, no_lag_column]
        else:
            no_lag_train = input_values[valid_idx_train]
            no_lag_valid = input_values[valid_idx_valid]
            no_lag_stats = SimplexKernelLearner._robust_scaling_stats(no_lag_train)
            no_lag_valid_scaled = (no_lag_valid - no_lag_stats.center) / no_lag_stats.scale
        self.total_valid_windows = int(x.shape[0])
        self.validation_windows = int(x_valid.shape[0])
        return _ValidationWindows(
            x_valid_scaled=x_valid_scaled,
            y_valid_scaled=y_valid_scaled,
            no_lag_valid_scaled=no_lag_valid_scaled,
        )

    def _numpy_loss(self, prediction: np.ndarray, target: np.ndarray) -> float:
        prediction_arr = np.asarray(prediction, dtype=np.float64)
        target_arr = np.asarray(target, dtype=np.float64)
        if self.loss == "mse":
            return float(np.mean((prediction_arr - target_arr) ** 2))
        residual = np.abs(prediction_arr - target_arr)
        quadratic = np.minimum(residual, self.huber_delta)
        linear = residual - quadratic
        return float(np.mean(0.5 * quadratic**2 + self.huber_delta * linear))


class _ValidationWindows:
    def __init__(
        self,
        *,
        x_valid_scaled: np.ndarray,
        y_valid_scaled: np.ndarray,
        no_lag_valid_scaled: np.ndarray,
    ) -> None:
        self.x_valid_scaled = x_valid_scaled
        self.y_valid_scaled = y_valid_scaled
        self.no_lag_valid_scaled = no_lag_valid_scaled


def _coerce_lag_value(value: str | int | float, *, param_name: str) -> str | int:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if float(value).is_integer():
        return int(value)
    raise ValueError(
        f"{param_name} must be an integer number of steps or a duration-like string."
    )


def kernel_comparison_table(comparison_result: KernelComparisonResult) -> pl.DataFrame:
    """Return deterministic comparison rows with a stable schema."""
    rows = [_comparison_row_for_result(result) for result in comparison_result.family_results]
    return _stable_comparison_table(rows)


def kernel_comparison_compact_dict(comparison_result: KernelComparisonResult) -> dict[str, Any]:
    """Return compact comparison summary without selecting a downstream model."""
    table = kernel_comparison_table(comparison_result)
    succeeded_table = table.filter(
        pl.col("succeeded")
        & pl.col("validation_loss").is_not_null()
        & pl.col("validation_loss").is_finite()
    )
    best_candidate_id = (
        str(succeeded_table["candidate_id"][0]) if succeeded_table.height > 0 else None
    )
    return {
        "candidate_set_id": comparison_result.candidate_set.candidate_set_id,
        "candidate_count": table.height,
        "succeeded_count": int(table["succeeded"].sum()),
        "failed_count": int((~table["succeeded"]).sum()),
        "best_candidate_id_by_validation_loss": best_candidate_id,
        "warnings": comparison_result.warnings,
    }


def kernel_comparison_compact_text(comparison_result: KernelComparisonResult) -> str:
    """Return one-line deterministic comparison summary."""
    table = kernel_comparison_table(comparison_result)
    parts: list[str] = []
    for row in table.select(["candidate_id", "validation_loss", "succeeded"]).to_dicts():
        if not bool(row["succeeded"]):
            parts.append(f"{row['candidate_id']}=failed")
            continue
        loss = row["validation_loss"]
        if loss is None:
            parts.append(f"{row['candidate_id']}=n/a")
            continue
        parts.append(f"{row['candidate_id']}={float(loss):.6g}")
    return "candidate validation losses: " + ", ".join(parts)


def _comparison_row_for_result(result: KernelFamilyFitResult) -> dict[str, Any]:
    fit_result = result.fit_result
    baseline_comparison: BaselineComparison | None = None
    if fit_result is not None:
        baseline_comparison = fit_result.baseline_comparison
    elif result.candidate.candidate_type == "fixed_kernel":
        baseline_comparison = result.fixed_baseline_comparison
    no_lag_loss = (
        float(baseline_comparison.no_lag_validation_loss)
        if baseline_comparison is not None
        else None
    )
    best_single_lag_loss = (
        float(baseline_comparison.best_single_lag_validation_loss)
        if baseline_comparison is not None
        else None
    )
    validation_loss = result.validation_loss
    beats_no_lag = (
        bool(validation_loss < no_lag_loss)
        if validation_loss is not None and no_lag_loss is not None
        else None
    )
    beats_best_single_lag = (
        bool(validation_loss < best_single_lag_loss)
        if validation_loss is not None and best_single_lag_loss is not None
        else None
    )
    warning_codes = tuple(result.warning_codes)
    return {
        "candidate_id": result.candidate.candidate_id,
        "family": result.candidate.family,
        "candidate_type": result.candidate.candidate_type,
        "succeeded": result.succeeded,
        "validation_loss": validation_loss,
        "train_loss": result.train_loss,
        "mean_lag": float(fit_result.fit_diagnostics.mean_lag) if fit_result is not None else None,
        "p50_lag": float(fit_result.fit_diagnostics.p50_lag) if fit_result is not None else None,
        "p90_lag": float(fit_result.fit_diagnostics.p90_lag) if fit_result is not None else None,
        "tail_mass": (
            float(fit_result.fit_diagnostics.tail_mass) if fit_result is not None else None
        ),
        "warning_count": len(warning_codes),
        "warning_codes": "|".join(warning_codes),
        "n_parameters": result.n_parameters,
        "beats_no_lag": beats_no_lag,
        "beats_best_single_lag": beats_best_single_lag,
        "error": result.error,
    }


def _stable_comparison_table(rows: list[dict[str, Any]]) -> pl.DataFrame:
    table = pl.DataFrame(
        rows,
        schema={
            "candidate_id": pl.String,
            "family": pl.String,
            "candidate_type": pl.String,
            "succeeded": pl.Boolean,
            "validation_loss": pl.Float64,
            "train_loss": pl.Float64,
            "mean_lag": pl.Float64,
            "p50_lag": pl.Float64,
            "p90_lag": pl.Float64,
            "tail_mass": pl.Float64,
            "warning_count": pl.Int64,
            "warning_codes": pl.String,
            "n_parameters": pl.Int64,
            "beats_no_lag": pl.Boolean,
            "beats_best_single_lag": pl.Boolean,
            "error": pl.String,
        },
    )
    if table.height == 0:
        return table.select(list(_COMPARISON_TABLE_COLUMNS))
    return (
        table.with_columns(
            pl.col("validation_loss")
            .is_not_null()
            .and_(pl.col("validation_loss").is_finite())
            .alias("_has_finite_loss"),
            pl.col("validation_loss").fill_null(float("inf")).alias("_loss_sort"),
            pl.col("n_parameters").fill_null(10**9).alias("_complexity_sort"),
        )
        .sort(
            by=["succeeded", "_has_finite_loss", "_loss_sort", "_complexity_sort", "candidate_id"],
            descending=[True, True, False, False, False],
        )
        .drop(["_has_finite_loss", "_loss_sort", "_complexity_sort"])
        .select(list(_COMPARISON_TABLE_COLUMNS))
    )
