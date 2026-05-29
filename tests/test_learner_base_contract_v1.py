"""Tests for learner base contract v1 (prep logic extraction)."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from rtdfeatures.learners._base import (
    LearnerConfig,
    prepare_fit_data,
    validate_fit_columns,
)
from rtdfeatures.learners.simplex import SimplexKernelLearner


def _make_time(n_rows: int) -> list[datetime]:
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [t0 + timedelta(minutes=i) for i in range(n_rows)]


def _simple_dataset(n_rows: int = 300) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    return pl.DataFrame(
        {
            "timestamp": _make_time(n_rows),
            "x": rng.normal(0.0, 1.0, size=n_rows).tolist(),
            "y": rng.normal(0.0, 1.0, size=n_rows).tolist(),
        }
    )


def _default_config(**overrides: object) -> LearnerConfig:
    kwargs: dict[str, object] = dict(
        max_lag="10m",
        min_lag=0,
        dt=None,
        loss="huber",
        validation_fraction=0.2,
        huber_delta=1.0,
    )
    kwargs.update(overrides)
    return LearnerConfig(
        max_lag=kwargs["max_lag"],  # type: ignore[arg-type]
        min_lag=kwargs["min_lag"],  # type: ignore[arg-type]
        dt=kwargs["dt"],  # type: ignore[arg-type]
        loss=kwargs["loss"],  # type: ignore[arg-type]
        validation_fraction=kwargs["validation_fraction"],  # type: ignore[arg-type]
        huber_delta=kwargs["huber_delta"],  # type: ignore[arg-type]
    )


class TestValidateFitColumns:
    def test_passes_valid_columns(self) -> None:
        df = _simple_dataset(50)
        validate_fit_columns(df, time_col="timestamp", input_col="x", target_col="y")

    def test_raises_on_missing_time_col(self) -> None:
        df = _simple_dataset(50).drop("timestamp")
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_fit_columns(df, time_col="timestamp", input_col="x", target_col="y")

    def test_raises_on_missing_input_col(self) -> None:
        df = _simple_dataset(50).drop("x")
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_fit_columns(df, time_col="timestamp", input_col="x", target_col="y")

    def test_raises_on_missing_target_col(self) -> None:
        df = _simple_dataset(50).drop("y")
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_fit_columns(df, time_col="timestamp", input_col="x", target_col="y")

    def test_raises_on_same_input_target(self) -> None:
        df = _simple_dataset(50)
        with pytest.raises(ValueError, match="must be different columns"):
            validate_fit_columns(df, time_col="timestamp", input_col="x", target_col="x")


class TestPrepareFitData:
    def test_prepare_fit_data_validates_required_columns(self) -> None:
        df = _simple_dataset(50).drop("x")
        config = _default_config()
        with pytest.raises(ValueError, match="Missing required columns"):
            prepare_fit_data(
                df,
                input_col="x",
                target_col="y",
                time_col="timestamp",
                order_by_time=False,
                config=config,
            )

    def test_prepare_fit_data_rejects_same_input_and_target(self) -> None:
        df = _simple_dataset(50)
        config = _default_config()
        with pytest.raises(ValueError, match="must be different columns"):
            prepare_fit_data(
                df,
                input_col="x",
                target_col="x",
                time_col="timestamp",
                order_by_time=False,
                config=config,
            )

    def test_prepare_fit_data_rejects_inverted_lag_window(self) -> None:
        df = _simple_dataset(50)
        config = _default_config(min_lag="10m", max_lag=0)
        with pytest.raises(ValueError, match="max_lag.*must be >= min_lag"):
            prepare_fit_data(
                df,
                input_col="x",
                target_col="y",
                time_col="timestamp",
                order_by_time=False,
                config=config,
            )

    def test_prepare_fit_data_requires_minimum_valid_windows(self) -> None:
        df = _simple_dataset(17)
        config = _default_config(max_lag="10m")
        with pytest.raises(ValueError, match="Not enough valid lag windows"):
            prepare_fit_data(
                df,
                input_col="x",
                target_col="y",
                time_col="timestamp",
                order_by_time=False,
                config=config,
            )

    def test_prepare_fit_data_outputs_expected_shapes(self) -> None:
        df = _simple_dataset(300)
        config = _default_config(max_lag="10m")
        result = prepare_fit_data(
            df,
            input_col="x",
            target_col="y",
            time_col="timestamp",
            order_by_time=False,
            config=config,
        )

        n_lags = 11
        N = 300
        max_steps = 10
        valid_windows = N - max_steps
        train_end = int(math.floor(valid_windows * 0.8))
        train_end = max(1, min(valid_windows - 1, train_end))

        assert result.design_matrix.shape == (valid_windows, n_lags)
        assert result.response_vector.shape == (valid_windows,)
        assert result.valid_indices.shape == (valid_windows,)
        assert result.x_train.shape == (train_end, n_lags)
        assert result.y_train.shape == (train_end,)
        assert result.x_valid.shape == (valid_windows - train_end, n_lags)
        assert result.y_valid.shape == (valid_windows - train_end,)
        assert result.x_train_scaled.shape == (train_end, n_lags)
        assert result.y_train_scaled.shape == (train_end,)
        assert result.x_valid_scaled.shape == (valid_windows - train_end, n_lags)
        assert result.y_valid_scaled.shape == (valid_windows - train_end,)
        assert result.no_lag_valid_scaled.shape == (valid_windows - train_end,)
        assert result.train_windows == train_end
        assert result.validation_windows == valid_windows - train_end
        assert result.total_valid_windows == valid_windows
        assert result.dt_seconds == 60.0
        assert result.min_lag_steps == 0
        assert result.max_lag_steps == 10

    def test_prepare_fit_data_preserves_existing_train_validation_split_rule(self) -> None:
        df = _simple_dataset(300)
        config = _default_config(max_lag="10m")
        result = prepare_fit_data(
            df,
            input_col="x",
            target_col="y",
            time_col="timestamp",
            order_by_time=False,
            config=config,
        )

        learner = SimplexKernelLearner(max_lag="10m", min_lag=0, dt=None)
        fit_result = learner.fit(
            df, input_col="x", target_col="y", time_col="timestamp"
        )

        coverage = fit_result.fit_data_coverage_summary
        assert coverage is not None
        assert result.train_windows == coverage.train_windows
        assert result.validation_windows == coverage.validation_windows
        assert result.total_valid_windows == coverage.valid_windows
