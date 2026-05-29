"""Tests for candidate-set bootstrap execution."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any

import polars as pl
import pytest

from rtdfeatures.bootstrap import (
    BlockedBootstrapConfig,
    bootstrap_kernel_candidates,
)
from rtdfeatures.bootstrap.sampling import _fit_bootstrap_kernel
from rtdfeatures.diagnostics import (
    KernelCandidate,
    KernelCandidateSet,
    KernelComparisonConfig,
)


def _toy_df(n_rows: int = 140) -> pl.DataFrame:
    start = datetime(2024, 1, 1)
    ts = [start + timedelta(seconds=idx) for idx in range(n_rows)]
    x = [float(idx % 19) for idx in range(n_rows)]
    y = [0.0] * n_rows
    for idx in range(3, n_rows):
        y[idx] = 0.2 * x[idx - 1] + 0.5 * x[idx - 2] + 0.3 * x[idx - 3]
    return pl.DataFrame({"ts": ts, "x": x, "y": y})


def _candidate_set() -> KernelCandidateSet:
    return KernelCandidateSet(
        candidate_set_id="set",
        input_col="x",
        target_col="y",
        time_col="ts",
        candidates=(
            KernelCandidate(
                candidate_id="simp",
                family="simplex",
                candidate_type="empirical_learner",
                min_lag=0,
                max_lag=4,
                learner_parameters={
                    "max_epochs": 30,
                    "learning_rate": 0.05,
                    "seed": 11,
                    "loss": "huber",
                    "huber_delta": 1.0,
                    "validation_fraction": 0.2,
                },
            ),
            KernelCandidate(
                candidate_id="exp",
                family="exponential",
                candidate_type="parametric_learner",
                min_lag=0,
                max_lag=4,
                learner_parameters={
                    "max_epochs": 30,
                    "learning_rate": 0.05,
                    "seed": 21,
                    "loss": "huber",
                    "huber_delta": 1.0,
                    "validation_fraction": 0.2,
                },
            ),
        ),
    )


def test_bootstrap_uses_same_train_sample_for_all_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _toy_df()
    candidate_set = _candidate_set()
    seen_train_indices: dict[int, dict[str, tuple[int, ...]]] = {}

    def _patched(**kwargs: Any) -> Any:
        family_result = kwargs["family_result"]
        bootstrap_id = kwargs["bootstrap_id"]
        train_indices = tuple(int(v) for v in kwargs["train_indices"])
        per_bootstrap = seen_train_indices.setdefault(bootstrap_id, {})
        per_bootstrap[family_result.candidate.candidate_id] = train_indices
        return _fit_bootstrap_kernel(**kwargs)

    monkeypatch.setattr(
        "rtdfeatures.bootstrap.sampling._fit_bootstrap_kernel", _patched
    )
    _ = bootstrap_kernel_candidates(
        df,
        candidate_set=candidate_set,
        comparison_config=KernelComparisonConfig(),
        config=BlockedBootstrapConfig(n_bootstrap=4, block_length=7, seed=13),
    )

    assert seen_train_indices
    for per_bootstrap in seen_train_indices.values():
        assert set(per_bootstrap) == {"simp", "exp"}
        assert per_bootstrap["simp"] == per_bootstrap["exp"]


def test_bootstrap_keeps_validation_window_fixed_and_counts_selection() -> None:
    df = _toy_df()
    candidate_set = _candidate_set()
    boot = bootstrap_kernel_candidates(
        df,
        candidate_set=candidate_set,
        comparison_config=KernelComparisonConfig(),
        config=BlockedBootstrapConfig(n_bootstrap=3, block_length=6, seed=9),
    )

    assert boot.n_bootstrap == 3
    assert boot.bootstrap_config["validation_window_size"] > 0
    assert sum(boot.family_selection_counts.values()) == boot.n_succeeded
    candidate_counts = boot.bootstrap_config["candidate_selection_counts"]
    assert isinstance(candidate_counts, dict)
    assert sum(candidate_counts.values()) == boot.n_succeeded


def test_failed_candidate_iteration_is_captured() -> None:
    df = _toy_df()
    broken = replace(
        _candidate_set().candidates[1],
        family="unsupported_family",
    )
    candidate_set = replace(
        _candidate_set(),
        candidates=(_candidate_set().candidates[0], broken),
    )

    boot = bootstrap_kernel_candidates(
        df,
        candidate_set=candidate_set,
        comparison_config=KernelComparisonConfig(),
        config=BlockedBootstrapConfig(n_bootstrap=2, block_length=6, seed=2),
    )

    assert boot.n_failed == 0
    assert boot.n_succeeded == 2
    assert any(item.get("candidate_id") == "exp" for item in boot.failures)


def test_candidate_and_config_immutability() -> None:
    df = _toy_df()
    candidate_set = _candidate_set()
    comparison_config = KernelComparisonConfig()
    before_candidates = tuple(candidate.to_dict() for candidate in candidate_set.candidates)
    before_cfg = comparison_config.to_dict()

    _ = bootstrap_kernel_candidates(
        df,
        candidate_set=candidate_set,
        comparison_config=comparison_config,
        config=BlockedBootstrapConfig(n_bootstrap=2, block_length=6, seed=5),
    )

    assert tuple(candidate.to_dict() for candidate in candidate_set.candidates) == before_candidates
    assert comparison_config.to_dict() == before_cfg


def test_unstable_family_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _toy_df()
    candidate_set = _candidate_set()

    def _patched(**kwargs: Any) -> Any:
        family_result = kwargs["family_result"]
        bootstrap_id = kwargs["bootstrap_id"]
        kernel, _valid, params = _fit_bootstrap_kernel(**kwargs)
        if family_result.candidate.candidate_id == "simp":
            valid_loss = 0.1 if bootstrap_id % 2 == 0 else 2.0
        else:
            valid_loss = 2.0 if bootstrap_id % 2 == 0 else 0.1
        return kernel, valid_loss, params

    monkeypatch.setattr(
        "rtdfeatures.bootstrap.sampling._fit_bootstrap_kernel", _patched
    )
    boot = bootstrap_kernel_candidates(
        df,
        candidate_set=candidate_set,
        comparison_config=KernelComparisonConfig(),
        config=BlockedBootstrapConfig(n_bootstrap=6, block_length=6, seed=11),
    )

    assert "BOOTSTRAP_FAMILY_UNSTABLE" in boot.warnings


def test_bootstrap_honors_loss_tolerance_fraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _toy_df()
    candidate_set = _candidate_set()

    def _patched(**kwargs: Any) -> Any:
        family_result = kwargs["family_result"]
        kernel, _valid, params = _fit_bootstrap_kernel(**kwargs)
        if family_result.candidate.candidate_id == "exp":
            valid_loss = 1.00
        else:
            valid_loss = 1.01
        return kernel, valid_loss, params

    monkeypatch.setattr(
        "rtdfeatures.bootstrap.sampling._fit_bootstrap_kernel", _patched
    )

    no_tolerance = bootstrap_kernel_candidates(
        df,
        candidate_set=candidate_set,
        comparison_config=KernelComparisonConfig(loss_tolerance_fraction=0.0),
        config=BlockedBootstrapConfig(n_bootstrap=4, block_length=6, seed=7),
    )
    with_tolerance = bootstrap_kernel_candidates(
        df,
        candidate_set=candidate_set,
        comparison_config=KernelComparisonConfig(loss_tolerance_fraction=0.02),
        config=BlockedBootstrapConfig(n_bootstrap=4, block_length=6, seed=7),
    )

    assert no_tolerance.bootstrap_config["candidate_selection_counts"] == {"exp": 4}
    assert with_tolerance.bootstrap_config["candidate_selection_counts"] == {"simp": 4}
