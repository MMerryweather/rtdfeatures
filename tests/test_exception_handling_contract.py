"""Contract tests for deliberate exception-handling behavior."""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import importlib.util
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from rtdfeatures.bootstrap import (
    BlockedBootstrapConfig,
    bootstrap_kernel_candidates,
    bootstrap_kernel_fit,
)
from rtdfeatures.candidates import fit_kernel_candidates
from rtdfeatures.diagnostics import (
    KernelCandidate,
    KernelCandidateSet,
    KernelComparisonConfig,
)
from rtdfeatures.learners import SharedSimplexKernelLearner


def _load_root_module(monkeypatch: pytest.MonkeyPatch, *, version_side_effect: Exception) -> Any:
    module_path = Path("src/rtdfeatures/__init__.py").resolve()
    spec = importlib.util.spec_from_file_location("rtdfeatures_contract_root", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    def _patched_version(_name: str) -> str:
        raise version_side_effect

    monkeypatch.setattr(importlib_metadata, "version", _patched_version)
    spec.loader.exec_module(module)
    return module


def _toy_df(n_rows: int = 120) -> pl.DataFrame:
    start = datetime(2024, 1, 1)
    ts = [start + timedelta(seconds=idx) for idx in range(n_rows)]
    x = [float(idx % 17) for idx in range(n_rows)]
    y = [0.0] * n_rows
    for idx in range(2, n_rows):
        y[idx] = 0.7 * x[idx - 1] + 0.3 * x[idx - 2]
    return pl.DataFrame({"ts": ts, "x": x, "y": y})


def _candidate_set() -> KernelCandidateSet:
    return KernelCandidateSet(
        candidate_set_id="exception-contract",
        input_col="x",
        target_col="y",
        time_col="ts",
        candidates=(
            KernelCandidate(
                candidate_id="simp",
                family="simplex",
                candidate_type="empirical_learner",
                min_lag=0,
                max_lag=3,
                learner_parameters={"max_epochs": 20, "seed": 11, "learning_rate": 0.05},
            ),
        ),
    )


def test_root_version_fallback_handles_package_not_found_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_root_module(
        monkeypatch,
        version_side_effect=importlib_metadata.PackageNotFoundError("rtdfeatures"),
    )
    assert module.__version__ == "0.0.0"


def test_root_version_fallback_does_not_hide_unexpected_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        _load_root_module(monkeypatch, version_side_effect=RuntimeError("boom"))


def test_fit_kernel_candidates_does_not_swallow_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _toy_df(12)

    def _raise_interrupt(*_args: Any, **_kwargs: Any) -> Any:
        raise KeyboardInterrupt("stop")

    monkeypatch.setattr("rtdfeatures.candidates.fitting._fit_one_candidate", _raise_interrupt)
    with pytest.raises(KeyboardInterrupt, match="stop"):
        fit_kernel_candidates(df, _candidate_set())


def test_bootstrap_kernel_fit_failure_records_include_error_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _toy_df()
    candidate_set = _candidate_set()
    family_result = fit_kernel_candidates(df, candidate_set).family_results[0]

    def _raise_failure(**_kwargs: Any) -> Any:
        raise RuntimeError("forced failure")

    monkeypatch.setattr("rtdfeatures.bootstrap.sampling._fit_bootstrap_kernel", _raise_failure)
    boot = bootstrap_kernel_fit(
        df,
        candidate_set=candidate_set,
        family_result=family_result,
        config=BlockedBootstrapConfig(n_bootstrap=2, block_length=5, seed=4),
    )

    assert boot.n_failed == 2
    assert boot.failures[0]["error_type"] == "RuntimeError"
    assert boot.failures[0]["error"].startswith("RuntimeError: forced failure")


def test_bootstrap_kernel_fit_does_not_swallow_system_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _toy_df()
    candidate_set = _candidate_set()
    family_result = fit_kernel_candidates(df, candidate_set).family_results[0]

    def _raise_exit(**_kwargs: Any) -> Any:
        raise SystemExit(2)

    monkeypatch.setattr("rtdfeatures.bootstrap.sampling._fit_bootstrap_kernel", _raise_exit)
    with pytest.raises(SystemExit):
        bootstrap_kernel_fit(
            df,
            candidate_set=candidate_set,
            family_result=family_result,
            config=BlockedBootstrapConfig(n_bootstrap=1, block_length=5, seed=4),
        )


def test_bootstrap_kernel_candidates_failure_records_include_error_type() -> None:
    df = _toy_df()
    broken = KernelCandidate(
        candidate_id="bad",
        family="unsupported_family",
        candidate_type="parametric_learner",
        min_lag=0,
        max_lag=3,
        learner_parameters={"max_epochs": 10, "seed": 1},
    )
    candidate_set = KernelCandidateSet(
        candidate_set_id="comparison-failure",
        input_col="x",
        target_col="y",
        time_col="ts",
        candidates=(_candidate_set().candidates[0], broken),
    )
    boot = bootstrap_kernel_candidates(
        df,
        candidate_set=candidate_set,
        comparison_config=KernelComparisonConfig(),
        config=BlockedBootstrapConfig(n_bootstrap=2, block_length=5, seed=4),
    )

    assert any(item.get("error_type") == "ComparisonStageFailure" for item in boot.failures)


def test_shared_learner_does_not_swallow_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _toy_df(40).rename({"ts": "timestamp", "x": "input_a", "y": "target_a"})
    learner = SharedSimplexKernelLearner(max_lag=3, min_lag=0, seed=7)

    def _raise_interrupt(self: Any, *_args: Any, **_kwargs: Any) -> Any:
        raise KeyboardInterrupt("interrupt")

    monkeypatch.setattr("rtdfeatures.learners.simplex.SimplexKernelLearner.fit", _raise_interrupt)
    with pytest.raises(KeyboardInterrupt, match="interrupt"):
        learner.fit(
            df,
            input_cols=["input_a"],
            target_cols=["target_a"],
            time_col="timestamp",
        )
