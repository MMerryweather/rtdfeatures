"""Tests for single-fit bootstrap execution."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any

import polars as pl
import pytest

from rtdfeatures.bootstrap import BlockedBootstrapConfig, bootstrap_kernel_fit
from rtdfeatures.candidates import fit_kernel_candidates
from rtdfeatures.diagnostics import KernelCandidate, KernelCandidateSet, KernelFamilyFitResult


def _toy_df(n_rows: int = 120) -> pl.DataFrame:
    start = datetime(2024, 1, 1)
    ts = [start + timedelta(seconds=idx) for idx in range(n_rows)]
    x = [float(idx % 17) for idx in range(n_rows)]
    y = [0.0] * n_rows
    for idx in range(2, n_rows):
        y[idx] = 0.7 * x[idx - 1] + 0.3 * x[idx - 2]
    return pl.DataFrame({"ts": ts, "x": x, "y": y})


def _fit_one(
    df: pl.DataFrame, candidate: KernelCandidate
) -> tuple[KernelCandidateSet, KernelFamilyFitResult]:
    candidate_set = KernelCandidateSet(
        candidate_set_id="set1",
        input_col="x",
        target_col="y",
        time_col="ts",
        candidates=(candidate,),
    )
    result = fit_kernel_candidates(df, candidate_set)
    return candidate_set, result.family_results[0]


def test_simplex_bootstrap_collects_weight_and_lag_samples() -> None:
    df = _toy_df()
    candidate = KernelCandidate(
        candidate_id="simp",
        family="simplex",
        candidate_type="empirical_learner",
        min_lag=0,
        max_lag=3,
        learner_parameters={"max_epochs": 50, "seed": 11, "learning_rate": 0.05},
    )
    candidate_set, family_result = _fit_one(df, candidate)

    boot = bootstrap_kernel_fit(
        df,
        candidate_set=candidate_set,
        family_result=family_result,
        config=BlockedBootstrapConfig(n_bootstrap=4, block_length=8, seed=7),
    )

    assert boot.n_bootstrap == 4
    assert boot.n_failed == 0
    assert boot.n_succeeded == 4
    assert len(boot.lag_summary_samples) == 4
    assert len(boot.weight_samples) == 4 * 4
    assert len(boot.parameter_samples) == 0
    assert "BOOTSTRAP_PARAMETER_PROVENANCE_MISSING" not in boot.warnings


def test_gamma_bootstrap_emits_parameter_samples() -> None:
    df = _toy_df()
    candidate = KernelCandidate(
        candidate_id="gamma",
        family="gamma",
        candidate_type="parametric_learner",
        min_lag=1,
        max_lag=4,
        learner_parameters={"max_epochs": 50, "seed": 3, "learning_rate": 0.05},
    )
    candidate_set, family_result = _fit_one(df, candidate)

    boot = bootstrap_kernel_fit(
        df,
        candidate_set=candidate_set,
        family_result=family_result,
        config=BlockedBootstrapConfig(n_bootstrap=3, block_length=7, seed=17),
    )

    assert boot.n_failed == 0
    names = {sample.parameter_name for sample in boot.parameter_samples}
    assert names == {"shape_alpha", "rate_beta"}


def test_exponential_bootstrap_is_deterministic_with_seed() -> None:
    df = _toy_df()
    candidate = KernelCandidate(
        candidate_id="exp",
        family="exponential",
        candidate_type="parametric_learner",
        min_lag=1,
        max_lag=4,
        learner_parameters={"max_epochs": 50, "seed": 21, "learning_rate": 0.05},
    )
    candidate_set, family_result = _fit_one(df, candidate)
    config = BlockedBootstrapConfig(n_bootstrap=3, block_length=6, seed=9)

    first = bootstrap_kernel_fit(
        df, candidate_set=candidate_set, family_result=family_result, config=config
    )
    second = bootstrap_kernel_fit(
        df, candidate_set=candidate_set, family_result=family_result, config=config
    )

    first_weights = [
        (s.bootstrap_id, s.lag_step, round(s.weight, 10)) for s in first.weight_samples
    ]
    second_weights = [
        (s.bootstrap_id, s.lag_step, round(s.weight, 10)) for s in second.weight_samples
    ]
    assert first_weights == second_weights


def test_fixed_kernel_bootstrap_supported() -> None:
    df = _toy_df()
    candidate = KernelCandidate(
        candidate_id="fixed",
        family="fixed_delay",
        candidate_type="fixed_kernel",
        min_lag=0,
        max_lag=3,
        fixed_parameters={"delay_steps": 1},
    )
    candidate_set, family_result = _fit_one(df, candidate)

    boot = bootstrap_kernel_fit(
        df,
        candidate_set=candidate_set,
        family_result=family_result,
        config=BlockedBootstrapConfig(n_bootstrap=3, block_length=6, seed=1),
    )

    assert boot.n_failed == 0
    assert boot.n_succeeded == 3
    assert len(boot.parameter_samples) == 0
    assert boot.family_selection_counts == {"fixed_delay": 3}


def test_failures_are_captured_with_error_text(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _toy_df()
    candidate = KernelCandidate(
        candidate_id="simp",
        family="simplex",
        candidate_type="empirical_learner",
        min_lag=0,
        max_lag=3,
        learner_parameters={"max_epochs": 20, "seed": 11, "learning_rate": 0.05},
    )
    candidate_set, family_result = _fit_one(df, candidate)

    from rtdfeatures.bootstrap.sampling import _fit_bootstrap_kernel as _original_fit

    original_fit = _original_fit

    def _patched(**kwargs: Any) -> Any:
        if kwargs["bootstrap_id"] == 1:
            raise RuntimeError("forced failure")
        return original_fit(**kwargs)

    monkeypatch.setattr(
        "rtdfeatures.bootstrap.sampling._fit_bootstrap_kernel", _patched
    )
    boot = bootstrap_kernel_fit(
        df,
        candidate_set=candidate_set,
        family_result=family_result,
        config=BlockedBootstrapConfig(n_bootstrap=3, block_length=5, seed=4),
    )

    assert boot.n_failed == 1
    assert boot.failures[0]["error"].startswith("RuntimeError: forced failure")


def test_original_result_and_candidate_are_not_mutated() -> None:
    df = _toy_df()
    candidate = KernelCandidate(
        candidate_id="exp",
        family="exponential",
        candidate_type="parametric_learner",
        min_lag=1,
        max_lag=4,
        learner_parameters={"max_epochs": 40, "seed": 21, "learning_rate": 0.05},
    )
    candidate_set, family_result = _fit_one(df, candidate)
    before_candidate = family_result.candidate.to_dict()
    assert family_result.fit_result is not None
    before_fit_prov = dict(family_result.fit_result.fit_provenance or {})

    _ = bootstrap_kernel_fit(
        df,
        candidate_set=candidate_set,
        family_result=family_result,
        config=BlockedBootstrapConfig(n_bootstrap=2, block_length=6, seed=3),
    )

    assert family_result.candidate.to_dict() == before_candidate
    assert family_result.fit_result is not None
    assert dict(family_result.fit_result.fit_provenance or {}) == before_fit_prov


def test_missing_parametric_parameter_provenance_warns() -> None:
    df = _toy_df()
    candidate = KernelCandidate(
        candidate_id="gamma",
        family="gamma",
        candidate_type="parametric_learner",
        min_lag=1,
        max_lag=4,
        learner_parameters={"max_epochs": 30, "seed": 3, "learning_rate": 0.05},
    )
    candidate_set, family_result = _fit_one(df, candidate)
    fit_result = family_result.fit_result
    assert fit_result is not None
    broken_prov = dict(fit_result.fit_provenance or {})
    broken_prov["parametric_family"] = "gamma"
    broken_prov.pop("parametric_parameters", None)
    patched_fit = replace(fit_result, fit_provenance=broken_prov)
    patched_family = replace(family_result, fit_result=patched_fit)

    boot = bootstrap_kernel_fit(
        df,
        candidate_set=candidate_set,
        family_result=patched_family,
        config=BlockedBootstrapConfig(n_bootstrap=2, block_length=6, seed=13),
    )

    assert "BOOTSTRAP_PARAMETER_PROVENANCE_MISSING" in boot.warnings


def test_empirical_bootstrap_has_no_parameter_samples_without_warning() -> None:
    df = _toy_df()
    candidate = KernelCandidate(
        candidate_id="simp",
        family="simplex",
        candidate_type="empirical_learner",
        min_lag=0,
        max_lag=3,
        learner_parameters={"max_epochs": 30, "seed": 11, "learning_rate": 0.05},
    )
    candidate_set, family_result = _fit_one(df, candidate)

    boot = bootstrap_kernel_fit(
        df,
        candidate_set=candidate_set,
        family_result=family_result,
        config=BlockedBootstrapConfig(n_bootstrap=2, block_length=5, seed=4),
    )

    assert len(boot.parameter_samples) == 0
    assert "BOOTSTRAP_PARAMETER_PROVENANCE_MISSING" not in boot.warnings
