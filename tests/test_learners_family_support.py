"""legacy milestone tests for baselines and identifiability diagnostics."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from rtdfeatures.learners import SimplexKernelLearner
from rtdfeatures.learners._base import _numpy_loss
from rtdfeatures.learners._identifiability import WARNING_DEFINITIONS


def _make_time(n_rows: int) -> list[datetime]:
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [t0 + timedelta(minutes=i) for i in range(n_rows)]


def _fixed_delay_df(
    delay: int, n_rows: int = 500, noise: float = 0.03, seed: int = 11
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=n_rows)
    y = np.zeros(n_rows, dtype=np.float64)
    for idx in range(delay, n_rows):
        y[idx] = x[idx - delay]
    y += rng.normal(0.0, noise, size=n_rows)
    return pl.DataFrame(
        {
            "timestamp": _make_time(n_rows),
            "input_signal": x,
            "target_signal": y,
        }
    )


def test_baseline_comparison_is_populated_and_explicit() -> None:
    df = _fixed_delay_df(delay=4)
    fit = SimplexKernelLearner(max_lag=7, min_lag=0, seed=7, loss="mse").fit(
        df,
        input_col="input_signal",
        target_col="target_signal",
        time_col="timestamp",
    )
    baseline = fit.baseline_comparison
    assert np.isfinite(baseline.no_lag_validation_loss)
    assert np.isfinite(baseline.best_single_lag_validation_loss)
    assert np.isfinite(baseline.learned_validation_loss)
    assert baseline.primary_ranking_metric == "validation_loss"
    assert baseline.best_single_lag_validation_loss <= baseline.no_lag_validation_loss

    if baseline.learned_validation_loss <= baseline.best_single_lag_validation_loss:
        assert (
            "best_single_lag beats the learned kernel."
            not in fit.identifiability_report.warnings
        )
    else:
        assert "best_single_lag beats the learned kernel." in fit.identifiability_report.warnings


def test_diagnostics_fields_populate_with_valid_ranges() -> None:
    df = _fixed_delay_df(delay=3)
    fit = SimplexKernelLearner(max_lag=6, min_lag=0, seed=9).fit(
        df,
        input_col="input_signal",
        target_col="target_signal",
        time_col="timestamp",
    )
    d = fit.fit_diagnostics
    assert d.train_loss >= 0.0
    assert d.validation_loss >= 0.0
    assert d.input_variance >= 0.0
    assert d.target_variance >= 0.0
    assert np.isclose(d.kernel_weight_sum, 1.0, atol=1e-6)
    assert d.p50_lag <= d.p90_lag
    assert 0.0 <= d.tail_mass <= 1.0
    assert 0.0 <= d.boundary_mass_fraction <= 1.0


def test_identifiability_warnings_cover_required_cases() -> None:
    flat_x = np.zeros(300, dtype=np.float64)
    noisy_y = np.random.default_rng(3).normal(0.0, 1.0, size=300)
    flat_df = pl.DataFrame(
        {
            "timestamp": _make_time(300),
            "input_signal": flat_x,
            "target_signal": noisy_y,
        }
    )
    flat_fit = SimplexKernelLearner(max_lag=4, min_lag=0, seed=1, max_epochs=60).fit(
        flat_df,
        input_col="input_signal",
        target_col="target_signal",
        time_col="timestamp",
    )
    assert "Input is too flat." in flat_fit.identifiability_report.warnings

    flat_target_df = pl.DataFrame(
        {
            "timestamp": _make_time(300),
            "input_signal": np.random.default_rng(5).normal(0.0, 1.0, size=300),
            "target_signal": np.zeros(300, dtype=np.float64),
        }
    )
    flat_target_fit = SimplexKernelLearner(max_lag=4, min_lag=0, seed=2, max_epochs=60).fit(
        flat_target_df,
        input_col="input_signal",
        target_col="target_signal",
        time_col="timestamp",
    )
    assert "Target signal is too flat." in flat_target_fit.identifiability_report.warnings

    boundary_df = _fixed_delay_df(delay=0, n_rows=500, noise=0.01, seed=33)
    boundary_fit = SimplexKernelLearner(max_lag=6, min_lag=0, seed=6).fit(
        boundary_df,
        input_col="input_signal",
        target_col="target_signal",
        time_col="timestamp",
    )
    assert "Kernel piles mass at the lag boundary." in boundary_fit.identifiability_report.warnings

    diffuse_df = _fixed_delay_df(delay=2, n_rows=450, noise=1.0, seed=21)
    diffuse_fit = SimplexKernelLearner(
        max_lag=8, min_lag=0, seed=8, smoothness_penalty=10.0, max_epochs=300
    ).fit(
        diffuse_df,
        input_col="input_signal",
        target_col="target_signal",
        time_col="timestamp",
    )
    assert (
        "Kernel is too diffuse to interpret confidently."
        in diffuse_fit.identifiability_report.warnings
    )

    beat_df = _fixed_delay_df(delay=2, n_rows=500, noise=0.2, seed=41)
    beat_fit = SimplexKernelLearner(
        max_lag=8, min_lag=0, seed=4, smoothness_penalty=250.0, max_epochs=120
    ).fit(
        beat_df,
        input_col="input_signal",
        target_col="target_signal",
        time_col="timestamp",
    )
    assert "best_single_lag beats the learned kernel." in beat_fit.identifiability_report.warnings


def test_warns_on_noisy_or_weakly_explained_target() -> None:
    rng = np.random.default_rng(301)
    n_rows = 420
    x = rng.normal(0.0, 1.0, size=n_rows)
    df = pl.DataFrame(
        {
            "timestamp": _make_time(n_rows),
            "input_signal": x,
            "target_signal": x + rng.normal(0.0, 0.05, size=n_rows),
        }
    )
    fit = SimplexKernelLearner(max_lag=7, min_lag=0, seed=12, max_epochs=120).fit(
        df,
        input_col="input_signal",
        target_col="target_signal",
        time_col="timestamp",
    )
    assert "Target signal appears noisy or weakly explained." in fit.identifiability_report.warnings


def test_warns_on_large_validation_gap() -> None:
    rng = np.random.default_rng(302)
    n_rows = 480
    x = rng.normal(0.0, 1.0, size=n_rows)
    y = np.zeros(n_rows, dtype=np.float64)
    split = int(0.65 * n_rows)
    for idx in range(2, n_rows):
        if idx < split:
            y[idx] = x[idx - 2] + rng.normal(0.0, 0.01)
        else:
            y[idx] = rng.normal(0.0, 2.0)
    df = pl.DataFrame(
        {
            "timestamp": _make_time(n_rows),
            "input_signal": x,
            "target_signal": y,
        }
    )
    fit = SimplexKernelLearner(max_lag=6, min_lag=0, seed=13, max_epochs=220).fit(
        df,
        input_col="input_signal",
        target_col="target_signal",
        time_col="timestamp",
    )
    assert (
        "Validation loss is much worse than training loss."
        in fit.identifiability_report.warnings
    )


def test_no_lag_baseline_handles_non_finite_current_time_input() -> None:
    rng = np.random.default_rng(303)
    n_rows = 260
    x = rng.normal(0.0, 1.0, size=n_rows)
    y = np.zeros(n_rows, dtype=np.float64)
    for idx in range(3, n_rows):
        y[idx] = 0.6 * x[idx - 1] + 0.4 * x[idx - 3]
    y += rng.normal(0.0, 0.02, size=n_rows)
    x[220] = np.nan
    x[235] = np.inf
    df = pl.DataFrame(
        {
            "timestamp": _make_time(n_rows),
            "input_signal": x,
            "target_signal": y,
        }
    )
    fit = SimplexKernelLearner(max_lag=5, min_lag=1, seed=14, max_epochs=180).fit(
        df,
        input_col="input_signal",
        target_col="target_signal",
        time_col="timestamp",
    )
    assert np.isfinite(fit.baseline_comparison.no_lag_validation_loss)


def test_uniform_baseline_matches_known_fixture_score() -> None:
    df = _fixed_delay_df(delay=3, n_rows=420, noise=0.02, seed=404)
    learner = SimplexKernelLearner(max_lag=6, min_lag=0, seed=21, loss="mse")
    fit = learner.fit(
        df,
        input_col="input_signal",
        target_col="target_signal",
        time_col="timestamp",
    )

    input_values = df.get_column("input_signal").cast(pl.Float64).to_numpy()
    target_values = df.get_column("target_signal").cast(pl.Float64).to_numpy()
    x, y, _ = learner._build_lagged_windows(  # noqa: SLF001
        input_values=input_values,
        target_values=target_values,
        min_lag_steps=0,
        max_lag_steps=6,
    )
    train_end = int(np.floor(x.shape[0] * (1.0 - learner.validation_fraction)))
    train_end = max(1, min(x.shape[0] - 1, train_end))
    x_train = x[:train_end]
    x_valid = x[train_end:]
    y_valid = y[train_end:]
    x_stats = learner._robust_scaling_stats(x_train)  # noqa: SLF001
    y_stats = learner._robust_scaling_stats(y[:train_end])  # noqa: SLF001
    x_valid_scaled = (x_valid - x_stats.center) / x_stats.scale
    y_valid_scaled = (y_valid - y_stats.center) / y_stats.scale
    expected_uniform_loss = _numpy_loss(
        np.mean(x_valid_scaled, axis=1),
        y_valid_scaled,
        loss=learner.loss,
        huber_delta=learner.huber_delta,
    )

    assert fit.baseline_comparison.uniform_validation_loss == pytest.approx(
        expected_uniform_loss, rel=1e-12
    )


def test_exponential_baseline_search_is_deterministic() -> None:
    df = _fixed_delay_df(delay=2, n_rows=430, noise=0.04, seed=405)
    fit_a = SimplexKernelLearner(max_lag=7, min_lag=1, seed=8, loss="huber").fit(
        df,
        input_col="input_signal",
        target_col="target_signal",
        time_col="timestamp",
    )
    fit_b = SimplexKernelLearner(max_lag=7, min_lag=1, seed=8, loss="huber").fit(
        df,
        input_col="input_signal",
        target_col="target_signal",
        time_col="timestamp",
    )
    assert fit_a.baseline_comparison.exponential_validation_loss == pytest.approx(
        fit_b.baseline_comparison.exponential_validation_loss,
        rel=0.0,
        abs=1e-12,
    )
    assert fit_a.fit_provenance is not None
    assert fit_b.fit_provenance is not None
    assert fit_a.fit_provenance["exponential_baseline_best_alpha"] == fit_b.fit_provenance[
        "exponential_baseline_best_alpha"
    ]


def test_v02_baselines_are_present_and_warn_when_beating_learned() -> None:
    df = _fixed_delay_df(delay=2, n_rows=520, noise=0.08, seed=406)
    fit = SimplexKernelLearner(
        max_lag=8,
        min_lag=0,
        seed=5,
        smoothness_penalty=500.0,
        max_epochs=40,
        loss="mse",
    ).fit(
        df,
        input_col="input_signal",
        target_col="target_signal",
        time_col="timestamp",
    )
    baseline = fit.baseline_comparison
    assert np.isfinite(baseline.no_lag_validation_loss)
    assert np.isfinite(baseline.best_single_lag_validation_loss)
    assert baseline.uniform_validation_loss is not None
    assert baseline.exponential_validation_loss is not None
    assert np.isfinite(baseline.uniform_validation_loss)
    assert np.isfinite(baseline.exponential_validation_loss)

    warning_set = set(fit.identifiability_report.warnings)
    if baseline.uniform_validation_loss <= (1.0 - 0.05) * baseline.learned_validation_loss:
        assert "uniform baseline beats the learned kernel." in warning_set
    if baseline.exponential_validation_loss <= (1.0 - 0.05) * baseline.learned_validation_loss:
        assert "exponential baseline beats the learned kernel." in warning_set


def test_warning_codes_and_severity_are_stable_and_deterministic() -> None:
    flat_x = np.zeros(320, dtype=np.float64)
    noisy_y = np.random.default_rng(777).normal(0.0, 1.0, size=320)
    df = pl.DataFrame(
        {
            "timestamp": _make_time(320),
            "input_signal": flat_x,
            "target_signal": noisy_y,
        }
    )
    fit_a = SimplexKernelLearner(max_lag=5, min_lag=0, seed=91, max_epochs=80).fit(
        df,
        input_col="input_signal",
        target_col="target_signal",
        time_col="timestamp",
    )
    fit_b = SimplexKernelLearner(max_lag=5, min_lag=0, seed=91, max_epochs=80).fit(
        df,
        input_col="input_signal",
        target_col="target_signal",
        time_col="timestamp",
    )
    report_a = fit_a.identifiability_report
    report_b = fit_b.identifiability_report
    assert report_a.warnings == report_b.warnings
    assert report_a.warning_codes == report_b.warning_codes
    assert report_a.warning_severity_by_code == report_b.warning_severity_by_code
    assert "INPUT_TOO_FLAT" in report_a.warning_codes
    assert "WEAK_NO_LAG_IMPROVEMENT" in report_a.warning_codes
    assert report_a.warning_severity_by_code["INPUT_TOO_FLAT"] == "high"
    assert report_a.warning_severity_by_code["WEAK_NO_LAG_IMPROVEMENT"] == "medium"
    assert report_a.warning_severity_by_code == {
        code: expected
        for code, expected in {
            "INPUT_TOO_FLAT": "high",
            "TARGET_TOO_FLAT": "high",
            "WEAK_NO_LAG_IMPROVEMENT": "medium",
            "LARGE_VALIDATION_GAP": "high",
            "BOUNDARY_PILED_KERNEL": "medium",
            "DIFFUSE_KERNEL": "medium",
            "BEST_SINGLE_LAG_BEATS_LEARNED": "medium",
            "UNIFORM_BASELINE_BEATS_LEARNED": "medium",
            "EXPONENTIAL_BASELINE_BEATS_LEARNED": "medium",
        }.items()
        if code in report_a.warning_codes
    }


def test_warning_code_severity_contract_map_is_exact() -> None:
    expected_map = {
        "INPUT_TOO_FLAT": ("Input is too flat.", "high"),
        "TARGET_TOO_FLAT": ("Target signal is too flat.", "high"),
        "WEAK_NO_LAG_IMPROVEMENT": ("Target signal appears noisy or weakly explained.", "medium"),
        "LARGE_VALIDATION_GAP": ("Validation loss is much worse than training loss.", "high"),
        "BOUNDARY_PILED_KERNEL": ("Kernel piles mass at the lag boundary.", "medium"),
        "DIFFUSE_KERNEL": ("Kernel is too diffuse to interpret confidently.", "medium"),
        "BEST_SINGLE_LAG_BEATS_LEARNED": ("best_single_lag beats the learned kernel.", "medium"),
        "UNIFORM_BASELINE_BEATS_LEARNED": ("uniform baseline beats the learned kernel.", "medium"),
        "EXPONENTIAL_BASELINE_BEATS_LEARNED": (
            "exponential baseline beats the learned kernel.",
            "medium",
        ),
    }
    assert WARNING_DEFINITIONS == expected_map


def test_baseline_kernel_shape_and_coverage_summaries_are_present() -> None:
    df = _fixed_delay_df(delay=3, n_rows=420, noise=0.03, seed=101)
    fit = SimplexKernelLearner(max_lag=7, min_lag=0, seed=17).fit(
        df,
        input_col="input_signal",
        target_col="target_signal",
        time_col="timestamp",
    )
    baseline_summary = fit.baseline_comparison.summary_by_baseline
    for baseline_name in ("no_lag", "best_single_lag", "uniform", "exponential"):
        assert baseline_name in baseline_summary
        row = baseline_summary[baseline_name]
        assert np.isfinite(float(row["baseline_validation_loss"]))
        assert np.isfinite(float(row["learned_validation_loss"]))
        assert isinstance(row["beats_learned_by_margin"], bool)

    assert fit.kernel_shape_summary is not None
    shape = fit.kernel_shape_summary
    assert 0.0 <= shape.normalized_entropy <= 1.0
    assert 0.0 <= shape.min_weight <= shape.max_weight <= 1.0
    assert shape.concentration_hhi > 0.0
    assert shape.effective_lag_count >= 1.0

    assert fit.fit_data_coverage_summary is not None
    coverage = fit.fit_data_coverage_summary
    assert coverage.total_rows == 420
    assert coverage.valid_windows == coverage.train_windows + coverage.validation_windows
    assert 0.0 < coverage.retained_row_fraction <= 1.0
    assert 0.0 < coverage.retained_window_fraction <= 1.0


def test_strong_case_avoids_warning_flood() -> None:
    df = _fixed_delay_df(delay=4, n_rows=520, noise=0.005, seed=1102)
    fit = SimplexKernelLearner(max_lag=7, min_lag=0, seed=33, loss="mse").fit(
        df,
        input_col="input_signal",
        target_col="target_signal",
        time_col="timestamp",
    )
    assert len(fit.identifiability_report.warnings) <= 2


def test_coverage_fraction_is_lag_aware_and_bounded_when_max_lag_zero() -> None:
    df = _fixed_delay_df(delay=0, n_rows=300, noise=0.02, seed=1201)
    fit = SimplexKernelLearner(max_lag=0, min_lag=0, seed=19, loss="mse").fit(
        df,
        input_col="input_signal",
        target_col="target_signal",
        time_col="timestamp",
    )
    coverage = fit.fit_data_coverage_summary
    assert coverage is not None
    assert coverage.total_rows == 300
    assert coverage.valid_windows <= coverage.total_rows
    assert 0.0 <= coverage.retained_window_fraction <= 1.0
    assert coverage.retained_window_fraction == pytest.approx(1.0)
