from __future__ import annotations

import re
from pathlib import Path

import pytest
from examples.nrtd_laminar_flow_worked_example import main as build_nrtd_worked_example
from examples.plant_first_example_gallery import (
    LearnerOutcome,
    _choose_recommended,
    _fit_quality_gate_for_positive,
    _kernel_lag_stats,
    _mini_recommendation,
    _shape_contradiction_for_mini,
    main,
)

from rtdfeatures.diagnostics.fit import (
    BaselineComparison,
    FitDiagnostics,
    IdentifiabilityReport,
    KernelFitResult,
    KernelShapeSummary,
)
from rtdfeatures.kernels.base import Kernel

ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = ROOT / "docs/examples/generated"
GALLERY_MD = ROOT / "docs/examples/plant_first_gallery.md"
SCENARIO_PNGS = (
    GENERATED_DIR / "fit_conveyor_signals.png",
    GENERATED_DIR / "fit_conveyor_kernel.png",
    GENERATED_DIR / "fit_conveyor_fit.png",
    GENERATED_DIR / "fit_cstr_signals.png",
    GENERATED_DIR / "fit_cstr_kernel.png",
    GENERATED_DIR / "fit_cstr_fit.png",
    GENERATED_DIR / "fit_flotation_banks_signals.png",
    GENERATED_DIR / "fit_flotation_banks_kernel.png",
    GENERATED_DIR / "fit_flotation_banks_fit.png",
    GENERATED_DIR / "fit_tanks_in_series_signals.png",
    GENERATED_DIR / "fit_tanks_in_series_kernel.png",
    GENERATED_DIR / "fit_tanks_in_series_fit.png",
    GENERATED_DIR / "fit_closed_loop_crushing_signals.png",
    GENERATED_DIR / "fit_closed_loop_crushing_kernel.png",
    GENERATED_DIR / "fit_closed_loop_crushing_fit.png",
    GENERATED_DIR / "fit_bounded_holdup_tank_signals.png",
    GENERATED_DIR / "fit_bounded_holdup_tank_kernel.png",
    GENERATED_DIR / "fit_bounded_holdup_tank_fit.png",
    GENERATED_DIR / "fit_mini_flowsheet_signals.png",
    GENERATED_DIR / "fit_mini_flowsheet_kernel.png",
    GENERATED_DIR / "fit_mini_flowsheet_fit.png",
)


def test_plant_first_generator_writes_all_artifacts() -> None:
    main()
    assert GALLERY_MD.exists()
    assert not (GENERATED_DIR / "plant_first_coverage_table.md").exists()
    for png_path in SCENARIO_PNGS:
        assert png_path.exists()


def test_mini_flowsheet_report_fields_and_mapping_present() -> None:
    main()
    report = GALLERY_MD.read_text()
    assert "## Mini Flowsheet" in report
    assert "recommended_kernel" in report
    assert "recommendation_status" in report
    assert "recommendation_reason" in report
    assert 'input_col="feed_copper_grade"' in report
    assert 'target_col="cleaner_copper_grade"' in report
    assert 'category_cols=["ore_type"]' in report
    assert 'weight_col="feed_mass"' in report
    assert "Ore campaign transition row" in report
    assert "Future work boundary" in report


def test_coverage_table_contains_each_public_kernel_family() -> None:
    main()
    report = GALLERY_MD.read_text()
    assert "## Coverage Table" in report
    for family in (
        "FixedDelayKernelLearner",
        "ExponentialKernelLearner",
        "GammaKernelLearner",
        "ErlangKernelLearner",
        "LogNormalKernelLearner",
        "DelayedExponentialKernelLearner",
        "UniformKernelLearner",
    ):
        assert f"`{family}`" in report


def test_generated_gallery_contains_fit_quality_gate_fields() -> None:
    main()
    report = GALLERY_MD.read_text()
    required_fields = (
        "fit_quality_gate_passed",
        "fit_quality_gate_reason",
        "fit_quality_selected_kernel",
        "fit_quality_selected_validation_loss",
        "fit_quality_selected_no_lag",
        "fit_quality_selected_best_single_lag",
        "fit_quality_selected_warning_codes",
        "fit_quality_selected_mean_lag_seconds",
        "fit_quality_selected_p50_lag_seconds",
        "fit_quality_selected_p90_lag_seconds",
    )
    for field in required_fields:
        assert field in report


def test_failed_fit_quality_gate_uses_failure_recommendation_reason() -> None:
    main()
    report = GALLERY_MD.read_text()
    assert "- `recommendation_status`: `not_recommended`" in report
    assert "- `recommendation_reason`: `fit quality gate failed:" in report


def test_coverage_table_header_uses_text_alignment_for_scenario() -> None:
    main()
    report = GALLERY_MD.read_text()
    assert "|---|---|---:|---|" in report


def test_nrtd_worked_example_keeps_laminar_as_end_to_end_boundary() -> None:
    wrapper = (ROOT / "docs/examples/nrtd_laminar_flow_worked_example.md").read_text()
    assert "laminar_flow" in wrapper
    assert "adler" in wrapper
    assert "cholette" in wrapper
    assert "dispersion" in wrapper
    assert "reference-only" in wrapper


def test_generated_docs_and_new_wrappers_do_not_use_internal_plan_labels() -> None:
    main()
    banned = re.compile(r"\b(ws\d+|wp\d+|ms\d+|workstream|milestone)\b", re.IGNORECASE)
    targets = (
        GALLERY_MD,
        ROOT / "docs/examples/plant_first_gallery.md",
        ROOT / "docs/examples/nrtd_laminar_flow_worked_example.md",
    )
    for path in targets:
        assert not banned.search(path.read_text()), f"internal label found in {path}"


def test_generated_pngs_are_embedded_in_gallery() -> None:
    main()
    report = GALLERY_MD.read_text()
    for png_path in SCENARIO_PNGS:
        assert png_path.exists()
        assert png_path.stat().st_size > 0
        assert f"(generated/{png_path.name})" in report


def test_nrtd_worked_example_embeds_plots_and_fit_evidence() -> None:
    build_nrtd_worked_example()
    report = (ROOT / "docs/examples/nrtd_laminar_flow_worked_example.md").read_text()
    assert "nrtd_laminar_intro_timeseries.png" in report
    assert "nrtd_laminar_observed_vs_fit.png" in report
    assert "nrtd_laminar_kernel_profile.png" in report
    assert "Fit RMSE" in report
    assert "Observed/predicted correlation" in report


def test_generated_example_docs_do_not_reference_svg_files() -> None:
    for md_path in GENERATED_DIR.glob("*.md"):
        text = md_path.read_text()
        assert ".svg" not in text, f"unexpected SVG reference in {md_path.name}"


def _k(weights: list[float], lag_steps: list[int] | None = None, dt: float = 1.0) -> Kernel:
    if lag_steps is None:
        lag_steps = list(range(len(weights)))
    return Kernel(
        weights=tuple(weights),
        lag_steps=tuple(lag_steps),
        dt=dt,
        min_lag_steps=min(lag_steps),
        max_lag_steps=max(lag_steps),
    )


def _fd(
    validation_loss: float = 0.1,
    kernel_weight_sum: float = 1.0,
) -> FitDiagnostics:
    return FitDiagnostics(
        train_loss=validation_loss,
        validation_loss=validation_loss,
        input_variance=1.0,
        target_variance=1.0,
        kernel_weight_sum=kernel_weight_sum,
        mean_lag=0.0,
        p50_lag=0.0,
        p90_lag=0.0,
        tail_mass=0.0,
        boundary_mass_fraction=0.0,
    )


def _ir(
    warning_codes: tuple[str, ...] = (),
    warning_severity_by_code: dict[str, str] | None = None,
) -> IdentifiabilityReport:
    return IdentifiabilityReport(
        warnings=(),
        is_reliable=True,
        warning_codes=warning_codes,
        warning_severity_by_code=warning_severity_by_code or {},
    )


def _bc(
    no_lag: float = 0.12,
    best_single_lag: float = 0.11,
    learned: float = 0.1,
) -> BaselineComparison:
    return BaselineComparison(
        no_lag_validation_loss=no_lag,
        best_single_lag_validation_loss=best_single_lag,
        learned_validation_loss=learned,
    )


def _kr(
    kernel: Kernel | None = None,
    validation_loss: float = 0.1,
    no_lag: float = 0.12,
    best_single_lag: float = 0.11,
    warning_codes: tuple[str, ...] = (),
    warning_severity_by_code: dict[str, str] | None = None,
    shape_summary: KernelShapeSummary | None = None,
) -> KernelFitResult:
    if kernel is None:
        kernel = _k([1.0], [0])
    return KernelFitResult(
        kernel=kernel,
        fit_diagnostics=_fd(
            validation_loss=validation_loss,
            kernel_weight_sum=float(sum(kernel.weights)),
        ),
        identifiability_report=_ir(
            warning_codes=warning_codes,
            warning_severity_by_code=warning_severity_by_code,
        ),
        baseline_comparison=_bc(
            no_lag=no_lag,
            best_single_lag=best_single_lag,
            learned=validation_loss,
        ),
        kernel_shape_summary=shape_summary,
    )


def _lo(
    name: str = "TestLearner",
    validation_loss: float = 0.1,
    no_lag: float = 0.12,
    best_single_lag: float = 0.11,
    warning_codes: tuple[str, ...] = (),
    warning_severity_by_code: dict[str, str] | None = None,
    kernel: Kernel | None = None,
    shape_summary: KernelShapeSummary | None = None,
    recommendation_status: str = "reference_only",
    recommendation_reason: str = "test fixture",
) -> LearnerOutcome:
    return LearnerOutcome(
        learner_name=name,
        fit_result=_kr(
            kernel=kernel,
            validation_loss=validation_loss,
            no_lag=no_lag,
            best_single_lag=best_single_lag,
            warning_codes=warning_codes,
            warning_severity_by_code=warning_severity_by_code,
            shape_summary=shape_summary,
        ),
        recommendation_status=recommendation_status,
        recommendation_reason=recommendation_reason,
    )


# - _kernel_lag_stats -

class TestKernelLagStats:

    def test_known_mean_p50_p90(self) -> None:
        kernel = _k([0.1, 0.3, 0.4, 0.2], [0, 1, 2, 3], dt=5.0)
        mean, p50, p90 = _kernel_lag_stats(kernel)
        assert mean == pytest.approx((0.1 * 0 + 0.3 * 1 + 0.4 * 2 + 0.2 * 3) * 5.0)
        assert p50 == pytest.approx(2 * 5.0)
        assert p90 == pytest.approx(3 * 5.0)

    def test_single_weight_kernel(self) -> None:
        kernel = _k([1.0], [4], dt=2.0)
        mean, p50, p90 = _kernel_lag_stats(kernel)
        assert mean == pytest.approx(4 * 2.0)
        assert p50 == pytest.approx(4 * 2.0)
        assert p90 == pytest.approx(4 * 2.0)

    def test_p50_and_p90_same_lag_different_p50_check(self) -> None:
        kernel = _k([0.6, 0.4], [0, 1], dt=3.0)
        mean, p50, p90 = _kernel_lag_stats(kernel)
        assert p50 == pytest.approx(0.0)
        assert p90 == pytest.approx(1 * 3.0)


# - _shape_contradiction_for_mini -

class TestShapeContradictionForMini:

    def test_few_weights_returns_true(self) -> None:
        result = _kr(kernel=_k([0.5, 0.5], [0, 1]))
        assert _shape_contradiction_for_mini(result) is True

    def test_high_entropy_low_max_weight_returns_true(self) -> None:
        w = tuple(0.125 for _ in range(8))
        result = _kr(
            kernel=_k(list(w), list(range(8))),
            shape_summary=KernelShapeSummary(
                normalized_entropy=0.99,
                max_weight=0.15,
                min_weight=0.125,
                concentration_hhi=0.125,
                effective_lag_count=8.0,
            ),
        )
        assert _shape_contradiction_for_mini(result) is True

    def test_low_coeff_var_returns_true(self) -> None:
        result = _kr(kernel=_k([0.32, 0.36, 0.32], [0, 1, 2]))
        assert _shape_contradiction_for_mini(result) is True

    def test_no_contradiction_returns_false(self) -> None:
        result = _kr(
            kernel=_k([0.01, 0.09, 0.80, 0.10], [0, 1, 2, 3]),
            shape_summary=KernelShapeSummary(
                normalized_entropy=0.5,
                max_weight=0.8,
                min_weight=0.01,
                concentration_hhi=0.65,
                effective_lag_count=2.0,
            ),
        )
        assert _shape_contradiction_for_mini(result) is False


# - _fit_quality_gate_for_positive -

class TestFitQualityGateForPositive:

    def test_empty_positive_outcomes(self) -> None:
        outcomes = (_lo(name="OtherLearner", validation_loss=0.3),)
        passed, reason, selected = _fit_quality_gate_for_positive(
            outcomes, positive_kernel_names=("PositiveLearner",)
        )
        assert passed is False
        assert "no configured positive kernel" in reason
        assert selected.learner_name == "OtherLearner"

    def test_kernel_validate_exception(self) -> None:
        bad_kernel = Kernel(
            weights=(1.0,), lag_steps=(0,), dt=0.0,
            min_lag_steps=0, max_lag_steps=0,
        )
        outcomes = (_lo(
            name="PositiveLearner", kernel=bad_kernel,
        ),)
        passed, reason, _ = _fit_quality_gate_for_positive(
            outcomes, positive_kernel_names=("PositiveLearner",)
        )
        assert passed is False
        assert "invalid kernel constraints" in reason

    def test_non_finite_validation_loss(self) -> None:
        outcomes = (_lo(
            name="PositiveLearner", validation_loss=float("nan"),
        ),)
        passed, reason, _ = _fit_quality_gate_for_positive(
            outcomes, positive_kernel_names=("PositiveLearner",)
        )
        assert passed is False
        assert "non-finite" in reason

    def test_best_single_lag_beats_warning(self) -> None:
        outcomes = (_lo(
            name="PositiveLearner",
            warning_codes=("BEST_SINGLE_LAG_BEATS_LEARNED",),
        ),)
        passed, reason, _ = _fit_quality_gate_for_positive(
            outcomes, positive_kernel_names=("PositiveLearner",)
        )
        assert passed is False
        assert "best_single_lag beats" in reason

    def test_best_single_lag_beats_by_margin(self) -> None:
        outcomes = (_lo(
            name="PositiveLearner",
            validation_loss=0.2,
            best_single_lag=0.1,
            no_lag=0.1,
        ),)
        passed, reason, _ = _fit_quality_gate_for_positive(
            outcomes, positive_kernel_names=("PositiveLearner",)
        )
        assert passed is False
        assert "best_single_lag beats selected positive kernel by margin" in reason

    def test_no_lag_beats_by_margin(self) -> None:
        outcomes = (_lo(
            name="PositiveLearner",
            validation_loss=0.2,
            no_lag=0.1,
        ),)
        passed, reason, _ = _fit_quality_gate_for_positive(
            outcomes, positive_kernel_names=("PositiveLearner",)
        )
        assert passed is False
        assert "no_lag beats" in reason

    def test_best_single_lag_not_finite_skips_margin_check(self) -> None:
        outcomes = (_lo(
            name="PositiveLearner",
            best_single_lag=float("inf"),
        ),)
        passed, reason, _ = _fit_quality_gate_for_positive(
            outcomes, positive_kernel_names=("PositiveLearner",)
        )
        assert "best_single_lag beats selected positive kernel by margin" not in reason

    def test_no_lag_not_finite_skips_margin_check(self) -> None:
        outcomes = (_lo(
            name="PositiveLearner",
            no_lag=float("inf"),
        ),)
        passed, reason, _ = _fit_quality_gate_for_positive(
            outcomes, positive_kernel_names=("PositiveLearner",)
        )
        assert "no_lag beats selected positive kernel by margin" not in reason

    def test_all_checks_pass(self) -> None:
        outcomes = (_lo(
            name="PositiveLearner",
            validation_loss=0.1,
            no_lag=0.12,
            best_single_lag=0.11,
        ),)
        passed, reason, selected = _fit_quality_gate_for_positive(
            outcomes, positive_kernel_names=("PositiveLearner",)
        )
        assert passed is True
        assert "passes" in reason
        assert selected.learner_name == "PositiveLearner"


# - _mini_recommendation -

class TestMiniRecommendation:

    def test_kernel_validate_exception(self) -> None:
        bad_kernel = Kernel(
            weights=(1.0,), lag_steps=(0,), dt=0.0,
            min_lag_steps=0, max_lag_steps=0,
        )
        outcomes = (_lo(kernel=bad_kernel),)
        reviewed = _mini_recommendation(outcomes)
        assert reviewed[0].recommendation_status == "not_recommended"
        assert "invalid kernel constraints" in reviewed[0].recommendation_reason

    def test_non_finite_validation_loss(self) -> None:
        outcomes = (_lo(validation_loss=float("nan")),)
        reviewed = _mini_recommendation(outcomes)
        assert reviewed[0].recommendation_status == "not_recommended"
        assert "non-finite" in reviewed[0].recommendation_reason

    def test_high_severity_warning(self) -> None:
        outcomes = (_lo(
            warning_codes=("IDENTIFIABILITY_LOW",),
            warning_severity_by_code={"IDENTIFIABILITY_LOW": "high"},
        ),)
        reviewed = _mini_recommendation(outcomes)
        assert reviewed[0].recommendation_status == "not_recommended"
        assert "high-severity" in reviewed[0].recommendation_reason

    def test_best_single_lag_beats_learned(self) -> None:
        outcomes = (_lo(
            warning_codes=("BEST_SINGLE_LAG_BEATS_LEARNED",),
        ),)
        reviewed = _mini_recommendation(outcomes)
        assert reviewed[0].recommendation_status == "not_recommended"
        assert "best_single_lag baseline beats" in reviewed[0].recommendation_reason

    def test_no_lag_baseline_beats_by_margin(self) -> None:
        outcomes = (_lo(
            validation_loss=0.2,
            no_lag=0.1,
        ),)
        reviewed = _mini_recommendation(outcomes)
        assert reviewed[0].recommendation_status == "not_recommended"
        assert "no_lag baseline beats" in reviewed[0].recommendation_reason

    def test_best_single_lag_baseline_beats_by_margin(self) -> None:
        outcomes = (_lo(
            validation_loss=0.2,
            best_single_lag=0.1,
        ),)
        reviewed = _mini_recommendation(outcomes)
        assert reviewed[0].recommendation_status == "not_recommended"
        assert "best_single_lag baseline beats" in reviewed[0].recommendation_reason

    def test_shape_contradiction(self) -> None:
        outcomes = (_lo(kernel=_k([0.5, 0.5], [0, 1])),)
        reviewed = _mini_recommendation(outcomes)
        assert reviewed[0].recommendation_status == "not_recommended"
        assert "shape check" in reviewed[0].recommendation_reason

    def test_no_lag_not_finite_skips_delta_check(self) -> None:
        outcomes = (_lo(
            validation_loss=0.2,
            no_lag=float("nan"),
        ),)
        reviewed = _mini_recommendation(outcomes)
        assert "no_lag baseline beats" not in reviewed[0].recommendation_reason

    def test_no_lag_small_delta_skips_reason(self) -> None:
        outcomes = (_lo(
            validation_loss=0.1,
            no_lag=0.099,
        ),)
        reviewed = _mini_recommendation(outcomes)
        assert "no_lag baseline beats" not in reviewed[0].recommendation_reason

    def test_best_single_lag_not_finite_skips_delta_check(self) -> None:
        outcomes = (_lo(
            validation_loss=0.2,
            best_single_lag=float("nan"),
        ),)
        reviewed = _mini_recommendation(outcomes)
        assert "best_single_lag baseline beats" not in reviewed[0].recommendation_reason

    def test_best_single_lag_small_delta_skips_reason(self) -> None:
        outcomes = (_lo(
            validation_loss=0.1,
            best_single_lag=0.099,
        ),)
        reviewed = _mini_recommendation(outcomes)
        assert "best_single_lag baseline beats" not in reviewed[0].recommendation_reason

    def test_no_high_severity_warning_skips_reason(self) -> None:
        outcomes = (_lo(
            warning_codes=("IDENTIFIABILITY_LOW",),
            warning_severity_by_code={"IDENTIFIABILITY_LOW": "medium"},
        ),)
        reviewed = _mini_recommendation(outcomes)
        assert "high-severity" not in reviewed[0].recommendation_reason

    def test_recommended_clean(self) -> None:
        outcomes = (_lo(
            validation_loss=0.1,
            no_lag=0.12,
            best_single_lag=0.11,
            kernel=_k([0.01, 0.09, 0.80, 0.10], [0, 1, 2, 3]),
            shape_summary=KernelShapeSummary(
                normalized_entropy=0.5,
                max_weight=0.8,
                min_weight=0.01,
                concentration_hhi=0.65,
                effective_lag_count=2.0,
            ),
        ),)
        reviewed = _mini_recommendation(outcomes)
        assert reviewed[0].recommendation_status == "recommended"


# - _choose_recommended -

class TestChooseRecommended:

    def test_with_recommended_picks_lowest_loss(self) -> None:
        outcomes = (
            _lo(name="High", validation_loss=0.5, recommendation_status="recommended"),
            _lo(name="Low", validation_loss=0.1, recommendation_status="recommended"),
            _lo(name="Other", validation_loss=0.3, recommendation_status="not_recommended"),
        )
        chosen = _choose_recommended(outcomes)
        assert chosen.learner_name == "Low"

    def test_without_recommended_picks_lowest_loss_from_all(self) -> None:
        outcomes = (
            _lo(name="A", validation_loss=0.5, recommendation_status="not_recommended"),
            _lo(name="B", validation_loss=0.1, recommendation_status="not_recommended"),
            _lo(name="C", validation_loss=0.3, recommendation_status="not_recommended"),
        )
        chosen = _choose_recommended(outcomes)
        assert chosen.learner_name == "B"
