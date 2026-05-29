"""legacy milestone tests for synthetic helper promotion in v0.3."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest

from rtdfeatures.features import KernelFeatureBuilder
from rtdfeatures.kernels import LearnedKernel
from rtdfeatures.learners import SimplexKernelLearner
from rtdfeatures.synthetic import (
    SyntheticDataset,
    make_baseline_challenge_dataset,
    make_boundary_kernel_dataset,
    make_delayed_exponential_kernel_dataset,
    make_diffuse_kernel_dataset,
    make_erlang_kernel_dataset,
    make_exponential_kernel_dataset,
    make_gamma_kernel_dataset,
    make_lognormal_kernel_dataset,
    make_missing_window_dataset,
    make_multi_pair_dataset,
    make_noisy_identifiable_dataset,
    make_single_delay_dataset,
    make_spread_delay_dataset,
    make_weak_identifiability_dataset,
)

ParametricFixtureHelper = Callable[..., SyntheticDataset]


def _assert_regular_grid(df: pl.DataFrame, *, time_col: str = "time") -> None:
    diffs = df.get_column(time_col).cast(pl.Int64).diff().drop_nulls().to_list()
    assert diffs
    assert len({round(float(d), 12) for d in diffs}) == 1


def test_helpers_return_stable_schema_and_row_count() -> None:
    helpers = (
        make_single_delay_dataset,
        make_spread_delay_dataset,
        make_noisy_identifiable_dataset,
        make_weak_identifiability_dataset,
        make_multi_pair_dataset,
        make_missing_window_dataset,
        make_boundary_kernel_dataset,
        make_diffuse_kernel_dataset,
        make_baseline_challenge_dataset,
    )
    for helper in helpers:
        out = helper(n_rows=120, dt=2.0, seed=5)
        assert isinstance(out, SyntheticDataset)
        assert isinstance(out.data, pl.DataFrame)
        assert out.data.height == 120
        assert out.data.columns[0] == "time"
        assert out.scenario["dt"] == 2.0


def test_seeded_generation_is_deterministic() -> None:
    a = make_multi_pair_dataset(n_rows=100, dt=1.0, seed=17)
    b = make_multi_pair_dataset(n_rows=100, dt=1.0, seed=17)
    c = make_multi_pair_dataset(n_rows=100, dt=1.0, seed=18)

    assert a.data.equals(b.data)
    assert a.true_kernels == b.true_kernels
    assert not a.data.equals(c.data)


def test_true_kernel_metadata_matches_contract() -> None:
    out = make_boundary_kernel_dataset(n_rows=140, dt=1.5, seed=9)
    kernel = out.true_kernels["input_signal->target_signal"]

    assert kernel["dt"] == 1.5
    assert kernel["min_lag"] == min(kernel["lag_steps"])
    assert kernel["max_lag"] == max(kernel["lag_steps"])
    assert abs(sum(kernel["weights"]) - 1.0) < 1e-9
    assert kernel["p50_lag"] <= kernel["p90_lag"]


def test_helpers_keep_regular_time_grid() -> None:
    helpers = (
        make_single_delay_dataset,
        make_missing_window_dataset,
        make_diffuse_kernel_dataset,
    )
    for helper in helpers:
        out = helper(n_rows=90, dt=0.5, seed=3)
        _assert_regular_grid(out.data)


def test_missing_window_introduces_nulls_without_time_irregularity() -> None:
    out = make_missing_window_dataset(
        n_rows=120,
        dt=1.0,
        seed=4,
        missing_window_start=20,
        missing_window_len=10,
    )
    assert out.data.get_column("input_signal").null_count() == 10
    assert out.data.get_column("target_signal").null_count() == 10
    _assert_regular_grid(out.data)


def test_helpers_are_usable_in_learner_and_builder_workflow() -> None:
    out = make_single_delay_dataset(n_rows=260, dt=1.0, seed=13, noise_std=0.02)
    fit = SimplexKernelLearner(max_lag=10, min_lag=0, dt="1s", seed=13, max_epochs=300).fit(
        out.data,
        input_col="input_signal",
        target_col="target_signal",
        time_col="time",
    )

    assert fit.kernel.name == "input_signal->target_signal"
    kernel = LearnedKernel(
        weights=fit.kernel.weights,
        lag_steps=fit.kernel.lag_steps,
        dt=fit.kernel.dt,
        min_lag_steps=fit.kernel.min_lag_steps,
        max_lag_steps=fit.kernel.max_lag_steps,
        name="learned",
    )
    builder = KernelFeatureBuilder(
        kernels={"learned": kernel},
        time_col="time",
        numeric_cols=["input_signal"],
    )
    features = builder.transform(out.data)
    assert "time" in features.columns
    assert any(col.startswith("learned_num_input_signal_") for col in features.columns)


def test_doc_links_and_benchmark_guidance_resolve() -> None:
    api_doc = Path("docs/08_api_design.md")
    examples_doc = Path("docs/10_examples_and_use_cases.md")
    benchmark_doc = Path("docs/benchmarks/nrtd_benchmark_layers.md")

    assert api_doc.exists()
    assert examples_doc.exists()
    assert benchmark_doc.exists()

    api_text = api_doc.read_text(encoding="utf-8")
    examples_text = examples_doc.read_text(encoding="utf-8")

    relative_ref = "benchmarks/nrtd_benchmark_layers.md"
    assert relative_ref in api_text
    assert relative_ref in examples_text
    assert "references, not synthetic helper ground truth" in api_text


def test_single_delay_rejects_negative_delay_steps() -> None:
    with pytest.raises(ValueError, match=r"delay_steps must be non-negative; got -1\."):
        make_single_delay_dataset(delay_steps=-1)


@pytest.mark.parametrize(
    ("helper", "expected_family"),
    [
        (make_exponential_kernel_dataset, "exponential"),
        (make_gamma_kernel_dataset, "gamma"),
        (make_delayed_exponential_kernel_dataset, "delayed_exponential"),
        (make_lognormal_kernel_dataset, "lognormal"),
        (make_erlang_kernel_dataset, "erlang"),
    ],
)
def test_parametric_family_fixtures_are_regular_and_expose_true_metadata(
    helper: ParametricFixtureHelper,
    expected_family: str,
) -> None:
    out = helper(n_rows=140, dt=30.0, seed=13, noise_std=0.02)
    _assert_regular_grid(out.data)
    kernel = out.true_kernels["input_signal->target_signal"]
    assert kernel["parametric_family"] == expected_family
    assert "parametric_parameters" in kernel
    assert out.scenario["params"]["family"] == expected_family


def test_new_parametric_fixtures_are_seed_deterministic() -> None:
    a = make_delayed_exponential_kernel_dataset(n_rows=120, dt=45.0, seed=17)
    b = make_delayed_exponential_kernel_dataset(n_rows=120, dt=45.0, seed=17)
    c = make_delayed_exponential_kernel_dataset(n_rows=120, dt=45.0, seed=18)
    assert a.data.equals(b.data)
    assert a.true_kernels == b.true_kernels
    assert not a.data.equals(c.data)


@pytest.mark.parametrize("invalid_shape_k", [2.5, 0, -1, "3", True])  # type: ignore[list-item]
def test_erlang_fixture_rejects_non_integer_shape_k(invalid_shape_k: object) -> None:
    with pytest.raises(
        ValueError, match=r"shape_k must be a positive integer for Erlang fixtures"
    ):
        make_erlang_kernel_dataset(shape_k=invalid_shape_k)  # type: ignore[arg-type]


def test_erlang_fixture_metadata_preserves_integer_shape_k() -> None:
    out = make_erlang_kernel_dataset(shape_k=4)
    params = out.true_kernels["input_signal->target_signal"]["parametric_parameters"]
    assert params["shape_k"] == 4
    assert isinstance(params["shape_k"], int)
