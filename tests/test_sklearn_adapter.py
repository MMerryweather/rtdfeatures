import pytest

pytest.importorskip("sklearn")

import numpy as np
import polars as pl
from sklearn.base import clone  # type: ignore[import-untyped, import-not-found]
from sklearn.exceptions import NotFittedError  # type: ignore[import-untyped, import-not-found]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped, import-not-found]

import rtdfeatures
from rtdfeatures.integrations.sklearn import KernelFeatureTransformer
from rtdfeatures.kernels import FixedDelayKernel
from rtdfeatures.learners import SimplexKernelLearner
from rtdfeatures.synthetic import make_single_delay_dataset


def test_adapter_not_root_exported() -> None:
    assert not hasattr(rtdfeatures, "KernelFeatureTransformer")


def test_fixed_kernel_mode_with_pandas_input() -> None:
    dataset = make_single_delay_dataset(n_rows=50)
    kernel = FixedDelayKernel(delay_steps=6, max_lag_steps=10, dt=1.0)
    transformer = KernelFeatureTransformer(
        time_col="time",
        kernels={"feature": kernel},
        numeric_cols=["input_signal"],
        return_type="pandas",
    )
    df_pd = dataset.data.to_pandas()
    transformer.fit(df_pd)
    result = transformer.transform(df_pd)
    import pandas as pd  # type: ignore[import-untyped, import-not-found]
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 50


def test_fixed_kernel_mode_with_polars_input() -> None:
    dataset = make_single_delay_dataset(n_rows=50)
    kernel = FixedDelayKernel(delay_steps=6, max_lag_steps=10, dt=1.0)
    transformer = KernelFeatureTransformer(
        time_col="time",
        kernels={"feature": kernel},
        numeric_cols=["input_signal"],
        return_type="polars",
    )
    transformer.fit(dataset.data)
    result = transformer.transform(dataset.data)
    assert isinstance(result, pl.DataFrame)
    assert result.shape[0] == 50


def test_learner_mode_with_pandas_input() -> None:
    dataset = make_single_delay_dataset(n_rows=800)
    learner = SimplexKernelLearner(max_lag="10m", max_epochs=20, seed=42)
    transformer = KernelFeatureTransformer(
        time_col="time",
        learner=learner,
        input_col="input_signal",
        target_col="target_signal",
        numeric_cols=["input_signal"],
        return_type="pandas",
    )
    df_pd = dataset.data.to_pandas()
    transformer.fit(df_pd)
    result = transformer.transform(df_pd)
    import pandas as pd  # type: ignore[import-untyped, import-not-found]
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 800


def test_learner_mode_with_polars_input() -> None:
    dataset = make_single_delay_dataset(n_rows=800)
    learner = SimplexKernelLearner(max_lag="10m", max_epochs=20, seed=42)
    transformer = KernelFeatureTransformer(
        time_col="time",
        learner=learner,
        input_col="input_signal",
        target_col="target_signal",
        numeric_cols=["input_signal"],
        return_type="polars",
    )
    transformer.fit(dataset.data)
    result = transformer.transform(dataset.data)
    assert isinstance(result, pl.DataFrame)
    assert result.shape[0] == 800


def test_pipeline_smoke() -> None:
    dataset = make_single_delay_dataset(n_rows=50)
    kernel = FixedDelayKernel(delay_steps=6, max_lag_steps=10, dt=1.0)
    pipeline = Pipeline([
        ("features", KernelFeatureTransformer(
            time_col="time",
            kernels={"feature": kernel},
            numeric_cols=["input_signal"],
        )),
    ])
    pipeline.fit(dataset.data)
    result = pipeline.transform(dataset.data)
    import pandas as pd  # type: ignore[import-untyped, import-not-found]
    assert isinstance(result, pd.DataFrame)


def test_sklearn_clone() -> None:
    kernel = FixedDelayKernel(delay_steps=6, max_lag_steps=10, dt=1.0)
    transformer = KernelFeatureTransformer(
        time_col="time",
        kernels={"feature": kernel},
        numeric_cols=["input_signal"],
    )
    cloned = clone(transformer)
    assert cloned.time_col == "time"
    assert cloned.kernels is not None
    assert "feature" in cloned.kernels


def test_get_feature_names_out_after_transform() -> None:
    dataset = make_single_delay_dataset(n_rows=50)
    kernel = FixedDelayKernel(delay_steps=6, max_lag_steps=10, dt=1.0)
    transformer = KernelFeatureTransformer(
        time_col="time",
        kernels={"feature": kernel},
        numeric_cols=["input_signal"],
    )
    transformer.fit(dataset.data)
    transformer.transform(dataset.data)
    names = transformer.get_feature_names_out()
    assert isinstance(names, np.ndarray)
    assert len(names) > 0
    assert "time" not in names


def test_get_feature_names_out_before_transform_raises() -> None:
    dataset = make_single_delay_dataset(n_rows=50)
    kernel = FixedDelayKernel(delay_steps=6, max_lag_steps=10, dt=1.0)
    transformer = KernelFeatureTransformer(
        time_col="time",
        kernels={"feature": kernel},
        numeric_cols=["input_signal"],
    )
    transformer.fit(dataset.data)
    with pytest.raises(NotFittedError):
        transformer.get_feature_names_out()


def test_return_type_pandas() -> None:
    dataset = make_single_delay_dataset(n_rows=50)
    kernel = FixedDelayKernel(delay_steps=6, max_lag_steps=10, dt=1.0)
    transformer = KernelFeatureTransformer(
        time_col="time",
        kernels={"feature": kernel},
        numeric_cols=["input_signal"],
        return_type="pandas",
    )
    transformer.fit(dataset.data)
    result = transformer.transform(dataset.data)
    import pandas as pd  # type: ignore[import-untyped, import-not-found]
    assert isinstance(result, pd.DataFrame)


def test_return_type_polars() -> None:
    dataset = make_single_delay_dataset(n_rows=50)
    kernel = FixedDelayKernel(delay_steps=6, max_lag_steps=10, dt=1.0)
    transformer = KernelFeatureTransformer(
        time_col="time",
        kernels={"feature": kernel},
        numeric_cols=["input_signal"],
        return_type="polars",
    )
    transformer.fit(dataset.data)
    result = transformer.transform(dataset.data)
    assert isinstance(result, pl.DataFrame)


def test_return_type_numpy() -> None:
    dataset = make_single_delay_dataset(n_rows=50)
    kernel = FixedDelayKernel(delay_steps=6, max_lag_steps=10, dt=1.0)
    transformer = KernelFeatureTransformer(
        time_col="time",
        kernels={"feature": kernel},
        numeric_cols=["input_signal"],
        return_type="numpy",
    )
    transformer.fit(dataset.data)
    result = transformer.transform(dataset.data)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 50


def test_include_time_col_false_removes_time_from_generated_output() -> None:
    dataset = make_single_delay_dataset(n_rows=50)
    kernel = FixedDelayKernel(delay_steps=6, max_lag_steps=10, dt=1.0)
    transformer = KernelFeatureTransformer(
        time_col="time",
        kernels={"feature": kernel},
        numeric_cols=["input_signal"],
        include_time_col=False,
        return_type="polars",
    )
    transformer.fit(dataset.data)
    result = transformer.transform(dataset.data)
    assert "time" not in result.columns


def test_passthrough_true_keeps_original_columns() -> None:
    dataset = make_single_delay_dataset(n_rows=50)
    kernel = FixedDelayKernel(delay_steps=6, max_lag_steps=10, dt=1.0)
    transformer = KernelFeatureTransformer(
        time_col="time",
        kernels={"feature": kernel},
        numeric_cols=["input_signal"],
        passthrough=True,
        return_type="polars",
    )
    transformer.fit(dataset.data)
    result = transformer.transform(dataset.data)
    assert "time" in result.columns
    assert "input_signal" in result.columns
    assert "target_signal" in result.columns
    assert result.shape[1] > 3


def test_bad_config_neither_learner_nor_kernels() -> None:
    dataset = make_single_delay_dataset(n_rows=20)
    transformer = KernelFeatureTransformer(time_col="time")
    with pytest.raises(ValueError, match="Exactly one of"):
        transformer.fit(dataset.data)


def test_bad_config_both_learner_and_kernels() -> None:
    dataset = make_single_delay_dataset(n_rows=20)
    kernel = FixedDelayKernel(delay_steps=6, max_lag_steps=10, dt=1.0)
    learner = SimplexKernelLearner(max_lag="10m", max_epochs=20, seed=42)
    transformer = KernelFeatureTransformer(
        time_col="time",
        learner=learner,
        kernels={"feature": kernel},
        input_col="input_signal",
        target_col="target_signal",
    )
    with pytest.raises(ValueError, match="Exactly one of"):
        transformer.fit(dataset.data)


def test_bad_config_learner_missing_input_col() -> None:
    dataset = make_single_delay_dataset(n_rows=20)
    learner = SimplexKernelLearner(max_lag="10m", max_epochs=20, seed=42)
    transformer = KernelFeatureTransformer(
        time_col="time",
        learner=learner,
        target_col="target_signal",
    )
    with pytest.raises(ValueError, match="input_col"):
        transformer.fit(dataset.data)


def test_bad_config_learner_missing_target_col() -> None:
    dataset = make_single_delay_dataset(n_rows=20)
    learner = SimplexKernelLearner(max_lag="10m", max_epochs=20, seed=42)
    transformer = KernelFeatureTransformer(
        time_col="time",
        learner=learner,
        input_col="input_signal",
    )
    with pytest.raises(ValueError, match="target_col"):
        transformer.fit(dataset.data)


def test_bad_config_unsupported_return_type() -> None:
    dataset = make_single_delay_dataset(n_rows=20)
    kernel = FixedDelayKernel(delay_steps=6, max_lag_steps=10, dt=1.0)
    transformer = KernelFeatureTransformer(
        time_col="time",
        kernels={"feature": kernel},
        numeric_cols=["input_signal"],
        return_type="json",
    )
    with pytest.raises(ValueError, match="return_type"):
        transformer.fit(dataset.data)


def test_numpy_input_rejected() -> None:
    dataset = make_single_delay_dataset(n_rows=20)
    kernel = FixedDelayKernel(delay_steps=6, max_lag_steps=10, dt=1.0)
    transformer = KernelFeatureTransformer(
        time_col="time",
        kernels={"feature": kernel},
        numeric_cols=["input_signal"],
    )
    with pytest.raises(ValueError, match="named pandas or Polars DataFrame"):
        transformer.fit(dataset.data.to_numpy())


def test_missing_required_column_rejected() -> None:
    dataset = make_single_delay_dataset(n_rows=20)
    kernel = FixedDelayKernel(delay_steps=6, max_lag_steps=10, dt=1.0)
    transformer = KernelFeatureTransformer(
        time_col="time",
        kernels={"feature": kernel},
        numeric_cols=["input_signal", "nonexistent_col"],
    )
    with pytest.raises(ValueError, match="Missing required columns"):
        transformer.fit(dataset.data)


def test_transform_does_not_require_target_col_after_learner_fit() -> None:
    dataset = make_single_delay_dataset(n_rows=800)
    learner = SimplexKernelLearner(max_lag="10m", max_epochs=20, seed=42)
    transformer = KernelFeatureTransformer(
        time_col="time",
        learner=learner,
        input_col="input_signal",
        target_col="target_signal",
        numeric_cols=["input_signal"],
        return_type="polars",
    )
    transformer.fit(dataset.data)
    transform_df = dataset.data.drop("target_signal")
    result = transformer.transform(transform_df)
    assert isinstance(result, pl.DataFrame)
    assert result.height == transform_df.height


def test_transform_requires_numeric_cols_after_learner_fit() -> None:
    dataset = make_single_delay_dataset(n_rows=800)
    learner = SimplexKernelLearner(max_lag="10m", max_epochs=20, seed=42)
    transformer = KernelFeatureTransformer(
        time_col="time",
        learner=learner,
        input_col="input_signal",
        target_col="target_signal",
        numeric_cols=["input_signal"],
        return_type="polars",
    )
    transformer.fit(dataset.data)
    transform_df = dataset.data.drop("input_signal")
    with pytest.raises(ValueError, match="Missing required columns for transform"):
        transformer.transform(transform_df)


def test_fit_error_message_mentions_fit_for_missing_target() -> None:
    dataset = make_single_delay_dataset(n_rows=120)
    learner = SimplexKernelLearner(max_lag="10m", max_epochs=20, seed=42)
    transformer = KernelFeatureTransformer(
        time_col="time",
        learner=learner,
        input_col="input_signal",
        target_col="target_signal",
        numeric_cols=["input_signal"],
    )
    with pytest.raises(ValueError, match="for fit"):
        transformer.fit(dataset.data.drop("target_signal"))


def test_transform_error_message_mentions_transform_for_missing_source() -> None:
    dataset = make_single_delay_dataset(n_rows=120)
    kernel = FixedDelayKernel(delay_steps=6, max_lag_steps=10, dt=1.0)
    transformer = KernelFeatureTransformer(
        time_col="time",
        kernels={"feature": kernel},
        numeric_cols=["input_signal"],
        return_type="polars",
    )
    transformer.fit(dataset.data)
    with pytest.raises(ValueError, match="for transform"):
        transformer.transform(dataset.data.drop("input_signal"))


def test_to_return_type_rejects_invalid_value_even_if_validation_bypassed() -> None:
    dataset = make_single_delay_dataset(n_rows=40)
    kernel = FixedDelayKernel(delay_steps=6, max_lag_steps=10, dt=1.0)
    transformer = KernelFeatureTransformer(
        time_col="time",
        kernels={"feature": kernel},
        numeric_cols=["input_signal"],
        return_type="pandas",
    )
    transformer.fit(dataset.data)
    transformer.return_type = "invalid"
    with pytest.raises(ValueError, match="Unsupported return_type"):
        transformer.transform(dataset.data)
