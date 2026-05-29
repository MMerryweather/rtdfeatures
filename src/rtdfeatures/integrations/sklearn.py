from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from sklearn.base import (  # type: ignore[import-untyped, import-not-found]
    BaseEstimator,
    TransformerMixin,
)
from sklearn.utils.validation import (  # type: ignore[import-untyped, import-not-found]
    check_is_fitted,
)

from rtdfeatures.features import KernelFeatureBuilder

try:
    import pandas as pd  # type: ignore[import-untyped, import-not-found]
except ImportError:
    pd = None


class KernelFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        *,
        time_col: str,
        learner: Any | None = None,
        kernels: dict[str, Any] | None = None,
        input_col: str | None = None,
        target_col: str | None = None,
        numeric_cols: list[str] | tuple[str, ...] | None = None,
        category_cols: list[str] | tuple[str, ...] | None = None,
        weight_col: str | None = None,
        kernel_name: str = "learned",
        order_by_time: bool = False,
        return_type: str = "pandas",
        include_time_col: bool = False,
        passthrough: bool = False,
    ) -> None:
        self.time_col = time_col
        self.learner = learner
        self.kernels = kernels
        self.input_col = input_col
        self.target_col = target_col
        self.numeric_cols = numeric_cols
        self.category_cols = category_cols
        self.weight_col = weight_col
        self.kernel_name = kernel_name
        self.order_by_time = order_by_time
        self.return_type = return_type
        self.include_time_col = include_time_col
        self.passthrough = passthrough

    # ------------------------------------------------------------------
    # Input conversion
    # ------------------------------------------------------------------

    def _to_polars(self, X: Any) -> pl.DataFrame:
        if isinstance(X, pl.DataFrame):
            return X
        if pd is not None and isinstance(X, pd.DataFrame):
            return pl.from_pandas(X)
        raise ValueError(
            f"X must be a named pandas or Polars DataFrame, got {type(X).__name__}."
        )

    # ------------------------------------------------------------------
    # Configuration validation
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        if (self.learner is None) == (self.kernels is None):
            raise ValueError(
                "Exactly one of 'learner' or 'kernels' must be provided."
            )
        if self.return_type not in {"pandas", "polars", "numpy"}:
            raise ValueError(
                f"return_type must be one of 'pandas', 'polars', 'numpy'; "
                f"got {self.return_type!r}."
            )
        if self.learner is not None:
            if self.input_col is None:
                raise ValueError("'input_col' is required when 'learner' is provided.")
            if self.target_col is None:
                raise ValueError("'target_col' is required when 'learner' is provided.")

    # ------------------------------------------------------------------
    # Required columns
    # ------------------------------------------------------------------

    def _required_fit_columns(self) -> set[str]:
        cols = {self.time_col}
        if self.numeric_cols:
            cols.update(self.numeric_cols)
        if self.category_cols:
            cols.update(self.category_cols)
        if self.weight_col is not None:
            cols.add(self.weight_col)
        if self.learner is not None:
            if self.input_col is not None:
                cols.add(self.input_col)
            if self.target_col is not None:
                cols.add(self.target_col)
        return cols

    def _required_transform_columns(self) -> set[str]:
        cols = {self.time_col}
        if self.numeric_cols:
            cols.update(self.numeric_cols)
        if self.category_cols:
            cols.update(self.category_cols)
        if self.weight_col is not None:
            cols.add(self.weight_col)
        return cols

    # ------------------------------------------------------------------
    # Column validation
    # ------------------------------------------------------------------

    def _validate_fit_columns(self, df: pl.DataFrame) -> None:
        required = self._required_fit_columns()
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns for fit: {sorted(missing)}. "
                f"Available columns: {sorted(df.columns)}."
            )

    def _validate_transform_columns(self, df: pl.DataFrame) -> None:
        required = self._required_transform_columns()
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns for transform: {sorted(missing)}. "
                f"Available columns: {sorted(df.columns)}."
            )

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X: Any, y: Any = None) -> KernelFeatureTransformer:
        self._validate_config()
        df = self._to_polars(X)
        self._validate_fit_columns(df)
        self.feature_names_in_ = np.array(df.columns, dtype=str)
        self.n_features_in_ = len(df.columns)
        if self.kernels is not None:
            self.kernels_ = dict(self.kernels)
        else:
            assert self.learner is not None
            self.fit_result_ = self.learner.fit(
                df,
                input_col=self.input_col,
                target_col=self.target_col,
                time_col=self.time_col,
                order_by_time=self.order_by_time,
            )
            self.kernels_ = {self.kernel_name: self.fit_result_.kernel}
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, X: Any) -> Any:
        check_is_fitted(self, "kernels_")
        df = self._to_polars(X)
        self._validate_transform_columns(df)
        builder = KernelFeatureBuilder(
            kernels=self.kernels_,
            time_col=self.time_col,
            numeric_cols=self.numeric_cols,
            category_cols=self.category_cols,
            weight_col=self.weight_col,
        )
        result = builder.transform_result(df, order_by_time=self.order_by_time)
        self.last_transform_report_ = result.report
        self.feature_registry_ = result.feature_registry
        feature_cols = [c for c in result.features.columns if c != self.time_col]
        if self.passthrough:
            out = df.clone()
            for col in feature_cols:
                out = out.with_columns(result.features[col])
            self.feature_names_out_ = np.array(out.columns, dtype=str)
        elif self.include_time_col:
            self.feature_names_out_ = np.array(result.features.columns, dtype=str)
            out = result.features
        else:
            out = result.features.select(feature_cols)
            self.feature_names_out_ = np.array(feature_cols, dtype=str)
        return self._to_return_type(out)

    def _to_return_type(self, df: pl.DataFrame) -> Any:
        if self.return_type == "polars":
            return df
        if self.return_type == "pandas":
            return df.to_pandas()
        if self.return_type == "numpy":
            return df.to_numpy()
        raise ValueError(
            f"Unsupported return_type {self.return_type!r}; expected one of "
            "'pandas', 'polars', 'numpy'."
        )

    # ------------------------------------------------------------------
    # Feature name output
    # ------------------------------------------------------------------

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_
