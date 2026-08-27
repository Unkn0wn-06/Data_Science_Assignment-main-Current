"""Frozen Building Name target-encoded PPSF LightGBM model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import KFold

from src.experimental_support.lightgbm_builder import build_lightgbm_regressor
from src.experimental_support.structured_features import (
    add_non_target_features,
    engineered_feature_lists,
)
from src.experimental_support.target_encoding import (
    DEFAULT_M_VALUES,
    MEstimateTargetEncoder,
)
from src.models.common.features import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


class NoncoordinatePPSFRegressor(RegressorMixin, BaseEstimator):
    """LightGBM PPSF estimator with inner-OOF high-cardinality encoding."""

    def __init__(
        self,
        te_columns: tuple[str, ...] = (),
        m_values: tuple[float, ...] = DEFAULT_M_VALUES,
        inner_splits: int = 5,
        random_state: int = 42,
    ):
        self.te_columns = te_columns
        self.m_values = m_values
        self.inner_splits = inner_splits
        self.random_state = random_state

    def _inner_cv(self) -> KFold:
        return KFold(
            n_splits=self.inner_splits,
            shuffle=True,
            random_state=self.random_state,
        )

    def _select_m(self, X: pd.DataFrame, ppsf: np.ndarray) -> tuple[float, dict]:
        scores = {}
        for m in self.m_values:
            encoded = MEstimateTargetEncoder(
                self.te_columns, m=float(m)
            ).fit_transform_oof(X, ppsf, self._inner_cv())
            proxy = encoded.mean(axis=1).to_numpy(float)
            scores[str(float(m))] = float(np.sqrt(np.mean(np.square(ppsf - proxy))))
        selected = min(self.m_values, key=lambda value: scores[str(float(value))])
        return float(selected), scores

    def feature_schema(self) -> tuple[list[str], list[str]]:
        numerical, categorical = engineered_feature_lists(
            NUMERICAL_FEATURES,
            CATEGORICAL_FEATURES,
            include_micro=False,
            include_interactions=True,
        )
        numerical.extend(f"{column}_te" for column in self.te_columns)
        categorical = [
            column for column in categorical if column not in self.te_columns
        ]
        return numerical, categorical

    def fit(self, X: pd.DataFrame, y):
        total_price = np.asarray(y, dtype=float)
        size = pd.to_numeric(X["property_size_sqft"], errors="coerce").to_numpy(float)
        if np.any(size <= 0) or not np.all(np.isfinite(size)):
            raise ValueError("Property size must be finite and positive.")
        ppsf = total_price / size
        if np.any(ppsf <= 0) or not np.all(np.isfinite(ppsf)):
            raise ValueError("PPSF must be finite and positive.")

        training = add_non_target_features(X, include_interactions=True)
        self.encoder_ = None
        if self.te_columns:
            self.selected_m_, self.smoothing_proxy_scores_ = self._select_m(X, ppsf)
            self.encoder_ = MEstimateTargetEncoder(
                self.te_columns, m=self.selected_m_
            )
            training = training.join(
                self.encoder_.fit_transform_oof(X, ppsf, self._inner_cv())
            )

        self.numerical_features_, self.categorical_features_ = self.feature_schema()
        self.regressor_ = clone(
            build_lightgbm_regressor(
                self.numerical_features_, self.categorical_features_
            )
        )
        self.regressor_.fit(training, ppsf)
        self.training_index_ = X.index.to_numpy(copy=True)
        self.training_prediction_ = (
            np.asarray(self.regressor_.predict(training), dtype=float) * size
        )
        self.n_features_in_ = X.shape[1]
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed = add_non_target_features(X, include_interactions=True)
        if self.encoder_ is not None:
            transformed = transformed.join(self.encoder_.transform(X))
        return transformed

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ppsf = np.asarray(self.regressor_.predict(self._transform(X)), dtype=float)
        size = pd.to_numeric(X["property_size_sqft"], errors="coerce").to_numpy(float)
        return ppsf * size

    def predict_training_oof_features(self, X: pd.DataFrame) -> np.ndarray:
        if not np.array_equal(X.index.to_numpy(), self.training_index_):
            raise ValueError("Training prediction rows do not match fitted rows.")
        return self.training_prediction_.copy()
