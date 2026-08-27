"""M-estimate PPSF target encoding with explicit inner-OOF training features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.model_selection import KFold

from experiments.advanced_real_estate_models.feature_engineering import (
    add_non_target_features,
    engineered_feature_lists,
)
from experiments.advanced_real_estate_models.model_builders import build_base_regressor
from src.models.common.features import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


MISSING_TOKEN = "__MISSING__"
DEFAULT_M_VALUES = (5.0, 10.0, 20.0, 50.0, 100.0)


def normalize_category(values: pd.Series) -> pd.Series:
    """Return deterministic string categories with one missing-value token."""
    return values.astype("string").fillna(MISSING_TOKEN)


class MEstimateTargetEncoder(TransformerMixin, BaseEstimator):
    """Encode selected categories from only the targets supplied to ``fit``."""

    def __init__(
        self,
        columns: tuple[str, ...],
        m: float = 20.0,
        add_log_count: bool = False,
    ):
        self.columns = columns
        self.m = m
        self.add_log_count = add_log_count

    def fit(self, X: pd.DataFrame, y):
        target = np.asarray(y, dtype=float)
        if len(X) != len(target):
            raise ValueError("X and y must have identical row counts.")
        if not np.all(np.isfinite(target)):
            raise ValueError("Target encoding requires finite targets.")
        self.global_mean_ = float(np.mean(target))
        self.mappings_ = {}
        self.count_mappings_ = {}
        for column in self.columns:
            category = normalize_category(X[column])
            grouped = pd.DataFrame(
                {"category": category.to_numpy(), "target": target}
            ).groupby("category")["target"].agg(["mean", "count"])
            smoothed = (
                grouped["count"] * grouped["mean"]
                + float(self.m) * self.global_mean_
            ) / (grouped["count"] + float(self.m))
            self.mappings_[column] = smoothed.to_dict()
            self.count_mappings_[column] = grouped["count"].astype(float).to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "mappings_"):
            raise ValueError("Encoder must be fitted before transform.")
        encoded = pd.DataFrame(index=X.index)
        for column in self.columns:
            category = normalize_category(X[column])
            encoded[f"{column}_te"] = (
                category.map(self.mappings_[column])
                .fillna(self.global_mean_)
                .astype(float)
            )
            if self.add_log_count:
                count = category.map(self.count_mappings_[column]).fillna(0.0)
                encoded[f"{column}_log_count"] = np.log1p(count.astype(float))
        return encoded

    def fit_transform_oof(self, X: pd.DataFrame, y, cv) -> pd.DataFrame:
        """Give every row an encoding learned without that row's target."""
        target = np.asarray(y, dtype=float)
        feature_names = [f"{column}_te" for column in self.columns]
        if self.add_log_count:
            feature_names.extend(
                f"{column}_log_count" for column in self.columns
            )
        result = pd.DataFrame(index=X.index, columns=feature_names, dtype=float)
        positions = np.arange(len(X))
        for train_position, validation_position in cv.split(positions):
            encoder = MEstimateTargetEncoder(
                columns=self.columns,
                m=self.m,
                add_log_count=self.add_log_count,
            ).fit(X.iloc[train_position], target[train_position])
            transformed = encoder.transform(X.iloc[validation_position])
            result.loc[X.index[validation_position], transformed.columns] = (
                transformed.to_numpy()
            )
        if result.isna().any().any():
            raise AssertionError("OOF target encoding produced missing values.")
        self.fit(X, target)
        return result.astype(float)


class AdvancedTargetEncodingPPSFRegressor(RegressorMixin, BaseEstimator):
    """Fit fixed LightGBM on PPSF with training-only M-estimate features."""

    def __init__(
        self,
        lightgbm_params: dict,
        te_columns: tuple[str, ...],
        m_values: tuple[float, ...] = DEFAULT_M_VALUES,
        retain_raw: bool = False,
        add_frequency: bool = False,
        inner_splits: int = 5,
        random_state: int = 42,
    ):
        self.lightgbm_params = lightgbm_params
        self.te_columns = te_columns
        self.m_values = m_values
        self.retain_raw = retain_raw
        self.add_frequency = add_frequency
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
            encoder = MEstimateTargetEncoder(self.te_columns, m=float(m))
            encoded = encoder.fit_transform_oof(X, ppsf, self._inner_cv())
            proxy = encoded[[f"{column}_te" for column in self.te_columns]].mean(axis=1)
            scores[str(float(m))] = float(
                np.sqrt(np.mean(np.square(ppsf - proxy.to_numpy(float))))
            )
        selected = min(self.m_values, key=lambda value: scores[str(float(value))])
        return float(selected), scores

    def _feature_schema(self) -> tuple[list[str], list[str]]:
        numerical, categorical = engineered_feature_lists(
            NUMERICAL_FEATURES,
            CATEGORICAL_FEATURES,
            include_micro=False,
            include_interactions=True,
        )
        numerical.extend(f"{column}_te" for column in self.te_columns)
        if self.add_frequency:
            numerical.extend(
                f"{column}_log_count" for column in self.te_columns
            )
        if not self.retain_raw:
            categorical = [
                column for column in categorical if column not in self.te_columns
            ]
        return numerical, categorical

    def fit(self, X: pd.DataFrame, y):
        total_price = np.asarray(y, dtype=float)
        size = pd.to_numeric(X["property_size_sqft"], errors="coerce").to_numpy(float)
        ppsf = total_price / size
        if not np.all(np.isfinite(ppsf)) or np.any(ppsf <= 0):
            raise ValueError("PPSF must be finite and strictly positive.")

        self.selected_m_, self.smoothing_proxy_scores_ = self._select_m(X, ppsf)
        self.encoder_ = MEstimateTargetEncoder(
            self.te_columns,
            m=self.selected_m_,
            add_log_count=self.add_frequency,
        )
        oof_encoded = self.encoder_.fit_transform_oof(X, ppsf, self._inner_cv())
        training = add_non_target_features(X, include_interactions=True).join(
            oof_encoded
        )
        self.numerical_features_, self.categorical_features_ = self._feature_schema()
        base = build_base_regressor(
            "lightgbm",
            self.lightgbm_params,
            self.numerical_features_,
            self.categorical_features_,
        )
        self.regressor_ = clone(base).fit(training, ppsf)
        self.training_index_ = X.index.to_numpy(copy=True)
        self.training_prediction_ = (
            np.asarray(self.regressor_.predict(training), dtype=float) * size
        )
        self.n_features_in_ = X.shape[1]
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return add_non_target_features(X, include_interactions=True).join(
            self.encoder_.transform(X)
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predicted_ppsf = np.asarray(
            self.regressor_.predict(self._transform(X)), dtype=float
        )
        size = pd.to_numeric(X["property_size_sqft"], errors="coerce").to_numpy(float)
        return predicted_ppsf * size

    def predict_training_oof_features(self, X: pd.DataFrame) -> np.ndarray:
        """Return in-sample model predictions built from cross-fitted TE features."""
        if not np.array_equal(X.index.to_numpy(), self.training_index_):
            raise ValueError("Training prediction rows do not match fitted rows.")
        return self.training_prediction_.copy()
