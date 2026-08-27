"""Non-coordinate, fold-fitted grouped features and PPSF model wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.model_selection import KFold

from experiments.advanced_real_estate_models.feature_engineering import (
    add_non_target_features,
    engineered_feature_lists,
)
from experiments.noncoordinate_target_encoding.model_builders import (
    build_lightgbm_regressor,
)
from experiments.noncoordinate_target_encoding.target_encoding import (
    CategoryCountEncoder,
    DEFAULT_M_VALUES,
    MEstimateTargetEncoder,
)
from src.models.common.features import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


CITY_AGGREGATE_FEATURES = (
    "city_median_sqft",
    "city_mean_sqft",
    "city_median_bedroom",
    "city_median_bathroom",
    "city_property_count",
    "ratio_to_city_median_sqft",
)


class CityContextEncoder(TransformerMixin, BaseEstimator):
    """Target-free city statistics learned from one training partition only."""

    def __init__(self, city_column: str = "city"):
        self.city_column = city_column

    def fit(self, X: pd.DataFrame, y=None):
        city = X[self.city_column].astype("string").fillna("__MISSING__")
        frame = pd.DataFrame(
            {
                "city": city.to_numpy(),
                "sqft": pd.to_numeric(X["property_size_sqft"], errors="coerce").to_numpy(),
                "bedroom": pd.to_numeric(X["bedroom"], errors="coerce").to_numpy(),
                "bathroom": pd.to_numeric(X["bathroom"], errors="coerce").to_numpy(),
            }
        )
        grouped = frame.groupby("city", dropna=False).agg(
            city_median_sqft=("sqft", "median"),
            city_mean_sqft=("sqft", "mean"),
            city_median_bedroom=("bedroom", "median"),
            city_median_bathroom=("bathroom", "median"),
            city_property_count=("sqft", "size"),
        )
        self.mappings_ = {
            column: grouped[column].astype(float).to_dict()
            for column in grouped.columns
        }
        self.fallbacks_ = {
            "city_median_sqft": float(frame["sqft"].median()),
            "city_mean_sqft": float(frame["sqft"].mean()),
            "city_median_bedroom": float(frame["bedroom"].median()),
            "city_median_bathroom": float(frame["bathroom"].median()),
            "city_property_count": 0.0,
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "mappings_"):
            raise ValueError("Encoder must be fitted before transform.")
        city = X[self.city_column].astype("string").fillna("__MISSING__")
        result = pd.DataFrame(index=X.index)
        for column, mapping in self.mappings_.items():
            result[column] = city.map(mapping).fillna(self.fallbacks_[column]).astype(float)
        size = pd.to_numeric(X["property_size_sqft"], errors="coerce")
        denominator = result["city_median_sqft"].replace(0.0, np.nan)
        result["ratio_to_city_median_sqft"] = (size / denominator).replace(
            [np.inf, -np.inf], np.nan
        )
        return result[list(CITY_AGGREGATE_FEATURES)]


class NoncoordinatePPSFRegressor(RegressorMixin, BaseEstimator):
    """LightGBM PPSF estimator with explicitly cross-fitted high-cardinality TE."""

    def __init__(
        self,
        te_columns: tuple[str, ...] = (),
        m_values: tuple[float, ...] = DEFAULT_M_VALUES,
        add_counts: bool = False,
        add_city_aggregates: bool = False,
        objective: str = "regression",
        weight_gamma: float | None = None,
        inner_splits: int = 5,
        random_state: int = 42,
    ):
        self.te_columns = te_columns
        self.m_values = m_values
        self.add_counts = add_counts
        self.add_city_aggregates = add_city_aggregates
        self.objective = objective
        self.weight_gamma = weight_gamma
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
            encoded = MEstimateTargetEncoder(self.te_columns, m=float(m)).fit_transform_oof(
                X, ppsf, self._inner_cv()
            )
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
        categorical = [column for column in categorical if column not in self.te_columns]
        if self.add_counts:
            for column in self.te_columns:
                count_name = "building_count" if column == "building_name" else f"{column}_count"
                numerical.extend([count_name, f"log_{count_name}"])
        if self.add_city_aggregates:
            numerical.extend(CITY_AGGREGATE_FEATURES)
        return numerical, categorical

    def _base_features(self, X: pd.DataFrame) -> pd.DataFrame:
        return add_non_target_features(X, include_interactions=True)

    def fit(self, X: pd.DataFrame, y):
        total_price = np.asarray(y, dtype=float)
        size = pd.to_numeric(X["property_size_sqft"], errors="coerce").to_numpy(float)
        if np.any(size <= 0) or not np.all(np.isfinite(size)):
            raise ValueError("Property size must be finite and positive.")
        ppsf = total_price / size
        if np.any(ppsf <= 0) or not np.all(np.isfinite(ppsf)):
            raise ValueError("PPSF must be finite and positive.")

        training = self._base_features(X)
        self.encoder_ = None
        self.count_encoder_ = None
        self.city_encoder_ = None
        if self.te_columns:
            self.selected_m_, self.smoothing_proxy_scores_ = self._select_m(X, ppsf)
            self.encoder_ = MEstimateTargetEncoder(
                self.te_columns, m=self.selected_m_
            )
            training = training.join(
                self.encoder_.fit_transform_oof(X, ppsf, self._inner_cv())
            )
        if self.add_counts:
            self.count_encoder_ = CategoryCountEncoder(self.te_columns).fit(X)
            training = training.join(self.count_encoder_.transform(X))
        if self.add_city_aggregates:
            self.city_encoder_ = CityContextEncoder().fit(X)
            training = training.join(self.city_encoder_.transform(X))

        self.numerical_features_, self.categorical_features_ = self.feature_schema()
        self.regressor_ = clone(
            build_lightgbm_regressor(
                self.numerical_features_, self.categorical_features_, self.objective
            )
        )
        fit_kwargs = {}
        if self.weight_gamma is not None:
            weights = 1.0 / np.power(total_price, float(self.weight_gamma))
            weights = weights / np.mean(weights)
            fit_kwargs["model__sample_weight"] = weights
            self.sample_weight_summary_ = {
                "gamma": float(self.weight_gamma),
                "minimum": float(np.min(weights)),
                "maximum": float(np.max(weights)),
                "mean": float(np.mean(weights)),
            }
        self.regressor_.fit(training, ppsf, **fit_kwargs)
        self.training_index_ = X.index.to_numpy(copy=True)
        self.training_prediction_ = (
            np.asarray(self.regressor_.predict(training), dtype=float) * size
        )
        self.n_features_in_ = X.shape[1]
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed = self._base_features(X)
        if self.encoder_ is not None:
            transformed = transformed.join(self.encoder_.transform(X))
        if self.count_encoder_ is not None:
            transformed = transformed.join(self.count_encoder_.transform(X))
        if self.city_encoder_ is not None:
            transformed = transformed.join(self.city_encoder_.transform(X))
        return transformed

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ppsf = np.asarray(self.regressor_.predict(self._transform(X)), dtype=float)
        size = pd.to_numeric(X["property_size_sqft"], errors="coerce").to_numpy(float)
        return ppsf * size

    def predict_training_oof_features(self, X: pd.DataFrame) -> np.ndarray:
        if not np.array_equal(X.index.to_numpy(), self.training_index_):
            raise ValueError("Training prediction rows do not match fitted rows.")
        return self.training_prediction_.copy()
