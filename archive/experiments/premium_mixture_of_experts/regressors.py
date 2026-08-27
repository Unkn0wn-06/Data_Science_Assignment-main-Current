"""Fixed-complexity standard and premium regression experts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from experiments.advanced_real_estate_models.feature_engineering import (
    add_non_target_features,
    engineered_feature_lists,
)
from experiments.advanced_real_estate_models.model_builders import (
    build_base_regressor as build_advanced_base,
    candidate_parameters,
)
from src.models.common.features import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from src.models.common.preprocessing import make_target_encoding_preprocessor
from src.models.random_forest.model import build_best_city_model


PREMIUM_SCOPES = (0.05, 0.10, 0.15, 0.20)


def premium_scope_mask(y_outer_train, scope: float) -> tuple[np.ndarray, float]:
    """Select an upper-market training scope from training prices only."""
    target = np.asarray(y_outer_train, dtype=float)
    threshold = float(np.quantile(target, 1.0 - float(scope)))
    return target >= threshold, threshold


def _canonical_pipeline(family: str, numerical, categorical):
    preprocessor = make_target_encoding_preprocessor(
        list(numerical),
        list(categorical),
        scale_numeric=family == "ridge",
    )
    if family == "random_forest":
        model = build_best_city_model()
    elif family == "ridge":
        model = Ridge(alpha=10.0)
    else:
        raise ValueError(f"Unsupported canonical expert family: {family}")
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


class ExpertRegressor(RegressorMixin, BaseEstimator):
    """Fit one expert on PPSF or direct price and return original-RM totals."""

    def __init__(
        self,
        family: str,
        target_strategy: str = "ppsf",
        interactions: bool = False,
        extra_numerical: tuple[str, ...] = (),
    ):
        self.family = family
        self.target_strategy = target_strategy
        self.interactions = interactions
        self.extra_numerical = extra_numerical

    def feature_schema(self):
        if self.interactions:
            numerical, categorical = engineered_feature_lists(
                NUMERICAL_FEATURES,
                CATEGORICAL_FEATURES,
                include_micro=False,
                include_interactions=True,
            )
        else:
            numerical, categorical = list(NUMERICAL_FEATURES), list(CATEGORICAL_FEATURES)
        numerical.extend(self.extra_numerical)
        return numerical, categorical

    def _transform(self, X):
        if self.interactions:
            return add_non_target_features(X, include_interactions=True)
        return X

    def _base(self):
        numerical, categorical = self.feature_schema()
        if self.family == "lightgbm":
            params = candidate_parameters()["lightgbm"][1].copy()
            params["objective"] = "regression"
            return build_advanced_base("lightgbm", params, numerical, categorical)
        return _canonical_pipeline(self.family, numerical, categorical)

    def fit(self, X: pd.DataFrame, y):
        total_price = np.asarray(y, dtype=float)
        size = pd.to_numeric(X["property_size_sqft"], errors="coerce").to_numpy(float)
        if not np.all(np.isfinite(total_price)) or np.any(total_price <= 0):
            raise ValueError("Expert targets must be finite and positive.")
        if not np.all(np.isfinite(size)) or np.any(size <= 0):
            raise ValueError("Expert property sizes must be finite and positive.")
        if self.target_strategy == "ppsf":
            transformed_target = total_price / size
        elif self.target_strategy == "direct_price":
            transformed_target = total_price
        else:
            raise ValueError(f"Unknown target strategy: {self.target_strategy}")
        self.model_ = clone(self._base())
        self.model_.fit(self._transform(X), transformed_target)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predicted = np.asarray(self.model_.predict(self._transform(X)), dtype=float)
        if self.target_strategy == "ppsf":
            size = pd.to_numeric(X["property_size_sqft"], errors="coerce").to_numpy(float)
            return predicted * size
        return predicted


def standard_rf(extra_numerical: tuple[str, ...] = ()) -> ExpertRegressor:
    return ExpertRegressor(
        "random_forest", target_strategy="ppsf", interactions=False,
        extra_numerical=extra_numerical,
    )


def standard_lightgbm(extra_numerical: tuple[str, ...] = ()) -> ExpertRegressor:
    return ExpertRegressor(
        "lightgbm", target_strategy="ppsf", interactions=True,
        extra_numerical=extra_numerical,
    )


def premium_expert(
    family: str = "lightgbm",
    target_strategy: str = "ppsf",
    extra_numerical: tuple[str, ...] = (),
) -> ExpertRegressor:
    return ExpertRegressor(
        family,
        target_strategy=target_strategy,
        interactions=family == "lightgbm",
        extra_numerical=extra_numerical,
    )
