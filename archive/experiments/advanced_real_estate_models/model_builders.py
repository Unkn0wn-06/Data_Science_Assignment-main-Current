"""Model factories and reversible target wrappers for the advanced experiment."""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import inv_boxcox
from scipy.stats import boxcox
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline

from src.models.common.preprocessing import make_target_encoding_preprocessor


OPTIONAL_PACKAGES = ("catboost", "xgboost", "lightgbm")


def dependency_status() -> dict[str, bool]:
    """Return availability without importing optional native libraries eagerly."""
    return {
        package: importlib.util.find_spec(package) is not None
        for package in OPTIONAL_PACKAGES
    }


class CatBoostNativeRegressor(RegressorMixin, BaseEstimator):
    """Keep categorical columns native while exposing a cloneable sklearn API."""

    def __init__(
        self,
        numerical_features: tuple[str, ...],
        categorical_features: tuple[str, ...],
        params: dict[str, Any] | None = None,
    ):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.params = params

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X[list(self.numerical_features + self.categorical_features)].copy()
        for column in self.numerical_features:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame[column] = frame[column].replace([np.inf, -np.inf], np.nan)
        for column in self.categorical_features:
            frame[column] = frame[column].astype("string").fillna("__MISSING__")
        return frame

    def fit(self, X: pd.DataFrame, y):
        from catboost import CatBoostRegressor

        parameters = dict(self.params or {})
        self.model_ = CatBoostRegressor(**parameters)
        prepared = self._prepare(X)
        self.model_.fit(
            prepared,
            np.asarray(y, dtype=float),
            cat_features=list(self.categorical_features),
        )
        self.n_features_in_ = prepared.shape[1]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model_.predict(self._prepare(X)), dtype=float)


class TargetStrategyRegressor(RegressorMixin, BaseEstimator):
    """Fit a fold-local reversible target without altering the market tail."""

    def __init__(
        self,
        regressor=None,
        strategy: str = "ppsf",
        size_column: str = "property_size_sqft",
    ):
        self.regressor = regressor
        self.strategy = strategy
        self.size_column = size_column

    def fit(self, X: pd.DataFrame, y):
        target = np.asarray(y, dtype=float)
        if not np.all(np.isfinite(target)) or np.any(target <= 0):
            raise ValueError("All target values must be finite and strictly positive.")
        size = pd.to_numeric(X[self.size_column], errors="coerce").to_numpy(float)
        if not np.all(np.isfinite(size)) or np.any(size <= 0):
            raise ValueError("Property size must be finite and strictly positive.")

        self.boxcox_lambda_ = None
        if self.strategy == "ppsf":
            transformed = target / size
        elif self.strategy == "log_ppsf":
            transformed = np.log1p(target / size)
        elif self.strategy == "log_price":
            transformed = np.log1p(target)
        elif self.strategy == "boxcox_price":
            transformed, self.boxcox_lambda_ = boxcox(target)
        elif self.strategy == "raw_price":
            transformed = target
        else:
            raise ValueError(f"Unknown target strategy: {self.strategy}")

        self.regressor_ = clone(self.regressor)
        self.regressor_.fit(X, transformed)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        transformed = np.asarray(self.regressor_.predict(X), dtype=float)
        size = pd.to_numeric(X[self.size_column], errors="coerce").to_numpy(float)
        if self.strategy == "ppsf":
            return transformed * size
        if self.strategy == "log_ppsf":
            return np.expm1(transformed) * size
        if self.strategy == "log_price":
            return np.expm1(transformed)
        if self.strategy == "boxcox_price":
            return inv_boxcox(transformed, self.boxcox_lambda_)
        return transformed


def _encoded_pipeline(family: str, params: dict, numerical, categorical):
    preprocessor = make_target_encoding_preprocessor(
        list(numerical), list(categorical), scale_numeric=False
    )
    if family == "xgboost":
        from xgboost import XGBRegressor

        model = XGBRegressor(**params)
    elif family == "lightgbm":
        from lightgbm import LGBMRegressor

        model = LGBMRegressor(**params)
    elif family == "huber":
        model = GradientBoostingRegressor(**params)
    else:
        raise KeyError(f"No encoded builder for {family}")
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def build_base_regressor(
    family: str, params: dict, numerical_features, categorical_features
):
    """Build a feature-to-target regressor without applying a target strategy."""
    numerical = tuple(numerical_features)
    categorical = tuple(categorical_features)
    if family == "catboost":
        return CatBoostNativeRegressor(numerical, categorical, params)
    return _encoded_pipeline(family, params, numerical, categorical)


def build_estimator(
    family: str,
    params: dict,
    numerical_features,
    categorical_features,
    target_strategy: str = "ppsf",
):
    """Build one model using native CatBoost or fold-fitted target encoding."""
    base = build_base_regressor(
        family, params, numerical_features, categorical_features
    )
    return TargetStrategyRegressor(base, strategy=target_strategy)


def candidate_parameters() -> dict[str, list[dict[str, Any]]]:
    """Return deliberately small, auditable searches rather than huge grids."""
    return {
        "catboost": [
            {
                "iterations": 400,
                "depth": 4,
                "learning_rate": 0.08,
                "loss_function": "RMSE",
                "l2_leaf_reg": 10,
                "random_seed": 42,
                "verbose": False,
                "allow_writing_files": False,
                "thread_count": -1,
            },
            {
                "iterations": 600,
                "depth": 6,
                "learning_rate": 0.05,
                "loss_function": "RMSE",
                "l2_leaf_reg": 5,
                "random_seed": 42,
                "verbose": False,
                "allow_writing_files": False,
                "thread_count": -1,
            },
            {
                "iterations": 400,
                "depth": 8,
                "learning_rate": 0.05,
                "loss_function": "RMSE",
                "l2_leaf_reg": 10,
                "random_seed": 42,
                "verbose": False,
                "allow_writing_files": False,
                "thread_count": -1,
            },
        ],
        "xgboost": [
            {
                "n_estimators": 500,
                "learning_rate": 0.05,
                "max_depth": 3,
                "min_child_weight": 5,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_alpha": 0.1,
                "reg_lambda": 5.0,
                "objective": "reg:squarederror",
                "random_state": 42,
                "n_jobs": -1,
            },
            {
                "n_estimators": 800,
                "learning_rate": 0.03,
                "max_depth": 5,
                "min_child_weight": 3,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_alpha": 0.1,
                "reg_lambda": 5.0,
                "objective": "reg:squarederror",
                "random_state": 42,
                "n_jobs": -1,
            },
            {
                "n_estimators": 1200,
                "learning_rate": 0.02,
                "max_depth": 7,
                "min_child_weight": 5,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "reg_alpha": 1.0,
                "reg_lambda": 10.0,
                "objective": "reg:squarederror",
                "random_state": 42,
                "n_jobs": -1,
            },
        ],
        "lightgbm": [
            {
                "n_estimators": 600,
                "learning_rate": 0.05,
                "num_leaves": 15,
                "max_depth": 6,
                "min_child_samples": 30,
                "subsample": 0.85,
                "subsample_freq": 1,
                "colsample_bytree": 0.85,
                "reg_alpha": 0.1,
                "reg_lambda": 5.0,
                "random_state": 42,
                "n_jobs": -1,
                "verbosity": -1,
            },
            {
                "n_estimators": 1000,
                "learning_rate": 0.03,
                "num_leaves": 31,
                "max_depth": -1,
                "min_child_samples": 20,
                "subsample": 0.8,
                "subsample_freq": 1,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": 42,
                "n_jobs": -1,
                "verbosity": -1,
            },
            {
                "n_estimators": 1400,
                "learning_rate": 0.02,
                "num_leaves": 31,
                "max_depth": 8,
                "min_child_samples": 35,
                "subsample": 0.7,
                "subsample_freq": 1,
                "colsample_bytree": 0.7,
                "reg_alpha": 1.0,
                "reg_lambda": 10.0,
                "random_state": 42,
                "n_jobs": -1,
                "verbosity": -1,
            },
        ],
        "huber": [
            {
                "loss": "huber", "alpha": 0.8, "n_estimators": 400,
                "learning_rate": 0.05, "max_depth": 2, "min_samples_leaf": 5,
                "random_state": 42,
            },
            {
                "loss": "huber", "alpha": 0.9, "n_estimators": 500,
                "learning_rate": 0.05, "max_depth": 3, "min_samples_leaf": 5,
                "random_state": 42,
            },
            {
                "loss": "huber", "alpha": 0.95, "n_estimators": 600,
                "learning_rate": 0.03, "max_depth": 3, "min_samples_leaf": 10,
                "random_state": 42,
            },
        ],
    }


def tweedie_candidates() -> list[dict[str, Any]]:
    """Build the requested compact variance-power sweep."""
    candidates = []
    for power in (1.1, 1.3, 1.5, 1.7, 1.9):
        params = candidate_parameters()["lightgbm"][1].copy()
        params.update(objective="tweedie", tweedie_variance_power=power)
        candidates.append(params)
    return candidates


def quantile_parameters(alpha: float) -> dict[str, Any]:
    """Use one stable LightGBM configuration for distributional estimates."""
    params = candidate_parameters()["lightgbm"][0].copy()
    params.update(objective="quantile", alpha=alpha, n_estimators=800)
    return params
