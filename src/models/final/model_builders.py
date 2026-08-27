"""Frozen builders for the four submitted models; no experiment imports."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.models.common.features import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from src.models.common.preprocessing import make_target_encoding_preprocessor
from src.models.common.utilities import PricePerSquareFootRegressor
from src.models.final.structured_features import (
    add_non_target_features,
    engineered_feature_lists,
)


def build_standard_ppsf_estimator(model_name: str) -> PricePerSquareFootRegressor:
    """Build the exact frozen Ridge, RF, or GB Scenario B pipeline."""
    if model_name == "Ridge Regression":
        estimator, scale = Ridge(alpha=10.0), True
    elif model_name == "Random Forest":
        estimator, scale = RandomForestRegressor(
            n_estimators=700,
            min_samples_split=6,
            min_samples_leaf=3,
            max_features=0.7,
            max_depth=24,
            criterion="squared_error",
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
        ), False
    elif model_name == "Gradient Boosting":
        estimator, scale = GradientBoostingRegressor(
            random_state=42,
            learning_rate=0.1,
            max_depth=5,
            n_estimators=200,
        ), False
    else:
        raise KeyError(f"Unknown final standard model: {model_name}")
    pipeline = Pipeline(
        [
            (
                "preprocessor",
                make_target_encoding_preprocessor(
                    NUMERICAL_FEATURES,
                    CATEGORICAL_FEATURES,
                    scale_numeric=scale,
                ),
            ),
            ("model", estimator),
        ]
    )
    return PricePerSquareFootRegressor(regressor=pipeline)


def build_position_lightgbm():
    numerical, categorical = engineered_feature_lists(
        NUMERICAL_FEATURES, CATEGORICAL_FEATURES
    )
    numerical.extend(
        [
            "is_high_floor_text",
            "is_low_floor_text",
            "is_top_floor_text",
            "has_balcony",
            "has_large_balcony",
        ]
    )
    pipeline = Pipeline(
        [
            (
                "preprocessor",
                make_target_encoding_preprocessor(numerical, categorical),
            ),
            (
                "model",
                LGBMRegressor(
                    n_estimators=1000,
                    learning_rate=0.03,
                    num_leaves=31,
                    max_depth=-1,
                    min_child_samples=20,
                    subsample=0.8,
                    subsample_freq=1,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1,
                    objective="regression",
                ),
            ),
        ]
    )
    return pipeline, numerical, categorical


def prepare_position_features(
    structured: pd.DataFrame, position: pd.DataFrame
) -> pd.DataFrame:
    result = add_non_target_features(structured)
    if not result.index.equals(position.index):
        raise ValueError("Position feature rows are not aligned with structured features.")
    return result.join(position)


def fit_position_fold(
    X_train,
    y_train,
    X_validation,
    train_position,
    validation_position,
):
    """Fit one frozen LightGBM PPSF outer fold and reconstruct total price."""
    estimator, numerical, categorical = build_position_lightgbm()
    train = prepare_position_features(X_train, train_position)
    validation = prepare_position_features(X_validation, validation_position)
    train_size = pd.to_numeric(
        X_train["property_size_sqft"], errors="coerce"
    ).to_numpy(float)
    validation_size = pd.to_numeric(
        X_validation["property_size_sqft"], errors="coerce"
    ).to_numpy(float)
    fitted = clone(estimator).fit(train, np.asarray(y_train, float) / train_size)
    importances = np.asarray(fitted.named_steps["model"].feature_importances_, float)
    names = list(numerical) + list(categorical)
    if len(names) != len(importances):
        raise AssertionError("LightGBM feature importance schema is misaligned.")
    return np.asarray(fitted.predict(validation), float) * validation_size
