"""Fixed-complexity LightGBM PPSF fold adapter for description indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.experimental_support.structured_features import (
    add_non_target_features,
    engineered_feature_lists,
)
from src.experimental_support.lightgbm_builder import build_lightgbm_regressor
from src.models.common.features import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


def feature_schema(extra_numerical=(), building_te: bool = False):
    numerical, categorical = engineered_feature_lists(
        NUMERICAL_FEATURES,
        CATEGORICAL_FEATURES,
        include_micro=False,
        include_interactions=True,
    )
    numerical.extend(list(extra_numerical))
    if building_te:
        if "building_name_te" not in numerical:
            numerical.append("building_name_te")
        categorical = [column for column in categorical if column != "building_name"]
    return numerical, categorical


def build_lightgbm(extra_numerical=(), building_te: bool = False):
    numerical, categorical = feature_schema(extra_numerical, building_te)
    return build_lightgbm_regressor(numerical, categorical), numerical, categorical


def prepare_features(structured: pd.DataFrame, dense_features: pd.DataFrame | None = None):
    result = add_non_target_features(structured, include_interactions=True)
    if dense_features is not None:
        if not result.index.equals(dense_features.index):
            raise ValueError("Dense text and structured feature rows are not aligned.")
        result = result.join(dense_features)
    return result


def fit_lightgbm_fold(
    X_train,
    y_train,
    X_validation,
    train_dense: pd.DataFrame | None = None,
    validation_dense: pd.DataFrame | None = None,
    building_te: bool = False,
):
    extra = () if train_dense is None else tuple(train_dense.columns)
    estimator, numerical, categorical = build_lightgbm(extra, building_te=building_te)
    train_features = prepare_features(X_train, train_dense)
    validation_features = prepare_features(X_validation, validation_dense)
    train_size = pd.to_numeric(X_train["property_size_sqft"], errors="coerce").to_numpy(float)
    validation_size = pd.to_numeric(X_validation["property_size_sqft"], errors="coerce").to_numpy(float)
    ppsf = np.asarray(y_train, dtype=float) / train_size
    fitted = clone(estimator).fit(train_features, ppsf)
    validation_prediction = np.asarray(fitted.predict(validation_features), dtype=float) * validation_size
    training_prediction = np.asarray(fitted.predict(train_features), dtype=float) * train_size
    importances = fitted.named_steps["model"].feature_importances_.astype(float)
    names = list(numerical) + list(categorical)
    if len(names) != len(importances):
        raise AssertionError("LightGBM feature importance schema is misaligned.")
    return {
        "validation_prediction": validation_prediction,
        "training_prediction": training_prediction,
        "feature_importance": dict(zip(names, importances)),
        "predictor_count": len(names),
    }
