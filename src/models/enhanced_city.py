"""Shared estimators and evaluation for enhanced City PPSF comparisons."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from src.cleaning.pipeline import PROJECT_ROOT
from src.models.common.evaluation import regression_metrics
from src.models.common.features import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
)
from src.models.common.parameters import load_params
from src.models.common.preprocessing import make_target_encoding_preprocessor
from src.models.common.utilities import PricePerSquareFootRegressor
from src.models.gradient_boosting.model import build_model as build_gradient_boosting
from src.models.knn.model import build_model as build_knn
from src.models.random_forest.model import build_best_city_model
from src.models.ridge.model import build_model as build_ridge


PARAMS_PATH = PROJECT_ROOT / "configs" / "best_params.json"
MODEL_NAMES = [
    "Ridge Regression",
    "Random Forest",
    "Gradient Boosting",
    "KNN",
]


def model_parameters() -> dict:
    """Return the existing parameters used without performing another search."""
    saved = load_params(PARAMS_PATH)
    return {
        "Ridge Regression": saved["Ridge Regression"],
        "Random Forest": {
            "n_estimators": 700,
            "min_samples_split": 6,
            "min_samples_leaf": 3,
            "max_features": 0.7,
            "max_depth": 24,
            "criterion": "squared_error",
            "bootstrap": True,
            "random_state": 42,
            "n_jobs": -1,
        },
        "Gradient Boosting": saved["Gradient Boosting"],
        "KNN": saved["KNN"],
    }


def build_base_regressor(model_name: str) -> Pipeline:
    """Build one fold-safe enhanced preprocessor and existing estimator."""
    parameters = model_parameters()
    estimators = {
        "Ridge Regression": build_ridge(parameters["Ridge Regression"]),
        "Random Forest": build_best_city_model(),
        "Gradient Boosting": build_gradient_boosting(
            parameters["Gradient Boosting"]
        ),
        "KNN": build_knn(parameters["KNN"]),
    }
    if model_name not in estimators:
        raise KeyError(f"Unknown enhanced City model: {model_name}")
    scale_numeric = model_name in {"Ridge Regression", "KNN"}
    return Pipeline(
        [
            (
                "preprocessor",
                make_target_encoding_preprocessor(
                    NUMERICAL_FEATURES,
                    CATEGORICAL_FEATURES,
                    scale_numeric=scale_numeric,
                ),
            ),
            ("model", estimators[model_name]),
        ]
    )


def build_ppsf_estimator(model_name: str) -> PricePerSquareFootRegressor:
    """Wrap one model so it learns training PPSF and predicts total price."""
    return PricePerSquareFootRegressor(regressor=build_base_regressor(model_name))


def shared_folds(row_count: int):
    """Materialize the exact five shuffled folds reused by every experiment."""
    return list(
        KFold(n_splits=5, shuffle=True, random_state=42).split(
            np.arange(row_count)
        )
    )


def adjusted_r2(r2: float, observations: int, predictors: int) -> float:
    """Calculate adjusted R-squared from OOF observations and input width."""
    if observations <= predictors + 1:
        return float("nan")
    return float(
        1.0
        - ((1.0 - r2) * (observations - 1) / (observations - predictors - 1))
    )


def out_of_fold_predictions(estimator, X, y, folds) -> np.ndarray:
    """Fit every preprocessing/model step on training rows and predict validation rows."""
    prediction = np.empty(len(y), dtype=float)
    for train_index, validation_index in folds:
        fitted = clone(estimator).fit(X.iloc[train_index], y.iloc[train_index])
        prediction[validation_index] = fitted.predict(X.iloc[validation_index])
    return prediction


def evaluate_estimator(estimator, data: pd.DataFrame, folds) -> tuple[dict, np.ndarray]:
    """Return common OOF price metrics and predictions for one model."""
    X = data[MODEL_FEATURES]
    y = data[TARGET_COLUMN]
    prediction = out_of_fold_predictions(estimator, X, y, folds)
    metrics = regression_metrics(y, prediction, include_distribution=True)
    metrics["Adjusted_R2"] = adjusted_r2(
        float(metrics["R2"]), len(data), len(MODEL_FEATURES)
    )
    return {
        "R2": float(metrics["R2"]),
        "Adjusted_R2": float(metrics["Adjusted_R2"]),
        "MAE_RM": float(metrics["MAE"]),
        "RMSE_RM": float(metrics["RMSE"]),
        "Median_AE_RM": float(metrics["median_absolute_error"]),
    }, prediction

