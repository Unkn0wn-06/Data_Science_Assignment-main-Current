"""Frozen LightGBM construction used by the final sensitivity experiment."""

from __future__ import annotations

from typing import Any

from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline

from src.models.common.preprocessing import make_target_encoding_preprocessor


NATIVE_OBJECTIVES = {
    "regression": "MSE / L2",
    "regression_l1": "MAE / L1",
    "huber": "Huber",
    "fair": "Fair",
}


def lightgbm_parameters(objective: str = "regression") -> dict[str, Any]:
    """Reuse the exact winning complexity and change only the objective."""
    if objective not in NATIVE_OBJECTIVES:
        raise ValueError(f"Unsupported controlled objective: {objective}")
    return {
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
        "objective": objective,
    }


def build_lightgbm_regressor(
    numerical_features: list[str],
    categorical_features: list[str],
    objective: str = "regression",
):
    preprocessor = make_target_encoding_preprocessor(
        list(numerical_features), list(categorical_features), scale_numeric=False
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", LGBMRegressor(**lightgbm_parameters(objective))),
        ]
    )
