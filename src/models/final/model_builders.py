"""Frozen builders for the four submitted models; no experiment imports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

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


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FINAL_TUNED_PARAMS_PATH = PROJECT_ROOT / "configs" / "final_tuned_params.json"
MODEL_SCALING_POLICY = {
    "Ridge Regression": "StandardScaler applied to numerical features",
    "Random Forest": "Not applied — tree-based model",
    "Gradient Boosting": "Not applied — tree-based model",
    "LightGBM + Position Features": "Not applied — tree-based model",
}
def load_final_tuned_config(path: Path = FINAL_TUNED_PARAMS_PATH) -> dict:
    """Load and validate the authoritative current four-model configuration."""
    if not path.is_file():
        raise FileNotFoundError(f"Authoritative tuned configuration is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    models = payload.get("models", {})
    if set(models) != set(MODEL_SCALING_POLICY):
        raise ValueError("Tuned configuration must contain the current four model families.")
    for model_name, spec in models.items():
        if not isinstance(spec.get("parameters"), dict) or not spec["parameters"]:
            raise ValueError(f"Tuned parameters are missing for {model_name}.")
        if spec.get("scaling") != MODEL_SCALING_POLICY[model_name]:
            raise ValueError(f"Scaling policy mismatch for {model_name}.")
    return payload


def get_final_model_parameters(model_name: str) -> dict:
    """Return a defensive copy of one frozen tuned parameter mapping."""
    if model_name not in MODEL_SCALING_POLICY:
        raise KeyError(f"Unknown final model: {model_name}")
    return dict(load_final_tuned_config()["models"][model_name]["parameters"])


def final_tuned_params_sha256(path: Path = FINAL_TUNED_PARAMS_PATH) -> str:
    """Fingerprint the tuned configuration for application cache invalidation."""
    if not path.is_file():
        raise FileNotFoundError(f"Authoritative tuned configuration is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_standard_ppsf_estimator(
    model_name: str,
    parameters: dict | None = None,
) -> PricePerSquareFootRegressor:
    """Build the exact frozen Ridge, RF, or GB Scenario B pipeline."""
    selected = get_final_model_parameters(model_name) if parameters is None else dict(parameters)
    if model_name == "Ridge Regression":
        estimator, scale = Ridge(**selected), True
    elif model_name == "Random Forest":
        selected.update(bootstrap=True, random_state=42, n_jobs=-1)
        estimator, scale = RandomForestRegressor(**selected), False
    elif model_name == "Gradient Boosting":
        selected.update(random_state=42)
        estimator, scale = GradientBoostingRegressor(**selected), False
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


def build_position_lightgbm(parameters: dict | None = None):
    selected = (
        get_final_model_parameters("LightGBM + Position Features")
        if parameters is None
        else dict(parameters)
    )
    selected.update(
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
        objective="regression",
    )
    selected["subsample_freq"] = 1 if float(selected.get("subsample", 1.0)) < 1.0 else 0
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
                LGBMRegressor(**selected),
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
    parameters: dict | None = None,
):
    """Fit one frozen LightGBM PPSF outer fold and reconstruct total price."""
    estimator, numerical, categorical = build_position_lightgbm(parameters)
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
