"""Frozen grouping definitions and model adapters for the final experiment."""

from __future__ import annotations

import numpy as np
from sklearn.base import clone

from src.experimental_support.building_te import NoncoordinatePPSFRegressor
from src.experimental_support.description_models import fit_lightgbm_fold
from src.experimental_support.regex_features import REGEX_GROUPS
from src.models.enhanced_city import build_ppsf_estimator


POSITION_FEATURES = list(REGEX_GROUPS["position"])
MAJOR_COLUMNS = [
    "price",
    "property_size_sqft",
    "bedroom",
    "bathroom",
    "property_type",
    "building_name",
    "developer",
    "city",
    "state",
]
MODEL_SPECS = {
    "random_forest": {"historical_variant": "random_forest_reference", "predictors": 32},
    "lightgbm_interaction": {"historical_variant": "lightgbm_structured_reference", "predictors": 42},
    "building_name_te": {"historical_variant": "building_name_te_reference", "predictors": 42},
    "position_regex_lightgbm": {"historical_variant": "regex_group_position", "predictors": 47},
}


def group_members(frame, columns: list[str]) -> list[np.ndarray]:
    """Return deterministic row-position arrays for keys occurring at least twice."""
    groups = []
    for _, index in frame.groupby(columns, dropna=False, sort=False).groups.items():
        positions = np.sort(np.asarray(index, dtype=int))
        if len(positions) > 1:
            groups.append(positions)
    return sorted(
        groups,
        key=lambda values: (int(values[0]), len(values), tuple(values.tolist())),
    )


def fit_model_fold(model: str, X, y, regex, train_index, validation_index):
    """Fit one exact, pre-existing model configuration on one outer fold."""
    X_train, X_validation = X.iloc[train_index], X.iloc[validation_index]
    y_train = y[train_index]
    if model == "random_forest":
        fitted = clone(build_ppsf_estimator("Random Forest")).fit(X_train, y_train)
        return np.asarray(fitted.predict(X_validation), float), {
            "target_encoding_outer_training_only": True,
            "validation_target_used": False,
        }
    if model == "building_name_te":
        fitted = clone(
            NoncoordinatePPSFRegressor(te_columns=("building_name",))
        ).fit(X_train, y_train)
        return np.asarray(fitted.predict(X_validation), float), {
            "target_encoding_outer_training_only": True,
            "target_encoding_inner_oof": True,
            "validation_target_used": False,
            "selected_m": float(fitted.selected_m_),
        }
    train_dense = None
    validation_dense = None
    if model == "position_regex_lightgbm":
        train_dense = regex.loc[train_index, POSITION_FEATURES]
        validation_dense = regex.loc[validation_index, POSITION_FEATURES]
    output = fit_lightgbm_fold(
        X_train, y_train, X_validation, train_dense, validation_dense
    )
    return output["validation_prediction"], {
        "target_encoding_outer_training_only": True,
        "validation_target_used": False,
        "regex_target_free": model == "position_regex_lightgbm",
    }
