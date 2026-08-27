"""Frozen deterministic structured features for the final LightGBM model."""

from __future__ import annotations

import numpy as np
import pandas as pd


SIZE_NUMERICAL_FEATURES = ("log1p_property_size_sqft", "property_size_sqft_squared")
INTERACTION_FEATURES = (
    "state_property_type",
    "city_property_type",
    "city_tenure_type",
    "city_building_name",
    "city_developer",
    "city_size_band",
    "property_type_size_band",
)


def _safe_category(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].astype("string").fillna("__MISSING__")


def _combine(frame: pd.DataFrame, left: str, right: str) -> pd.Series:
    return _safe_category(frame, left) + "__" + _safe_category(frame, right)


def add_non_target_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the exact target-free feature engineering used during evaluation."""
    result = frame.copy()
    size = pd.to_numeric(result["property_size_sqft"], errors="coerce")
    result["log1p_property_size_sqft"] = np.log1p(size)
    result["property_size_sqft_squared"] = np.square(size)
    result["size_band"] = pd.cut(
        size,
        bins=[-np.inf, 600, 800, 1000, 1300, 1800, np.inf],
        labels=["xs", "s", "m", "l", "xl", "xxl"],
    ).astype("string").fillna("__MISSING__")
    result["state_property_type"] = _combine(result, "state", "property_type")
    result["city_property_type"] = _combine(result, "city", "property_type")
    result["city_tenure_type"] = _combine(result, "city", "tenure_type")
    result["city_building_name"] = _combine(result, "city", "building_name")
    result["city_developer"] = _combine(result, "city", "developer")
    result["city_size_band"] = _combine(result, "city", "size_band")
    result["property_type_size_band"] = _combine(
        result, "property_type", "size_band"
    )
    return result


def engineered_feature_lists(numerical_features, categorical_features):
    """Return the unchanged 42-feature structured LightGBM schema."""
    return (
        list(numerical_features) + list(SIZE_NUMERICAL_FEATURES),
        list(categorical_features) + ["size_band"] + list(INTERACTION_FEATURES),
    )
