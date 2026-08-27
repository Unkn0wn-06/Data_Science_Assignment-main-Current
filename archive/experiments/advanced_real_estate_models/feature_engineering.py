"""Leakage-safe size, location, and micro-market feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import KFold


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
MICRO_LEVELS = (
    "state",
    "city",
    "state_property_type",
    "city_property_type",
    "building_name",
    "developer",
)
MICRO_STATS = ("median", "mean", "count", "iqr")
MICRO_FEATURES = tuple(
    f"micro_{level}_{stat}" for level in MICRO_LEVELS for stat in MICRO_STATS
)


def _safe_category(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].astype("string").fillna("__MISSING__")


def _combine(frame: pd.DataFrame, left: str, right: str) -> pd.Series:
    return _safe_category(frame, left) + "__" + _safe_category(frame, right)


def add_non_target_features(
    frame: pd.DataFrame, include_interactions: bool = True
) -> pd.DataFrame:
    """Add deterministic features that never consult price or validation rows."""
    result = frame.copy()
    size = pd.to_numeric(result["property_size_sqft"], errors="coerce")
    result["log1p_property_size_sqft"] = np.log1p(size)
    result["property_size_sqft_squared"] = np.square(size)
    result["size_band"] = pd.cut(
        size,
        bins=[-np.inf, 600, 800, 1000, 1300, 1800, np.inf],
        labels=["xs", "s", "m", "l", "xl", "xxl"],
    ).astype("string").fillna("__MISSING__")
    if include_interactions:
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


class MicroMarketPPSFEncoder:
    """Map fold-trained PPSF aggregates with explicit hierarchical fallbacks."""

    fallback_levels = {
        "state": (),
        "city": ("state",),
        "state_property_type": ("state",),
        "city_property_type": ("city", "state_property_type", "state"),
        "building_name": ("city_property_type", "city", "state"),
        "developer": ("city_property_type", "city", "state"),
    }

    @staticmethod
    def _keys(frame: pd.DataFrame) -> dict[str, pd.Series]:
        return {
            "state": _safe_category(frame, "state"),
            "city": _safe_category(frame, "city"),
            "state_property_type": _combine(frame, "state", "property_type"),
            "city_property_type": _combine(frame, "city", "property_type"),
            "building_name": _safe_category(frame, "building_name"),
            "developer": _safe_category(frame, "developer"),
        }

    def fit(self, X: pd.DataFrame, y) -> "MicroMarketPPSFEncoder":
        price = np.asarray(y, dtype=float)
        size = pd.to_numeric(X["property_size_sqft"], errors="coerce").to_numpy(float)
        ppsf = price / size
        if not np.all(np.isfinite(ppsf)) or np.any(ppsf <= 0):
            raise ValueError("Micro-market PPSF must be finite and positive.")
        self.global_stats_ = {
            "median": float(np.median(ppsf)),
            "mean": float(np.mean(ppsf)),
            "count": float(len(ppsf)),
            "iqr": float(np.quantile(ppsf, 0.75) - np.quantile(ppsf, 0.25)),
        }
        self.tables_ = {}
        keys = self._keys(X)
        for level, key in keys.items():
            source = pd.DataFrame({"key": key.to_numpy(), "ppsf": ppsf})
            grouped = source.groupby("key", dropna=False)["ppsf"]
            table = grouped.agg(median="median", mean="mean", count="count")
            table["iqr"] = grouped.quantile(0.75) - grouped.quantile(0.25)
            table["iqr"] = table["iqr"].fillna(0.0)
            self.tables_[level] = table
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        keys = self._keys(X)
        encoded = pd.DataFrame(index=X.index)
        for level in MICRO_LEVELS:
            for stat in MICRO_STATS:
                values = keys[level].map(self.tables_[level][stat])
                if stat == "count":
                    values = values.fillna(0.0)
                else:
                    for fallback in self.fallback_levels[level]:
                        values = values.fillna(
                            keys[fallback].map(self.tables_[fallback][stat])
                        )
                    values = values.fillna(self.global_stats_[stat])
                encoded[f"micro_{level}_{stat}"] = values.astype(float)
        return encoded


def oof_micro_market_features(
    X: pd.DataFrame, y, n_splits: int = 5, random_state: int = 42
) -> pd.DataFrame:
    """Encode each training row without using that row's own target."""
    result = pd.DataFrame(index=X.index, columns=MICRO_FEATURES, dtype=float)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    positions = np.arange(len(X))
    for train_position, validation_position in splitter.split(positions):
        encoder = MicroMarketPPSFEncoder().fit(
            X.iloc[train_position], np.asarray(y, dtype=float)[train_position]
        )
        encoded = encoder.transform(X.iloc[validation_position])
        result.loc[X.index[validation_position], :] = encoded.to_numpy()
    if result.isna().any().any():
        raise AssertionError("OOF micro-market features contain missing values.")
    return result.astype(float)


def engineered_feature_lists(
    numerical_features,
    categorical_features,
    *,
    include_micro: bool,
    include_interactions: bool,
    remove_city: bool = False,
    remove_building_developer: bool = False,
) -> tuple[list[str], list[str]]:
    """Return the exact schema for a full or ablated engineered model."""
    numerical = list(numerical_features) + list(SIZE_NUMERICAL_FEATURES)
    categorical = list(categorical_features) + ["size_band"]
    if include_interactions:
        categorical.extend(INTERACTION_FEATURES)
    if include_micro:
        numerical.extend(MICRO_FEATURES)

    if remove_city:
        categorical = [
            name
            for name in categorical
            if name != "city" and not name.startswith("city_")
        ]
        numerical = [
            name
            for name in numerical
            if not name.startswith("micro_city_")
            and not name.startswith("micro_city_property_type_")
        ]
    if remove_building_developer:
        categorical = [
            name
            for name in categorical
            if name not in {"building_name", "developer", "city_building_name", "city_developer"}
        ]
        numerical = [
            name
            for name in numerical
            if not name.startswith("micro_building_name_")
            and not name.startswith("micro_developer_")
        ]
    return numerical, categorical


class FeatureEngineeringPPSFRegressor(RegressorMixin, BaseEstimator):
    """Fit engineered training rows and reconstruct uncapped total prices."""

    def __init__(
        self,
        regressor=None,
        include_micro: bool = True,
        include_interactions: bool = True,
        micro_folds: int = 5,
        random_state: int = 42,
    ):
        self.regressor = regressor
        self.include_micro = include_micro
        self.include_interactions = include_interactions
        self.micro_folds = micro_folds
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y):
        target = np.asarray(y, dtype=float)
        training = add_non_target_features(X, self.include_interactions)
        self.micro_encoder_ = None
        if self.include_micro:
            micro = oof_micro_market_features(
                X,
                target,
                n_splits=self.micro_folds,
                random_state=self.random_state,
            )
            training = training.join(micro)
            self.micro_encoder_ = MicroMarketPPSFEncoder().fit(X, target)
        size = pd.to_numeric(X["property_size_sqft"], errors="coerce").to_numpy(float)
        self.regressor_ = clone(self.regressor)
        self.regressor_.fit(training, target / size)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        transformed = add_non_target_features(X, self.include_interactions)
        if self.micro_encoder_ is not None:
            transformed = transformed.join(self.micro_encoder_.transform(X))
        predicted_ppsf = np.asarray(self.regressor_.predict(transformed), dtype=float)
        size = pd.to_numeric(X["property_size_sqft"], errors="coerce").to_numpy(float)
        return predicted_ppsf * size
