"""Fold-local premium classifiers and deterministic description indicators."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import fbeta_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from experiments.advanced_real_estate_models.feature_engineering import (
    add_non_target_features,
    engineered_feature_lists,
)
from src.cleaning.categorical_cleaning import standardize_text_columns
from src.cleaning.duplicate_cleaning import remove_exact_duplicates, remove_property_duplicates
from src.cleaning.invalid_values import handle_invalid_values, handle_outliers
from src.cleaning.missing_values import standardize_missing_values
from src.cleaning.numeric_cleaning import (
    clean_numeric_columns,
    clean_price,
    clean_property_size,
    handle_basic_missing_values,
)
from src.models.common.features import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


ROUTING_THRESHOLDS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70)
DESCRIPTION_PATTERNS = {
    "is_penthouse": r"\bpenthouse\b",
    "is_duplex_text": r"\bduplex\b",
    "is_triplex": r"\btriplex\b",
    "is_corner_unit": r"\bcorner\s+unit\b",
    "has_private_lift": r"\bprivate\s+lift\b",
    "has_private_pool": r"\bprivate\s+(?:swimming\s+)?pool\b",
    "has_balcony": r"\bbalcon(?:y|ies)\b",
    "has_sea_view": r"\bsea\s+view\b",
    "has_city_view": r"\bcity\s+view\b",
    "has_klcc_view": r"\bklcc\s+view\b",
    "is_high_floor_text": r"\bhigh\s+floor\b",
    "is_luxury_text": r"\b(?:luxury|luxurious)\b",
    "is_designer_unit": r"\bdesigner\s+(?:unit|home|interior)\b",
    "is_fully_renovated_text": r"\bfully\s+renovated\b",
    "is_fully_furnished_text": r"\bfully\s+furnished\b",
    "is_dual_key": r"\bdual[ -]?key\b",
    "is_garden_unit": r"\bgarden\s+unit\b",
    "is_loft": r"\bloft\b",
}


def fold_premium_threshold(y_outer_train, quantile: float = 0.95) -> float:
    """Derive a routing label threshold from outer-training prices only."""
    values = np.asarray(y_outer_train, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Premium thresholds require finite training prices.")
    return float(np.quantile(values, quantile))


def _cleaned_raw(raw_path) -> pd.DataFrame:
    frame = pd.read_csv(raw_path)
    frame = standardize_missing_values(frame)
    frame = clean_price(frame)
    frame = clean_property_size(frame)
    frame = clean_numeric_columns(frame)
    frame = handle_basic_missing_values(frame)
    frame = standardize_text_columns(frame)
    frame = remove_exact_duplicates(frame)
    frame = remove_property_duplicates(frame)
    frame = handle_invalid_values(frame)
    return handle_outliers(frame).reset_index(drop=True)


def description_feature_table(raw_path, canonical_listing_ids, minimum_count: int = 10):
    """Return all predefined flags, frequencies, and modeling-eligible columns."""
    cleaned = _cleaned_raw(raw_path)
    ids = cleaned["Ad List"].astype(int).to_numpy()
    expected = np.asarray(canonical_listing_ids, dtype=int)
    if not np.array_equal(ids, expected):
        raise AssertionError("Cleaned raw descriptions do not align to canonical IDs.")
    text = cleaned["description"].astype("string").fillna("")
    features = pd.DataFrame(index=np.arange(len(cleaned)))
    for name, pattern in DESCRIPTION_PATTERNS.items():
        features[name] = text.str.contains(
            re.compile(pattern, flags=re.IGNORECASE), na=False
        ).astype(int)
    frequencies = {column: int(features[column].sum()) for column in features}
    eligible = [column for column, count in frequencies.items() if count >= minimum_count]
    return features, frequencies, eligible


def _target_free_preprocessor(numerical, categorical):
    return ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), list(numerical)),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ordinal",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                        ),
                    ]
                ),
                list(categorical),
            ),
        ]
    )


def build_classifier_pipeline(family: str, numerical, categorical):
    if family == "lightgbm":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=6,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=2.0,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
    elif family == "random_forest":
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_leaf=3,
            max_features=0.7,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown classifier family: {family}")
    return Pipeline(
        [("preprocessor", _target_free_preprocessor(numerical, categorical)), ("model", model)]
    )


class PremiumClassifier(ClassifierMixin, BaseEstimator):
    """Classifier using target-free canonical features and interactions only."""

    def __init__(self, family: str = "lightgbm", extra_numerical: tuple[str, ...] = ()):
        self.family = family
        self.extra_numerical = extra_numerical

    def feature_schema(self):
        numerical, categorical = engineered_feature_lists(
            NUMERICAL_FEATURES,
            CATEGORICAL_FEATURES,
            include_micro=False,
            include_interactions=True,
        )
        numerical.extend(self.extra_numerical)
        return numerical, categorical

    def fit(self, X: pd.DataFrame, y):
        labels = np.asarray(y, dtype=int)
        if np.unique(labels).size != 2:
            raise ValueError("Premium classifier requires both classes.")
        numerical, categorical = self.feature_schema()
        self.model_ = clone(build_classifier_pipeline(self.family, numerical, categorical))
        self.model_.fit(add_non_target_features(X, include_interactions=True), labels)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X: pd.DataFrame):
        return self.model_.predict_proba(add_non_target_features(X, include_interactions=True))

    def predict(self, X: pd.DataFrame):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def inner_oof_probabilities(
    X: pd.DataFrame,
    labels,
    classifier: PremiumClassifier,
    n_splits: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """Cross-fit probabilities entirely inside one outer-training partition."""
    target = np.asarray(labels, dtype=int)
    result = np.empty(len(target), dtype=float)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_position, validation_position in cv.split(np.arange(len(target))):
        fitted = clone(classifier).fit(X.iloc[train_position], target[train_position])
        result[validation_position] = fitted.predict_proba(X.iloc[validation_position])[:, 1]
    return result


def select_routing_threshold(labels, probabilities, candidates=ROUTING_THRESHOLDS):
    """Select by training-only F2, which weights premium recall over precision."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    scores = {
        float(threshold): float(
            fbeta_score(labels, probabilities >= threshold, beta=2, zero_division=0)
        )
        for threshold in candidates
    }
    selected = max(candidates, key=lambda threshold: (scores[float(threshold)], -threshold))
    return float(selected), scores
