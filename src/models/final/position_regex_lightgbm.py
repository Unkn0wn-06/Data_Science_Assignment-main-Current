"""Deployment interface for the final Position-regex LightGBM PPSF model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.cleaning.pipeline import PROJECT_ROOT
from src.models.final.description_linkage import (
    clean_description_text,
    link_descriptions,
)
from src.models.final.model_builders import (
    build_position_lightgbm,
    prepare_position_features,
)
from src.models.final.regex_features import (
    POSITION_FEATURES,
    extract_position_features as _extract_position_features,
)
from src.models.common.features import MODEL_FEATURES


FINAL_MODEL_NAME = "LightGBM + Position Features"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "enhanced_city_dataset.csv"
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "houses.csv"
PROPERTY_AGE_REFERENCE_YEAR = 2026
POSITION_DISPLAY_NAMES = {
    "is_high_floor_text": "High Floor",
    "is_low_floor_text": "Low Floor",
    "is_top_floor_text": "Top Floor",
    "has_balcony": "Balcony",
    "has_large_balcony": "Large Balcony",
}


def _aligned_clean_descriptions(descriptions, index: pd.Index) -> pd.Series:
    values = pd.Series(descriptions, index=index, dtype="string")
    return clean_description_text(values)


def extract_position_features(descriptions, index: pd.Index | None = None) -> pd.DataFrame:
    """Extract exactly the five established target-free position indicators."""
    if index is None:
        index = pd.RangeIndex(len(descriptions))
    cleaned = _aligned_clean_descriptions(descriptions, index)
    return _extract_position_features(cleaned).loc[:, list(POSITION_FEATURES)]


class PositionRegexLightGBM:
    """Fit PPSF on structured interactions plus five position regex features."""

    def fit(self, X: pd.DataFrame, y, descriptions) -> "PositionRegexLightGBM":
        structured = X.loc[:, MODEL_FEATURES].copy()
        price = np.asarray(y, dtype=float)
        size = pd.to_numeric(
            structured["property_size_sqft"], errors="coerce"
        ).to_numpy(float)
        if len(structured) != len(price) or len(structured) != len(descriptions):
            raise ValueError("Structured rows, targets, and descriptions must align.")
        if np.any(size <= 0) or not np.all(np.isfinite(size)):
            raise ValueError("Property size must be finite and strictly positive.")
        if np.any(price <= 0) or not np.all(np.isfinite(price)):
            raise ValueError("Price must be finite and strictly positive.")

        position = extract_position_features(descriptions, structured.index)
        estimator, numerical, categorical = build_position_lightgbm()
        training = prepare_position_features(structured, position)
        self.estimator_ = clone(estimator).fit(training, price / size)
        self.feature_names_ = list(numerical) + list(categorical)
        importances = self.estimator_.named_steps["model"].feature_importances_
        if len(self.feature_names_) != len(importances):
            raise AssertionError(
                "Final model feature schema does not match LightGBM importances: "
                f"{len(self.feature_names_)} names vs {len(importances)} values."
            )
        self.training_rows_ = len(structured)
        self.description_length_median_ = float(
            pd.to_numeric(structured["description_length"], errors="coerce").median()
        )
        return self

    def _matrix(self, X: pd.DataFrame, descriptions) -> tuple[pd.DataFrame, np.ndarray]:
        if not hasattr(self, "estimator_"):
            raise ValueError("Final model must be fitted before prediction.")
        structured = X.loc[:, MODEL_FEATURES].copy()
        position = extract_position_features(descriptions, structured.index)
        transformed = prepare_position_features(structured, position)
        size = pd.to_numeric(
            structured["property_size_sqft"], errors="coerce"
        ).to_numpy(float)
        if np.any(size <= 0) or not np.all(np.isfinite(size)):
            raise ValueError("Property size must be finite and strictly positive.")
        return transformed, size

    def predict_ppsf(self, X: pd.DataFrame, descriptions) -> np.ndarray:
        transformed, _ = self._matrix(X, descriptions)
        return np.asarray(self.estimator_.predict(transformed), dtype=float)

    def predict(self, X: pd.DataFrame, descriptions) -> np.ndarray:
        transformed, size = self._matrix(X, descriptions)
        predicted_ppsf = np.asarray(self.estimator_.predict(transformed), dtype=float)
        return predicted_ppsf * size

    def feature_importance(self) -> pd.DataFrame:
        values = self.estimator_.named_steps["model"].feature_importances_.astype(float)
        if len(self.feature_names_) != len(values):
            raise AssertionError("Stored final-model feature mapping is misaligned.")
        return pd.DataFrame({"Feature": self.feature_names_, "Importance": values})


def fit_final_model(
    data_path: Path = DATA_PATH,
    raw_path: Path = RAW_PATH,
) -> PositionRegexLightGBM:
    """Fit the deployment model once on all 3,791 canonical eligible rows."""
    data = pd.read_csv(data_path).reset_index(drop=True)
    if len(data) != 3_791 or data["listing_id"].nunique() != 3_791:
        raise AssertionError("Final deployment training requires all 3,791 listings.")
    descriptions, _ = link_descriptions(raw_path, data["listing_id"])
    return PositionRegexLightGBM().fit(
        data[MODEL_FEATURES], data["price"].to_numpy(float), descriptions
    )


def prepare_live_features(
    values: dict,
    description: str,
    description_length_fallback: float,
) -> tuple[pd.DataFrame, pd.Series, dict[str, bool]]:
    """Build one canonical structured row and its target-free position features."""
    row = dict(values)
    size = float(row.get("property_size_sqft", np.nan))
    if not np.isfinite(size) or size <= 0:
        raise ValueError("Property size must be greater than zero.")
    for column in ("bedroom", "bathroom", "parking_lot"):
        value = float(row.get(column, np.nan))
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{column.replace('_', ' ').title()} cannot be negative.")

    completion_year = row.get("completion_year", np.nan)
    if completion_year is None or not np.isfinite(float(completion_year)):
        row["completion_year"] = np.nan
        row["property_age"] = np.nan
    else:
        completion_year = float(completion_year)
        if not 1800 <= completion_year <= 2030:
            raise ValueError("Completion year must be between 1800 and 2030.")
        row["completion_year"] = completion_year
        row["property_age"] = float(PROPERTY_AGE_REFERENCE_YEAR - completion_year)

    for column in ("building_name", "developer", "city"):
        row[column] = str(row.get(column, "")).strip() or "Unknown"
    cleaned = clean_description_text(pd.Series([description])).iloc[0]
    row["description_length"] = (
        float(len(cleaned)) if cleaned else float(description_length_fallback)
    )
    missing = sorted(set(MODEL_FEATURES).difference(row))
    if missing:
        raise ValueError(f"Live input is missing required fields: {missing}")
    structured = pd.DataFrame([row], columns=MODEL_FEATURES)
    descriptions = pd.Series([cleaned], index=structured.index)
    detected_row = extract_position_features(descriptions, structured.index).iloc[0]
    detected = {
        POSITION_DISPLAY_NAMES[name]: bool(detected_row[name])
        for name in POSITION_FEATURES
    }
    return structured, descriptions, detected


def predict_total_price(
    model: PositionRegexLightGBM,
    values: dict,
    description: str,
    description_length_fallback: float,
) -> dict:
    """Return total price, PPSF, and detected position indicators for one listing."""
    structured, descriptions, detected = prepare_live_features(
        values, description, description_length_fallback
    )
    predicted_ppsf = float(model.predict_ppsf(structured, descriptions)[0])
    total_price = predicted_ppsf * float(structured.iloc[0]["property_size_sqft"])
    if not np.isfinite(total_price) or total_price <= 0:
        raise ValueError("The fitted model returned a non-positive or non-finite estimate.")
    return {
        "total_price_RM": float(total_price),
        "ppsf_RM": predicted_ppsf,
        "detected_position_features": detected,
        "description_length": float(structured.iloc[0]["description_length"]),
    }
