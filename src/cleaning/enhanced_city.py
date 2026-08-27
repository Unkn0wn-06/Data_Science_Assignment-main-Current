"""Build the canonical pre-encoding enhanced City feature table."""

from pathlib import Path

import pandas as pd

from src.cleaning.categorical_cleaning import standardize_text_columns
from src.cleaning.duplicate_cleaning import (
    remove_exact_duplicates,
    remove_property_duplicates,
)
from src.cleaning.feature_engineering import build_model_dataset
from src.cleaning.invalid_values import handle_invalid_values, handle_outliers
from src.cleaning.location_cleaning import extract_city
from src.cleaning.missing_values import standardize_missing_values
from src.cleaning.numeric_cleaning import (
    clean_numeric_columns,
    clean_price,
    clean_property_size,
    handle_basic_missing_values,
)
from src.cleaning.pipeline import PROJECT_ROOT, RAW_DATA_PATH
from src.models.common.features import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
)


ENHANCED_CITY_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "enhanced_city_dataset.csv"
)
HISTORICAL_CITY_ROWS = 3791


def prepare_enhanced_dataset() -> pd.DataFrame:
    """Reproduce the enhanced experiment fields and retain City plus address."""
    cleaned = pd.read_csv(RAW_DATA_PATH)
    cleaned = standardize_missing_values(cleaned)
    cleaned = clean_price(cleaned)
    cleaned = clean_property_size(cleaned)
    cleaned = clean_numeric_columns(cleaned)
    cleaned = handle_basic_missing_values(cleaned)
    cleaned = standardize_text_columns(cleaned)
    cleaned = remove_exact_duplicates(cleaned)
    cleaned = remove_property_duplicates(cleaned)
    cleaned = handle_invalid_values(cleaned)
    cleaned = handle_outliers(cleaned).reset_index(drop=True)

    # Start from the proven production fields, then add only historical enhanced fields.
    output = build_model_dataset(cleaned)
    output.insert(0, "listing_id", cleaned["Ad List"].astype(int).to_numpy())
    output["completion_year"] = cleaned["Completion Year"].astype(float)
    output["property_age"] = 2026.0 - output["completion_year"]
    output["number_of_floors"] = cleaned["# of Floors"].astype(float)
    output["total_units"] = cleaned["Total Units"].astype(float)

    description = cleaned["description"].astype("string").fillna("").str.lower()
    output["description_length"] = description.str.len().astype(float)
    facilities = cleaned["Facilities"].astype("string").fillna("").str.lower()
    output["has_swimming_pool"] = facilities.str.contains(
        "swimming pool", regex=False
    ).astype(int)
    output["has_security"] = facilities.str.contains("security", regex=False).astype(int)
    output["has_lift"] = facilities.str.contains("lift", regex=False).astype(int)
    output["has_gym"] = facilities.str.contains(r"gym|gymnasium", regex=True).astype(int)
    output["has_playground"] = facilities.str.contains(
        "playground", regex=False
    ).astype(int)
    output["is_furnished"] = description.str.contains(
        r"furnished|furnishing", regex=True
    ).astype(int)
    output["is_renovated"] = description.str.contains(
        r"renovated|renovation", regex=True
    ).astype(int)
    output["building_name"] = (
        cleaned["Building Name"]
        .astype("string")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .fillna("Unknown")
    )
    output["developer"] = (
        cleaned["Developer"]
        .astype("string")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .fillna("Unknown")
    )
    output["detailed_address"] = cleaned["Address"].astype("string").fillna("Unknown")
    output["city"] = output["detailed_address"].map(extract_city)

    # Detailed address is retained only for historical experiments, never canonical input.
    columns = ["listing_id", TARGET_COLUMN, *MODEL_FEATURES, "detailed_address"]
    return output[columns].reset_index(drop=True)


def canonical_enhanced_city_table() -> pd.DataFrame:
    """Return identifier, untouched price target, and the canonical 32 features."""
    enhanced = prepare_enhanced_dataset()
    canonical = enhanced[["listing_id", TARGET_COLUMN, *MODEL_FEATURES]].copy()
    if len(canonical) != HISTORICAL_CITY_ROWS:
        raise ValueError(
            "Enhanced City row count differs from the historical experiment: "
            f"expected {HISTORICAL_CITY_ROWS}, found {len(canonical)}."
        )
    return canonical


def dataset_profile(data: pd.DataFrame) -> dict:
    """Return the requested schema, missingness, duplication, and distributions."""
    target = data[TARGET_COLUMN]
    size = data["property_size_sqft"]

    def distribution(series: pd.Series) -> dict:
        return {
            "count": int(series.count()),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "p25": float(series.quantile(0.25)),
            "median": float(series.median()),
            "p75": float(series.quantile(0.75)),
            "p95": float(series.quantile(0.95)),
            "max": float(series.max()),
            "skew": float(series.skew()),
        }

    return {
        "rows": len(data),
        "columns": len(data.columns),
        "missing_values_total": int(data.isna().sum().sum()),
        "missing_values_by_column": {
            column: int(count)
            for column, count in data.isna().sum().items()
            if count
        },
        "duplicate_rows": int(data.duplicated().sum()),
        "duplicate_model_rows_excluding_id": int(
            data[[TARGET_COLUMN, *MODEL_FEATURES]].duplicated().sum()
        ),
        "numerical_columns": NUMERICAL_FEATURES,
        "categorical_columns": CATEGORICAL_FEATURES,
        "model_features": MODEL_FEATURES,
        "target_distribution": distribution(target),
        "property_size_distribution": distribution(size),
        "city_unique_count": int(data["city"].nunique()),
    }


def build_enhanced_city_dataset(
    output_path: Path = ENHANCED_CITY_DATA_PATH,
) -> tuple[pd.DataFrame, dict]:
    """Write the canonical unencoded, non-Winsorized enhanced City dataset."""
    data = canonical_enhanced_city_table()
    profile = dataset_profile(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    return data, profile

