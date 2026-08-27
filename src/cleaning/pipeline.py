"""Coordinate the established cleaning stages and production dataset output."""

from pathlib import Path

import pandas as pd

from src.cleaning.categorical_cleaning import standardize_text_columns
from src.cleaning.duplicate_cleaning import (
    remove_exact_duplicates,
    remove_property_duplicates,
)
from src.cleaning.feature_engineering import build_model_dataset, remove_irrelevant_columns
from src.cleaning.invalid_values import handle_invalid_values, handle_outliers
from src.cleaning.missing_values import standardize_missing_values
from src.cleaning.numeric_cleaning import (
    clean_numeric_columns,
    clean_price,
    clean_property_size,
    handle_basic_missing_values,
)
from src.cleaning.validation import validate_prepared_dataset, validate_raw_dataset


# Resolve canonical paths from the repository, never from the launch directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "houses.csv"
PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "production_prepared_dataset.csv"
)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Run the original production cleaning rules in their established order."""
    cleaned = df.copy()
    cleaned = standardize_missing_values(cleaned)
    cleaned = clean_price(cleaned)
    cleaned = clean_property_size(cleaned)
    cleaned = clean_numeric_columns(cleaned)
    cleaned = handle_basic_missing_values(cleaned)
    cleaned = standardize_text_columns(cleaned)
    cleaned = remove_exact_duplicates(cleaned)
    cleaned = remove_property_duplicates(cleaned)
    cleaned = handle_invalid_values(cleaned)
    cleaned = handle_outliers(cleaned)
    cleaned = remove_irrelevant_columns(cleaned)
    return cleaned.reset_index(drop=True)


def clean_and_prepare_dataset(
    source_path: Path = RAW_DATA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Read raw data, run all stages, and return raw/prepared data plus validation."""
    if not source_path.exists():
        raise FileNotFoundError(f"Required raw dataset not found: {source_path}")
    raw = pd.read_csv(source_path)
    raw_validation = validate_raw_dataset(raw)
    cleaned = clean_dataset(raw)
    prepared = build_model_dataset(cleaned)
    prepared_validation = validate_prepared_dataset(prepared)
    validation = {"raw": raw_validation, "prepared": prepared_validation}
    return raw, prepared, validation


def build_production_dataset(
    source_path: Path = RAW_DATA_PATH,
    output_path: Path = PROCESSED_DATA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Rebuild and write the canonical production prepared dataset."""
    raw, prepared, validation = clean_and_prepare_dataset(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False)
    return raw, prepared, validation

