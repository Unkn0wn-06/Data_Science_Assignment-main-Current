"""Central validation checks for raw, cleaned, and model-ready datasets."""

import numpy as np
import pandas as pd

from src.models.common.features import (
    FEATURES,
    PRODUCTION_NUMERICAL_FEATURES,
    TARGET_COLUMN,
)


# Validate every raw field referenced by the staged production cleaner.
RAW_REQUIRED_COLUMNS = {
    "Ad List",
    "price",
    "Property Size",
    "Bedroom",
    "Bathroom",
    "Parking Lot",
    "Completion Year",
    "# of Floors",
    "Total Units",
    "Facilities",
    "Property Type",
    "Tenure Type",
    "Land Title",
    "Floor Range",
    "Address",
    "Nearby School",
    "School",
    "Nearby Mall",
    "Mall",
    "Hospital",
    "Nearby Railway Station",
    "Railway Station",
    "Bus Stop",
    "Park",
    "Highway",
}


def _require_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    """Raise a clear error when a dataset omits required fields."""
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")


def validate_raw_dataset(df: pd.DataFrame) -> dict:
    """Validate source shape and required columns without changing the source data."""
    _require_columns(df, RAW_REQUIRED_COLUMNS, "Raw dataset")
    if df.empty:
        raise ValueError("Raw dataset contains no rows.")
    return {
        "valid": True,
        "rows": len(df),
        "columns": len(df.columns),
        "exact_duplicates": int(df.duplicated().sum()),
    }


def validate_prepared_dataset(df: pd.DataFrame) -> dict:
    """Validate the production schema, target, duplicates, and finite numerics."""
    _require_columns(df, {TARGET_COLUMN, *FEATURES}, "Prepared dataset")
    if df.empty:
        raise ValueError("Prepared dataset contains no rows.")
    target = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    if target.isna().any() or not np.isfinite(target.to_numpy(dtype=float)).all():
        raise ValueError("Prepared target contains missing or non-finite prices.")
    if (target <= 0).any():
        raise ValueError("Prepared target contains a non-positive price.")
    numeric = df[PRODUCTION_NUMERICAL_FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )
    infinite_count = int(np.isinf(numeric.to_numpy(dtype=float)).sum())
    if infinite_count:
        raise ValueError(f"Prepared numeric features contain {infinite_count} infinite values.")
    return {
        "valid": True,
        "rows": len(df),
        "columns": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": int(df.isna().sum().sum()),
        "infinite_numeric_values": infinite_count,
    }
