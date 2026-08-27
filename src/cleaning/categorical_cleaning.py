"""Normalize model-facing categorical listing fields."""

import pandas as pd


# Keep the exact categorical columns used by the established production cleaner.
MODEL_TEXT_COLUMNS = [
    "Property Type",
    "Tenure Type",
    "Land Title",
    "Floor Range",
]


def standardize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse spacing and label missing model categories as ``Unknown``."""
    cleaned = df.copy()
    for column in [*MODEL_TEXT_COLUMNS, "Address", "Facilities"]:
        text = (
            cleaned[column]
            .astype("string")
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        cleaned[column] = text.fillna("Unknown")
    return cleaned

