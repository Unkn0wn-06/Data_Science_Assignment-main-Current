"""Standardize textual missing-value markers before field-specific cleaning."""

import pandas as pd


# Preserve every missing marker recognized by the established production cleaner.
MISSING_MARKERS = {
    "",
    "-",
    "--",
    "n/a",
    "na",
    "nan",
    "none",
    "null",
    "unknown",
}


def standardize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy whose textual missing markers use pandas missing values."""
    cleaned = df.copy()
    # Normalize each text-like column without modifying numeric source fields.
    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        text = cleaned[column].astype("string").str.strip()
        cleaned[column] = text.mask(text.str.lower().isin(MISSING_MARKERS), pd.NA)
    return cleaned

