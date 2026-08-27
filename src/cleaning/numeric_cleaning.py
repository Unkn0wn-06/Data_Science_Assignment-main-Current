"""Parse numeric listing fields and apply verified advertisement corrections."""

import numpy as np
import pandas as pd


# These are the raw count, year, and development fields used by current features.
RAW_NUMERIC_COLUMNS = [
    "Bedroom",
    "Bathroom",
    "Parking Lot",
    "Completion Year",
    "# of Floors",
    "Total Units",
]

# Preserve the manually verified source corrections as auditable constants.
VERIFIED_PRICE_CORRECTIONS = {
    103798808: 480000.0,
}

VERIFIED_SIZE_CORRECTIONS = {
    103423738: 991.0,
    103788197: 850.0,
    101812262: 1227.74,
    102897216: 1450.0,
    103729938: np.nan,
}


def extract_number(series: pd.Series) -> pd.Series:
    """Extract the first ordinary integer or decimal number from each value."""
    return pd.to_numeric(
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.extract(r"([0-9]+(?:\.[0-9]+)?)")[0],
        errors="coerce",
    )


def extract_price(series: pd.Series) -> pd.Series:
    """Convert formatted Ringgit text such as ``RM 340,000`` to 340000."""
    digits = series.astype("string").str.replace(r"[^0-9]", "", regex=True)
    return pd.to_numeric(digits.replace("", pd.NA), errors="coerce")


def clean_price(df: pd.DataFrame) -> pd.DataFrame:
    """Parse positive prices and apply the verified missing-zero correction."""
    cleaned = df.copy()
    cleaned["price"] = extract_price(cleaned["price"])
    for advertisement_id, corrected_price in VERIFIED_PRICE_CORRECTIONS.items():
        cleaned.loc[cleaned["Ad List"].eq(advertisement_id), "price"] = corrected_price
    cleaned.loc[cleaned["price"].le(0), "price"] = np.nan
    return cleaned


def clean_property_size(df: pd.DataFrame) -> pd.DataFrame:
    """Parse square footage and apply all manually verified listing corrections."""
    cleaned = df.copy()
    cleaned["Property Size"] = extract_number(cleaned["Property Size"]).astype(float)
    for advertisement_id, corrected_size in VERIFIED_SIZE_CORRECTIONS.items():
        cleaned.loc[cleaned["Ad List"].eq(advertisement_id), "Property Size"] = corrected_size
    return cleaned


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce raw count, year, floor, and unit fields to numeric values."""
    cleaned = df.copy()
    for column in RAW_NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    return cleaned


def handle_basic_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Median-impute only the three established room and parking count fields."""
    cleaned = df.copy()
    for column in ["Bedroom", "Bathroom", "Parking Lot"]:
        cleaned[column] = cleaned[column].fillna(cleaned[column].median())
    return cleaned

