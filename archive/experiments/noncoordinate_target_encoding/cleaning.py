"""Auditable raw-field cleaning helpers for the isolated experiment.

The measured models use the protected canonical enhanced dataset so the baseline
remains exactly comparable. These helpers mirror and extend the repository's raw
parsers without writing a replacement dataset.
"""

from __future__ import annotations

import re

import pandas as pd


FURNISHED_PATTERN = re.compile(
    r"\b(?:furnish(?:ed|ing)?|fully furnished|partly furnished|sofa|bed|fridge|refrigerator)\b",
    flags=re.IGNORECASE,
)
RENOVATED_PATTERN = re.compile(
    r"\b(?:renovat(?:ed|ion|e)|kitchen cabinets?)\b",
    flags=re.IGNORECASE,
)
MISSING_TEXT = {"", "-", "nan", "none", "n/a", "na", "unknown"}


def clean_price(series: pd.Series) -> pd.Series:
    """Parse Ringgit strings to positive floats without clipping the upper tail."""
    normalized = series.astype("string").str.replace(r"[^0-9.]", "", regex=True)
    values = pd.to_numeric(normalized.replace("", pd.NA), errors="coerce")
    return values.astype(float).where(values > 0)


def clean_property_size(series: pd.Series) -> pd.Series:
    """Extract the first positive numeric square-foot value from listing text."""
    normalized = series.astype("string").str.replace(",", "", regex=False)
    extracted = normalized.str.extract(r"([0-9]+(?:\.[0-9]+)?)", expand=False)
    values = pd.to_numeric(extracted, errors="coerce")
    return values.astype(float).where(values > 0)


def clean_numeric(series: pd.Series) -> pd.Series:
    """Coerce one raw numeric field while retaining missing placeholders as NaN."""
    normalized = series.astype("string").str.strip().replace(
        {"": pd.NA, "-": pd.NA, "N/A": pd.NA, "n/a": pd.NA}
    )
    return pd.to_numeric(normalized, errors="coerce")


def description_features(series: pd.Series) -> pd.DataFrame:
    """Build deterministic, target-free furnishing and renovation indicators."""
    text = series.astype("string").fillna("")
    return pd.DataFrame(
        {
            "description_length": text.str.len().astype(float),
            "is_furnished": text.str.contains(FURNISHED_PATTERN, na=False).astype(int),
            "is_renovated": text.str.contains(RENOVATED_PATTERN, na=False).astype(int),
        },
        index=series.index,
    )


def count_facilities(value: object) -> int:
    """Count unique, meaningful comma-separated facility entries."""
    if pd.isna(value):
        return 0
    items = {
        item.strip().lower()
        for item in str(value).split(",")
        if item.strip() and item.strip().lower() not in MISSING_TEXT
    }
    return len(items)


def presence_flag(series: pd.Series) -> pd.Series:
    """Flag whether a raw amenity field records a meaningful value."""
    text = series.astype("string").fillna("").str.strip().str.lower()
    return (~text.isin(MISSING_TEXT)).astype(int)
