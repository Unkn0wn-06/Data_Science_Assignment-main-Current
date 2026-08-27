"""Frozen target-free description indicators used by the submitted model."""

from __future__ import annotations

import re

import pandas as pd


# Promoted unchanged from the completed description-feature experiment.
POSITION_REGEXES = {
    "is_high_floor_text": r"\bhigh\s+floor\b",
    "is_low_floor_text": r"\blow\s+floor\b",
    "is_top_floor_text": r"\btop\s+floor\b",
    "has_balcony": r"\bbalcon(?:y|ies)\b",
    "has_large_balcony": r"\b(?:large|huge|spacious)\s+balcon(?:y|ies)\b",
}
POSITION_FEATURES = tuple(POSITION_REGEXES)
POSITION_PATTERNS = {
    name: re.compile(pattern, flags=re.IGNORECASE)
    for name, pattern in POSITION_REGEXES.items()
}


def extract_position_features(cleaned_text: pd.Series) -> pd.DataFrame:
    """Return the five deterministic indicators in frozen schema order."""
    text = cleaned_text.fillna("").astype(str)
    return pd.DataFrame(
        {
            name: text.str.contains(pattern, na=False).astype(int)
            for name, pattern in POSITION_PATTERNS.items()
        },
        index=text.index,
    )
