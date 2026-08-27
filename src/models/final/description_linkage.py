"""Deterministic source-description linkage for final-model inference."""

from __future__ import annotations

import numpy as np
import pandas as pd

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


def clean_description_text(series: pd.Series) -> pd.Series:
    """Normalize description text exactly as in the frozen experiment."""
    text = series.fillna("").astype(str).str.lower()
    return text.str.replace(r"\s+", " ", regex=True).str.strip()


def clean_raw_to_canonical(raw_path) -> pd.DataFrame:
    """Replay the cleaning sequence that established canonical eligibility."""
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


def link_descriptions(raw_path, canonical_listing_ids) -> tuple[pd.Series, dict]:
    """Recover descriptions by exact canonical order; never fuzzy-match."""
    raw = pd.read_csv(raw_path)
    cleaned = clean_raw_to_canonical(raw_path)
    source_ids = cleaned["Ad List"].astype(int).to_numpy()
    canonical_ids = np.asarray(canonical_listing_ids, dtype=int)
    if len(canonical_ids) != len(np.unique(canonical_ids)):
        raise AssertionError("Canonical listing IDs are not unique.")
    if len(source_ids) != len(np.unique(source_ids)):
        raise AssertionError("Cleaned raw listing IDs are not unique.")
    if not np.array_equal(source_ids, canonical_ids):
        raise AssertionError("Cleaned raw IDs do not match canonical IDs in exact order.")
    descriptions = clean_description_text(cleaned["description"])
    audit = {
        "linkage_method": "replay canonical cleaning pipeline, then exact Ad List == listing_id order assertion",
        "fuzzy_join_used": False,
        "raw_rows": int(len(raw)),
        "cleaned_raw_rows": int(len(cleaned)),
        "canonical_rows": int(len(canonical_ids)),
        "linked_rows": int(len(descriptions)),
        "unique_canonical_ids": int(len(np.unique(canonical_ids))),
        "unique_cleaned_raw_ids": int(len(np.unique(source_ids))),
        "missing_description_count": int(descriptions.eq("").sum()),
        "many_to_many_join": False,
        "exact_order_match": True,
        "rows_reintroduced": 0,
    }
    return descriptions.reset_index(drop=True), audit
