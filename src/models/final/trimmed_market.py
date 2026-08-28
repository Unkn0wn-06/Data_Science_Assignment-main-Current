"""Experimental deployment fits for the saved restricted-market trim levels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.cleaning.pipeline import PROJECT_ROOT
from src.models.common.features import MODEL_FEATURES
from src.models.final.description_linkage import link_descriptions
from src.models.final.position_regex_lightgbm import (
    DATA_PATH,
    RAW_PATH,
    PositionRegexLightGBM,
)


TRIMMING_RESULTS_DIR = PROJECT_ROOT / "results" / "outlier_trimming"
TRIM_DISTRIBUTION_PATH = TRIMMING_RESULTS_DIR / "distribution_shift.csv"
SUPPORTED_TRIM_LEVELS = (0.0, 0.5, 1.0, 2.5, 5.0, 10.0)
EXPERIMENTAL_TRIM_LEVELS = SUPPORTED_TRIM_LEVELS[1:]
EXPECTED_CANONICAL_ROWS = 3_791


def _validated_trim_level(trim_level: float) -> float:
    level = float(trim_level)
    if level not in SUPPORTED_TRIM_LEVELS:
        choices = ", ".join(f"{value:g}%" for value in SUPPORTED_TRIM_LEVELS)
        raise ValueError(f"Trim level must be one of: {choices}.")
    return level


def _trim_result_row(
    trim_level: float,
    distribution_path: Path = TRIM_DISTRIBUTION_PATH,
) -> pd.Series:
    """Return the authoritative saved distribution row for one trim level."""
    level = _validated_trim_level(trim_level)
    distribution = pd.read_csv(distribution_path)
    required = {
        "Removal_Percent",
        "Full_Population_Cutoff_RM",
        "Before_Row_Count",
        "After_Row_Count",
        "After_Maximum_Price_RM",
        "After_Mean_Price_RM",
    }
    missing = sorted(required.difference(distribution.columns))
    if missing:
        raise ValueError(f"Saved trimming distribution is missing columns: {missing}")
    matched = distribution["Removal_Percent"].astype(float).eq(level)
    if int(matched.sum()) != 1:
        raise ValueError(f"Saved trimming distribution has no unique {level:g}% row.")
    return distribution.loc[matched].iloc[0]


def get_trimmed_population(
    data: pd.DataFrame,
    trim_level: float,
    distribution_path: Path = TRIM_DISTRIBUTION_PATH,
) -> pd.DataFrame:
    """Return retained canonical rows using the completed experiment's saved cutoff.

    Trimming uses only known training-market prices. The input frame is never
    mutated, and no filtering rule is applied later to a user's new property.
    """
    level = _validated_trim_level(trim_level)
    if "price" not in data.columns:
        raise ValueError("Training-market data must contain the known price target.")
    result_row = _trim_result_row(level, distribution_path)
    expected_original = int(result_row["Before_Row_Count"])
    if len(data) != expected_original:
        raise ValueError(
            f"Saved {level:g}% trimming results require {expected_original:,} "
            f"canonical rows; received {len(data):,}."
        )

    if level == 0.0:
        retained = data.copy(deep=True)
    else:
        cutoff = float(result_row["Full_Population_Cutoff_RM"])
        prices = pd.to_numeric(data["price"], errors="coerce")
        if prices.isna().any() or not np.isfinite(prices.to_numpy(float)).all():
            raise ValueError("Canonical training prices must all be finite.")
        retained = data.loc[prices.le(cutoff)].copy(deep=True)

    expected_retained = int(result_row["After_Row_Count"])
    if len(retained) != expected_retained:
        raise AssertionError(
            f"{level:g}% trimming retained {len(retained):,} rows, but the saved "
            f"experiment requires {expected_retained:,}."
        )
    return retained


def get_trim_market_metadata(
    trim_level: float,
    distribution_path: Path = TRIM_DISTRIBUTION_PATH,
) -> dict[str, float | int | str | None]:
    """Return saved market-scope metadata for one restricted-market trim."""
    level = _validated_trim_level(trim_level)
    row = _trim_result_row(level, distribution_path)
    original_rows = int(row["Before_Row_Count"])
    retained_rows = int(row["After_Row_Count"])
    cutoff = row["Full_Population_Cutoff_RM"]
    return {
        "trim_level": level,
        "trim_label": f"{level:g}%",
        "original_rows": original_rows,
        "removed_rows": original_rows - retained_rows,
        "retained_rows": retained_rows,
        "retention_percentage": 100.0 * retained_rows / original_rows,
        "cutoff_RM": None if pd.isna(cutoff) else float(cutoff),
        "maximum_retained_price_RM": float(row["After_Maximum_Price_RM"]),
        "mean_retained_price_RM": float(row["After_Mean_Price_RM"]),
    }


def fit_trimmed_market_model(
    trim_level: float,
    data_path: Path = DATA_PATH,
    raw_path: Path = RAW_PATH,
    distribution_path: Path = TRIM_DISTRIBUTION_PATH,
) -> PositionRegexLightGBM:
    """Fit the final architecture once on the selected retained market."""
    level = _validated_trim_level(trim_level)
    if level == 0.0:
        raise ValueError("Use fit_final_model for the official 0% deployment model.")

    canonical = pd.read_csv(data_path).reset_index(drop=True)
    required = {"listing_id", "price", *MODEL_FEATURES}
    missing = sorted(required.difference(canonical.columns))
    if missing:
        raise ValueError(f"Canonical dataset is missing required columns: {missing}")
    if (
        len(canonical) != EXPECTED_CANONICAL_ROWS
        or canonical["listing_id"].nunique() != EXPECTED_CANONICAL_ROWS
    ):
        raise AssertionError(
            "Experimental deployment fitting requires all 3,791 canonical listings "
            "before trimming."
        )

    descriptions, _ = link_descriptions(raw_path, canonical["listing_id"])
    retained = get_trimmed_population(canonical, level, distribution_path)
    retained_index = retained.index
    retained_descriptions = descriptions.loc[retained_index].reset_index(drop=True)
    retained = retained.reset_index(drop=True)

    model = PositionRegexLightGBM().fit(
        retained[MODEL_FEATURES],
        retained["price"].to_numpy(float),
        retained_descriptions,
    )
    metadata = get_trim_market_metadata(level, distribution_path)
    if model.training_rows_ != metadata["retained_rows"]:
        raise AssertionError("Trimmed model training rows do not match saved results.")
    model.trim_level_ = level
    model.trim_cutoff_RM_ = metadata["cutoff_RM"]
    model.original_training_rows_ = EXPECTED_CANONICAL_ROWS
    return model
