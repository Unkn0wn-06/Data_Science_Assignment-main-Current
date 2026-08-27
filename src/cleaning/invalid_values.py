"""Apply the existing broad validity ranges and conservative outlier rule."""

import numpy as np
import pandas as pd


def handle_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    """Invalidate impossible values and drop rows missing target or property size."""
    cleaned = df.copy()
    cleaned.loc[~cleaned["Property Size"].between(100, 20000), "Property Size"] = np.nan
    cleaned.loc[~cleaned["Bedroom"].between(0, 20), "Bedroom"] = np.nan
    cleaned.loc[~cleaned["Bathroom"].between(0, 20), "Bathroom"] = np.nan
    cleaned.loc[~cleaned["Parking Lot"].between(0, 20), "Parking Lot"] = np.nan
    cleaned.loc[~cleaned["Completion Year"].between(1900, 2030), "Completion Year"] = np.nan
    cleaned.loc[~cleaned["# of Floors"].between(1, 200), "# of Floors"] = np.nan
    cleaned.loc[~cleaned["Total Units"].between(1, 20000), "Total Units"] = np.nan
    return cleaned.dropna(subset=["price", "Property Size"])


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove only the established implausible RM50–RM5,000 price/sq.ft. rows."""
    cleaned = df.copy()
    price_per_square_foot = cleaned["price"] / cleaned["Property Size"]
    return cleaned.loc[price_per_square_foot.between(50, 5000)].copy()

