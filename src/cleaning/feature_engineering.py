"""Construct the established model-ready features from cleaned raw listings."""

import pandas as pd

from src.cleaning.location_cleaning import extract_city, extract_state
from src.models.common.features import FEATURES


# Retain raw fields needed to construct the production model schema.
RELEVANT_COLUMNS = [
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
]


def remove_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select relevant raw fields in their stable established order."""
    return df[RELEVANT_COLUMNS].copy()


def recorded(*series: pd.Series) -> pd.Series:
    """Combine one or more amenity sources into a binary presence indicator."""
    present = pd.Series(False, index=series[0].index)
    for values in series:
        text = values.astype("string").str.strip()
        present |= text.notna() & ~text.str.lower().isin(["", "-", "nan", "none"])
    return present.astype(int)


def build_model_dataset(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Convert cleaned raw-grain listings to the exact production feature schema."""
    output = pd.DataFrame(index=cleaned.index)
    output["price"] = cleaned["price"].astype(float)
    output["property_size_sqft"] = cleaned["Property Size"].astype(float)
    output["bedroom"] = cleaned["Bedroom"].astype(float)
    output["bathroom"] = cleaned["Bathroom"].astype(float)
    output["parking_lot"] = cleaned["Parking Lot"].astype(float)
    output["completion_year"] = cleaned["Completion Year"].astype(float)
    output["number_of_floors"] = cleaned["# of Floors"].astype(float)
    output["total_units"] = cleaned["Total Units"].astype(float)
    output["property_type"] = cleaned["Property Type"].astype(str)
    output["tenure_type"] = cleaned["Tenure Type"].astype(str)
    output["land_title"] = cleaned["Land Title"].astype(str)
    output["floor_range"] = cleaned["Floor Range"].astype(str)
    output["state"] = cleaned["Address"].map(extract_state)
    output["city"] = cleaned["Address"].map(extract_city)

    # Count normalized unique facilities while treating Unknown as empty.
    facilities = cleaned["Facilities"].astype("string").str.strip()
    output["facilities_count"] = facilities.map(
        lambda value: (
            0
            if pd.isna(value) or value.lower() in {"", "unknown"}
            else len({item.strip().lower() for item in value.split(",") if item.strip()})
        )
    )
    output["has_school"] = recorded(cleaned["Nearby School"], cleaned["School"])
    output["has_mall"] = recorded(cleaned["Nearby Mall"], cleaned["Mall"])
    output["has_hospital"] = recorded(cleaned["Hospital"])
    output["has_railway"] = recorded(
        cleaned["Nearby Railway Station"], cleaned["Railway Station"]
    )
    output["has_bus_stop"] = recorded(cleaned["Bus Stop"])
    output["has_park"] = recorded(cleaned["Park"])
    output["has_highway"] = recorded(cleaned["Highway"])
    output = output[["price", *FEATURES]].reset_index(drop=True)
    output["price"] = output["price"].astype(int)
    return output

