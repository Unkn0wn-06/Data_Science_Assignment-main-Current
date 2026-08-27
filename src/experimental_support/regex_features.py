"""Predefined, target-free real-estate description indicators."""

from __future__ import annotations

import re

import pandas as pd


REGEX_GROUPS = {
    "layout": {
        "is_penthouse": r"\bpent\s*house\b|\bpenthouse\b",
        "is_duplex_text": r"\bduplex\b",
        "is_triplex": r"\btriplex\b",
        "is_loft": r"\bloft\b",
        "is_dual_key": r"\bdual[ -]?key\b",
        "is_studio_text": r"\bstudio(?:\s+unit)?\b",
        "is_corner_unit": r"\bcorner\s+(?:unit|lot)\b",
        "is_end_lot_text": r"\bend\s+lot\b",
        "is_garden_unit": r"\bgarden\s+unit\b",
    },
    "private_luxury": {
        "has_private_lift": r"\bprivate\s+(?:lift|elevator)\b",
        "has_private_lift_lobby": r"\bprivate\s+(?:lift|elevator)\s+lobb(?:y|ies)\b",
        "has_private_pool": r"\bprivate\s+(?:swimming\s+)?pool\b",
        "has_private_garden": r"\bprivate\s+garden\b",
        "has_private_entrance": r"\bprivate\s+entrance\b",
        "is_luxury_text": r"\bluxury\b",
        "is_premium_text": r"\bpremium\b",
        "is_exclusive_text": r"\bexclusive\b",
        "is_luxurious_text": r"\bluxurious\b",
        "is_branded_residence": r"\bbranded\s+residence(?:s)?\b",
    },
    "views": {
        "has_sea_view": r"\bsea\s+view\b",
        "has_city_view": r"\bcity\s+view\b",
        "has_klcc_view": r"\bklcc\s+view\b",
        "has_pool_view": r"\bpool\s+view\b",
        "has_golf_view": r"\bgolf(?:\s+course)?\s+view\b",
        "has_unblocked_view": r"\bunblock(?:ed)?\s+view\b",
        "has_panorama_view": r"\bpanoram(?:a|ic)\s+view\b",
    },
    "position": {
        "is_high_floor_text": r"\bhigh\s+floor\b",
        "is_low_floor_text": r"\blow\s+floor\b",
        "is_top_floor_text": r"\btop\s+floor\b",
        "has_balcony": r"\bbalcon(?:y|ies)\b",
        "has_large_balcony": r"\b(?:large|huge|spacious)\s+balcon(?:y|ies)\b",
    },
    "renovation_furnishing": {
        "is_fully_furnished_text": r"\bfully\s+furnished\b",
        "is_partly_furnished_text": r"\bpart(?:ly|ially)\s+furnished\b|\bsemi[ -]?furnished\b",
        "is_unfurnished_text": r"\bunfurnished\b|\bnot\s+furnished\b",
        "is_fully_renovated_text": r"\bfully\s+renovated\b",
        "is_designer_unit": r"\bdesigner\s+(?:unit|home|interior|renovation)\b",
        "has_kitchen_cabinet": r"\bkitchen\s+cabinet(?:s)?\b",
        "has_built_in_wardrobe": r"\bbuilt[ -]?in\s+wardrobe(?:s)?\b",
        "has_air_conditioning": r"\bair[ -]?(?:cond(?:ition(?:ing|er)?)?|con)\b",
    },
    "sale_context": {
        "is_urgent_sale": r"\burgent\s+sale\b|\burgent\s+sell\b|\bmust\s+sell\b",
        "is_below_market": r"\bbelow\s+market(?:\s+(?:price|value))?\b",
        "is_owner_listing": r"\b(?:direct\s+)?owner\s+(?:listing|sale)\b|\bowner\s+direct\b",
        "is_tenanted": r"\btenanted\b|\bwith\s+tenant\b",
        "is_vacant": r"\bvacant\b|\bvacant\s+possession\b",
    },
}

PATTERNS = {
    name: re.compile(pattern, flags=re.IGNORECASE)
    for group in REGEX_GROUPS.values()
    for name, pattern in group.items()
}
FEATURE_TO_GROUP = {
    name: group_name for group_name, group in REGEX_GROUPS.items() for name in group
}


def extract_regex_features(cleaned_text: pd.Series) -> pd.DataFrame:
    text = cleaned_text.fillna("").astype(str)
    return pd.DataFrame(
        {name: text.str.contains(pattern, na=False).astype(int) for name, pattern in PATTERNS.items()}
    )


def frequency_table(features: pd.DataFrame, minimum_count: int = 10) -> pd.DataFrame:
    rows = []
    total = len(features)
    for column in features.columns:
        count = int(features[column].sum())
        if count < 3:
            frequency_class = "extremely rare"
        elif count < minimum_count:
            frequency_class = "rare"
        elif count < 100:
            frequency_class = "moderate"
        else:
            frequency_class = "common"
        rows.append(
            {
                "feature": column,
                "group": FEATURE_TO_GROUP[column],
                "count": count,
                "percentage": count / total * 100.0,
                "frequency_class": frequency_class,
                "modeled": count >= minimum_count,
            }
        )
    return pd.DataFrame(rows)
