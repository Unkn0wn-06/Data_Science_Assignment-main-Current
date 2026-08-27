"""Keep exact-row removal distinct from repeated-advertisement handling."""

import pandas as pd


def remove_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the first occurrence of rows identical across every raw field."""
    return df.drop_duplicates(keep="first").copy()


def remove_property_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one version of each repeated ``Ad List`` advertisement identifier."""
    return df.drop_duplicates(subset=["Ad List"], keep="first").copy()

