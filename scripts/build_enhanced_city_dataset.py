"""Rebuild and report the canonical unencoded enhanced City dataset."""

if __package__:
    from scripts import _bootstrap  # noqa: F401
else:
    import _bootstrap  # noqa: F401

from src.cleaning.enhanced_city import (
    ENHANCED_CITY_DATA_PATH,
    build_enhanced_city_dataset,
)


def main() -> None:
    """Write the dataset and print the requested quality profile."""
    data, profile = build_enhanced_city_dataset()
    print(f"Enhanced City dataset: {ENHANCED_CITY_DATA_PATH}")
    print(f"Shape: {data.shape}")
    print(f"Missing values: {profile['missing_values_total']}")
    print(f"Duplicate rows: {profile['duplicate_rows']}")
    print(f"Numerical columns: {profile['numerical_columns']}")
    print(f"Categorical columns: {profile['categorical_columns']}")
    print(f"Target distribution: {profile['target_distribution']}")
    print(f"Property-size distribution: {profile['property_size_distribution']}")
    print(f"City unique count: {profile['city_unique_count']}")


if __name__ == "__main__":
    main()

