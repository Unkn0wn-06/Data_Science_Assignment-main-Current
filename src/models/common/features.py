"""Canonical enhanced City features plus explicit legacy-production features."""


# The promoted enhanced City numerical schema is shared by every new fair model run.
NUMERICAL_FEATURES = [
    "property_size_sqft",
    "bedroom",
    "bathroom",
    "parking_lot",
    "facilities_count",
    "has_school",
    "has_mall",
    "has_hospital",
    "has_railway",
    "has_bus_stop",
    "has_park",
    "has_highway",
    "completion_year",
    "property_age",
    "number_of_floors",
    "total_units",
    "description_length",
    "has_swimming_pool",
    "has_security",
    "has_lift",
    "has_gym",
    "has_playground",
    "is_furnished",
    "is_renovated",
]

# Use the City-only location representation that produced the strongest experiment.
CATEGORICAL_FEATURES = [
    "property_type",
    "tenure_type",
    "land_title",
    "floor_range",
    "state",
    "building_name",
    "developer",
    "city",
]

# This is the only feature list allowed in the new enhanced comparison/experiment.
MODEL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "price"
PRICE_PER_SQUARE_FOOT_TARGET = "price_per_square_foot"
SIZE_COLUMN = "property_size_sqft"

# Preserve the currently deployed 21-feature dashboard and historical assignment
# comparison until the user explicitly authorizes a production switch.
PRODUCTION_NUMERICAL_FEATURES = [
    "property_size_sqft",
    "bedroom",
    "bathroom",
    "parking_lot",
    "facilities_count",
    "has_school",
    "has_mall",
    "has_hospital",
    "has_railway",
    "has_bus_stop",
    "has_park",
    "has_highway",
    "completion_year",
    "number_of_floors",
    "total_units",
]
PRODUCTION_CATEGORICAL_FEATURES = [
    "property_type",
    "tenure_type",
    "land_title",
    "floor_range",
    "state",
    "city",
]
PRODUCTION_FEATURES = PRODUCTION_NUMERICAL_FEATURES + PRODUCTION_CATEGORICAL_FEATURES

# Keep the old public name for production consumers without weakening the new
# canonical MODEL_FEATURES contract used by enhanced comparisons.
FEATURES = PRODUCTION_FEATURES
