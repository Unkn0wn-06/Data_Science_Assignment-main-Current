"""Construct production and best-experiment Random Forest estimators."""

from sklearn.ensemble import RandomForestRegressor


def build_model(parameters: dict) -> RandomForestRegressor:
    """Build the assignment forest with reproducible full-core fitting."""
    return RandomForestRegressor(random_state=42, n_jobs=-1, **parameters)


def build_best_city_model() -> RandomForestRegressor:
    """Recreate the selected city + price/sq.ft. experiment forest exactly."""
    return RandomForestRegressor(
        n_estimators=700,
        min_samples_split=6,
        min_samples_leaf=3,
        max_features=0.7,
        max_depth=24,
        criterion="squared_error",
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )

