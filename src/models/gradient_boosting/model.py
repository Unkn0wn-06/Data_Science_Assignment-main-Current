"""Construct the assignment Gradient Boosting estimator."""

from sklearn.ensemble import GradientBoostingRegressor


def build_model(parameters: dict) -> GradientBoostingRegressor:
    """Build Gradient Boosting with its fixed reproducibility seed."""
    return GradientBoostingRegressor(random_state=42, **parameters)

