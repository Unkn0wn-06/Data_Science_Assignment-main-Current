"""Construct the assignment K-Nearest Neighbors estimator."""

from sklearn.neighbors import KNeighborsRegressor


def build_model(parameters: dict) -> KNeighborsRegressor:
    """Build KNN with the supplied current/default estimator parameters."""
    return KNeighborsRegressor(**parameters)

