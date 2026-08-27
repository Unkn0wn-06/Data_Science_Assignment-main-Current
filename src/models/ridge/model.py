"""Construct the assignment Ridge Regression estimator."""

from sklearn.linear_model import Ridge


def build_model(parameters: dict) -> Ridge:
    """Build Ridge with the supplied current/default estimator parameters."""
    return Ridge(**parameters)

