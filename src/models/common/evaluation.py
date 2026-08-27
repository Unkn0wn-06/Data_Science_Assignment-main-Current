"""Central regression metrics for production and historical experiments."""

import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.common.utilities import PricePerSquareFootRegressor


def regression_metrics(actual, prediction, include_distribution: bool = False) -> dict:
    """Calculate unchanged RMSE, MAE, R-squared, and optional diagnostics."""
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(prediction, dtype=float)
    metrics = {
        "RMSE": mean_squared_error(actual_values, predicted_values) ** 0.5,
        "MAE": mean_absolute_error(actual_values, predicted_values),
        "R2": r2_score(actual_values, predicted_values),
    }
    if include_distribution:
        absolute_error = np.abs(actual_values - predicted_values)
        metrics.update(
            {
                "median_absolute_error": float(np.median(absolute_error)),
                "within_RM_1000_rate": float(np.mean(absolute_error < 1000)),
            }
        )
        # Match the historical JSON ordering used by the experiment files.
        metrics = {
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "median_absolute_error": metrics["median_absolute_error"],
            "within_RM_1000_rate": metrics["within_RM_1000_rate"],
            "R2": metrics["R2"],
        }
    return metrics


def evaluate(pipeline, x_test, y_test):
    """Return the production dashboard metrics and row-level predictions."""
    prediction = pipeline.predict(x_test)
    metrics = regression_metrics(y_test, prediction)
    fitted_pipeline = (
        pipeline.regressor_
        if isinstance(pipeline, PricePerSquareFootRegressor)
        else pipeline
    )
    transformed_features = (
        fitted_pipeline.named_steps["preprocessor"].get_feature_names_out().size
    )
    observations = len(y_test)
    adjusted = (
        np.nan
        if observations <= transformed_features + 1
        else 1
        - (
            (1 - metrics["R2"])
            * (observations - 1)
            / (observations - transformed_features - 1)
        )
    )
    return {
        "R2": metrics["R2"],
        "Adjusted R2": adjusted,
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
    }, prediction


def out_of_fold_metrics(estimator, X, y, folds) -> dict:
    """Calculate historical total-price metrics on fixed outer folds."""
    prediction = np.empty(len(y), dtype=float)
    for train_index, test_index in folds:
        fitted = clone(estimator).fit(X.iloc[train_index], y.iloc[train_index])
        prediction[test_index] = fitted.predict(X.iloc[test_index])
    return regression_metrics(y, prediction, include_distribution=True)

