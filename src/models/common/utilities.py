"""Small shared helpers for safe input and target-normalized regressors."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

from src.models.common.features import (
    FEATURES,
    PRODUCTION_NUMERICAL_FEATURES,
    TARGET_COLUMN,
)


def sanitize_model_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return required model columns with finite numerics and a valid price target."""
    required = {TARGET_COLUMN, *FEATURES}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    cleaned = data[[TARGET_COLUMN, *FEATURES]].copy()
    for column in [TARGET_COLUMN, *PRODUCTION_NUMERICAL_FEATURES]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        cleaned[column] = cleaned[column].replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    if cleaned.empty:
        raise ValueError("Dataset has no rows with a finite numeric price target.")
    return cleaned


class PricePerSquareFootRegressor(RegressorMixin, BaseEstimator):
    """Learn price per square foot while exposing total-price predictions."""

    def __init__(self, regressor=None, size_column="property_size_sqft"):
        self.regressor = regressor
        self.size_column = size_column

    def fit(self, X, y):
        """Clone and fit the inner pipeline against the normalized target."""
        self.regressor_ = clone(self.regressor)
        size = X[self.size_column].to_numpy(dtype=float)
        normalized_target = np.asarray(y, dtype=float) / size
        self.regressor_.fit(X, normalized_target)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        """Restore normalized predictions to non-negative total Ringgit prices."""
        normalized_prediction = self.regressor_.predict(X)
        size = X[self.size_column].to_numpy(dtype=float)
        return np.clip(normalized_prediction * size, 0.0, None)


class WinsorizedPricePerSquareFootRegressor(RegressorMixin, BaseEstimator):
    """Cap training-fold PPSF while preserving original validation targets."""

    def __init__(
        self,
        regressor=None,
        size_column="property_size_sqft",
        lower_quantile=None,
        upper_quantile=None,
    ):
        self.regressor = regressor
        self.size_column = size_column
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y):
        """Learn quantile limits and cap PPSF using this training fold only."""
        self._validate_quantiles()
        size = X[self.size_column].to_numpy(dtype=float)
        training_ppsf = np.asarray(y, dtype=float) / size
        self.lower_bound_ = (
            None
            if self.lower_quantile is None
            else float(np.quantile(training_ppsf, self.lower_quantile))
        )
        self.upper_bound_ = (
            None
            if self.upper_quantile is None
            else float(np.quantile(training_ppsf, self.upper_quantile))
        )
        self.lower_clipped_count_ = (
            0
            if self.lower_bound_ is None
            else int(np.sum(training_ppsf < self.lower_bound_))
        )
        self.upper_clipped_count_ = (
            0
            if self.upper_bound_ is None
            else int(np.sum(training_ppsf > self.upper_bound_))
        )
        clipped_ppsf = np.clip(
            training_ppsf, self.lower_bound_, self.upper_bound_
        )
        self.training_rows_ = len(training_ppsf)
        self.rows_after_clipping_ = len(clipped_ppsf)
        self.regressor_ = clone(self.regressor)
        self.regressor_.fit(X, clipped_ppsf)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        """Restore predicted PPSF using untouched validation property sizes."""
        predicted_ppsf = self.regressor_.predict(X)
        size = X[self.size_column].to_numpy(dtype=float)
        return np.clip(predicted_ppsf * size, 0.0, None)

    def _validate_quantiles(self) -> None:
        for name, value in [
            ("lower_quantile", self.lower_quantile),
            ("upper_quantile", self.upper_quantile),
        ]:
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one.")
        if (
            self.lower_quantile is not None
            and self.upper_quantile is not None
            and self.lower_quantile >= self.upper_quantile
        ):
            raise ValueError("lower_quantile must be smaller than upper_quantile.")


def select_balanced_candidate(cv_results: dict) -> int:
    """Select the candidate with the smallest combined RMSE and MAE ranks."""
    combined_rank = cv_results["rank_test_rmse"] + cv_results["rank_test_mae"]
    return int(np.argmin(combined_rank))
