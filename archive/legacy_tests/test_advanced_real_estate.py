"""Focused leakage and target-transform tests for the advanced experiment."""

import unittest

import numpy as np
import pandas as pd
from scipy.stats import boxcox
from sklearn.dummy import DummyRegressor

from experiments.advanced_real_estate_models.feature_engineering import (
    MICRO_FEATURES,
    MicroMarketPPSFEncoder,
    oof_micro_market_features,
)
from experiments.advanced_real_estate_models.model_builders import (
    TargetStrategyRegressor,
)


def synthetic_properties(rows: int = 15) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "property_size_sqft": np.linspace(600, 1300, rows),
            "state": [f"state_{index}" for index in range(rows)],
            "city": [f"city_{index}" for index in range(rows)],
            "property_type": ["Condo"] * rows,
            "building_name": [f"building_{index}" for index in range(rows)],
            "developer": [f"developer_{index}" for index in range(rows)],
        }
    )


class AdvancedExperimentTests(unittest.TestCase):
    def test_micro_market_training_feature_excludes_own_target(self) -> None:
        X = synthetic_properties()
        y = np.linspace(200_000, 900_000, len(X))
        first = oof_micro_market_features(X, y)
        changed = y.copy()
        changed[0] *= 50
        second = oof_micro_market_features(X, changed)
        np.testing.assert_allclose(first.iloc[0], second.iloc[0])
        self.assertEqual(tuple(first.columns), MICRO_FEATURES)

    def test_unseen_micro_market_categories_use_training_fallbacks(self) -> None:
        X = synthetic_properties()
        y = np.linspace(200_000, 900_000, len(X))
        encoder = MicroMarketPPSFEncoder().fit(X.iloc[:-1], y[:-1])
        encoded = encoder.transform(X.iloc[[-1]])
        self.assertFalse(encoded.isna().any().any())
        self.assertEqual(float(encoded["micro_building_name_count"].iloc[0]), 0.0)

    def test_boxcox_lambda_is_fitted_from_passed_training_target(self) -> None:
        X = synthetic_properties(8)
        y = np.array([100_000, 130_000, 170_000, 220_000, 300_000, 410_000, 560_000, 800_000])
        estimator = TargetStrategyRegressor(
            DummyRegressor(strategy="mean"), strategy="boxcox_price"
        ).fit(X, y)
        _, expected_lambda = boxcox(y)
        self.assertAlmostEqual(estimator.boxcox_lambda_, expected_lambda, places=12)
        self.assertTrue(np.all(np.isfinite(estimator.predict(X))))


if __name__ == "__main__":
    unittest.main()
