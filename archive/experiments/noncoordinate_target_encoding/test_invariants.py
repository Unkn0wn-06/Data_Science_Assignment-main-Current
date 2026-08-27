"""Dependency-free unit tests for cleaning and leakage-sensitive encoders."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from experiments.noncoordinate_target_encoding.cleaning import (
    clean_price,
    clean_property_size,
    count_facilities,
    description_features,
    presence_flag,
)
from experiments.noncoordinate_target_encoding.feature_engineering import (
    CityContextEncoder,
)
from experiments.noncoordinate_target_encoding.target_encoding import (
    CategoryCountEncoder,
    MEstimateTargetEncoder,
)


class CleaningTests(unittest.TestCase):
    def test_price_cleaning(self):
        result = clean_price(pd.Series(["RM 340 000", "RM340,000", "RM 1 250 000", "-"]))
        np.testing.assert_allclose(result.iloc[:3], [340000.0, 340000.0, 1250000.0])
        self.assertTrue(pd.isna(result.iloc[3]))

    def test_property_size_cleaning(self):
        result = clean_property_size(pd.Series(["1000 sq.ft.", "1,200 sq.ft.", "bad", "0 sqft"]))
        np.testing.assert_allclose(result.iloc[:2], [1000.0, 1200.0])
        self.assertTrue(pd.isna(result.iloc[2]))
        self.assertTrue(pd.isna(result.iloc[3]))

    def test_description_regexes(self):
        result = description_features(
            pd.Series(["Fully furnished with fridge", "Renovated kitchen cabinets", None])
        )
        self.assertEqual(result["is_furnished"].tolist(), [1, 0, 0])
        self.assertEqual(result["is_renovated"].tolist(), [0, 1, 0])
        self.assertEqual(result["description_length"].iloc[2], 0.0)

    def test_facility_count_and_presence(self):
        self.assertEqual(count_facilities("Pool, Gym, pool, -"), 2)
        self.assertEqual(count_facilities(None), 0)
        self.assertEqual(presence_flag(pd.Series(["School", "-", None])).tolist(), [1, 0, 0])


class EncodingTests(unittest.TestCase):
    def test_smoothing_formula(self):
        X = pd.DataFrame({"city": ["A", "A", "B"]})
        y = np.array([100.0, 200.0, 300.0])
        encoded = MEstimateTargetEncoder(("city",), m=2.0).fit(X, y).transform(
            pd.DataFrame({"city": ["A"]})
        )
        self.assertAlmostEqual(encoded.loc[0, "city_te"], 175.0)

    def test_oof_encoding_does_not_use_own_target(self):
        X = pd.DataFrame(
            {"building_name": ["a", "a", "b", "b", "c", "c"]},
            index=[10, 11, 12, 13, 14, 15],
        )
        y = np.array([100.0, 110.0, 200.0, 210.0, 300.0, 310.0])
        folds = KFold(n_splits=3, shuffle=False)
        first = MEstimateTargetEncoder(("building_name",), m=5).fit_transform_oof(X, y, folds)
        changed = y.copy(); changed[0] = 1_000_000.0
        second = MEstimateTargetEncoder(("building_name",), m=5).fit_transform_oof(
            X, changed, folds
        )
        self.assertAlmostEqual(
            first.loc[10, "building_name_te"], second.loc[10, "building_name_te"]
        )
        self.assertFalse(np.allclose(first.to_numpy(), second.to_numpy()))

    def test_unseen_and_missing_fallback(self):
        X = pd.DataFrame({"developer": ["A", None, "B"]})
        y = np.array([100.0, 200.0, 300.0])
        encoder = MEstimateTargetEncoder(("developer",), m=5).fit(X, y)
        transformed = encoder.transform(pd.DataFrame({"developer": ["new", None]}))
        self.assertAlmostEqual(transformed.loc[0, "developer_te"], 200.0)
        self.assertTrue(np.isfinite(transformed.loc[1, "developer_te"]))

    def test_count_unseen_is_zero(self):
        encoder = CategoryCountEncoder(("city",)).fit(pd.DataFrame({"city": ["A", "A", "B"]}))
        transformed = encoder.transform(pd.DataFrame({"city": ["A", "new"]}))
        self.assertEqual(transformed.loc[0, "city_count"], 2.0)
        self.assertEqual(transformed.loc[1, "city_count"], 0.0)
        self.assertEqual(transformed.loc[1, "log_city_count"], 0.0)

    def test_city_aggregates_are_training_only(self):
        training = pd.DataFrame(
            {
                "city": ["A", "A", "B"],
                "property_size_sqft": [100.0, 300.0, 500.0],
                "bedroom": [1.0, 3.0, 5.0],
                "bathroom": [1.0, 2.0, 4.0],
            }
        )
        validation = pd.DataFrame(
            {
                "city": ["A", "unseen"],
                "property_size_sqft": [900.0, 400.0],
                "bedroom": [9.0, 4.0],
                "bathroom": [9.0, 3.0],
            }
        )
        transformed = CityContextEncoder().fit(training).transform(validation)
        self.assertEqual(transformed.loc[0, "city_median_sqft"], 200.0)
        self.assertEqual(transformed.loc[0, "city_property_count"], 2.0)
        self.assertEqual(transformed.loc[1, "city_median_sqft"], 300.0)
        self.assertEqual(transformed.loc[1, "city_property_count"], 0.0)


if __name__ == "__main__":
    unittest.main()
