"""Focused invariants for leakage-safe target and spatial feature builders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import unittest
from sklearn.model_selection import KFold

from experiments.spatial_target_encoding.spatial_features import (
    SpatialGeometryFeatures,
    SpatialPPSFNeighborEncoder,
    haversine_km,
)
from experiments.spatial_target_encoding.target_encoding import MEstimateTargetEncoder


class LeakageInvariantTests(unittest.TestCase):
    def test_oof_target_encoding_does_not_use_rows_own_target(self):
        frame = pd.DataFrame(
            {"developer": ["a", "a", "b", "b", "c", "c"]},
            index=[10, 11, 12, 13, 14, 15],
        )
        target = np.array([100.0, 110.0, 200.0, 210.0, 300.0, 310.0])
        folds = KFold(n_splits=3, shuffle=False)
        original = MEstimateTargetEncoder(("developer",), m=10).fit_transform_oof(
            frame, target, folds
        )
        mutated_target = target.copy()
        mutated_target[0] = 1_000_000.0
        mutated = MEstimateTargetEncoder(("developer",), m=10).fit_transform_oof(
            frame, mutated_target, folds
        )
        self.assertAlmostEqual(
            original.loc[10, "developer_te"], mutated.loc[10, "developer_te"]
        )
        self.assertFalse(np.allclose(original.to_numpy(), mutated.to_numpy()))

    def test_target_encoder_handles_missing_and_unseen_without_target(self):
        training = pd.DataFrame({"city": ["KL", None, "KL", "PJ"]})
        target = np.array([100.0, 200.0, 300.0, 400.0])
        encoder = MEstimateTargetEncoder(("city",), m=5).fit(training, target)
        transformed = encoder.transform(pd.DataFrame({"city": ["unseen", None]}))
        self.assertAlmostEqual(transformed.loc[0, "city_te"], target.mean())
        self.assertTrue(np.isfinite(transformed.loc[1, "city_te"]))
        self.assertEqual(transformed.shape, (2, 1))

    def test_haversine_zero_and_known_one_degree_equator_distance(self):
        self.assertAlmostEqual(
            float(haversine_km(3.139, 101.6869, 3.139, 101.6869)), 0.0
        )
        self.assertAlmostEqual(
            float(haversine_km(0.0, 0.0, 0.0, 1.0)), 111.195, delta=0.01
        )

    def test_geometry_training_transform_excludes_only_exact_row(self):
        frame = pd.DataFrame(
            {
                "latitude": [3.0, 3.0, 3.01],
                "longitude": [101.0, 101.0, 101.01],
            }
        )
        features = SpatialGeometryFeatures().fit_transform_training(frame)
        self.assertAlmostEqual(features.loc[0, "nearest_property_distance_km"], 0.0)
        self.assertAlmostEqual(features.loc[1, "nearest_property_distance_km"], 0.0)
        self.assertGreater(features.loc[2, "nearest_property_distance_km"], 0.0)

    def test_spatial_ppsf_training_transform_cannot_use_own_price(self):
        frame = pd.DataFrame(
            {
                "latitude": [3.00, 3.01, 3.02],
                "longitude": [101.00, 101.01, 101.02],
                "property_size_sqft": [100.0, 100.0, 100.0],
            }
        )
        price = np.array([10_000.0, 20_000.0, 30_000.0])
        encoded = (
            SpatialPPSFNeighborEncoder()
            .fit(frame, price)
            .transform_training_excluding_self(frame)
        )
        self.assertAlmostEqual(encoded.loc[0, "knn_5_mean_ppsf"], 250.0)
        self.assertAlmostEqual(encoded.loc[1, "knn_5_mean_ppsf"], 200.0)
        self.assertAlmostEqual(encoded.loc[2, "knn_5_mean_ppsf"], 150.0)

    def test_spatial_ppsf_invalid_coordinate_uses_training_median_only(self):
        training = pd.DataFrame(
            {
                "latitude": [3.0, 3.1, 3.2],
                "longitude": [101.0, 101.1, 101.2],
                "property_size_sqft": [100.0, 100.0, 100.0],
            }
        )
        price = np.array([10_000.0, 20_000.0, 50_000.0])
        query = pd.DataFrame(
            {"latitude": [np.nan], "longitude": [101.0], "property_size_sqft": [100.0]}
        )
        encoder = SpatialPPSFNeighborEncoder().fit(training, price)
        transformed = encoder.transform(query)
        self.assertEqual(encoder.last_fallback_count_, 1)
        self.assertTrue(np.allclose(transformed.to_numpy(), 200.0))

    def test_spatial_ppsf_inner_oof_row_does_not_use_own_target(self):
        frame = pd.DataFrame(
            {
                "latitude": [3.00, 3.01, 3.02, 3.03, 3.04, 3.05],
                "longitude": [101.00, 101.01, 101.02, 101.03, 101.04, 101.05],
                "property_size_sqft": [100.0] * 6,
            },
            index=[10, 11, 12, 13, 14, 15],
        )
        price = np.array([10_000.0, 20_000.0, 30_000.0, 40_000.0, 50_000.0, 60_000.0])
        folds = KFold(n_splits=3, shuffle=False)
        original = SpatialPPSFNeighborEncoder().fit_transform_oof(frame, price, folds)
        mutated_price = price.copy()
        mutated_price[0] = 9_000_000.0
        mutated = SpatialPPSFNeighborEncoder().fit_transform_oof(
            frame, mutated_price, folds
        )
        np.testing.assert_allclose(original.loc[10], mutated.loc[10])
        self.assertFalse(np.allclose(original.to_numpy(), mutated.to_numpy()))

    def test_spatial_outer_validation_transform_has_no_target_input(self):
        training = pd.DataFrame(
            {
                "latitude": [3.0, 3.1],
                "longitude": [101.0, 101.1],
                "property_size_sqft": [100.0, 100.0],
            }
        )
        validation = pd.DataFrame(
            {"latitude": [3.05], "longitude": [101.05], "property_size_sqft": [100.0]}
        )
        encoder = SpatialPPSFNeighborEncoder().fit(training, [10_000.0, 20_000.0])
        first = encoder.transform(validation)
        second = encoder.transform(validation.copy())
        np.testing.assert_allclose(first, second)


if __name__ == "__main__":
    unittest.main()
