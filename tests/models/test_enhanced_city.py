"""Validate the canonical enhanced dataset and final comparison artifact."""

import unittest

import pandas as pd

from prototype.enhanced_results import load_enhanced_comparison_table
from src.cleaning.enhanced_city import ENHANCED_CITY_DATA_PATH
from src.models.common.features import MODEL_FEATURES


class EnhancedCityArtifactTests(unittest.TestCase):
    """Check canonical schema and the final Scenario B comparison artifact."""

    def test_canonical_dataset_schema(self) -> None:
        data = pd.read_csv(ENHANCED_CITY_DATA_PATH)
        self.assertEqual(data.shape, (3791, 34))
        self.assertEqual(list(data.columns), ["listing_id", "price", *MODEL_FEATURES])
        self.assertNotIn("price_per_square_foot", data.columns)
        self.assertNotIn("detailed_address", data.columns)

    def test_streamlit_table_is_final_scenario_b_comparison(self) -> None:
        table = load_enhanced_comparison_table()
        self.assertEqual(len(table), 4)
        self.assertEqual(
            list(table.columns),
            ["Model", "R²", "Adjusted R²", "MAE (RM)", "RMSE (RM)"],
        )
        self.assertTrue(table["RMSE (RM)"].is_monotonic_increasing)
        self.assertNotIn("KNN", set(table["Model"]))
        self.assertIn("LightGBM + Position Features", set(table["Model"]))


if __name__ == "__main__":
    unittest.main()
