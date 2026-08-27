"""Validate the final Scenario B artifacts and deployment model interface."""

from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.cleaning.pipeline import PROJECT_ROOT
from src.models.common.features import MODEL_FEATURES
from src.models.final.final_evaluation import (
    FINAL_MODELS,
    FOLD_PATH,
    PREDICTION_COLUMNS,
    RESULTS_DIR,
)
from src.models.final.position_regex_lightgbm import (
    FINAL_MODEL_NAME,
    POSITION_FEATURES,
    extract_position_features,
    fit_final_model,
    predict_total_price,
    prepare_live_features,
)


class FinalModelArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(
            PROJECT_ROOT / "data" / "processed" / "enhanced_city_dataset.csv"
        ).reset_index(drop=True)
        cls.assignments = pd.read_csv(FOLD_PATH).sort_values("row_index")
        cls.comparison = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
        cls.oof = pd.read_csv(RESULTS_DIR / "oof_predictions.csv")
        cls.fold_metrics = pd.read_csv(RESULTS_DIR / "fold_metrics.csv")
        cls.metadata = json.loads(
            (RESULTS_DIR / "metadata.json").read_text(encoding="utf-8")
        )
        cls.model = fit_final_model()

    def test_final_model_set_and_selection(self):
        self.assertEqual(set(FINAL_MODELS), set(self.comparison["Model"]))
        self.assertNotIn("KNN", set(self.comparison["Model"]))
        self.assertNotIn("Building TE", set(self.comparison["Model"]))
        self.assertEqual(FINAL_MODEL_NAME, self.metadata["selected_final_model"])
        self.assertEqual(3_791, self.metadata["full_data_deployment_training_rows"])

    def test_oof_rows_and_scenario_b_folds_are_exact(self):
        self.assertEqual(3_791, len(self.oof))
        self.assertEqual(3_791, self.oof["listing_id"].nunique())
        np.testing.assert_array_equal(
            self.oof["listing_id"].to_numpy(int), self.data["listing_id"].to_numpy(int)
        )
        np.testing.assert_array_equal(
            self.oof["scenario_b_fold"].to_numpy(int),
            self.assignments["fold"].to_numpy(int),
        )
        for column in PREDICTION_COLUMNS.values():
            self.assertTrue(np.isfinite(self.oof[column]).all())
        repeated = self.assignments[self.assignments["is_grouped_repeat"]]
        self.assertEqual(
            0,
            repeated.groupby("repeat_group_id")["fold"].nunique().gt(1).sum(),
        )

    def test_saved_metrics_recompute_from_total_price_oof(self):
        actual = self.oof["actual_price"].to_numpy(float)
        stored = self.comparison.set_index("Model")
        for model_name, column in PREDICTION_COLUMNS.items():
            predicted = self.oof[column].to_numpy(float)
            row = stored.loc[model_name]
            self.assertAlmostEqual(
                np.sqrt(mean_squared_error(actual, predicted)), row["RMSE_RM"], places=8
            )
            self.assertAlmostEqual(
                mean_absolute_error(actual, predicted), row["MAE_RM"], places=8
            )
            self.assertAlmostEqual(r2_score(actual, predicted), row["R2"], places=10)

    def test_fold_metrics_cover_same_validation_rows(self):
        self.assertEqual(20, len(self.fold_metrics))
        for model_name, rows in self.fold_metrics.groupby("Model"):
            self.assertEqual(set(range(1, 6)), set(rows["Fold"]))
            self.assertEqual(3_791, int(rows["Validation_Rows"].sum()))

    def test_position_feature_schema_and_regexes(self):
        self.assertEqual(47, len(self.model.feature_names_))
        self.assertTrue(set(POSITION_FEATURES).issubset(self.model.feature_names_))
        extracted = extract_position_features(
            ["Spacious high floor top floor home with a large balcony"]
        ).iloc[0]
        self.assertEqual(1, extracted["is_high_floor_text"])
        self.assertEqual(1, extracted["is_top_floor_text"])
        self.assertEqual(1, extracted["has_balcony"])
        self.assertEqual(1, extracted["has_large_balcony"])
        self.assertEqual(0, extracted["is_low_floor_text"])

    def test_live_helper_handles_unseen_identity_values(self):
        values = self.data.iloc[0][MODEL_FEATURES].to_dict()
        values["building_name"] = "Never Seen Residence 999"
        values["developer"] = "Never Seen Developer 999"
        values["city"] = "Never Seen Locality 999"
        result = predict_total_price(
            self.model,
            values,
            "High floor unit with a spacious balcony",
            float(self.data["description_length"].median()),
        )
        self.assertTrue(np.isfinite(result["total_price_RM"]))
        self.assertGreater(result["total_price_RM"], 0)
        self.assertGreater(result["ppsf_RM"], 0)
        self.assertTrue(result["detected_position_features"]["High Floor"])
        self.assertTrue(result["detected_position_features"]["Large Balcony"])

    def test_blank_description_uses_training_fallback(self):
        values = self.data.iloc[1][MODEL_FEATURES].to_dict()
        fallback = float(self.data["description_length"].median())
        structured, _, detected = prepare_live_features(values, "", fallback)
        self.assertEqual(fallback, float(structured.iloc[0]["description_length"]))
        self.assertFalse(any(detected.values()))


if __name__ == "__main__":
    unittest.main()
