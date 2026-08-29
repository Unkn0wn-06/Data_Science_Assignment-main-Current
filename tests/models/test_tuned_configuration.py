"""Validate the authoritative tuned model configuration and rebuilt evidence."""

from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.cleaning.pipeline import PROJECT_ROOT
from src.models.final.final_evaluation import FINAL_MODELS
from src.models.final.model_builders import (
    FINAL_TUNED_PARAMS_PATH,
    build_position_lightgbm,
    build_standard_ppsf_estimator,
    final_tuned_params_sha256,
    get_final_model_parameters,
    load_final_tuned_config,
)


TUNING_DIR = PROJECT_ROOT / "results" / "tuning"
FINAL_RESULTS_DIR = PROJECT_ROOT / "results" / "final_models"
TRIMMING_DIR = PROJECT_ROOT / "results" / "outlier_trimming"


class TunedConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_final_tuned_config()
        cls.tuning_metadata = json.loads(
            (TUNING_DIR / "metadata.json").read_text(encoding="utf-8")
        )
        cls.summary = pd.read_csv(TUNING_DIR / "tuning_summary.csv").set_index("Model")
        cls.selected = pd.read_csv(TUNING_DIR / "tuned_cv_results.csv").set_index("Model")
        cls.official = pd.read_csv(FINAL_RESULTS_DIR / "model_comparison.csv").set_index("Model")

    def test_tuning_is_complete_reproducible_and_covers_only_final_four(self):
        self.assertTrue(FINAL_TUNED_PARAMS_PATH.is_file())
        self.assertEqual("complete", self.tuning_metadata["status"])
        self.assertEqual(42, self.tuning_metadata["random_state"])
        self.assertEqual(5, self.tuning_metadata["folds"])
        self.assertEqual(set(FINAL_MODELS), set(self.config["models"]))
        self.assertNotIn("KNN", self.config["models"])
        self.assertNotIn("Building TE", self.config["models"])
        self.assertEqual(
            {
                "Ridge Regression": 31,
                "Random Forest": 50,
                "Gradient Boosting": 60,
                "LightGBM + Position Features": 80,
            },
            self.tuning_metadata["candidate_counts"],
        )

    def test_builders_use_selected_parameters_and_correct_scaling(self):
        for model_name in FINAL_MODELS[:3]:
            estimator = build_standard_ppsf_estimator(model_name)
            pipeline = estimator.regressor
            numeric = pipeline.named_steps["preprocessor"].transformers[0][1]
            has_scaler = any(
                isinstance(step, StandardScaler) for _, step in numeric.steps
            )
            self.assertEqual(model_name == "Ridge Regression", has_scaler)
            actual = pipeline.named_steps["model"].get_params()
            for parameter, expected in get_final_model_parameters(model_name).items():
                self.assertEqual(expected, actual[parameter], (model_name, parameter))

        lightgbm, _, _ = build_position_lightgbm()
        numeric = lightgbm.named_steps["preprocessor"].transformers[0][1]
        self.assertFalse(any(isinstance(step, StandardScaler) for _, step in numeric.steps))
        actual = lightgbm.named_steps["model"].get_params()
        for parameter, expected in get_final_model_parameters(
            "LightGBM + Position Features"
        ).items():
            self.assertEqual(expected, actual[parameter], parameter)

    def test_every_model_improved_and_official_metrics_match_selected_tuning(self):
        self.assertTrue((self.summary["RMSE_Improvement_RM"] > 0).all())
        self.assertTrue((self.summary["MAE_Improvement_RM"] > 0).all())
        self.assertTrue((self.summary["R2_Change"] > 0).all())
        self.assertTrue((self.summary["Adjusted_R2_Change"] > 0).all())
        for tuned_column, official_column, tolerance in (
            ("CV_RMSE_RM", "RMSE_RM", 1e-6),
            ("CV_MAE_RM", "MAE_RM", 1e-6),
            ("CV_R2", "R2", 1e-12),
            ("CV_Adjusted_R2", "Adjusted_R2", 1e-12),
        ):
            np.testing.assert_allclose(
                self.selected.loc[list(FINAL_MODELS), tuned_column],
                self.official.loc[list(FINAL_MODELS), official_column],
                rtol=1e-10,
                atol=tolerance,
            )

    def test_all_rebuilt_artifacts_record_same_tuned_configuration(self):
        expected_hash = final_tuned_params_sha256()
        final_metadata = json.loads(
            (FINAL_RESULTS_DIR / "metadata.json").read_text(encoding="utf-8")
        )
        self.assertEqual(expected_hash, final_metadata["tuned_configuration_sha256"])
        trimming_metadata_path = TRIMMING_DIR / "all_models_trimmed_market_metadata.json"
        self.assertTrue(trimming_metadata_path.is_file())
        trimming_metadata = json.loads(trimming_metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(expected_hash, trimming_metadata["tuned_config_sha256"])
        self.assertEqual(
            ["0%", "0.5%", "1%", "2.5%", "5%", "10%"],
            trimming_metadata["trim_levels"],
        )


if __name__ == "__main__":
    unittest.main()
