"""Validate the saved all-model restricted-market trimming artifacts."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.cleaning.pipeline import PROJECT_ROOT
from src.models.final.final_evaluation import FINAL_MODELS, PREDICTOR_COUNTS
from src.models.final.position_regex_lightgbm import FINAL_MODEL_NAME


RESULTS_DIR = PROJECT_ROOT / "results" / "outlier_trimming"
SUMMARY_PATH = RESULTS_DIR / "all_models_trimmed_market_summary.csv"
FOLD_METRICS_PATH = RESULTS_DIR / "all_models_trimmed_market_fold_metrics.csv"
OOF_PATH = RESULTS_DIR / "all_models_trimmed_market_oof.csv"
OFFICIAL_PATH = PROJECT_ROOT / "results" / "final_models" / "model_comparison.csv"
LIGHTGBM_REFERENCE_PATH = RESULTS_DIR / "trimmed_population_comparison.csv"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "enhanced_city_dataset.csv"
FOLD_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "repeat_group_sensitivity"
    / "scenario_b_fold_assignments.csv"
)
EXPECTED_LEVELS = ["0%", "0.5%", "1%", "2.5%", "5%", "10%"]
METRIC_COLUMNS = ["RMSE_RM", "MAE_RM", "R2", "Adjusted_R2"]


class AllModelsTrimmedMarketArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = pd.read_csv(SUMMARY_PATH)
        cls.fold_metrics = pd.read_csv(FOLD_METRICS_PATH)
        cls.oof = pd.read_csv(OOF_PATH)
        cls.official = pd.read_csv(OFFICIAL_PATH)
        cls.lightgbm_reference = pd.read_csv(LIGHTGBM_REFERENCE_PATH)
        cls.data = pd.read_csv(DATA_PATH, usecols=["listing_id", "price"])
        cls.assignments = pd.read_csv(FOLD_PATH)

    def test_summary_shape_models_levels_and_finite_metrics(self):
        self.assertTrue(SUMMARY_PATH.is_file())
        self.assertEqual(24, len(self.summary))
        self.assertEqual(set(FINAL_MODELS), set(self.summary["Model"]))
        self.assertFalse(self.summary.duplicated(["Model", "Trim_Level"]).any())
        self.assertTrue(
            (self.summary.groupby("Model")["Trim_Level"].nunique() == 6).all()
        )
        for model_name in FINAL_MODELS:
            self.assertEqual(
                EXPECTED_LEVELS,
                self.summary.loc[
                    self.summary["Model"].eq(model_name), "Trim_Level"
                ].tolist(),
            )
        self.assertTrue(
            np.isfinite(self.summary[METRIC_COLUMNS].to_numpy(float)).all()
        )

    def test_all_models_share_retained_counts_and_listing_ids(self):
        self.assertTrue(
            (
                self.summary.groupby("Trim_Level", sort=False)["Retained_Rows"].nunique()
                == 1
            ).all()
        )
        for trim_level, rows in self.oof.groupby("Trim_Level", sort=False):
            ids_by_model = {
                model_name: set(
                    rows.loc[rows["Model"].eq(model_name), "listing_id"].astype(int)
                )
                for model_name in FINAL_MODELS
            }
            first = ids_by_model[FINAL_MODELS[0]]
            for model_name in FINAL_MODELS[1:]:
                self.assertEqual(first, ids_by_model[model_name], trim_level)

    def test_zero_percent_reproduces_official_four_model_metrics(self):
        zero = self.summary[self.summary["Removal_Percent"].eq(0.0)].set_index("Model")
        official = self.official.set_index("Model")
        np.testing.assert_allclose(
            zero.loc[list(FINAL_MODELS), METRIC_COLUMNS],
            official.loc[list(FINAL_MODELS), METRIC_COLUMNS],
            rtol=1e-10,
            atol=1e-6,
        )

    def test_lightgbm_reproduces_existing_restricted_market_results(self):
        current = self.summary[self.summary["Model"].eq(FINAL_MODEL_NAME)].merge(
            self.lightgbm_reference,
            on="Removal_Percent",
            validate="one_to_one",
        )
        self.assertEqual(6, len(current))
        np.testing.assert_array_equal(current["Retained_Rows"], current["Retained_OOF_Rows"])
        for new_column, reference_column, tolerance in (
            ("RMSE_RM", "Matched_Retrained_RMSE_RM", 1e-6),
            ("MAE_RM", "Matched_Retrained_MAE_RM", 1e-6),
            ("R2", "Retrained_R2", 1e-12),
            ("Adjusted_R2", "Retrained_Adjusted_R2", 1e-12),
        ):
            np.testing.assert_allclose(
                current[new_column],
                current[reference_column],
                rtol=1e-10,
                atol=tolerance,
            )

    def test_adjusted_r2_uses_official_model_specific_predictor_counts(self):
        predictors = self.summary["Model"].map(PREDICTOR_COUNTS)
        expected = 1.0 - (1.0 - self.summary["R2"]) * (
            self.summary["Retained_Rows"] - 1
        ) / (self.summary["Retained_Rows"] - predictors - 1)
        np.testing.assert_allclose(
            self.summary["Adjusted_R2"], expected, rtol=1e-12, atol=1e-12
        )
        self.assertEqual(
            {
                "Ridge Regression": 32,
                "Random Forest": 32,
                "Gradient Boosting": 32,
                FINAL_MODEL_NAME: 47,
            },
            PREDICTOR_COUNTS,
        )

    def test_oof_coverage_folds_and_group_safety(self):
        self.assertTrue(OOF_PATH.is_file())
        self.assertTrue(FOLD_METRICS_PATH.is_file())
        self.assertEqual(120, len(self.fold_metrics))
        self.assertFalse(
            self.fold_metrics.duplicated(["Model", "Trim_Level", "Fold"]).any()
        )
        self.assertTrue(
            (
                self.fold_metrics.groupby(["Model", "Trim_Level"])["Fold"].nunique()
                == 5
            ).all()
        )
        self.assertFalse(self.oof.duplicated(["Model", "Trim_Level", "listing_id"]).any())
        expected_rows = self.summary.set_index(["Model", "Trim_Level"])["Retained_Rows"]
        actual_rows = self.oof.groupby(["Model", "Trim_Level"]).size()
        pd.testing.assert_series_equal(
            actual_rows.sort_index(), expected_rows.sort_index(), check_names=False
        )

        fold_lookup = self.assignments.set_index("listing_id")["fold"]
        expected_folds = self.oof["listing_id"].map(fold_lookup)
        np.testing.assert_array_equal(
            self.oof["scenario_b_fold"].astype(int), expected_folds.astype(int)
        )
        repeated = self.assignments[self.assignments["is_grouped_repeat"]]
        self.assertFalse(
            repeated.groupby("repeat_group_id")["fold"].nunique().gt(1).any()
        )

        distribution = pd.read_csv(RESULTS_DIR / "distribution_shift.csv")
        canonical = self.data.set_index("listing_id")
        for _, reference in distribution.iterrows():
            label = f"{float(reference['Removal_Percent']):g}%"
            expected_ids = (
                set(canonical.index.astype(int))
                if float(reference["Removal_Percent"]) == 0.0
                else set(
                    canonical.loc[
                        canonical["price"].le(reference["Full_Population_Cutoff_RM"])
                    ].index.astype(int)
                )
            )
            for model_name in FINAL_MODELS:
                observed_ids = set(
                    self.oof.loc[
                        self.oof["Trim_Level"].eq(label)
                        & self.oof["Model"].eq(model_name),
                        "listing_id",
                    ].astype(int)
                )
                self.assertEqual(expected_ids, observed_ids)


if __name__ == "__main__":
    unittest.main()
