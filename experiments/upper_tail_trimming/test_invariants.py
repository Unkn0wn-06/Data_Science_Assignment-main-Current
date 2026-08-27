"""Independent invariants for the isolated upper-tail trimming experiment."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.upper_tail_trimming.run_experiment import (
    BOOTSTRAP_SAMPLES,
    DATA_PATH,
    EXPECTED_ROWS,
    EXPERIMENT,
    FOLD_PATH,
    MODEL_PARAMETERS,
    MODELS,
    TRIM_LEVELS,
    sha256,
)


REQUIRED_FILES = (
    "results.json",
    "training_only_comparison.csv",
    "trimmed_population_comparison.csv",
    "fold_metrics.csv",
    "training_cutoffs.csv",
    "oof_predictions.csv",
    "segment_metrics.csv",
    "distribution_shift.csv",
    "bootstrap_results.csv",
    "run_experiment.py",
    "test_invariants.py",
)


class UpperTailTrimmingInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(DATA_PATH).reset_index(drop=True)
        cls.assignments = pd.read_csv(FOLD_PATH).sort_values("row_index").reset_index(drop=True)
        cls.results = json.loads((EXPERIMENT / "results.json").read_text(encoding="utf-8"))
        cls.training = pd.read_csv(EXPERIMENT / "training_only_comparison.csv")
        cls.trimmed = pd.read_csv(EXPERIMENT / "trimmed_population_comparison.csv")
        cls.folds = pd.read_csv(EXPERIMENT / "fold_metrics.csv")
        cls.cutoffs = pd.read_csv(EXPERIMENT / "training_cutoffs.csv")
        cls.oof = pd.read_csv(EXPERIMENT / "oof_predictions.csv")
        cls.segments = pd.read_csv(EXPERIMENT / "segment_metrics.csv")
        cls.distribution = pd.read_csv(EXPERIMENT / "distribution_shift.csv")
        cls.bootstrap = pd.read_csv(EXPERIMENT / "bootstrap_results.csv")

    def test_01_required_outputs_exist(self):
        self.assertEqual([], [name for name in REQUIRED_FILES if not (EXPERIMENT / name).is_file()])

    def test_02_canonical_dataset_is_complete_and_unchanged(self):
        self.assertEqual(EXPECTED_ROWS, len(self.data))
        self.assertEqual(EXPECTED_ROWS, self.data["listing_id"].nunique())
        self.assertEqual(self.results["canonical_dataset"]["sha256"], sha256(DATA_PATH))

    def test_03_scenario_b_assignments_are_reused_exactly(self):
        self.assertEqual("B", self.results["validation"]["scenario"])
        self.assertEqual(5, self.results["validation"]["folds"])
        self.assertEqual(
            self.results["validation"]["fold_assignment_sha256"], sha256(FOLD_PATH)
        )
        np.testing.assert_array_equal(
            self.assignments["listing_id"].astype(int), self.data["listing_id"].astype(int)
        )

    def test_04_training_only_variants_have_full_oof_coverage(self):
        rows = self.oof[self.oof["Experiment_Type"] == "training_only"]
        expected_variants = len(MODELS) * len(TRIM_LEVELS)
        self.assertEqual(EXPECTED_ROWS * expected_variants, len(rows))
        counts = rows.groupby(["Model", "Trim_Level"])["listing_id"].agg(["size", "nunique"])
        self.assertTrue((counts["size"] == EXPECTED_ROWS).all())
        self.assertTrue((counts["nunique"] == EXPECTED_ROWS).all())
        self.assertTrue(np.isfinite(rows["predicted_price_RM"]).all())

    def test_05_every_training_only_listing_uses_its_scenario_b_fold(self):
        expected = self.assignments.set_index("listing_id")["fold"]
        rows = self.oof[self.oof["Experiment_Type"] == "training_only"]
        for _, variant in rows.groupby(["Model", "Trim_Level"]):
            observed = variant.set_index("listing_id")["scenario_b_fold"].sort_index()
            np.testing.assert_array_equal(observed, expected.sort_index())

    def test_06_validation_rows_are_never_trimmed_training_only(self):
        training_folds = self.folds[self.folds["Experiment_Type"] == "training_only"]
        self.assertTrue((training_folds["Validation_Rows_Removed"] == 0).all())
        self.assertTrue((self.cutoffs["Validation_Rows_Removed"] == 0).all())
        expected = self.assignments["fold"].value_counts().sort_index()
        for fold, count in expected.items():
            self.assertTrue(
                (self.cutoffs[self.cutoffs["Fold"] == fold]["Validation_Rows"] == count).all()
            )

    def test_07_thresholds_recompute_from_outer_training_only(self):
        price = self.data["price"].to_numpy(float)
        fold_values = self.assignments["fold"].to_numpy(int)
        for _, row in self.cutoffs.iterrows():
            training = np.flatnonzero(fold_values != int(row["Fold"]))
            validation = np.flatnonzero(fold_values == int(row["Fold"]))
            self.assertEqual("outer_training_fold_prices_only", row["Cutoff_Source"])
            self.assertEqual(len(training), row["Training_Rows_Before"])
            self.assertEqual(len(validation), row["Validation_Rows"])
            if row["Removal_Percent"] == 0:
                self.assertTrue(pd.isna(row["Training_Derived_Cutoff_RM"]))
                retained = training
            else:
                expected_cutoff = np.quantile(
                    price[training], 1.0 - row["Removal_Percent"] / 100.0
                )
                self.assertAlmostEqual(expected_cutoff, row["Training_Derived_Cutoff_RM"], places=8)
                retained = training[price[training] <= expected_cutoff]
            self.assertEqual(len(retained), row["Training_Rows_Retained"])
            self.assertEqual(len(training) - len(retained), row["Training_Rows_Removed"])

    def test_08_no_full_dataset_percentile_is_used_for_training_thresholds(self):
        self.assertEqual(
            {"outer_training_fold_prices_only"}, set(self.cutoffs["Cutoff_Source"])
        )
        for level, percent in TRIM_LEVELS[1:]:
            # Five independently recomputed fold thresholds must be recorded.
            rows = self.cutoffs[self.cutoffs["Trim_Level"] == level]
            self.assertEqual(5, len(rows))
            self.assertEqual(5, rows["Fold"].nunique())
            self.assertFalse(rows["Training_Derived_Cutoff_RM"].isna().any())

    def test_09_frozen_models_and_ppsf_strategy_are_recorded(self):
        methodology = self.results["methodology"]
        self.assertEqual(MODEL_PARAMETERS, methodology["frozen_model_parameters"])
        self.assertEqual("price / property_size_sqft", methodology["target"])
        self.assertEqual("total listing price", methodology["trimming_variable"])
        self.assertEqual(5, len(methodology["position_features"]))

    def test_10_no_other_outlier_treatment_is_applied(self):
        controls = self.results["methodology"]["other_outlier_treatment"]
        self.assertTrue(controls)
        self.assertFalse(any(controls.values()))

    def test_11_training_only_overall_metrics_recompute(self):
        rows = self.oof[self.oof["Experiment_Type"] == "training_only"]
        stored = self.training.set_index(["Model", "Trim_Level"])
        for key, variant in rows.groupby(["Model", "Trim_Level"]):
            actual = variant["actual_price_RM"].to_numpy(float)
            predicted = variant["predicted_price_RM"].to_numpy(float)
            record = stored.loc[key]
            self.assertAlmostEqual(np.sqrt(mean_squared_error(actual, predicted)), record["RMSE_RM"], places=8)
            self.assertAlmostEqual(mean_absolute_error(actual, predicted), record["MAE_RM"], places=8)
            self.assertAlmostEqual(r2_score(actual, predicted), record["R2"], places=10)

    def test_12_segment_metrics_recompute(self):
        rows = self.oof[self.oof["Experiment_Type"] == "training_only"]
        p95 = np.quantile(self.data["price"], 0.95)
        p99 = np.quantile(self.data["price"], 0.99)
        masks = {
            "Remaining 95%": lambda values: values < p95,
            "Top 5%": lambda values: values >= p95,
            "95-99%": lambda values: (values >= p95) & (values < p99),
            "99-100%": lambda values: values >= p99,
        }
        stored = self.segments.set_index(["Model", "Trim_Level", "Segment"])
        for (model, level), variant in rows.groupby(["Model", "Trim_Level"]):
            actual = variant["actual_price_RM"].to_numpy(float)
            predicted = variant["predicted_price_RM"].to_numpy(float)
            for segment, mask_fn in masks.items():
                mask = mask_fn(actual)
                record = stored.loc[(model, level, segment)]
                error = predicted[mask] - actual[mask]
                self.assertEqual(mask.sum(), record["Rows"])
                self.assertAlmostEqual(np.sqrt(np.mean(error ** 2)), record["RMSE_RM"], places=8)
                self.assertAlmostEqual(np.mean(np.abs(error)), record["MAE_RM"], places=8)
                self.assertAlmostEqual(np.mean(error < 0) * 100, record["Underprediction_Pct"], places=10)

    def test_13_trimmed_population_matched_metrics_recompute(self):
        rows = self.oof[self.oof["Experiment_Type"] == "trimmed_population"]
        stored = self.trimmed.set_index("Trim_Level")
        for level, variant in rows.groupby("Trim_Level"):
            actual = variant["actual_price_RM"]
            retrained = variant["predicted_price_RM"]
            original = variant["matched_original_prediction_RM"]
            record = stored.loc[level]
            self.assertAlmostEqual(np.sqrt(mean_squared_error(actual, original)), record["Matched_Original_RMSE_RM"], places=8)
            self.assertAlmostEqual(np.sqrt(mean_squared_error(actual, retrained)), record["Matched_Retrained_RMSE_RM"], places=8)
            self.assertAlmostEqual(mean_absolute_error(actual, original), record["Matched_Original_MAE_RM"], places=8)
            self.assertAlmostEqual(mean_absolute_error(actual, retrained), record["Matched_Retrained_MAE_RM"], places=8)

    def test_14_fold_metrics_recompute(self):
        stored = self.folds.set_index(["Experiment_Type", "Model", "Trim_Level", "Fold"])
        for key, variant in self.oof.groupby(
            ["Experiment_Type", "Model", "Trim_Level", "scenario_b_fold"]
        ):
            actual = variant["actual_price_RM"]
            predicted = variant["predicted_price_RM"]
            record = stored.loc[key]
            self.assertAlmostEqual(np.sqrt(mean_squared_error(actual, predicted)), record["RMSE_RM"], places=8)
            self.assertAlmostEqual(mean_absolute_error(actual, predicted), record["MAE_RM"], places=8)

    def test_15_bootstrap_schema_and_point_differences(self):
        self.assertEqual(len(MODELS) * len(TRIM_LEVELS), len(self.bootstrap))
        self.assertTrue((self.bootstrap["Bootstrap_Samples"] == BOOTSTRAP_SAMPLES).all())
        self.assertTrue((self.bootstrap["Difference_Definition"] == "trimmed_minus_untrimmed").all())
        self.assertTrue((self.bootstrap["RMSE_CI95_Lower_RM"] <= self.bootstrap["RMSE_CI95_Upper_RM"]).all())
        self.assertTrue((self.bootstrap["MAE_CI95_Lower_RM"] <= self.bootstrap["MAE_CI95_Upper_RM"]).all())
        comparison = self.training.set_index(["Model", "Trim_Level"])
        for _, row in self.bootstrap.iterrows():
            baseline = comparison.loc[(row["Model"], "A")]
            candidate = comparison.loc[(row["Model"], row["Trim_Level"])]
            self.assertAlmostEqual(candidate["RMSE_RM"] - baseline["RMSE_RM"], row["RMSE_Difference_RM"], places=8)
            self.assertAlmostEqual(candidate["MAE_RM"] - baseline["MAE_RM"], row["MAE_Difference_RM"], places=8)

    def test_16_protected_production_files_are_unchanged(self):
        # This invariant records what happened during the isolated experiment.
        # Later, intentional application or documentation changes must not rewrite
        # that historical before/after snapshot or make the experiment unarchivable.
        safety = self.results["production_safety"]
        self.assertGreater(safety["protected_file_count"], 0)
        self.assertTrue(safety["all_protected_files_unchanged"])
        self.assertEqual(safety["before_manifest_sha256"], safety["after_manifest_sha256"])
        self.assertFalse(self.results["production_model_changed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
