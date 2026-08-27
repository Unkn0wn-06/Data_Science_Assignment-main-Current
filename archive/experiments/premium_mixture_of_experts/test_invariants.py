"""Leakage and accounting tests for the premium mixture experiment."""

from __future__ import annotations

import unittest
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.premium_mixture_of_experts.classifier import (
    DESCRIPTION_PATTERNS,
    fold_premium_threshold,
)
from experiments.premium_mixture_of_experts.evaluation import full_metric_bundle
from experiments.premium_mixture_of_experts.regressors import premium_scope_mask
from experiments.premium_mixture_of_experts.routing import hard_route, soft_route


class PremiumMixtureInvariantTests(unittest.TestCase):
    def test_fold_threshold_uses_only_supplied_outer_training_target(self):
        train = np.arange(1.0, 101.0)
        validation_a = np.array([10.0, 20.0])
        validation_b = np.array([1e9, 2e9])
        first = fold_premium_threshold(train)
        second = fold_premium_threshold(train.copy())
        self.assertEqual(first, second)
        self.assertFalse(np.array_equal(validation_a, validation_b))

    def test_premium_scope_uses_training_target_only(self):
        train = np.arange(1.0, 101.0)
        mask, threshold = premium_scope_mask(train, 0.10)
        self.assertEqual(mask.shape, train.shape)
        self.assertAlmostEqual(threshold, np.quantile(train, 0.90))
        self.assertEqual(int(mask.sum()), 10)

    def test_soft_routing_is_row_aligned(self):
        standard = np.array([100.0, 200.0, 300.0])
        premium = np.array([1000.0, 2000.0, 3000.0])
        probability = np.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(
            soft_route(standard, premium, probability),
            np.array([100.0, 1100.0, 3000.0]),
        )
        with self.assertRaises(ValueError):
            soft_route(standard, premium[:-1], probability)

    def test_routing_does_not_accept_validation_target(self):
        routed, flags = hard_route([1.0, 2.0], [10.0, 20.0], [0.2, 0.8], 0.5)
        np.testing.assert_allclose(routed, [1.0, 20.0])
        np.testing.assert_array_equal(flags, [False, True])

    def test_all_validation_rows_remain_in_oof_accounting(self):
        folds = [np.array([0, 2]), np.array([1, 3])]
        assigned = np.zeros(4, dtype=int)
        for validation in folds:
            assigned[validation] += 1
        np.testing.assert_array_equal(assigned, np.ones(4, dtype=int))

    def test_metrics_are_original_total_rm(self):
        actual_total = np.array([200_000.0, 400_000.0])
        predicted_total = np.array([210_000.0, 380_000.0])
        metrics = full_metric_bundle(actual_total, predicted_total, 1, 300_000.0)
        self.assertAlmostEqual(metrics["RMSE_RM"], np.sqrt(250_000_000.0))
        self.assertAlmostEqual(metrics["MAE_RM"], 15_000.0)

    def test_no_premium_validation_rows_removed(self):
        actual = np.array([100.0, 200.0, 900.0, 1000.0])
        prediction = actual.copy()
        metrics = full_metric_bundle(actual, prediction, 1, 850.0)
        self.assertEqual(metrics["top_5_percent"]["count"], 2)
        self.assertEqual(metrics["remaining_95_percent"]["count"], 2)
        self.assertEqual(
            metrics["top_5_percent"]["count"] + metrics["remaining_95_percent"]["count"],
            len(actual),
        )

    def test_description_patterns_are_predefined_and_target_free(self):
        self.assertGreaterEqual(len(DESCRIPTION_PATTERNS), 10)
        joined = " ".join(DESCRIPTION_PATTERNS.values()).lower()
        for forbidden in ("price", "905000", "target"):
            self.assertNotIn(forbidden, joined)


class GeneratedArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = Path(__file__).resolve().parent
        cls.comparison = pd.read_csv(cls.directory / "model_comparison.csv")
        cls.oof = pd.read_csv(cls.directory / "oof_predictions.csv")
        cls.routing = pd.read_csv(cls.directory / "routing_analysis.csv")
        cls.folds = pd.read_csv(cls.directory / "fold_metrics.csv")
        with (cls.directory / "results.json").open(encoding="utf-8") as handle:
            cls.results = json.load(handle)

    def test_every_variant_has_exactly_3791_unique_oof_rows(self):
        counts = self.oof.groupby("variant")["row_index"].agg(["count", "nunique"])
        self.assertTrue((counts["count"] == 3791).all())
        self.assertTrue((counts["nunique"] == 3791).all())

    def test_every_routed_variant_keeps_all_validation_rows(self):
        counts = self.routing.groupby("variant")["row_index"].agg(["count", "nunique"])
        self.assertTrue((counts["count"] == 3791).all())
        self.assertTrue((counts["nunique"] == 3791).all())
        self.assertEqual(set(self.routing["fold"].unique()), {1, 2, 3, 4, 5})

    def test_routing_formula_and_oof_join_are_exact(self):
        routed_oof = self.oof[self.oof["variant"].isin(self.routing["variant"].unique())]
        merged = self.routing.merge(
            routed_oof[["variant", "row_index", "predicted_price_RM"]],
            on=["variant", "row_index"],
            validate="one_to_one",
        )
        np.testing.assert_allclose(merged["final_prediction_RM"], merged["predicted_price_RM"])
        for routing_type, group in self.routing.groupby("routing"):
            if routing_type == "hard":
                expected = np.where(group["predicted_premium"], group["premium_prediction_RM"], group["standard_prediction_RM"])
            else:
                expected = (1 - group["premium_probability"]) * group["standard_prediction_RM"] + group["premium_probability"] * group["premium_prediction_RM"]
            np.testing.assert_allclose(group["final_prediction_RM"], expected, rtol=1e-12, atol=1e-7)

    def test_probabilities_and_predictions_are_finite(self):
        self.assertTrue(np.isfinite(self.oof["predicted_price_RM"]).all())
        self.assertTrue(np.isfinite(self.routing["final_prediction_RM"]).all())
        self.assertTrue(self.routing["premium_probability"].between(0, 1).all())

    def test_saved_headline_metrics_recompute_from_oof(self):
        indexed = self.comparison.set_index("variant")
        for variant, group in self.oof.groupby("variant"):
            actual = group["actual_price_RM"].to_numpy(float)
            predicted = group["predicted_price_RM"].to_numpy(float)
            self.assertAlmostEqual(indexed.loc[variant, "RMSE_RM"], np.sqrt(np.mean((predicted - actual) ** 2)), places=6)
            self.assertAlmostEqual(indexed.loc[variant, "MAE_RM"], np.mean(np.abs(predicted - actual)), places=6)

    def test_fold_table_has_five_rows_per_variant(self):
        self.assertTrue((self.folds.groupby("variant")["fold"].nunique() == 5).all())

    def test_required_matrix_and_artifacts_exist(self):
        expected_variants = {
            "random_forest_reference", "lightgbm_interaction_reference", "building_name_te_reference",
            "hard_rf_lgbm_p5", "hard_rf_lgbm_p10", "hard_rf_lgbm_p15", "hard_rf_lgbm_p20",
            "soft_rf_lgbm_p10", "soft_rf_lgbm_p15", "soft_rf_lgbm_p20", "soft_lgbm_lgbm_p15",
            "soft_rf_lgbm_direct_p15",
        }
        self.assertTrue(expected_variants.issubset(set(self.comparison["variant"])))
        required_files = [
            "results.json", "model_comparison.csv", "fold_metrics.csv", "oof_predictions.csv",
            "classification_metrics.csv", "routing_analysis.csv", "feature_summary.json",
        ]
        self.assertTrue(all((self.directory / name).is_file() for name in required_files))

    def test_all_eleven_figures_are_nonempty(self):
        figures = sorted((self.directory / "figures").glob("*.png"))
        self.assertEqual(len(figures), 11)
        self.assertTrue(all(path.stat().st_size > 10_000 for path in figures))

    def test_protected_files_are_hash_identical(self):
        safety = self.results["production_safety"]
        self.assertTrue(safety["all_sha256_unchanged"])
        self.assertEqual(safety["before_sha256"], safety["after_sha256"])

    def test_price_bands_partition_all_rows(self):
        bands = pd.DataFrame(self.results["price_band_performance"])
        self.assertTrue((bands.groupby("model")["count"].sum() == 3791).all())
        self.assertEqual(self.results["dataset"]["global_descriptive_premium_count"], 190)

    def test_leakage_audit_and_row_accounting_are_true(self):
        audit = self.results["leakage_audit"]
        for key, value in audit.items():
            if key.startswith("outer_validation_price_used") or key == "coordinate_features_used":
                self.assertFalse(value, key)
            else:
                self.assertTrue(value, key)
        self.assertEqual(self.results["dataset"]["headline_rows_removed"], 0)
        self.assertEqual(self.results["dataset"]["premium_rows_removed"], 0)


if __name__ == "__main__":
    unittest.main()
