"""Leakage, linkage, text-feature, and artifact tests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.description_text_features.evaluation import complete_metrics
from experiments.description_text_features.regex_features import extract_regex_features
from experiments.description_text_features.text_cleaning import clean_description_text, link_descriptions
from experiments.description_text_features.tfidf_features import FoldTextTransformer


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent


class TextFeatureInvariantTests(unittest.TestCase):
    def test_cleaning_handles_missing_case_and_whitespace(self):
        source = pd.Series([None, "  HIGH   Floor\nUnit  ", "Already clean"])
        self.assertEqual(clean_description_text(source).tolist(), ["", "high floor unit", "already clean"])

    def test_regex_positive_and_negative_examples(self):
        features = extract_regex_features(pd.Series(["beautiful penthouse with private lift", "normal condominium unit"]))
        self.assertEqual(features.loc[0, "is_penthouse"], 1)
        self.assertEqual(features.loc[0, "has_private_lift"], 1)
        self.assertEqual(int(features.loc[1].sum()), 0)

    def test_phrase_boundaries_avoid_obvious_false_positives(self):
        features = extract_regex_features(pd.Series(["exclusive unit", "nonexclusive arrangement", "balcony", "balconies"]))
        self.assertEqual(features.loc[0, "is_exclusive_text"], 1)
        self.assertEqual(features.loc[1, "is_exclusive_text"], 0)
        self.assertEqual(features.loc[2, "has_balcony"], 1)
        self.assertEqual(features.loc[3, "has_balcony"], 1)

    def test_validation_vocabulary_does_not_change_training_fit(self):
        train = pd.Series(["alpha condo", "alpha apartment", "beta condo", "beta apartment", "alpha beta"] * 2)
        transformer = FoldTextTransformer(1).fit(train)
        before = dict(transformer.vectorizer_.vocabulary_)
        transformer.transform(pd.Series(["validationonlytoken penthouse"]))
        self.assertEqual(before, transformer.vectorizer_.vocabulary_)
        self.assertNotIn("validationonlytoken", transformer.vectorizer_.vocabulary_)

    def test_svd_is_deterministic_finite_and_fixed_dimension(self):
        text = pd.Series([f"property unit token{i % 7} city view" for i in range(70)])
        first = FoldTextTransformer(5).fit(text).transform(text)
        second = FoldTextTransformer(5).fit(text).transform(text)
        self.assertEqual(first.shape, (70, 5))
        self.assertTrue(np.isfinite(first.to_numpy()).all())
        np.testing.assert_allclose(first, second)

    def test_description_linkage_is_exactly_one_to_one(self):
        canonical = pd.read_csv(ROOT / "data" / "processed" / "enhanced_city_dataset.csv")
        descriptions, audit = link_descriptions(ROOT / "data" / "raw" / "houses.csv", canonical["listing_id"])
        self.assertEqual(len(descriptions), 3791)
        self.assertEqual(audit["linked_rows"], 3791)
        self.assertEqual(audit["rows_reintroduced"], 0)
        self.assertTrue(audit["exact_order_match"])

    def test_official_metrics_are_original_total_rm(self):
        actual = np.array([100_000.0, 200_000.0, 1_000_000.0])
        predicted = np.array([110_000.0, 180_000.0, 900_000.0])
        result = complete_metrics(actual, predicted, predictors=1, premium_threshold=900_000.0)
        self.assertAlmostEqual(result["RMSE_RM"], np.sqrt((10_000**2 + 20_000**2 + 100_000**2) / 3))
        self.assertAlmostEqual(result["MAE_RM"], 130_000 / 3)


class GeneratedArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (EXPERIMENT / "results.json").exists():
            raise unittest.SkipTest("Experiment artifacts have not been generated yet.")
        cls.comparison = pd.read_csv(EXPERIMENT / "model_comparison.csv")
        cls.oof = pd.read_csv(EXPERIMENT / "oof_predictions.csv")
        cls.folds = pd.read_csv(EXPERIMENT / "fold_metrics.csv")
        with (EXPERIMENT / "results.json").open(encoding="utf-8") as handle:
            cls.results = json.load(handle)

    def test_every_variant_has_complete_unique_oof_rows(self):
        counts = self.oof.groupby("variant")["row_id"].agg(["count", "nunique"])
        self.assertTrue((counts["count"] == 3791).all())
        self.assertTrue((counts["nunique"] == 3791).all())

    def test_saved_metrics_recompute_from_oof(self):
        expected = self.comparison.set_index("variant")
        for variant, group in self.oof.groupby("variant"):
            actual = group["actual_price_RM"].to_numpy(float)
            predicted = group["predicted_price_RM"].to_numpy(float)
            self.assertAlmostEqual(expected.loc[variant, "RMSE_RM"], np.sqrt(np.mean((predicted - actual) ** 2)), places=6)
            self.assertAlmostEqual(expected.loc[variant, "MAE_RM"], np.mean(np.abs(predicted - actual)), places=6)

    def test_shared_folds_and_finite_predictions(self):
        self.assertTrue((self.folds.groupby("variant")["fold"].nunique() == 5).all())
        self.assertTrue(np.isfinite(self.oof["predicted_price_RM"]).all())

    def test_required_outputs_and_figures_exist(self):
        files = ["results.json", "model_comparison.csv", "fold_metrics.csv", "oof_predictions.csv", "feature_summary.json", "regex_feature_frequencies.csv", "svd_component_summary.csv"]
        self.assertTrue(all((EXPERIMENT / name).is_file() for name in files))
        figures = sorted((EXPERIMENT / "figures").glob("*.png"))
        self.assertEqual(len(figures), 10)
        self.assertTrue(all(path.stat().st_size > 10_000 for path in figures))

    def test_required_experiment_matrix_is_present(self):
        expected = {
            "structured_baseline_reproduced", "basic_text_existing_control", "regex_expanded",
            "tfidf_svd_10", "tfidf_svd_20", "tfidf_svd_30", "tfidf_svd_50",
            "regex_svd_nested", "regex_svd_building_te_nested",
        }
        self.assertTrue(expected.issubset(set(self.comparison["variant"])))

    def test_baseline_reproduction_is_exact(self):
        reproduction = self.results["baseline_reproduction"]
        self.assertTrue(reproduction["matched_at_1e-8"])
        self.assertTrue(all(abs(value) <= 1e-8 for value in reproduction["differences_vs_verified"].values()))

    def test_linkage_and_price_band_counts_preserve_grain(self):
        linkage = self.results["description_linkage"]
        self.assertEqual(linkage["linked_rows"], 3791)
        self.assertEqual(linkage["unique_canonical_ids"], 3791)
        self.assertEqual(linkage["rows_reintroduced"], 0)
        bands = pd.DataFrame(self.results["price_band_performance"])
        self.assertTrue((bands.groupby("variant")["count"].sum() == 3791).all())

    def test_svd_summary_is_finite_and_fold_complete(self):
        svd = pd.read_csv(EXPERIMENT / "svd_component_summary.csv")
        word = svd[svd["analyzer"] == "word"]
        self.assertEqual(set(word["fold"].unique()), {1, 2, 3, 4, 5})
        self.assertEqual(set(word["n_components"].unique()), {10, 20, 30, 50})
        self.assertTrue(np.isfinite(word[["explained_variance_ratio", "cumulative_explained_variance_ratio"]]).all().all())
        self.assertTrue((word["vocabulary_size"] <= 5000).all())

    def test_protected_files_unchanged_and_leakage_audit_passes(self):
        safety = self.results["production_safety"]
        self.assertTrue(safety["all_sha256_unchanged"])
        self.assertEqual(safety["before_sha256"], safety["after_sha256"])
        expected_false = {
            "validation_descriptions_in_transformer_fit",
            "regex_keywords_target_selected",
            "validation_price_used_for_feature_engineering",
            "validation_price_used_for_model_tuning",
            "duplicate_canonical_identifiers_introduced",
            "many_to_many_raw_join",
        }
        for key, value in self.results["leakage_audit"].items():
            self.assertEqual(value, key not in expected_false, key)


if __name__ == "__main__":
    unittest.main()
