"""Artifact-level invariants for the isolated missing-data quality experiment."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments" / "missing_data_quality"
REQUIRED = {
    "results.json",
    "dataset_variant_summary.csv",
    "missingness_summary.csv",
    "missingness_by_price_band.csv",
    "fold_metrics.csv",
    "oof_predictions.csv",
}
EXPECTED_VARIANTS = {
    "A_current",
    "B_valid_critical",
    "C_complete_core",
    "D_missing_lt3",
    "E_missing_lt5",
    "F_completeness_ge80",
    "G_completeness_ge90",
    "H_missing_indicators",
}


class MissingDataQualityArtifactsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = json.loads((EXPERIMENT / "results.json").read_text(encoding="utf-8"))
        cls.summary = pd.read_csv(EXPERIMENT / "dataset_variant_summary.csv")
        cls.missing = pd.read_csv(EXPERIMENT / "missingness_summary.csv")
        cls.by_price = pd.read_csv(EXPERIMENT / "missingness_by_price_band.csv")
        cls.folds = pd.read_csv(EXPERIMENT / "fold_metrics.csv")
        cls.oof = pd.read_csv(EXPERIMENT / "oof_predictions.csv")

    def test_required_artifacts_exist(self):
        self.assertEqual(REQUIRED, {path.name for path in EXPERIMENT.iterdir() if path.name in REQUIRED})

    def test_only_prescribed_model_variants(self):
        self.assertEqual(EXPECTED_VARIANTS, set(self.summary["variant"]))
        self.assertEqual(EXPECTED_VARIANTS, set(self.oof["variant"]))

    def test_reference_metrics_match_declared_baseline(self):
        reference = self.results["canonical_reference"]
        self.assertTrue(reference["expected_metrics_reproduced"])
        self.assertAlmostEqual(reference["metrics"]["rmse_rm"], 118750.19350875785, places=6)
        self.assertAlmostEqual(reference["metrics"]["mae_rm"], 60967.10968555279, places=6)
        self.assertAlmostEqual(reference["metrics"]["r2"], 0.8702674858843791, places=10)

    def test_row_accounting_and_oof_uniqueness(self):
        for row in self.summary.itertuples(index=False):
            self.assertEqual(row.rows_retained + row.rows_removed, 3791)
            selected = self.oof[self.oof["variant"].eq(row.variant)]
            self.assertEqual(len(selected), row.rows_retained)
            self.assertEqual(selected["canonical_row_index"].nunique(), row.rows_retained)
            self.assertTrue(np.isfinite(selected["retrained_model_oof_prediction_rm"]).all())

    def test_five_fold_oof_for_both_metric_sources(self):
        counts = self.folds.groupby(["variant", "metric_source"])["fold"].nunique()
        self.assertTrue(counts.eq(5).all())

    def test_gain_definitions_recompute(self):
        rmse_gain = self.summary["matched_original_rmse_rm"] - self.summary["rmse_rm"]
        mae_gain = self.summary["matched_original_mae_rm"] - self.summary["mae_rm"]
        np.testing.assert_allclose(rmse_gain, self.summary["retraining_rmse_gain_rm"], rtol=0, atol=1e-8)
        np.testing.assert_allclose(mae_gain, self.summary["retraining_mae_gain_rm"], rtol=0, atol=1e-8)

    def test_zero_values_are_not_counted_missing(self):
        numeric = self.missing[self.missing["zero_count"].fillna(0).gt(0)]
        self.assertTrue(numeric["zero_treated_as_missing"].eq(False).all())

    def test_missingness_output_covers_requested_features_and_price_bands(self):
        requested = {
            "price", "property_size_sqft", "bedroom", "bathroom", "parking_lot",
            "completion_year", "number_of_floors", "total_units", "property_type",
            "tenure_type", "land_title", "floor_range", "state", "city",
            "building_name", "developer",
        }
        self.assertTrue(requested.issubset(set(self.missing["feature"])))
        self.assertEqual(6, self.by_price["price_band"].nunique())

    def test_conditional_bootstrap_contract(self):
        bootstrap = self.results["bootstrap"]
        self.assertEqual(set(bootstrap["eligible_variants"]), set(bootstrap["comparisons"]))
        for interval in bootstrap["comparisons"].values():
            self.assertEqual(interval["draws"], 5000)
            self.assertIn("ci95_lower", interval["rmse_difference_rm"])
            self.assertIn("ci95_upper", interval["mae_difference_rm"])

    def test_production_files_unchanged(self):
        safety = self.results["production_safety"]
        self.assertTrue(safety["all_protected_files_unchanged"])
        self.assertEqual([], safety["changed_protected_files"])
        self.assertEqual(safety["before_manifest_sha256"], safety["after_manifest_sha256"])


if __name__ == "__main__":
    unittest.main()
