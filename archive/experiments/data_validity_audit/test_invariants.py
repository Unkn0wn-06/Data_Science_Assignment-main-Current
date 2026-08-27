"""Safety, audit, and metric invariants for the validity-first experiment."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments" / "data_validity_audit"
REQUIRED = {
    "results.json", "validity_audit.csv", "clearly_invalid_rows.csv",
    "suspicious_rows.csv", "model_comparison.csv", "fold_metrics.csv",
    "oof_predictions.csv", "run_experiment.py", "test_invariants.py",
}
REQUIRED_AUDIT_COLUMNS = {
    "row_id", "price", "property_size_sqft", "ppsf", "property_type",
    "building_name", "developer", "city", "state", "bedroom", "bathroom",
    "parking_lot", "completion_year", "number_of_floors", "total_units",
    "validity_status", "flag_reason", "recommended_action",
}


class DataValidityAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = json.loads((EXPERIMENT / "results.json").read_text(encoding="utf-8"))
        cls.audit = pd.read_csv(EXPERIMENT / "validity_audit.csv")
        cls.invalid = pd.read_csv(EXPERIMENT / "clearly_invalid_rows.csv")
        cls.suspicious = pd.read_csv(EXPERIMENT / "suspicious_rows.csv")
        cls.comparison = pd.read_csv(EXPERIMENT / "model_comparison.csv")
        cls.folds = pd.read_csv(EXPERIMENT / "fold_metrics.csv")
        cls.oof = pd.read_csv(EXPERIMENT / "oof_predictions.csv")

    def test_required_files_exist(self):
        self.assertTrue(REQUIRED.issubset({path.name for path in EXPERIMENT.iterdir()}))

    def test_canonical_grain_and_audit_schema(self):
        self.assertEqual(len(self.audit), 3791)
        self.assertEqual(self.audit["row_id"].nunique(), 3791)
        self.assertTrue(REQUIRED_AUDIT_COLUMNS.issubset(self.audit.columns))
        self.assertTrue(set(self.audit["validity_status"]).issubset({"VALID", "SUSPICIOUS", "CLEARLY_INVALID"}))

    def test_exact_duplicates_and_possible_repeats_are_separate(self):
        exact = self.results["exact_duplicates"]
        repeats = self.results["possible_repeats"]
        self.assertEqual(exact["rows_found"], int(self.audit["exact_duplicate_flag"].sum()))
        self.assertEqual(repeats["rows"], int(self.audit["possible_repeat_flag"].sum()))
        self.assertFalse(repeats["automatic_deletion"])
        possible_only = self.audit[self.audit["flag_reason"].str.contains("POSSIBLE_REPEAT_DIFFERENT_LISTING_ID")]
        self.assertFalse(possible_only["recommended_action"].eq("REMOVE_FROM_EXPERIMENT_ONLY").any())

    def test_no_forbidden_removal_basis(self):
        deletion_rules = {
            name: rule for name, rule in self.results["validity_rules"].items() if rule["deletion_rule"]
        }
        for name, rule in deletion_rules.items():
            self.assertNotIn("percentile", rule["criterion"].lower())
            self.assertNotIn("ppsf threshold", rule["criterion"].lower())
        reasons = ";".join(self.invalid["flag_reason"].fillna("").tolist()).upper()
        self.assertNotIn("PREMIUM_PROPERTY", reasons)
        self.assertNotIn("VALID_BUT_EXTREME_PPSF", reasons)
        self.assertNotIn("OPTIONAL_MISSING", reasons)

    def test_every_removed_row_has_documented_clear_rule(self):
        clear_rule_names = {
            name for name, rule in self.results["validity_rules"].items()
            if rule["classification"] == "CLEARLY_INVALID" and rule["deletion_rule"]
        }
        self.assertTrue(self.invalid["flag_reason"].notna().all())
        for value in self.invalid["flag_reason"]:
            self.assertTrue(set(str(value).split(";")).intersection(clear_rule_names))

    def test_optional_missingness_does_not_cause_removal(self):
        optional_only = self.audit[
            self.audit["audit_evidence"].fillna("").str.contains("OPTIONAL_MISSING_COUNT")
            & self.audit["flag_reason"].fillna("").eq("NO_INVALIDITY_OR_REVIEW_FLAG")
        ]
        self.assertFalse(optional_only["recommended_action"].eq("REMOVE_FROM_EXPERIMENT_ONLY").any())

    def test_oof_complete_unique_and_five_fold(self):
        expected = {
            "A_current_canonical": 3791,
            "B_clearly_invalid_removed": 3791 - len(self.invalid),
        }
        for variant, count in expected.items():
            selected = self.oof[self.oof["variant"].eq(variant)]
            self.assertEqual(len(selected), count)
            self.assertEqual(selected["row_id"].nunique(), count)
            self.assertEqual(selected["fold"].nunique(), 5)
            self.assertTrue(np.isfinite(selected["predicted_price_rm"]).all())
        self.assertTrue(self.folds.groupby("variant")["fold"].nunique().eq(5).all())

    def test_metrics_recompute_from_oof(self):
        for row in self.comparison.itertuples(index=False):
            source_variant = (
                "A_current_canonical"
                if row.variant in {"A_current_canonical", "A_original_on_B_retained"}
                else "B_clearly_invalid_removed"
            )
            selected = self.oof[self.oof["variant"].eq(source_variant)].copy()
            if row.variant == "A_original_on_B_retained":
                selected = selected[selected["retained_after_audit"]]
            actual = selected["actual_price_rm"].to_numpy(float)
            predicted = selected["predicted_price_rm"].to_numpy(float)
            self.assertAlmostEqual(np.sqrt(mean_squared_error(actual, predicted)), row.rmse_rm, places=7)
            self.assertAlmostEqual(mean_absolute_error(actual, predicted), row.mae_rm, places=7)
            self.assertAlmostEqual(r2_score(actual, predicted), row.r2, places=10)

    def test_matched_gain_definition(self):
        rows = self.comparison.set_index("variant")
        candidate = rows.loc["B_clearly_invalid_removed"]
        matched = rows.loc["A_original_on_B_retained"]
        gains = self.results["model"]["retraining_gain"]
        self.assertAlmostEqual(matched.rmse_rm - candidate.rmse_rm, gains["rmse_gain_rm"], places=7)
        self.assertAlmostEqual(matched.mae_rm - candidate.mae_rm, gains["mae_gain_rm"], places=7)

    def test_bootstrap_gate(self):
        gains = self.results["model"]["retraining_gain"]
        expected = gains["rmse_gain_rm"] > 0 and gains["mae_gain_rm"] > 0
        self.assertEqual(expected, self.results["bootstrap"]["eligible"])
        if expected:
            self.assertEqual(self.results["bootstrap"]["result"]["draws"], 5000)
        else:
            self.assertIsNone(self.results["bootstrap"]["result"])

    def test_protected_files_unchanged(self):
        safety = self.results["production_safety"]
        self.assertTrue(safety["all_protected_files_unchanged"])
        self.assertEqual([], safety["changed_protected_files"])
        self.assertEqual(safety["before_manifest_sha256"], safety["after_manifest_sha256"])


if __name__ == "__main__":
    unittest.main()
