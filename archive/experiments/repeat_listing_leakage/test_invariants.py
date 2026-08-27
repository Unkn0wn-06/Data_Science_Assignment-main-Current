"""Independent invariants for the repeat-listing leakage experiment."""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.description_text_features.regex_features import extract_regex_features
from experiments.noncoordinate_target_encoding.target_encoding import MEstimateTargetEncoder


EXPERIMENT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed" / "enhanced_city_dataset.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(actual, predicted):
    actual = np.asarray(actual, float)
    predicted = np.asarray(predicted, float)
    return {
        "RMSE_RM": float(np.sqrt(np.mean(np.square(predicted - actual)))),
        "MAE_RM": float(np.mean(np.abs(predicted - actual))),
        "Median_AE_RM": float(np.median(np.abs(predicted - actual))),
        "R2": float(1 - np.square(predicted - actual).sum() / np.square(actual - actual.mean()).sum()),
    }


class ExperimentInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = json.loads((EXPERIMENT / "results.json").read_text(encoding="utf-8"))
        cls.data = pd.read_csv(DATA)
        cls.repeats = pd.read_csv(EXPERIMENT / "repeat_groups.csv")
        cls.assignments = pd.read_csv(EXPERIMENT / "group_safe_fold_assignments.csv")
        cls.comparison = pd.read_csv(EXPERIMENT / "model_comparison.csv")
        cls.folds = pd.read_csv(EXPERIMENT / "fold_metrics.csv")
        cls.oof = pd.read_csv(EXPERIMENT / "oof_predictions.csv")
        cls.invalid = pd.read_csv(EXPERIMENT / "invalid_row_fixed_fold_comparison.csv")

    def test_01_canonical_dataset_unchanged(self):
        self.assertEqual(len(self.data), 3_791)
        self.assertEqual(self.data["listing_id"].nunique(), 3_791)
        self.assertEqual(sha256(DATA), self.results["dataset"]["sha256"])

    def test_02_protected_files_unchanged(self):
        safety = self.results["production_safety"]
        self.assertTrue(safety["all_protected_files_unchanged"])
        self.assertEqual(safety["before_manifest_sha256"], safety["after_manifest_sha256"])

    def test_03_oof_prediction_counts_and_uniqueness(self):
        group_safe = self.oof[self.oof["cv_scheme"] == "group_safe"]
        self.assertEqual(len(group_safe), 4 * 3_791)
        counts = group_safe.groupby("model")["listing_id"].agg(["size", "nunique"])
        self.assertTrue((counts["size"] == 3_791).all())
        self.assertTrue((counts["nunique"] == 3_791).all())
        fixed = self.oof[self.oof["cv_scheme"] == "invalid_fixed_original_folds"]
        self.assertEqual(len(fixed), 2 * 3_788)
        self.assertTrue((fixed.groupby("model")["listing_id"].nunique() == 3_788).all())

    def test_04_repeat_groups_never_cross_group_safe_folds(self):
        repeat = self.assignments[self.assignments["is_repeat_like"]]
        self.assertEqual(int(repeat.groupby("repeat_group_id")["group_safe_fold"].nunique().max()), 1)
        self.assertEqual(self.results["group_safe_cv"]["repeat_groups_crossing_folds"], 0)

    def test_05_all_models_use_identical_group_safe_folds(self):
        group_safe = self.oof[self.oof["cv_scheme"] == "group_safe"]
        pivot = group_safe.pivot(index="listing_id", columns="model", values="fold")
        self.assertTrue(pivot.nunique(axis=1).eq(1).all())
        expected = self.assignments.set_index("listing_id")["group_safe_fold"]
        for model in group_safe["model"].unique():
            observed = group_safe[group_safe["model"] == model].set_index("listing_id")["fold"]
            self.assertTrue(observed.sort_index().equals(expected.sort_index()))

    def test_06_target_encoding_is_training_only_and_inner_oof(self):
        training = pd.DataFrame({"building_name": ["A", "A", "B", "B", "C", "C"]})
        target = np.array([1.0, 3.0, 10.0, 12.0, 20.0, 24.0])
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        first = MEstimateTargetEncoder(("building_name",), m=5).fit_transform_oof(training, target, cv)
        changed = target.copy()
        changed[0] = 1_000_000.0
        second = MEstimateTargetEncoder(("building_name",), m=5).fit_transform_oof(training, changed, cv)
        # The changed row's own OOF encoding cannot depend on its held-out target.
        self.assertEqual(float(first.iloc[0, 0]), float(second.iloc[0, 0]))
        for record in self.results["leakage_controls"]["fit_audit"]:
            self.assertFalse(record["validation_target_used"])
            self.assertEqual(record["group_overlap"], 0)

    def test_07_regex_extraction_is_target_free(self):
        text = pd.Series(["High floor with large balcony", "ordinary unit"])
        first = extract_regex_features(text)
        arbitrary_target = np.array([1.0, 999999.0])
        arbitrary_target[:] = arbitrary_target[::-1]
        second = extract_regex_features(text)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(list(first.columns), [name for group in __import__("experiments.description_text_features.regex_features", fromlist=["REGEX_GROUPS"]).REGEX_GROUPS.values() for name in group])

    def test_08_original_folds_preserved_in_invalid_subtest(self):
        fixed = self.oof[self.oof["cv_scheme"] == "invalid_fixed_original_folds"]
        original = pd.read_csv(ROOT / "experiments" / "description_text_features" / "oof_predictions.csv")
        original = original[original["variant"] == "regex_group_position"].set_index("row_id")["fold"]
        for model in fixed["model"].unique():
            observed = fixed[fixed["model"] == model].set_index("listing_id")["fold"]
            self.assertTrue(observed.sort_index().equals(original.loc[observed.index].sort_index()))
        self.assertTrue(self.results["invalid_row_fixed_fold_subtest"]["original_fold_membership_preserved"])

    def test_09_all_overall_metrics_recompute(self):
        for _, row in self.comparison.iterrows():
            selected = self.oof[(self.oof["cv_scheme"] == "group_safe") & (self.oof["model"] == row["model"])]
            recomputed = metrics(selected["actual_price_RM"], selected["predicted_price_RM"])
            for key, value in recomputed.items():
                self.assertAlmostEqual(value, row[f"group_safe_{key}"], places=7)
        for _, row in self.invalid.iterrows():
            selected = self.oof[(self.oof["cv_scheme"] == "invalid_fixed_original_folds") & (self.oof["model"] == row["model"])]
            recomputed = metrics(selected["actual_price_RM"], selected["predicted_price_RM"])
            for key, value in recomputed.items():
                self.assertAlmostEqual(value, row[key], places=7)

    def test_10_fold_metrics_recompute(self):
        for _, row in self.folds.iterrows():
            selected = self.oof[
                (self.oof["cv_scheme"] == row["cv_scheme"])
                & (self.oof["model"] == row["model"])
                & (self.oof["fold"] == row["fold"])
            ]
            recomputed = metrics(selected["actual_price_RM"], selected["predicted_price_RM"])
            self.assertAlmostEqual(recomputed["RMSE_RM"], row["RMSE_RM"], places=7)
            self.assertAlmostEqual(recomputed["MAE_RM"], row["MAE_RM"], places=7)

    def test_11_repeat_counts_recompute(self):
        audit = self.results["repeat_audit"]
        self.assertEqual(len(self.repeats), audit["total_repeat_like_rows"])
        self.assertEqual(self.repeats["repeat_group_id"].nunique(), audit["total_repeat_groups_used_for_cv"])
        self.assertEqual(self.repeats["level1_match_group_id"].replace("", np.nan).nunique(), audit["level1_exact_duplicate_groups"])
        self.assertEqual(self.repeats["level2_match_group_id"].replace("", np.nan).nunique(), audit["level2_strong_repeat_groups"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
