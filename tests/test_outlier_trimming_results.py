"""Validate retained-population presentation artifacts without model retraining."""

from __future__ import annotations

import json
import unittest

import pandas as pd

from src.cleaning.pipeline import PROJECT_ROOT
from src.models.final.position_regex_lightgbm import FINAL_MODEL_NAME


RESULTS_DIR = PROJECT_ROOT / "results" / "outlier_trimming"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "enhanced_city_dataset.csv"
FOLD_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "repeat_group_sensitivity"
    / "scenario_b_fold_assignments.csv"
)
EXPECTED_LEVELS = ["0%", "0.5%", "1%", "2.5%", "5%", "10%"]


class RetainedCVArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary_path = RESULTS_DIR / "retained_cv_summary.csv"
        cls.summary = pd.read_csv(cls.summary_path)
        cls.canonical = pd.read_csv(DATA_PATH, usecols=["listing_id", "price"])
        cls.assignments = pd.read_csv(FOLD_PATH)
        cls.distribution = pd.read_csv(RESULTS_DIR / "distribution_shift.csv")
        cls.restricted = pd.read_csv(
            RESULTS_DIR / "trimmed_population_comparison.csv"
        )
        cls.metadata = json.loads(
            (RESULTS_DIR / "metadata.json").read_text(encoding="utf-8")
        )
        cls.joined = cls.canonical.merge(
            cls.assignments[["listing_id", "fold", "repeat_group_id"]],
            on="listing_id",
            how="left",
            validate="one_to_one",
        )

    def test_summary_schema_levels_and_fold_counts(self):
        self.assertTrue(self.summary_path.is_file())
        self.assertEqual(
            [
                "trim_level",
                "fold",
                "original_rows",
                "retained_rows",
                "removed_rows",
                "training_rows",
                "validation_rows",
                "retention_percentage",
            ],
            list(self.summary.columns),
        )
        self.assertEqual(EXPECTED_LEVELS, list(self.summary["trim_level"].unique()))
        self.assertTrue(
            (self.summary.groupby("trim_level", sort=False)["fold"].nunique() == 5).all()
        )
        self.assertEqual(30, len(self.summary))

    def test_fold_counts_cover_each_retained_population_exactly(self):
        self.assertTrue(
            (
                self.summary["training_rows"] + self.summary["validation_rows"]
                == self.summary["retained_rows"]
            ).all()
        )
        grouped = self.summary.groupby("trim_level", sort=False)
        self.assertTrue(
            (
                grouped["validation_rows"].sum()
                == grouped["retained_rows"].first()
            ).all()
        )
        self.assertTrue(
            (
                grouped["training_rows"].sum()
                == 4 * grouped["retained_rows"].first()
            ).all()
        )

    def test_retained_ids_preserve_scenario_b_partitions(self):
        distribution = self.distribution.sort_values("Removal_Percent")
        for _, level in distribution.iterrows():
            label = f"{float(level['Removal_Percent']):g}%"
            cutoff = level["Full_Population_Cutoff_RM"]
            retained = (
                self.joined
                if float(level["Removal_Percent"]) == 0
                else self.joined[self.joined["price"] <= cutoff]
            )
            retained_ids = set(retained["listing_id"].astype(int))
            validation_occurrences: list[int] = []
            for fold in range(1, 6):
                validation_ids = set(
                    retained.loc[retained["fold"].eq(fold), "listing_id"].astype(int)
                )
                training_ids = retained_ids.difference(validation_ids)
                self.assertTrue(training_ids.isdisjoint(validation_ids))
                self.assertEqual(retained_ids, training_ids.union(validation_ids))
                validation_occurrences.extend(validation_ids)
                saved = self.summary[
                    self.summary["trim_level"].eq(label)
                    & self.summary["fold"].eq(fold)
                ].iloc[0]
                self.assertEqual(len(training_ids), int(saved["training_rows"]))
                self.assertEqual(len(validation_ids), int(saved["validation_rows"]))

            validation_counts = pd.Series(validation_occurrences).value_counts()
            self.assertEqual(retained_ids, set(validation_counts.index.astype(int)))
            self.assertTrue((validation_counts == 1).all())
            repeated = retained.dropna(subset=["repeat_group_id"])
            self.assertFalse(
                repeated.groupby("repeat_group_id")["fold"].nunique().gt(1).any()
            )

    def test_retained_counts_match_restricted_results_and_final_decision(self):
        saved_counts = (
            self.summary.groupby("trim_level", sort=False)["retained_rows"].first()
        )
        expected_counts = {
            f"{float(row['Removal_Percent']):g}%": int(row["Retained_OOF_Rows"])
            for _, row in self.restricted.iterrows()
        }
        self.assertEqual(expected_counts, saved_counts.astype(int).to_dict())
        self.assertEqual("0%", self.metadata["recommended_trimming"])
        self.assertEqual(FINAL_MODEL_NAME, self.metadata["production_model"])
        self.assertFalse(self.metadata["production_model_changed"])
        self.assertFalse(self.metadata["streamlit_retraining"])


if __name__ == "__main__":
    unittest.main()
