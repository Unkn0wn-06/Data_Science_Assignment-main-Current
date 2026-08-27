"""Independent invariants for the repeat-group sensitivity experiment."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experimental_support.description_linkage import link_descriptions
from src.experimental_support.target_encoding import MEstimateTargetEncoder
from experiments.repeat_group_sensitivity.run_experiment import (
    DATA_PATH,
    EXPECTED_ROWS,
    EXPERIMENT,
    MODEL_SPECS,
    RAW_PATH,
    SCENARIOS,
    build_level_groups,
    compose_scenario_groups,
    sha256,
)


REQUIRED_FILES = (
    "results.json",
    "sensitivity_model_comparison.csv",
    "fold_metrics.csv",
    "oof_predictions.csv",
    "scenario_a_fold_assignments.csv",
    "scenario_b_fold_assignments.csv",
    "scenario_c_fold_assignments.csv",
    "repeat_diagnostics.csv",
    "bootstrap_results.csv",
    "run_experiment.py",
    "test_invariants.py",
)


class RepeatGroupSensitivityInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = json.loads((EXPERIMENT / "results.json").read_text(encoding="utf-8"))
        cls.frame = pd.read_csv(DATA_PATH).reset_index(drop=True)
        cls.comparison = pd.read_csv(EXPERIMENT / "sensitivity_model_comparison.csv")
        cls.fold_metrics = pd.read_csv(EXPERIMENT / "fold_metrics.csv")
        cls.oof = pd.read_csv(EXPERIMENT / "oof_predictions.csv")
        cls.diagnostics = pd.read_csv(EXPERIMENT / "repeat_diagnostics.csv")
        cls.bootstrap = pd.read_csv(EXPERIMENT / "bootstrap_results.csv")
        cls.assignments = {
            scenario: pd.read_csv(
                EXPERIMENT / f"scenario_{scenario.lower()}_fold_assignments.csv"
            )
            for scenario in SCENARIOS
        }

    def test_01_required_artifacts_exist(self):
        self.assertEqual([], [name for name in REQUIRED_FILES if not (EXPERIMENT / name).is_file()])

    def test_02_canonical_dataset_is_unchanged_and_complete(self):
        dataset = self.results["dataset"]
        self.assertEqual(EXPECTED_ROWS, len(self.frame))
        self.assertEqual(EXPECTED_ROWS, self.frame["listing_id"].nunique())
        self.assertEqual(EXPECTED_ROWS, dataset["rows"])
        self.assertEqual(0, dataset["rows_deleted_as_repeats"])
        self.assertEqual(dataset["sha256"], sha256(DATA_PATH))
        self.assertTrue(self.results["production_safety"]["canonical_dataset_unchanged"])

    def test_03_production_and_prior_outputs_unchanged(self):
        safety = self.results["production_safety"]
        self.assertTrue(safety["all_files_outside_new_experiment_unchanged"])
        self.assertEqual(safety["before_manifest_sha256"], safety["after_manifest_sha256"])
        self.assertGreater(safety["protected_file_count"], 0)

    def test_04_repeat_definitions_and_counts_recompute(self):
        descriptions, _ = link_descriptions(RAW_PATH, self.frame["listing_id"])
        levels = build_level_groups(self.frame, descriptions)
        stored = self.results["grouping"]["level_counts"]
        for level, groups in levels.items():
            self.assertEqual(len(groups), stored[f"level_{level}"]["groups"])
            self.assertEqual(sum(map(len, groups)), stored[f"level_{level}"]["rows"])
        self.assertTrue(self.results["grouping"]["definitions_reused_without_change"])

    def test_05_fold_assignments_match_recomputed_group_relations(self):
        descriptions, _ = link_descriptions(RAW_PATH, self.frame["listing_id"])
        levels = build_level_groups(self.frame, descriptions)
        for scenario, spec in SCENARIOS.items():
            expected_group, expected_repeat, _ = compose_scenario_groups(
                self.frame, levels, spec["levels"], scenario
            )
            assignment = self.assignments[scenario].sort_values("row_index")
            np.testing.assert_array_equal(assignment["group_id"].to_numpy(), expected_group)
            np.testing.assert_array_equal(
                assignment["repeat_group_id"].fillna("").to_numpy(), expected_repeat
            )

    def test_06_no_repeat_group_crosses_folds_and_folds_are_balanced(self):
        for scenario, assignment in self.assignments.items():
            repeated = assignment[assignment["is_grouped_repeat"]]
            self.assertEqual(0, repeated.groupby("repeat_group_id")["fold"].nunique().gt(1).sum())
            self.assertEqual(set(range(1, 6)), set(assignment["fold"]))
            sizes = assignment["fold"].value_counts()
            # A sub-1% spread is a direct, scale-aware balance criterion. GroupKFold
            # may differ by slightly more than one largest group after shuffling.
            self.assertLessEqual(
                int(sizes.max() - sizes.min()), int(np.ceil(0.01 * EXPECTED_ROWS))
            )
            self.assertEqual(0, self.results["grouping"]["scenarios"][scenario]["repeat_groups_crossing_folds"])

    def test_07_stricter_scenarios_never_split_weaker_repeat_components(self):
        for weaker, stronger in (("A", "B"), ("B", "C")):
            joined = self.assignments[weaker][["listing_id", "group_id"]].merge(
                self.assignments[stronger][["listing_id", "group_id"]],
                on="listing_id",
                suffixes=("_weak", "_strong"),
                validate="one_to_one",
            )
            repeated = joined[joined["group_id_weak"].str.contains("_RG_", regex=False)]
            self.assertTrue(
                (repeated.groupby("group_id_weak")["group_id_strong"].nunique() == 1).all()
            )

    def test_08_each_listing_has_exactly_one_oof_prediction(self):
        expected_pairs = len(SCENARIOS) * len(MODEL_SPECS)
        self.assertEqual(EXPECTED_ROWS * expected_pairs, len(self.oof))
        counts = self.oof.groupby(["scenario", "model"])["listing_id"].agg(["size", "nunique"])
        self.assertTrue((counts["size"] == EXPECTED_ROWS).all())
        self.assertTrue((counts["nunique"] == EXPECTED_ROWS).all())
        self.assertFalse(self.oof["predicted_price_RM"].isna().any())

    def test_09_all_models_within_scenario_use_identical_folds(self):
        for scenario, scenario_oof in self.oof.groupby("scenario"):
            pivot = scenario_oof.pivot(index="listing_id", columns="model", values="fold")
            for model in pivot.columns[1:]:
                np.testing.assert_array_equal(pivot.iloc[:, 0], pivot[model])
            expected = self.assignments[scenario].set_index("listing_id")["fold"].sort_index()
            np.testing.assert_array_equal(pivot.sort_index().iloc[:, 0], expected)
        self.assertTrue(
            self.results["leakage_controls"]["all_models_within_scenario_use_identical_folds"]
        )

    def test_10_all_aggregate_metrics_recompute(self):
        thresholds = self.results["metric_thresholds"]
        comparison = self.comparison.set_index(["Scenario", "Model"])
        for (scenario, model), rows in self.oof.groupby(["scenario", "model"]):
            actual = rows["actual_price_RM"].to_numpy(float)
            predicted = rows["predicted_price_RM"].to_numpy(float)
            record = comparison.loc[(scenario, model)]
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            mae = mean_absolute_error(actual, predicted)
            r2 = r2_score(actual, predicted)
            adjusted = 1 - (1 - r2) * (len(actual) - 1) / (
                len(actual) - MODEL_SPECS[model]["predictors"] - 1
            )
            self.assertAlmostEqual(rmse, record["RMSE"], places=8)
            self.assertAlmostEqual(mae, record["MAE"], places=8)
            self.assertAlmostEqual(r2, record["R2"], places=10)
            self.assertAlmostEqual(adjusted, record["Adjusted R2"], places=10)
            self.assertAlmostEqual(np.median(np.abs(predicted - actual)), record["Median Absolute Error"], places=8)
            masks = {
                "Top5 RMSE": actual >= thresholds["p95"],
                "Top5 MAE": actual >= thresholds["p95"],
                "95-99% RMSE": (actual >= thresholds["p95"]) & (actual < thresholds["p99"]),
                "99-100% RMSE": actual >= thresholds["p99"],
            }
            for column, mask in masks.items():
                value = (
                    mean_absolute_error(actual[mask], predicted[mask])
                    if column == "Top5 MAE"
                    else np.sqrt(mean_squared_error(actual[mask], predicted[mask]))
                )
                self.assertAlmostEqual(value, record[column], places=8)

    def test_11_fold_metrics_recompute(self):
        stored = self.fold_metrics.set_index(["scenario", "model", "fold"])
        for key, rows in self.oof.groupby(["scenario", "model", "fold"]):
            actual = rows["actual_price_RM"]
            predicted = rows["predicted_price_RM"]
            record = stored.loc[key]
            self.assertEqual(len(rows), record["validation_rows"])
            self.assertAlmostEqual(np.sqrt(mean_squared_error(actual, predicted)), record["RMSE_RM"], places=8)
            self.assertAlmostEqual(mean_absolute_error(actual, predicted), record["MAE_RM"], places=8)

    def test_12_repeat_diagnostics_recompute(self):
        stored = self.diagnostics.set_index(["scenario", "model"])
        for key, rows in self.oof.groupby(["scenario", "model"]):
            record = stored.loc[key]
            mask = rows["is_grouped_repeat"].to_numpy(bool)
            actual = rows["actual_price_RM"].to_numpy(float)
            predicted = rows["predicted_price_RM"].to_numpy(float)
            self.assertEqual(int(mask.sum()), record["grouped_repeat_rows"])
            self.assertEqual(EXPECTED_ROWS, int(record["grouped_repeat_rows"] + record["non_repeat_rows"]))
            for name, selected in (("repeat_row", mask), ("non_repeat_row", ~mask)):
                self.assertAlmostEqual(
                    np.sqrt(mean_squared_error(actual[selected], predicted[selected])),
                    record[f"{name}_RMSE_RM"], places=8,
                )
                self.assertAlmostEqual(
                    mean_absolute_error(actual[selected], predicted[selected]),
                    record[f"{name}_MAE_RM"], places=8,
                )

    def test_13_bootstrap_point_differences_recompute_and_intervals_are_ordered(self):
        indexed = self.oof.set_index(["scenario", "model", "listing_id"])
        for _, record in self.bootstrap.iterrows():
            scenario = record["scenario"]
            actual = indexed.loc[(scenario, record["candidate"]), "actual_price_RM"].sort_index()
            candidate = indexed.loc[(scenario, record["candidate"]), "predicted_price_RM"].sort_index()
            reference = indexed.loc[(scenario, record["reference"]), "predicted_price_RM"].sort_index()
            candidate_rmse = np.sqrt(mean_squared_error(actual, candidate))
            reference_rmse = np.sqrt(mean_squared_error(actual, reference))
            candidate_mae = mean_absolute_error(actual, candidate)
            reference_mae = mean_absolute_error(actual, reference)
            self.assertAlmostEqual(candidate_rmse - reference_rmse, record["RMSE_difference_RM"], places=8)
            self.assertAlmostEqual(candidate_mae - reference_mae, record["MAE_difference_RM"], places=8)
            self.assertLessEqual(record["RMSE_CI95_lower_RM"], record["RMSE_CI95_upper_RM"])
            self.assertLessEqual(record["MAE_CI95_lower_RM"], record["MAE_CI95_upper_RM"])
            self.assertEqual(5_000, record["bootstrap_samples"])

    def test_14_target_encoder_oof_value_excludes_own_target(self):
        X = pd.DataFrame({"building_name": ["A", "A", "B", "B", "C", "C"]})
        target = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        original = MEstimateTargetEncoder(("building_name",), m=10.0).fit_transform_oof(X, target, cv)
        for row in range(len(X)):
            mutated = target.copy()
            mutated[row] += 1_000_000.0
            changed = MEstimateTargetEncoder(("building_name",), m=10.0).fit_transform_oof(
                X, mutated, KFold(n_splits=3, shuffle=True, random_state=42)
            )
            self.assertAlmostEqual(original.iloc[row, 0], changed.iloc[row, 0], places=12)
        controls = self.results["leakage_controls"]
        self.assertTrue(controls["outer_preprocessing_training_fold_only"])
        self.assertTrue(controls["building_target_encoding_inner_oof"])
        self.assertFalse(controls["validation_targets_used_for_fit_or_features"])
        self.assertTrue(all(row["group_overlap"] == 0 for row in controls["fit_audit"]))

    def test_15_rank_columns_recompute(self):
        for _, rows in self.comparison.groupby("Scenario"):
            expected_rmse = rows["RMSE"].rank(method="min").astype(int)
            expected_mae = rows["MAE"].rank(method="min").astype(int)
            np.testing.assert_array_equal(rows["RMSE Rank"], expected_rmse)
            np.testing.assert_array_equal(rows["MAE Rank"], expected_mae)


if __name__ == "__main__":
    unittest.main(verbosity=2)
