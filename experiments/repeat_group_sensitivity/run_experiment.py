"""Sensitivity of frozen model results to repeat-listing grouping strength."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experimental_support.description_linkage import link_descriptions
from src.experimental_support.regex_features import extract_regex_features
from src.experimental_support.repeat_models import (
    MAJOR_COLUMNS,
    MODEL_SPECS,
    POSITION_FEATURES,
    fit_model_fold,
    group_members,
)
from src.models.common.features import MODEL_FEATURES


EXPERIMENT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "enhanced_city_dataset.csv"
RAW_PATH = ROOT / "data" / "raw" / "houses.csv"
EXPECTED_ROWS = 3_791
RANDOM_STATE = 42
BOOTSTRAP_DRAWS = 5_000
SCENARIOS = {
    "A": {
        "name": "Scenario A - Level 1 only",
        "interpretation": "minimal duplicate protection",
        "levels": [1],
    },
    "B": {
        "name": "Scenario B - Level 1 + Level 2",
        "interpretation": "moderate / strong-repeat protection",
        "levels": [1, 2],
    },
    "C": {
        "name": "Scenario C - Level 1 + Level 2 + Level 3",
        "interpretation": "conservative protection including possible repeats",
        "levels": [1, 2, 3],
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_snapshot() -> dict[str, str]:
    """Hash everything outside this new experiment, excluding generated caches."""
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        try:
            path.relative_to(EXPERIMENT)
        except ValueError:
            files.append(path)
    return {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in sorted(files)
    }


def manifest_digest(snapshot: dict[str, str]) -> str:
    payload = "\n".join(f"{name}:{digest}" for name, digest in sorted(snapshot.items()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def json_clean(value):
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_clean(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if value is pd.NA:
        return None
    return value


class DisjointSet:
    """Deterministic union-find used only to compose existing match relations."""

    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=int)

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = int(self.parent[value])
        return value

    def union_group(self, positions: np.ndarray) -> None:
        anchor = self.find(int(positions[0]))
        for value in positions[1:]:
            left = self.find(anchor)
            right = self.find(int(value))
            if left != right:
                self.parent[max(left, right)] = min(left, right)
                anchor = min(left, right)


def build_level_groups(frame: pd.DataFrame, descriptions: pd.Series) -> dict[int, list[np.ndarray]]:
    """Reuse the exact inclusive definitions from repeat_listing_leakage."""
    comparison = frame.copy()
    comparison["_normalized_description"] = descriptions.to_numpy()
    exact_columns = [column for column in frame.columns if column != "listing_id"]
    return {
        1: group_members(comparison, exact_columns),
        2: group_members(comparison, MAJOR_COLUMNS + ["_normalized_description"]),
        3: group_members(comparison, MAJOR_COLUMNS),
    }


def compose_scenario_groups(
    frame: pd.DataFrame,
    level_groups: dict[int, list[np.ndarray]],
    levels: list[int],
    scenario: str,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Take the union of the requested pre-existing grouping relations."""
    dsu = DisjointSet(len(frame))
    for level in levels:
        for positions in level_groups[level]:
            dsu.union_group(positions)
    components: dict[int, list[int]] = {}
    for row in range(len(frame)):
        components.setdefault(dsu.find(row), []).append(row)
    repeated = [np.asarray(rows, dtype=int) for rows in components.values() if len(rows) > 1]
    repeated = sorted(repeated, key=lambda rows: (int(rows[0]), len(rows), tuple(rows.tolist())))
    repeat_id = np.full(len(frame), "", dtype=object)
    for number, rows in enumerate(repeated, 1):
        repeat_id[rows] = f"{scenario}_RG_{number:04d}"
    group_id = np.asarray(
        [
            repeat_id[row]
            if repeat_id[row]
            else f"{scenario}_UNIQUE_{int(frame.iloc[row]['listing_id'])}"
            for row in range(len(frame))
        ],
        dtype=object,
    )
    return group_id, repeat_id, repeated


def create_folds(frame: pd.DataFrame, group_id: np.ndarray):
    splitter = GroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    folds = list(splitter.split(frame, frame["price"], groups=group_id))
    fold_id = np.zeros(len(frame), dtype=int)
    for fold, (train_index, validation_index) in enumerate(folds, 1):
        fold_id[validation_index] = fold
        overlap = set(group_id[train_index]).intersection(group_id[validation_index])
        if overlap:
            raise AssertionError(f"Repeat-group leakage in fold {fold}: {sorted(overlap)[:3]}")
    if np.any(fold_id == 0):
        raise AssertionError("Every row must have exactly one validation fold.")
    return folds, fold_id


def basic_metrics(actual, predicted) -> dict:
    actual = np.asarray(actual, float)
    predicted = np.asarray(predicted, float)
    error = predicted - actual
    return {
        "count": int(len(actual)),
        "RMSE_RM": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAE_RM": float(mean_absolute_error(actual, predicted)),
        "Median_AE_RM": float(np.median(np.abs(error))),
        "R2": float(r2_score(actual, predicted)) if len(actual) > 1 else None,
    }


def complete_metrics(actual, predicted, predictors: int, thresholds: dict[str, float]) -> dict:
    result = basic_metrics(actual, predicted)
    count = len(actual)
    result["Adjusted_R2"] = float(
        1.0 - (1.0 - result["R2"]) * (count - 1) / (count - predictors - 1)
    )
    actual = np.asarray(actual, float)
    predicted = np.asarray(predicted, float)
    masks = {
        "Top5": actual >= thresholds["p95"],
        "P95_P99": (actual >= thresholds["p95"]) & (actual < thresholds["p99"]),
        "P99_P100": actual >= thresholds["p99"],
    }
    for label, mask in masks.items():
        segment = basic_metrics(actual[mask], predicted[mask])
        result[f"{label}_count"] = segment["count"]
        result[f"{label}_RMSE_RM"] = segment["RMSE_RM"]
        result[f"{label}_MAE_RM"] = segment["MAE_RM"]
    return result


def bootstrap_difference(actual, candidate, reference, seed: int) -> dict:
    """Paired row bootstrap; Position-regex minus reference, so negative is better."""
    actual = np.asarray(actual, float)
    candidate_sq = np.square(np.asarray(candidate, float) - actual)
    reference_sq = np.square(np.asarray(reference, float) - actual)
    absolute_delta = (
        np.abs(np.asarray(candidate, float) - actual)
        - np.abs(np.asarray(reference, float) - actual)
    )
    rng = np.random.default_rng(seed)
    rmse_draws = np.empty(BOOTSTRAP_DRAWS, float)
    mae_draws = np.empty(BOOTSTRAP_DRAWS, float)
    batch = 100
    for start in range(0, BOOTSTRAP_DRAWS, batch):
        count = min(batch, BOOTSTRAP_DRAWS - start)
        sampled = rng.integers(0, len(actual), size=(count, len(actual)))
        rmse_draws[start:start + count] = (
            np.sqrt(candidate_sq[sampled].mean(axis=1))
            - np.sqrt(reference_sq[sampled].mean(axis=1))
        )
        mae_draws[start:start + count] = absolute_delta[sampled].mean(axis=1)
    rmse_ci = np.quantile(rmse_draws, [0.025, 0.975])
    mae_ci = np.quantile(mae_draws, [0.025, 0.975])
    return {
        "bootstrap_samples": BOOTSTRAP_DRAWS,
        "seed": seed,
        "difference_definition": "Position-regex LightGBM minus reference; negative is better",
        "RMSE_difference_RM": float(np.sqrt(candidate_sq.mean()) - np.sqrt(reference_sq.mean())),
        "RMSE_CI95_lower_RM": float(rmse_ci[0]),
        "RMSE_CI95_upper_RM": float(rmse_ci[1]),
        "RMSE_zero_inside_CI": bool(rmse_ci[0] <= 0 <= rmse_ci[1]),
        "MAE_difference_RM": float(absolute_delta.mean()),
        "MAE_CI95_lower_RM": float(mae_ci[0]),
        "MAE_CI95_upper_RM": float(mae_ci[1]),
        "MAE_zero_inside_CI": bool(mae_ci[0] <= 0 <= mae_ci[1]),
        "fixed_oof_limitation": "Row bootstrap does not refit models or capture grouping/model-selection uncertainty.",
    }


def evaluate_scenario(
    scenario: str,
    frame: pd.DataFrame,
    descriptions: pd.Series,
    folds,
    fold_id: np.ndarray,
    group_id: np.ndarray,
    repeat_id: np.ndarray,
    thresholds: dict[str, float],
):
    X = frame[MODEL_FEATURES]
    y = frame["price"].to_numpy(float)
    regex = extract_regex_features(descriptions)
    predictions = {model: np.full(len(frame), np.nan, float) for model in MODEL_SPECS}
    seen = {model: np.zeros(len(frame), dtype=int) for model in MODEL_SPECS}
    fold_rows = []
    fit_audit = []
    for model in MODEL_SPECS:
        for fold, (train_index, validation_index) in enumerate(folds, 1):
            predicted, audit = fit_model_fold(
                model, X, y, regex, train_index, validation_index
            )
            predictions[model][validation_index] = predicted
            seen[model][validation_index] += 1
            metrics = basic_metrics(y[validation_index], predicted)
            fold_rows.append(
                {
                    "scenario": scenario,
                    "scenario_name": SCENARIOS[scenario]["name"],
                    "model": model,
                    "fold": fold,
                    "training_rows": len(train_index),
                    "validation_rows": len(validation_index),
                    "RMSE_RM": metrics["RMSE_RM"],
                    "MAE_RM": metrics["MAE_RM"],
                    "R2": metrics["R2"],
                }
            )
            fit_audit.append(
                {
                    "scenario": scenario,
                    "model": model,
                    "fold": fold,
                    "group_overlap": len(
                        set(group_id[train_index]).intersection(group_id[validation_index])
                    ),
                    **audit,
                }
            )
            print(f"Completed scenario {scenario}, {model}, fold {fold}/5.", flush=True)
        if not np.all(seen[model] == 1) or not np.all(np.isfinite(predictions[model])):
            raise AssertionError(f"Invalid OOF coverage for scenario {scenario}, model {model}.")

    metrics = {
        model: complete_metrics(y, prediction, MODEL_SPECS[model]["predictors"], thresholds)
        for model, prediction in predictions.items()
    }
    repeat_mask = repeat_id != ""
    diagnostics = []
    for model, prediction in predictions.items():
        repeat_metrics = basic_metrics(y[repeat_mask], prediction[repeat_mask])
        non_repeat_metrics = basic_metrics(y[~repeat_mask], prediction[~repeat_mask])
        diagnostics.append(
            {
                "scenario": scenario,
                "scenario_name": SCENARIOS[scenario]["name"],
                "model": model,
                "grouped_repeat_rows": int(repeat_mask.sum()),
                "grouped_repeat_percentage": float(100.0 * repeat_mask.mean()),
                "non_repeat_rows": int((~repeat_mask).sum()),
                "repeat_row_RMSE_RM": repeat_metrics["RMSE_RM"],
                "non_repeat_row_RMSE_RM": non_repeat_metrics["RMSE_RM"],
                "repeat_row_MAE_RM": repeat_metrics["MAE_RM"],
                "non_repeat_row_MAE_RM": non_repeat_metrics["MAE_RM"],
            }
        )
    oof_parts = []
    for model, prediction in predictions.items():
        oof_parts.append(
            pd.DataFrame(
                {
                    "scenario": scenario,
                    "scenario_name": SCENARIOS[scenario]["name"],
                    "model": model,
                    "row_index": np.arange(len(frame)),
                    "listing_id": frame["listing_id"].astype(int),
                    "fold": fold_id,
                    "group_id": group_id,
                    "repeat_group_id": repeat_id,
                    "is_grouped_repeat": repeat_mask,
                    "actual_price_RM": y,
                    "predicted_price_RM": prediction,
                    "residual_RM": prediction - y,
                    "absolute_error_RM": np.abs(prediction - y),
                }
            )
        )
    return predictions, metrics, fold_rows, diagnostics, pd.concat(oof_parts), fit_audit


def metric_row(scenario: str, model: str, metrics: dict, group_count: int, repeat_rows: int):
    return {
        "Grouping Scenario": SCENARIOS[scenario]["name"],
        "Scenario": scenario,
        "Model": model,
        "Repeat Groups Included": "+".join(f"Level {level}" for level in SCENARIOS[scenario]["levels"]),
        "Repeat Group Count": group_count,
        "Repeat Rows Grouped": repeat_rows,
        "RMSE": metrics["RMSE_RM"],
        "MAE": metrics["MAE_RM"],
        "R2": metrics["R2"],
        "Adjusted R2": metrics["Adjusted_R2"],
        "Median Absolute Error": metrics["Median_AE_RM"],
        "Top5 RMSE": metrics["Top5_RMSE_RM"],
        "Top5 MAE": metrics["Top5_MAE_RM"],
        "95-99% RMSE": metrics["P95_P99_RMSE_RM"],
        "99-100% RMSE": metrics["P99_P100_RMSE_RM"],
    }


def main() -> None:
    started = time.perf_counter()
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    protected_before = protected_snapshot()
    canonical_before = sha256(DATA_PATH)
    frame = pd.read_csv(DATA_PATH).reset_index(drop=True)
    if len(frame) != EXPECTED_ROWS or frame["listing_id"].nunique() != EXPECTED_ROWS:
        raise AssertionError("Canonical data must contain 3,791 distinct listings.")
    descriptions, linkage = link_descriptions(RAW_PATH, frame["listing_id"])
    level_groups = build_level_groups(frame, descriptions)
    level_counts = {
        f"level_{level}": {
            "groups": len(groups),
            "rows": int(sum(len(group) for group in groups)),
            "definition_is_inclusive": True,
        }
        for level, groups in level_groups.items()
    }
    y = frame["price"].to_numpy(float)
    thresholds = {
        "p95": float(np.quantile(y, 0.95)),
        "p99": float(np.quantile(y, 0.99)),
    }

    all_fold_rows = []
    all_oof = []
    all_diagnostics = []
    all_fit_audit = []
    scenario_metrics = {}
    scenario_predictions = {}
    scenario_summaries = {}
    sensitivity_rows = []
    bootstrap_rows = []

    for scenario, spec in SCENARIOS.items():
        group_id, repeat_id, repeated = compose_scenario_groups(
            frame, level_groups, spec["levels"], scenario
        )
        folds, fold_id = create_folds(frame, group_id)
        repeat_mask = repeat_id != ""
        crossing = int(
            pd.DataFrame({"repeat_id": repeat_id[repeat_mask], "fold": fold_id[repeat_mask]})
            .groupby("repeat_id")["fold"].nunique().gt(1).sum()
        )
        if crossing != 0:
            raise AssertionError(f"Scenario {scenario} has {crossing} crossing repeat groups.")
        assignments = pd.DataFrame(
            {
                "scenario": scenario,
                "row_index": np.arange(len(frame)),
                "listing_id": frame["listing_id"].astype(int),
                "group_id": group_id,
                "repeat_group_id": repeat_id,
                "is_grouped_repeat": repeat_mask,
                "group_size": pd.Series(group_id).map(pd.Series(group_id).value_counts()).to_numpy(int),
                "fold": fold_id,
            }
        )
        assignments.to_csv(
            EXPERIMENT / f"scenario_{scenario.lower()}_fold_assignments.csv", index=False
        )
        predictions, metrics, fold_rows, diagnostics, oof, fit_audit = evaluate_scenario(
            scenario, frame, descriptions, folds, fold_id, group_id, repeat_id, thresholds
        )
        scenario_predictions[scenario] = predictions
        scenario_metrics[scenario] = metrics
        all_fold_rows.extend(fold_rows)
        all_diagnostics.extend(diagnostics)
        all_oof.append(oof)
        all_fit_audit.extend(fit_audit)
        scenario_summaries[scenario] = {
            "name": spec["name"],
            "interpretation": spec["interpretation"],
            "levels": spec["levels"],
            "repeat_group_count": len(repeated),
            "repeat_rows_grouped": int(repeat_mask.sum()),
            "repeat_percentage": float(100.0 * repeat_mask.mean()),
            "fold_sizes": {str(fold): int(np.sum(fold_id == fold)) for fold in range(1, 6)},
            "repeat_groups_crossing_folds": crossing,
            "all_models_use_identical_folds": True,
        }
        for model, model_metrics in metrics.items():
            sensitivity_rows.append(
                metric_row(scenario, model, model_metrics, len(repeated), int(repeat_mask.sum()))
            )
        position = predictions["position_regex_lightgbm"]
        for comparison_number, reference in enumerate(
            ("random_forest", "lightgbm_interaction", "building_name_te"), 1
        ):
            result = bootstrap_difference(y, position, predictions[reference], RANDOM_STATE)
            bootstrap_rows.append(
                {
                    "scenario": scenario,
                    "scenario_name": spec["name"],
                    "candidate": "position_regex_lightgbm",
                    "reference": reference,
                    **result,
                }
            )

    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity["RMSE Rank"] = (
        sensitivity.groupby("Scenario")["RMSE"].rank(method="min").astype(int)
    )
    sensitivity["MAE Rank"] = (
        sensitivity.groupby("Scenario")["MAE"].rank(method="min").astype(int)
    )
    sensitivity = sensitivity.sort_values(["Scenario", "RMSE Rank", "MAE Rank"]).reset_index(drop=True)
    fold_metrics = pd.DataFrame(all_fold_rows).sort_values(["scenario", "model", "fold"])
    oof_predictions = pd.concat(all_oof, ignore_index=True).sort_values(
        ["scenario", "model", "row_index"]
    )
    repeat_diagnostics = pd.DataFrame(all_diagnostics).sort_values(["scenario", "model"])
    bootstrap_results = pd.DataFrame(bootstrap_rows).sort_values(["scenario", "reference"])

    sensitivity.to_csv(EXPERIMENT / "sensitivity_model_comparison.csv", index=False)
    fold_metrics.to_csv(EXPERIMENT / "fold_metrics.csv", index=False)
    oof_predictions.to_csv(EXPERIMENT / "oof_predictions.csv", index=False)
    repeat_diagnostics.to_csv(EXPERIMENT / "repeat_diagnostics.csv", index=False)
    bootstrap_results.to_csv(EXPERIMENT / "bootstrap_results.csv", index=False)

    stability = {}
    for model in MODEL_SPECS:
        values = {
            scenario: {
                "RMSE_RM": scenario_metrics[scenario][model]["RMSE_RM"],
                "MAE_RM": scenario_metrics[scenario][model]["MAE_RM"],
            }
            for scenario in SCENARIOS
        }
        stability[model] = {
            "scenarios": values,
            "B_minus_A": {
                metric: values["B"][metric] - values["A"][metric]
                for metric in ("RMSE_RM", "MAE_RM")
            },
            "C_minus_B": {
                metric: values["C"][metric] - values["B"][metric]
                for metric in ("RMSE_RM", "MAE_RM")
            },
            "C_minus_A": {
                metric: values["C"][metric] - values["A"][metric]
                for metric in ("RMSE_RM", "MAE_RM")
            },
        }

    rankings = {}
    for scenario in SCENARIOS:
        rows = sensitivity[sensitivity["Scenario"] == scenario]
        best_rmse = rows.sort_values(["RMSE", "MAE"]).iloc[0]["Model"]
        best_mae = rows.sort_values(["MAE", "RMSE"]).iloc[0]["Model"]
        rankings[scenario] = {
            "RMSE_order": rows.sort_values("RMSE")["Model"].tolist(),
            "MAE_order": rows.sort_values("MAE")["Model"].tolist(),
            "best_by_RMSE": best_rmse,
            "best_by_MAE": best_mae,
            "position_best_by_RMSE": best_rmse == "position_regex_lightgbm",
            "position_best_by_MAE": best_mae == "position_regex_lightgbm",
            "position_best_by_both": best_rmse == best_mae == "position_regex_lightgbm",
        }
    position_best_all = all(item["position_best_by_both"] for item in rankings.values())
    reliable = {
        scenario: {
            reference: {
                "RMSE": bool(
                    bootstrap_results.loc[
                        (bootstrap_results["scenario"] == scenario)
                        & (bootstrap_results["reference"] == reference),
                        "RMSE_CI95_upper_RM",
                    ].iloc[0] < 0
                ),
                "MAE": bool(
                    bootstrap_results.loc[
                        (bootstrap_results["scenario"] == scenario)
                        & (bootstrap_results["reference"] == reference),
                        "MAE_CI95_upper_RM",
                    ].iloc[0] < 0
                ),
            }
            for reference in ("random_forest", "lightgbm_interaction", "building_name_te")
        }
        for scenario in SCENARIOS
    }

    protected_after = protected_snapshot()
    canonical_after = sha256(DATA_PATH)
    if protected_before != protected_after:
        changed = sorted(set(protected_before) | set(protected_after))
        changed = [name for name in changed if protected_before.get(name) != protected_after.get(name)]
        raise AssertionError(f"Files outside the new experiment changed: {changed}")
    if canonical_before != canonical_after:
        raise AssertionError("Canonical dataset changed during the experiment.")

    level3_material = bool(
        abs(stability["position_regex_lightgbm"]["C_minus_B"]["RMSE_RM"])
        >= 0.05 * stability["position_regex_lightgbm"]["scenarios"]["B"]["RMSE_RM"]
    )
    results = {
        "question": "Does Position-regex LightGBM remain strongest as repeat-listing protection varies from minimal to conservative?",
        "dataset": {
            "path": DATA_PATH.relative_to(ROOT).as_posix(),
            "rows": len(frame),
            "unique_listing_ids": int(frame["listing_id"].nunique()),
            "sha256": canonical_after,
            "rows_deleted_as_repeats": 0,
        },
        "description_linkage": linkage,
        "grouping": {
            "source": "src/experimental_support/repeat_models.py",
            "definitions_reused_without_change": True,
            "counts_are_inclusive": True,
            "level_counts": level_counts,
            "scenarios": scenario_summaries,
        },
        "models": {
            "configurations_reused_without_retuning": True,
            "specifications": MODEL_SPECS,
            "metrics": scenario_metrics,
        },
        "stability": stability,
        "rankings": rankings,
        "bootstrap": {
            "samples": BOOTSTRAP_DRAWS,
            "paired_on_common_oof_rows": True,
            "difference_definition": "Position-regex LightGBM minus reference; negative is better",
            "statistically_reliable_advantages": reliable,
        },
        "answers": {
            "position_regex_best_by_both_in_every_scenario": position_best_all,
            "position_regex_advantage_reliable_on_both_metrics_vs_every_reference_in_every_scenario": bool(
                all(
                    metrics["RMSE"] and metrics["MAE"]
                    for scenario_values in reliable.values()
                    for metrics in scenario_values.values()
                )
            ),
            "level3_materially_changes_position_rmse_vs_scenario_b_at_5_percent_threshold": level3_material,
            "level3_interpretation": (
                "Level 3 materially changes RMSE under the pre-specified 5% threshold; it may include legitimate separate units sharing structured attributes."
                if level3_material
                else "Level 3 does not materially change Position-regex RMSE under the pre-specified 5% threshold; it may still include legitimate separate units sharing structured attributes."
            ),
            "most_defensible_reporting_scenario": "Scenario B",
            "reporting_rationale": "Scenario B protects exact and strong description-matched repeats while avoiding the broader Level-3 assumption that identical structured attributes necessarily identify the same listing/unit. Report Scenario C alongside it as a conservative sensitivity bound.",
        },
        "leakage_controls": {
            "outer_preprocessing_training_fold_only": True,
            "building_target_encoding_outer_training_only": True,
            "building_target_encoding_inner_oof": True,
            "regex_features_target_free": True,
            "validation_targets_used_for_fit_or_features": False,
            "all_models_within_scenario_use_identical_folds": True,
            "fit_audit": all_fit_audit,
        },
        "metric_thresholds": thresholds,
        "reproducibility": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "random_state": RANDOM_STATE,
            "bootstrap_samples": BOOTSTRAP_DRAWS,
        },
        "production_safety": {
            "protected_file_count": len(protected_before),
            "before_manifest_sha256": manifest_digest(protected_before),
            "after_manifest_sha256": manifest_digest(protected_after),
            "all_files_outside_new_experiment_unchanged": protected_before == protected_after,
            "canonical_dataset_unchanged": canonical_before == canonical_after,
        },
        "artifacts": [
            f"experiments/repeat_group_sensitivity/{name}"
            for name in (
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
        ],
        "runtime_seconds": time.perf_counter() - started,
    }
    (EXPERIMENT / "results.json").write_text(
        json.dumps(json_clean(results), indent=2), encoding="utf-8"
    )
    print("\n" + sensitivity.to_string(index=False), flush=True)
    print(f"\nCompleted in {results['runtime_seconds']:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
