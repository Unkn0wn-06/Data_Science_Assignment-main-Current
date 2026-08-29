"""Build all-model restricted-market trimming artifacts from frozen pipelines."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.common.features import MODEL_FEATURES
from src.models.final.description_linkage import link_descriptions
from src.models.final.final_evaluation import (
    EXPECTED_ROWS,
    FINAL_MODELS,
    PREDICTION_COLUMNS,
    PREDICTOR_COUNTS,
    load_scenario_b,
)
from src.models.final.model_builders import (
    build_standard_ppsf_estimator,
    final_tuned_params_sha256,
    fit_position_fold,
    load_final_tuned_config,
)
from src.models.final.position_regex_lightgbm import FINAL_MODEL_NAME
from src.models.final.regex_features import extract_position_features


DATA_PATH = ROOT / "data" / "processed" / "enhanced_city_dataset.csv"
RAW_PATH = ROOT / "data" / "raw" / "houses.csv"
OFFICIAL_DIR = ROOT / "results" / "final_models"
OUTPUT_DIR = ROOT / "results" / "outlier_trimming"
OFFICIAL_COMPARISON_PATH = OFFICIAL_DIR / "model_comparison.csv"
OFFICIAL_OOF_PATH = OFFICIAL_DIR / "oof_predictions.csv"
LIGHTGBM_REFERENCE_PATH = OUTPUT_DIR / "trimmed_population_comparison.csv"
SUMMARY_PATH = OUTPUT_DIR / "all_models_trimmed_market_summary.csv"
FOLD_METRICS_PATH = OUTPUT_DIR / "all_models_trimmed_market_fold_metrics.csv"
OOF_PATH = OUTPUT_DIR / "all_models_trimmed_market_oof.csv"
METADATA_PATH = OUTPUT_DIR / "all_models_trimmed_market_metadata.json"
TRIM_LEVELS = (
    ("0%", 0.0),
    ("0.5%", 0.5),
    ("1%", 1.0),
    ("2.5%", 2.5),
    ("5%", 5.0),
    ("10%", 10.0),
)
PROTECTED_PATHS = (
    DATA_PATH,
    OFFICIAL_DIR / "model_comparison.csv",
    OFFICIAL_DIR / "model_comparison.json",
    OFFICIAL_DIR / "oof_predictions.csv",
    OFFICIAL_DIR / "fold_metrics.csv",
    OFFICIAL_DIR / "feature_importance.csv",
    OUTPUT_DIR / "training_only_comparison.csv",
    OUTPUT_DIR / "trimmed_population_comparison.csv",
    OUTPUT_DIR / "distribution_shift.csv",
    OUTPUT_DIR / "bootstrap_results.csv",
    OUTPUT_DIR / "segment_metrics.csv",
    OUTPUT_DIR / "retained_cv_summary.csv",
    OUTPUT_DIR / "metadata.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_manifest() -> dict[str, str]:
    missing = [path for path in PROTECTED_PATHS if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Protected input artifacts are missing: {missing}")
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in PROTECTED_PATHS}


def regression_metrics(actual, predicted, predictors: int) -> dict[str, float]:
    """Use the official final-evaluation adjusted-R2 definition."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    r2 = float(r2_score(actual, predicted))
    adjusted_r2 = float(
        1.0
        - (1.0 - r2)
        * (len(actual) - 1)
        / (len(actual) - predictors - 1)
    )
    return {
        "RMSE_RM": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAE_RM": float(mean_absolute_error(actual, predicted)),
        "R2": r2,
        "Adjusted_R2": adjusted_r2,
    }


def load_inputs():
    data = pd.read_csv(DATA_PATH).reset_index(drop=True)
    assignments, folds_list = load_scenario_b(data)
    folds = {fold: positions for fold, positions in enumerate(folds_list, 1)}
    official_oof = pd.read_csv(OFFICIAL_OOF_PATH).reset_index(drop=True)
    official_comparison = pd.read_csv(OFFICIAL_COMPARISON_PATH)
    expected_ids = data["listing_id"].astype(int).to_numpy()
    if not np.array_equal(official_oof["listing_id"].astype(int), expected_ids):
        raise AssertionError("Official OOF rows do not align with the canonical listings.")
    if not np.array_equal(
        official_oof["scenario_b_fold"].astype(int), assignments["fold"].astype(int)
    ):
        raise AssertionError("Official OOF rows do not use the saved Scenario B folds.")
    if set(official_comparison["Model"]) != set(FINAL_MODELS):
        raise AssertionError("Official comparison does not contain the frozen final four models.")
    descriptions, _ = link_descriptions(RAW_PATH, data["listing_id"])
    position = extract_position_features(descriptions)
    return data, assignments, folds, official_oof, official_comparison, position


def retained_populations(price: np.ndarray) -> dict[str, dict[str, object]]:
    """Calculate each retained population once for reuse by all four models."""
    populations: dict[str, dict[str, object]] = {}
    for label, removal_percent in TRIM_LEVELS:
        cutoff = (
            np.nan
            if removal_percent == 0.0
            else float(np.quantile(price, 1.0 - removal_percent / 100.0))
        )
        retained = (
            np.arange(len(price), dtype=int)
            if removal_percent == 0.0
            else np.flatnonzero(price <= cutoff)
        )
        mask = np.zeros(len(price), dtype=bool)
        mask[retained] = True
        populations[label] = {
            "removal_percent": removal_percent,
            "cutoff": cutoff,
            "positions": retained,
            "mask": mask,
        }
    return populations


def fit_fold(
    model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    position: pd.DataFrame,
    training: np.ndarray,
    validation: np.ndarray,
) -> np.ndarray:
    if model_name == FINAL_MODEL_NAME:
        return fit_position_fold(
            X.iloc[training],
            y[training],
            X.iloc[validation],
            position.iloc[training],
            position.iloc[validation],
        )
    fitted = clone(build_standard_ppsf_estimator(model_name)).fit(
        X.iloc[training], y[training]
    )
    return np.asarray(fitted.predict(X.iloc[validation]), dtype=float)


def build_results():
    data, assignments, folds, official_oof, official_comparison, position = load_inputs()
    X = data[MODEL_FEATURES]
    y = data["price"].to_numpy(float)
    listing_ids = data["listing_id"].astype(int).to_numpy()
    fold_values = assignments["fold"].astype(int).to_numpy()
    populations = retained_populations(y)
    summary_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    oof_parts: list[pd.DataFrame] = []

    for label, removal_percent in TRIM_LEVELS:
        population = populations[label]
        retained = np.asarray(population["positions"], dtype=int)
        retained_mask = np.asarray(population["mask"], dtype=bool)
        for model_name in FINAL_MODELS:
            prediction = np.full(EXPECTED_ROWS, np.nan)
            coverage = np.zeros(EXPECTED_ROWS, dtype=int)
            frozen = official_oof[PREDICTION_COLUMNS[model_name]].to_numpy(float)
            for fold, (outer_training, outer_validation) in folds.items():
                training = outer_training[retained_mask[outer_training]]
                validation = outer_validation[retained_mask[outer_validation]]
                if removal_percent == 0.0:
                    fold_prediction = frozen[validation]
                else:
                    fold_prediction = fit_fold(
                        model_name, X, y, position, training, validation
                    )
                prediction[validation] = fold_prediction
                coverage[validation] += 1
                fold_metric = regression_metrics(
                    y[validation], fold_prediction, PREDICTOR_COUNTS[model_name]
                )
                fold_rows.append(
                    {
                        "Model": model_name,
                        "Trim_Level": label,
                        "Removal_Percent": removal_percent,
                        "Fold": fold,
                        "Training_Rows": len(training),
                        "Validation_Rows": len(validation),
                        **fold_metric,
                    }
                )
                print(
                    f"Completed {model_name}, trim {label}, Scenario B fold {fold}/5.",
                    flush=True,
                )
            if not np.all(coverage[retained] == 1):
                raise AssertionError(f"Incomplete OOF coverage for {model_name}, trim {label}.")
            if np.any(coverage[~retained_mask] != 0):
                raise AssertionError(f"Excluded rows received predictions for {model_name}, trim {label}.")
            retained_prediction = prediction[retained]
            if not np.isfinite(retained_prediction).all():
                raise AssertionError(f"Non-finite OOF prediction for {model_name}, trim {label}.")
            overall = regression_metrics(
                y[retained], retained_prediction, PREDICTOR_COUNTS[model_name]
            )
            summary_rows.append(
                {
                    "Model": model_name,
                    "Trim_Level": label,
                    "Removal_Percent": removal_percent,
                    "Original_Rows": EXPECTED_ROWS,
                    "Retained_Rows": len(retained),
                    "Removed_Rows": EXPECTED_ROWS - len(retained),
                    "Retention_Percentage": 100.0 * len(retained) / EXPECTED_ROWS,
                    **overall,
                }
            )
            oof_parts.append(
                pd.DataFrame(
                    {
                        "Model": model_name,
                        "Trim_Level": label,
                        "Removal_Percent": removal_percent,
                        "listing_id": listing_ids[retained],
                        "scenario_b_fold": fold_values[retained],
                        "actual_price_RM": y[retained],
                        "predicted_price_RM": retained_prediction,
                    }
                )
            )

    summary = pd.DataFrame(summary_rows)
    fold_metrics = pd.DataFrame(fold_rows)
    oof = pd.concat(oof_parts, ignore_index=True)
    validate_results(summary, fold_metrics, oof, official_comparison, assignments)
    return summary, fold_metrics, oof


def _assert_close(label: str, actual, expected, *, atol: float) -> None:
    if not np.allclose(actual, expected, rtol=1e-10, atol=atol):
        maximum = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
        raise AssertionError(f"{label} does not reproduce its reference; max delta={maximum}.")


def validate_results(
    summary: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    oof: pd.DataFrame,
    official_comparison: pd.DataFrame,
    assignments: pd.DataFrame,
) -> None:
    expected_levels = [label for label, _ in TRIM_LEVELS]
    if len(summary) != 24 or set(summary["Model"]) != set(FINAL_MODELS):
        raise AssertionError("Summary must contain exactly 24 rows and the four final models.")
    if summary.duplicated(["Model", "Trim_Level"]).any():
        raise AssertionError("Summary contains duplicate model/trim pairs.")
    level_counts = summary.groupby("Model")["Trim_Level"].nunique()
    if not (level_counts == 6).all():
        raise AssertionError("Every model must contain all six trim levels.")
    for model_name in FINAL_MODELS:
        levels = summary.loc[summary["Model"].eq(model_name), "Trim_Level"].tolist()
        if levels != expected_levels:
            raise AssertionError(f"Unexpected trim order for {model_name}: {levels}")
    retained_by_level = summary.groupby("Trim_Level", sort=False)["Retained_Rows"]
    if not (retained_by_level.nunique() == 1).all():
        raise AssertionError("Models did not use identical retained counts at each trim level.")
    metric_columns = ["RMSE_RM", "MAE_RM", "R2", "Adjusted_R2"]
    if not np.isfinite(summary[metric_columns].to_numpy(float)).all():
        raise AssertionError("Summary contains non-finite metrics.")
    recomputed_adjusted = 1.0 - (1.0 - summary["R2"]) * (
        summary["Retained_Rows"] - 1
    ) / (
        summary["Retained_Rows"]
        - summary["Model"].map(PREDICTOR_COUNTS)
        - 1
    )
    _assert_close(
        "Adjusted R2", summary["Adjusted_R2"], recomputed_adjusted, atol=1e-12
    )

    zero = summary[summary["Removal_Percent"].eq(0.0)].set_index("Model")
    official = official_comparison.set_index("Model")
    for metric in metric_columns:
        tolerance = 1e-6 if metric in {"RMSE_RM", "MAE_RM"} else 1e-12
        _assert_close(
            f"0% {metric}",
            zero.loc[list(FINAL_MODELS), metric],
            official.loc[list(FINAL_MODELS), metric],
            atol=tolerance,
        )

    reference = pd.read_csv(LIGHTGBM_REFERENCE_PATH)
    lightgbm = summary[summary["Model"].eq(FINAL_MODEL_NAME)].merge(
        reference,
        on="Removal_Percent",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_Reference"),
    )
    if len(lightgbm) != 6:
        raise AssertionError("LightGBM reference does not contain all six trim levels.")
    _assert_close("LightGBM retained rows", lightgbm["Retained_Rows"], lightgbm["Retained_OOF_Rows"], atol=0.0)
    for current, saved, tolerance in (
        ("RMSE_RM", "Matched_Retrained_RMSE_RM", 1e-6),
        ("MAE_RM", "Matched_Retrained_MAE_RM", 1e-6),
        ("R2", "Retrained_R2", 1e-12),
        ("Adjusted_R2", "Retrained_Adjusted_R2", 1e-12),
    ):
        _assert_close(f"LightGBM {current}", lightgbm[current], lightgbm[saved], atol=tolerance)

    if len(fold_metrics) != 120 or fold_metrics.duplicated(
        ["Model", "Trim_Level", "Fold"]
    ).any():
        raise AssertionError("Fold metrics must contain 120 unique model/trim/fold rows.")
    for (model_name, trim_level), rows in oof.groupby(["Model", "Trim_Level"], sort=False):
        if rows["listing_id"].duplicated().any():
            raise AssertionError(f"Duplicate retained OOF listing for {model_name}, {trim_level}.")
        expected = int(
            summary.loc[
                summary["Model"].eq(model_name)
                & summary["Trim_Level"].eq(trim_level),
                "Retained_Rows",
            ].iloc[0]
        )
        if len(rows) != expected:
            raise AssertionError(f"OOF coverage count mismatch for {model_name}, {trim_level}.")
    for trim_level, rows in oof.groupby("Trim_Level", sort=False):
        id_sets = [
            set(rows.loc[rows["Model"].eq(model_name), "listing_id"].astype(int))
            for model_name in FINAL_MODELS
        ]
        if any(ids != id_sets[0] for ids in id_sets[1:]):
            raise AssertionError(f"Models used different retained IDs at {trim_level}.")
    repeated = assignments[assignments["is_grouped_repeat"]]
    if repeated.groupby("repeat_group_id")["fold"].nunique().gt(1).any():
        raise AssertionError("A Scenario B repeat group crosses folds.")


def run_experiment():
    before = protected_manifest()
    summary, fold_metrics, oof = build_results()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    fold_metrics.to_csv(FOLD_METRICS_PATH, index=False)
    oof.to_csv(OOF_PATH, index=False)
    tuned_config = load_final_tuned_config()
    metadata = {
        "purpose": "All-model restricted-market Scenario B OOF comparison",
        "model_configuration_policy": (
            "The selected tuned configuration for each model is frozen and reused "
            "unchanged at every trimming level."
        ),
        "tuned_config_sha256": final_tuned_params_sha256(),
        "selected_parameters": tuned_config["models"],
        "trim_levels": [label for label, _ in TRIM_LEVELS],
        "folds": 5,
        "random_seed": 42,
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    after = protected_manifest()
    if before != after:
        changed = sorted(path for path in before if before[path] != after[path])
        raise AssertionError(f"Protected official/existing trimming files changed: {changed}")
    return summary, fold_metrics, oof


def main() -> None:
    summary, fold_metrics, oof = run_experiment()
    print("\nALL-MODEL TRIMMED-MARKET SUMMARY")
    print(summary.to_string(index=False))
    print(
        f"\nSaved {len(summary)} summary rows, {len(fold_metrics)} fold rows, "
        f"and {len(oof)} retained-market OOF rows."
    )


if __name__ == "__main__":
    main()
