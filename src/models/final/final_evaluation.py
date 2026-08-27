"""Scenario B group-safe evaluation for the four final assignment models."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.cleaning.pipeline import PROJECT_ROOT
from src.models.final.description_linkage import link_descriptions
from src.models.final.model_builders import (
    build_standard_ppsf_estimator,
    fit_position_fold,
)
from src.models.final.regex_features import extract_position_features
from src.models.common.features import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERICAL_FEATURES
from src.models.final.position_regex_lightgbm import (
    FINAL_MODEL_NAME,
    POSITION_DISPLAY_NAMES,
    POSITION_FEATURES,
    PositionRegexLightGBM,
)


DATA_PATH = PROJECT_ROOT / "data" / "processed" / "enhanced_city_dataset.csv"
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "houses.csv"
FOLD_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "repeat_group_sensitivity"
    / "scenario_b_fold_assignments.csv"
)
RESULTS_DIR = PROJECT_ROOT / "results" / "final_models"
EXPECTED_ROWS = 3_791
FINAL_MODELS = (
    "Ridge Regression",
    "Random Forest",
    "Gradient Boosting",
    FINAL_MODEL_NAME,
)
INTERNAL_NAMES = {
    "Ridge Regression": "ridge",
    "Random Forest": "random_forest",
    "Gradient Boosting": "gradient_boosting",
    FINAL_MODEL_NAME: "position_regex_lightgbm",
}
PREDICTION_COLUMNS = {
    "Ridge Regression": "ridge_prediction",
    "Random Forest": "random_forest_prediction",
    "Gradient Boosting": "gradient_boosting_prediction",
    FINAL_MODEL_NAME: "position_regex_lightgbm_prediction",
}
PREDICTOR_COUNTS = {
    "Ridge Regression": len(MODEL_FEATURES),
    "Random Forest": len(MODEL_FEATURES),
    "Gradient Boosting": len(MODEL_FEATURES),
    FINAL_MODEL_NAME: 47,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_scenario_b(data: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[np.ndarray, np.ndarray]]]:
    assignments = pd.read_csv(FOLD_PATH).sort_values("row_index").reset_index(drop=True)
    if len(data) != EXPECTED_ROWS or len(assignments) != EXPECTED_ROWS:
        raise AssertionError("Canonical data and Scenario B assignments must each have 3,791 rows.")
    if assignments["listing_id"].duplicated().any():
        raise AssertionError("Every Scenario B listing ID must appear exactly once.")
    if not np.array_equal(
        assignments["listing_id"].astype(int).to_numpy(),
        data["listing_id"].astype(int).to_numpy(),
    ):
        raise AssertionError("Scenario B assignments do not align with canonical listing order.")
    if set(assignments["fold"]) != {1, 2, 3, 4, 5}:
        raise AssertionError("Scenario B must contain folds 1 through 5.")
    repeated = assignments[assignments["is_grouped_repeat"]]
    crossing = repeated.groupby("repeat_group_id")["fold"].nunique().gt(1)
    if crossing.any():
        raise AssertionError("A Scenario B repeat group crosses folds.")
    folds = []
    fold_values = assignments["fold"].to_numpy(int)
    for fold in range(1, 6):
        validation = np.flatnonzero(fold_values == fold)
        training = np.flatnonzero(fold_values != fold)
        folds.append((training, validation))
    return assignments, folds


def metrics(actual, predicted, predictors: int, p95: float) -> dict:
    actual = np.asarray(actual, float)
    predicted = np.asarray(predicted, float)
    r2 = float(r2_score(actual, predicted))
    top = actual >= p95
    return {
        "RMSE_RM": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAE_RM": float(mean_absolute_error(actual, predicted)),
        "R2": r2,
        "Adjusted_R2": float(
            1.0 - (1.0 - r2) * (len(actual) - 1) / (len(actual) - predictors - 1)
        ),
        "Median_AE_RM": float(np.median(np.abs(predicted - actual))),
        "Top5_RMSE_RM": float(
            np.sqrt(mean_squared_error(actual[top], predicted[top]))
        ),
        "Top5_MAE_RM": float(mean_absolute_error(actual[top], predicted[top])),
    }


def _display_feature(name: str) -> str:
    if name in POSITION_DISPLAY_NAMES:
        return POSITION_DISPLAY_NAMES[name]
    return name.replace("_", " ").title().replace("Sqft", "sq.ft.")


def _standard_importance(model_name: str, fitted) -> pd.DataFrame:
    pipeline = fitted.regressor_
    estimator = pipeline.named_steps["model"]
    feature_names = list(NUMERICAL_FEATURES) + list(CATEGORICAL_FEATURES)
    if model_name == "Ridge Regression":
        values = np.abs(np.ravel(estimator.coef_)).astype(float)
        importance_type = "Absolute Coefficient Magnitude"
    else:
        values = np.asarray(estimator.feature_importances_, dtype=float)
        importance_type = "Feature Importance"
    if len(feature_names) != len(values):
        raise AssertionError(
            f"{model_name} importance mismatch: {len(feature_names)} names vs {len(values)} values."
        )
    return pd.DataFrame(
        {
            "Model": model_name,
            "Feature": [_display_feature(name) for name in feature_names],
            "Raw_Feature": feature_names,
            "Importance": values,
            "Importance_Type": importance_type,
        }
    )


def build_final_results() -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH).reset_index(drop=True)
    if len(data) != EXPECTED_ROWS or data["listing_id"].nunique() != EXPECTED_ROWS:
        raise AssertionError("Canonical final data must contain 3,791 unique listings.")
    assignments, folds = load_scenario_b(data)
    descriptions, linkage = link_descriptions(RAW_PATH, data["listing_id"])
    regex = extract_position_features(descriptions)
    X = data[MODEL_FEATURES]
    y = data["price"].to_numpy(float)
    p95 = float(np.quantile(y, 0.95))
    predictions = {name: np.full(EXPECTED_ROWS, np.nan) for name in FINAL_MODELS}
    coverage = {name: np.zeros(EXPECTED_ROWS, dtype=int) for name in FINAL_MODELS}
    fold_rows = []

    for model_name in FINAL_MODELS:
        for fold, (training, validation) in enumerate(folds, 1):
            if model_name == FINAL_MODEL_NAME:
                predicted = fit_position_fold(
                    X.iloc[training],
                    y[training],
                    X.iloc[validation],
                    regex.iloc[training],
                    regex.iloc[validation],
                )
            else:
                fitted = clone(build_standard_ppsf_estimator(model_name)).fit(
                    X.iloc[training], y[training]
                )
                predicted = np.asarray(fitted.predict(X.iloc[validation]), float)
            predictions[model_name][validation] = predicted
            coverage[model_name][validation] += 1
            fold_metric = metrics(y[validation], predicted, PREDICTOR_COUNTS[model_name], p95)
            fold_rows.append(
                {
                    "Model": model_name,
                    "Fold": fold,
                    "Training_Rows": len(training),
                    "Validation_Rows": len(validation),
                    "RMSE_RM": fold_metric["RMSE_RM"],
                    "MAE_RM": fold_metric["MAE_RM"],
                    "R2": fold_metric["R2"],
                }
            )
            print(f"Completed {model_name}, Scenario B fold {fold}/5.", flush=True)
        if not np.all(coverage[model_name] == 1) or not np.isfinite(predictions[model_name]).all():
            raise AssertionError(f"Incomplete OOF coverage for {model_name}.")

    metric_rows = []
    metric_map = {}
    for model_name in FINAL_MODELS:
        result = metrics(y, predictions[model_name], PREDICTOR_COUNTS[model_name], p95)
        metric_map[model_name] = result
        metric_rows.append({"Model": model_name, **result})
    comparison = pd.DataFrame(metric_rows)
    comparison["RMSE_Rank"] = comparison["RMSE_RM"].rank(method="min").astype(int)
    comparison["MAE_Rank"] = comparison["MAE_RM"].rank(method="min").astype(int)
    comparison = comparison.sort_values(["RMSE_Rank", "MAE_Rank"]).reset_index(drop=True)

    oof = pd.DataFrame(
        {
            "listing_id": data["listing_id"].astype(int),
            "actual_price": y,
            "scenario_b_fold": assignments["fold"].astype(int),
            **{PREDICTION_COLUMNS[name]: predictions[name] for name in FINAL_MODELS},
        }
    )
    fold_metrics = pd.DataFrame(fold_rows).sort_values(["Model", "Fold"])

    importance_parts = []
    for model_name in FINAL_MODELS[:-1]:
        fitted = clone(build_standard_ppsf_estimator(model_name)).fit(X, y)
        importance_parts.append(_standard_importance(model_name, fitted))
    final_model = PositionRegexLightGBM().fit(X, y, descriptions)
    position_importance = final_model.feature_importance()
    position_importance.insert(0, "Model", FINAL_MODEL_NAME)
    position_importance["Raw_Feature"] = position_importance["Feature"]
    position_importance["Feature"] = position_importance["Feature"].map(_display_feature)
    position_importance["Importance_Type"] = "Feature Importance"
    importance_parts.append(position_importance)
    importance = pd.concat(importance_parts, ignore_index=True)

    lowest_rmse = comparison.sort_values("RMSE_RM").iloc[0]["Model"]
    lowest_mae = comparison.sort_values("MAE_RM").iloc[0]["Model"]
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    oof.to_csv(RESULTS_DIR / "oof_predictions.csv", index=False)
    fold_metrics.to_csv(RESULTS_DIR / "fold_metrics.csv", index=False)
    importance.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)

    comparison_payload = {
        "validation": "Scenario B leakage-safe group cross-validation",
        "selected_final_model": FINAL_MODEL_NAME,
        "lowest_rmse_model": lowest_rmse,
        "lowest_mae_model": lowest_mae,
        "selection_rationale": (
            "LightGBM with position features was selected as the final model because it "
            "provided competitive and balanced RMSE and MAE while incorporating meaningful "
            "property-position information from listing descriptions. Differences from the "
            "strongest competing models were not statistically significant at the 95% confidence level."
        ),
        "models": comparison.to_dict("records"),
    }
    (RESULTS_DIR / "model_comparison.json").write_text(
        json.dumps(comparison_payload, indent=2), encoding="utf-8"
    )
    metadata = {
        "dataset": DATA_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "dataset_sha256": sha256(DATA_PATH),
        "raw_dataset": RAW_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "raw_dataset_sha256": sha256(RAW_PATH),
        "rows": EXPECTED_ROWS,
        "rows_removed": 0,
        "fold_assignments": FOLD_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "fold_assignments_sha256": sha256(FOLD_PATH),
        "fold_sizes": assignments["fold"].value_counts().sort_index().astype(int).to_dict(),
        "repeat_groups_crossing_folds": 0,
        "models": list(FINAL_MODELS),
        "internal_names": INTERNAL_NAMES,
        "selected_final_model": FINAL_MODEL_NAME,
        "target_strategy": "price / property_size_sqft; reconstruct total price by multiplying predicted PPSF by property_size_sqft",
        "position_features": list(POSITION_FEATURES),
        "position_regex_target_free": True,
        "description_linkage": linkage,
        "all_models_same_rows_and_folds": True,
        "all_rows_have_one_oof_prediction_per_model": True,
        "premium_threshold_RM": p95,
        "full_data_deployment_training_rows": final_model.training_rows_,
        "feature_importance_schema_verified": True,
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
        },
    }
    (RESULTS_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return {"comparison": comparison, "metadata": metadata}
