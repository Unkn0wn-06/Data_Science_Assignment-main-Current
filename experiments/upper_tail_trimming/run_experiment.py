"""Evaluate upper-price-tail trimming without changing the production workflow."""

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
from src.models.final.model_builders import (
    build_standard_ppsf_estimator,
    fit_position_fold,
)
from src.models.final.regex_features import extract_position_features


EXPERIMENT = ROOT / "experiments" / "upper_tail_trimming"
DATA_PATH = ROOT / "data" / "processed" / "enhanced_city_dataset.csv"
RAW_PATH = ROOT / "data" / "raw" / "houses.csv"
FOLD_PATH = (
    ROOT
    / "experiments"
    / "repeat_group_sensitivity"
    / "scenario_b_fold_assignments.csv"
)
FINAL_OOF_PATH = ROOT / "results" / "final_models" / "oof_predictions.csv"
EXPECTED_ROWS = 3_791
BOOTSTRAP_SAMPLES = 5_000
TRIM_LEVELS = (
    ("A", 0.0),
    ("B", 0.5),
    ("C", 1.0),
    ("D", 2.5),
    ("E", 5.0),
    ("F", 10.0),
)
MODELS = ("LightGBM + Position Features", "Random Forest")
BASELINE_COLUMNS = {
    "LightGBM + Position Features": "position_regex_lightgbm_prediction",
    "Random Forest": "random_forest_prediction",
}
PREDICTOR_COUNTS = {
    "LightGBM + Position Features": 47,
    "Random Forest": 32,
}
PROTECTED_FILES = (
    ROOT / "data" / "processed" / "enhanced_city_dataset.csv",
    ROOT / "data" / "raw" / "houses.csv",
    ROOT / "app.py",
)
PROTECTED_DIRECTORIES = (
    ROOT / "results" / "final_models",
    ROOT / "prototype",
    ROOT / "archive",
    ROOT / "src" / "models" / "final",
)
MODEL_PARAMETERS = {
    "LightGBM + Position Features": {
        "n_estimators": 1000,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
        "objective": "regression",
    },
    "Random Forest": {
        "n_estimators": 700,
        "min_samples_split": 6,
        "min_samples_leaf": 3,
        "max_features": 0.7,
        "max_depth": 24,
        "criterion": "squared_error",
        "bootstrap": True,
        "random_state": 42,
        "n_jobs": -1,
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_generated(path: Path) -> bool:
    return (
        "__pycache__" in path.parts
        or ".ipynb_checkpoints" in path.parts
        or path.suffix == ".pyc"
        or path.name in {".DS_Store", "Thumbs.db"}
    )


def protected_manifest() -> tuple[dict[str, str], str]:
    files = list(PROTECTED_FILES)
    for directory in PROTECTED_DIRECTORIES:
        files.extend(
            path
            for path in directory.rglob("*")
            if path.is_file() and not _is_generated(path)
        )
    manifest = {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in sorted(set(files))
    }
    payload = "\n".join(f"{name}:{digest}" for name, digest in manifest.items())
    return manifest, hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_inputs():
    data = pd.read_csv(DATA_PATH).reset_index(drop=True)
    assignments = pd.read_csv(FOLD_PATH).sort_values("row_index").reset_index(drop=True)
    baseline = pd.read_csv(FINAL_OOF_PATH).reset_index(drop=True)
    if len(data) != EXPECTED_ROWS or data["listing_id"].nunique() != EXPECTED_ROWS:
        raise AssertionError("Canonical data must contain 3,791 unique listings.")
    if len(assignments) != EXPECTED_ROWS or assignments["listing_id"].duplicated().any():
        raise AssertionError("Scenario B assignments must cover every listing once.")
    expected_ids = data["listing_id"].astype(int).to_numpy()
    if not np.array_equal(assignments["listing_id"].astype(int), expected_ids):
        raise AssertionError("Scenario B assignments do not match canonical row order.")
    if not np.array_equal(baseline["listing_id"].astype(int), expected_ids):
        raise AssertionError("Frozen OOF predictions do not match canonical row order.")
    if not np.array_equal(
        baseline["scenario_b_fold"].astype(int), assignments["fold"].astype(int)
    ):
        raise AssertionError("Frozen predictions do not use the Scenario B folds.")
    if set(assignments["fold"]) != {1, 2, 3, 4, 5}:
        raise AssertionError("Scenario B must contain folds 1 through 5.")
    repeated = assignments[assignments["is_grouped_repeat"]]
    if repeated.groupby("repeat_group_id")["fold"].nunique().gt(1).any():
        raise AssertionError("A protected repeat group crosses Scenario B folds.")
    descriptions, linkage = link_descriptions(RAW_PATH, data["listing_id"])
    position = extract_position_features(descriptions)
    return data, assignments, baseline, position, linkage


def regression_metrics(actual, predicted, predictors: int) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    r2 = float(r2_score(actual, predicted))
    return {
        "RMSE_RM": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAE_RM": float(mean_absolute_error(actual, predicted)),
        "R2": r2,
        "Adjusted_R2": float(
            1.0
            - (1.0 - r2)
            * (len(actual) - 1)
            / (len(actual) - predictors - 1)
        ),
        "Median_AE_RM": float(np.median(np.abs(predicted - actual))),
    }


def segment_definitions(price: np.ndarray) -> tuple[dict[str, np.ndarray], float, float]:
    p95 = float(np.quantile(price, 0.95))
    p99 = float(np.quantile(price, 0.99))
    return {
        "Remaining 95%": price < p95,
        "Top 5%": price >= p95,
        "95-99%": (price >= p95) & (price < p99),
        "99-100%": price >= p99,
    }, p95, p99


def segment_metric(actual, predicted, mask) -> dict:
    selected_actual = np.asarray(actual, float)[mask]
    selected_prediction = np.asarray(predicted, float)[mask]
    error = selected_prediction - selected_actual
    return {
        "Rows": int(mask.sum()),
        "RMSE_RM": float(np.sqrt(np.mean(np.square(error)))),
        "MAE_RM": float(np.mean(np.abs(error))),
        "Mean_Prediction_Error_RM": float(np.mean(error)),
        "Underprediction_Pct": float(np.mean(error < 0) * 100.0),
    }


def fold_positions(assignments: pd.DataFrame):
    values = assignments["fold"].to_numpy(int)
    return {
        fold: (
            np.flatnonzero(values != fold),
            np.flatnonzero(values == fold),
        )
        for fold in range(1, 6)
    }


def build_training_cutoffs(price: np.ndarray, folds) -> tuple[pd.DataFrame, dict]:
    rows = []
    lookup = {}
    for level, removal_percent in TRIM_LEVELS:
        for fold, (training, validation) in folds.items():
            training_price = price[training]
            if removal_percent == 0:
                cutoff = np.nan
                retained = training
            else:
                cutoff = float(
                    np.quantile(training_price, 1.0 - removal_percent / 100.0)
                )
                retained = training[training_price <= cutoff]
            removed = len(training) - len(retained)
            lookup[(level, fold)] = {
                "training": training,
                "validation": validation,
                "retained": retained,
                "cutoff": cutoff,
            }
            rows.append(
                {
                    "Trim_Level": level,
                    "Removal_Percent": removal_percent,
                    "Fold": fold,
                    "Threshold_Quantile": (
                        np.nan if removal_percent == 0 else 1.0 - removal_percent / 100.0
                    ),
                    "Cutoff_Source": "outer_training_fold_prices_only",
                    "Training_Rows_Before": len(training),
                    "Training_Rows_Removed": removed,
                    "Training_Rows_Retained": len(retained),
                    "Actual_Removal_Pct": removed / len(training) * 100.0,
                    "Training_Derived_Cutoff_RM": cutoff,
                    "Maximum_Retained_Training_Price_RM": float(price[retained].max()),
                    "Validation_Rows": len(validation),
                    "Validation_Rows_Removed": 0,
                }
            )
    return pd.DataFrame(rows), lookup


def fit_model(
    model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    position: pd.DataFrame,
    training: np.ndarray,
    validation: np.ndarray,
) -> np.ndarray:
    if model_name == "LightGBM + Position Features":
        return fit_position_fold(
            X.iloc[training],
            y[training],
            X.iloc[validation],
            position.iloc[training],
            position.iloc[validation],
        )
    fitted = clone(build_standard_ppsf_estimator("Random Forest")).fit(
        X.iloc[training], y[training]
    )
    return np.asarray(fitted.predict(X.iloc[validation]), dtype=float)


def run_training_only(
    data,
    assignments,
    baseline,
    position,
    cutoffs,
    cutoff_lookup,
    folds,
):
    X = data[MODEL_FEATURES]
    y = data["price"].to_numpy(float)
    listing_ids = data["listing_id"].astype(int).to_numpy()
    fold_values = assignments["fold"].to_numpy(int)
    segments, _, _ = segment_definitions(y)
    predictions_by_variant = {}
    oof_parts = []
    fold_rows = []
    segment_rows = []
    comparison_rows = []

    for model_name in MODELS:
        frozen_prediction = baseline[BASELINE_COLUMNS[model_name]].to_numpy(float)
        for level, removal_percent in TRIM_LEVELS:
            prediction = np.full(EXPECTED_ROWS, np.nan)
            coverage = np.zeros(EXPECTED_ROWS, dtype=int)
            for fold, (_, validation) in folds.items():
                spec = cutoff_lookup[(level, fold)]
                if removal_percent == 0:
                    fold_prediction = frozen_prediction[validation]
                else:
                    fold_prediction = fit_model(
                        model_name,
                        X,
                        y,
                        position,
                        spec["retained"],
                        validation,
                    )
                prediction[validation] = fold_prediction
                coverage[validation] += 1
                fold_result = regression_metrics(
                    y[validation], fold_prediction, PREDICTOR_COUNTS[model_name]
                )
                fold_rows.append(
                    {
                        "Experiment_Type": "training_only",
                        "Model": model_name,
                        "Trim_Level": level,
                        "Removal_Percent": removal_percent,
                        "Fold": fold,
                        **{
                            key: value
                            for key, value in cutoffs[
                                (cutoffs["Trim_Level"] == level)
                                & (cutoffs["Fold"] == fold)
                            ].iloc[0].to_dict().items()
                            if key
                            in {
                                "Training_Rows_Before",
                                "Training_Rows_Removed",
                                "Training_Rows_Retained",
                                "Actual_Removal_Pct",
                                "Training_Derived_Cutoff_RM",
                                "Maximum_Retained_Training_Price_RM",
                                "Validation_Rows",
                                "Validation_Rows_Removed",
                            }
                        },
                        **fold_result,
                    }
                )
                print(
                    f"Training-only: {model_name}, trim {level}, fold {fold}/5",
                    flush=True,
                )
            if not np.all(coverage == 1) or not np.isfinite(prediction).all():
                raise AssertionError(f"Incomplete training-only OOF coverage: {model_name} {level}")
            predictions_by_variant[(model_name, level)] = prediction
            oof_parts.append(
                pd.DataFrame(
                    {
                        "Experiment_Type": "training_only",
                        "Model": model_name,
                        "Trim_Level": level,
                        "Removal_Percent": removal_percent,
                        "listing_id": listing_ids,
                        "scenario_b_fold": fold_values,
                        "actual_price_RM": y,
                        "predicted_price_RM": prediction,
                        "matched_original_prediction_RM": frozen_prediction,
                        "Threshold_Scope": "outer_training_fold_only",
                    }
                )
            )
            overall = regression_metrics(y, prediction, PREDICTOR_COUNTS[model_name])
            cutoff_subset = cutoffs[cutoffs["Trim_Level"] == level]
            comparison_rows.append(
                {
                    "Model": model_name,
                    "Trim_Level": level,
                    "Removal_Percent": removal_percent,
                    "Mean_Training_Rows_Removed": float(
                        cutoff_subset["Training_Rows_Removed"].mean()
                    ),
                    "Mean_Training_Rows_Retained": float(
                        cutoff_subset["Training_Rows_Retained"].mean()
                    ),
                    "Mean_Training_Removal_Pct": float(
                        cutoff_subset["Actual_Removal_Pct"].mean()
                    ),
                    "Mean_Training_Cutoff_RM": (
                        np.nan
                        if removal_percent == 0
                        else float(cutoff_subset["Training_Derived_Cutoff_RM"].mean())
                    ),
                    "OOF_Rows": EXPECTED_ROWS,
                    **overall,
                }
            )
            for segment_name, mask in segments.items():
                segment_rows.append(
                    {
                        "Model": model_name,
                        "Trim_Level": level,
                        "Removal_Percent": removal_percent,
                        "Segment": segment_name,
                        **segment_metric(y, prediction, mask),
                    }
                )

    comparison = pd.DataFrame(comparison_rows)
    fold_metrics = pd.DataFrame(fold_rows)
    segment_metrics = pd.DataFrame(segment_rows)
    for model_name in MODELS:
        baseline_row = comparison[
            (comparison["Model"] == model_name)
            & (comparison["Trim_Level"] == "A")
        ].iloc[0]
        model_mask = comparison["Model"] == model_name
        comparison.loc[model_mask, "RMSE_Change_vs_0_RM"] = (
            comparison.loc[model_mask, "RMSE_RM"] - baseline_row["RMSE_RM"]
        )
        comparison.loc[model_mask, "MAE_Change_vs_0_RM"] = (
            comparison.loc[model_mask, "MAE_RM"] - baseline_row["MAE_RM"]
        )
        baseline_folds = fold_metrics[
            (fold_metrics["Model"] == model_name)
            & (fold_metrics["Trim_Level"] == "A")
        ].set_index("Fold")
        for row_index in comparison[model_mask].index:
            level = comparison.loc[row_index, "Trim_Level"]
            candidate = fold_metrics[
                (fold_metrics["Model"] == model_name)
                & (fold_metrics["Trim_Level"] == level)
            ].set_index("Fold")
            comparison.loc[row_index, "RMSE_Fold_Wins_vs_0"] = int(
                (candidate["RMSE_RM"] < baseline_folds["RMSE_RM"]).sum()
            )
            comparison.loc[row_index, "MAE_Fold_Wins_vs_0"] = int(
                (candidate["MAE_RM"] < baseline_folds["MAE_RM"]).sum()
            )
    segment_pivot = segment_metrics.pivot_table(
        index=["Model", "Trim_Level"],
        columns="Segment",
        values=["RMSE_RM", "Underprediction_Pct"],
    )
    for row_index, row in comparison.iterrows():
        key = (row["Model"], row["Trim_Level"])
        comparison.loc[row_index, "Top5_RMSE_RM"] = segment_pivot.loc[
            key, ("RMSE_RM", "Top 5%")
        ]
        comparison.loc[row_index, "P95_99_RMSE_RM"] = segment_pivot.loc[
            key, ("RMSE_RM", "95-99%")
        ]
        comparison.loc[row_index, "P99_100_RMSE_RM"] = segment_pivot.loc[
            key, ("RMSE_RM", "99-100%")
        ]
        comparison.loc[row_index, "Top5_Underprediction_Pct"] = segment_pivot.loc[
            key, ("Underprediction_Pct", "Top 5%")
        ]
    return (
        comparison.sort_values(["Model", "Removal_Percent"]),
        fold_metrics,
        segment_metrics,
        pd.concat(oof_parts, ignore_index=True),
        predictions_by_variant,
    )


def price_distribution(values: np.ndarray, prefix: str) -> dict:
    series = pd.Series(np.asarray(values, float))
    return {
        f"{prefix}_Row_Count": int(len(series)),
        f"{prefix}_Mean_Price_RM": float(series.mean()),
        f"{prefix}_Median_Price_RM": float(series.median()),
        f"{prefix}_Std_Price_RM": float(series.std(ddof=1)),
        f"{prefix}_P90_Price_RM": float(series.quantile(0.90)),
        f"{prefix}_P95_Price_RM": float(series.quantile(0.95)),
        f"{prefix}_P99_Price_RM": float(series.quantile(0.99)),
        f"{prefix}_Maximum_Price_RM": float(series.max()),
        f"{prefix}_Skewness": float(series.skew()),
    }


def run_trimmed_population(data, assignments, baseline, position, folds):
    X = data[MODEL_FEATURES]
    y = data["price"].to_numpy(float)
    listing_ids = data["listing_id"].astype(int).to_numpy()
    fold_values = assignments["fold"].to_numpy(int)
    original_prediction = baseline[
        BASELINE_COLUMNS["LightGBM + Position Features"]
    ].to_numpy(float)
    _, p95, p99 = segment_definitions(y)
    comparison_rows = []
    distribution_rows = []
    fold_rows = []
    oof_parts = []

    for level, removal_percent in TRIM_LEVELS:
        cutoff = (
            np.nan
            if removal_percent == 0
            else float(np.quantile(y, 1.0 - removal_percent / 100.0))
        )
        retained = np.arange(EXPECTED_ROWS) if removal_percent == 0 else np.flatnonzero(y <= cutoff)
        retained_mask = np.zeros(EXPECTED_ROWS, dtype=bool)
        retained_mask[retained] = True
        retrained_prediction = np.full(EXPECTED_ROWS, np.nan)
        coverage = np.zeros(EXPECTED_ROWS, dtype=int)
        for fold, (outer_training, outer_validation) in folds.items():
            training = outer_training[retained_mask[outer_training]]
            validation = outer_validation[retained_mask[outer_validation]]
            if removal_percent == 0:
                fold_prediction = original_prediction[validation]
            else:
                fold_prediction = fit_model(
                    "LightGBM + Position Features",
                    X,
                    y,
                    position,
                    training,
                    validation,
                )
            retrained_prediction[validation] = fold_prediction
            coverage[validation] += 1
            fold_rows.append(
                {
                    "Experiment_Type": "trimmed_population",
                    "Model": "LightGBM + Position Features",
                    "Trim_Level": level,
                    "Removal_Percent": removal_percent,
                    "Fold": fold,
                    "Training_Rows_Before": len(outer_training),
                    "Training_Rows_Removed": len(outer_training) - len(training),
                    "Training_Rows_Retained": len(training),
                    "Actual_Removal_Pct": (
                        (len(outer_training) - len(training)) / len(outer_training) * 100.0
                    ),
                    "Training_Derived_Cutoff_RM": cutoff,
                    "Maximum_Retained_Training_Price_RM": float(y[training].max()),
                    "Validation_Rows": len(validation),
                    "Validation_Rows_Removed": len(outer_validation) - len(validation),
                    **regression_metrics(y[validation], fold_prediction, 47),
                }
            )
            print(f"Trimmed population: trim {level}, fold {fold}/5", flush=True)
        if not np.all(coverage[retained] == 1) or not np.isfinite(retrained_prediction[retained]).all():
            raise AssertionError(f"Incomplete retained-population OOF coverage: {level}")
        actual = y[retained]
        matched_original = original_prediction[retained]
        matched_retrained = retrained_prediction[retained]
        original_metrics = regression_metrics(actual, matched_original, 47)
        retrained_metrics = regression_metrics(actual, matched_retrained, 47)
        comparison_rows.append(
            {
                "Model": "LightGBM + Position Features",
                "Trim_Level": level,
                "Removal_Percent": removal_percent,
                "Full_Population_Cutoff_RM": cutoff,
                "Retained_OOF_Rows": len(retained),
                "Removed_Evaluation_Rows": EXPECTED_ROWS - len(retained),
                "Matched_Original_RMSE_RM": original_metrics["RMSE_RM"],
                "Matched_Retrained_RMSE_RM": retrained_metrics["RMSE_RM"],
                "Matched_RMSE_Gain_RM": original_metrics["RMSE_RM"] - retrained_metrics["RMSE_RM"],
                "Matched_Original_MAE_RM": original_metrics["MAE_RM"],
                "Matched_Retrained_MAE_RM": retrained_metrics["MAE_RM"],
                "Matched_MAE_Gain_RM": original_metrics["MAE_RM"] - retrained_metrics["MAE_RM"],
                "Retrained_R2": retrained_metrics["R2"],
                "Retrained_Adjusted_R2": retrained_metrics["Adjusted_R2"],
                "Retrained_Median_AE_RM": retrained_metrics["Median_AE_RM"],
            }
        )
        distribution_rows.append(
            {
                "Trim_Level": level,
                "Removal_Percent": removal_percent,
                "Full_Population_Cutoff_RM": cutoff,
                **price_distribution(y, "Before"),
                **price_distribution(actual, "After"),
                "Premium_Threshold_RM": p95,
                "Premium_Rows_Retained": int(np.sum(actual >= p95)),
                "Premium_Rows_Removed": int(np.sum(y >= p95) - np.sum(actual >= p95)),
                "Top5_Rows_Retained": int(np.sum(actual >= p95)),
                "Top1_Rows_Retained": int(np.sum(actual >= p99)),
            }
        )
        oof_parts.append(
            pd.DataFrame(
                {
                    "Experiment_Type": "trimmed_population",
                    "Model": "LightGBM + Position Features",
                    "Trim_Level": level,
                    "Removal_Percent": removal_percent,
                    "listing_id": listing_ids[retained],
                    "scenario_b_fold": fold_values[retained],
                    "actual_price_RM": actual,
                    "predicted_price_RM": matched_retrained,
                    "matched_original_prediction_RM": matched_original,
                    "Threshold_Scope": "full_population_for_restricted_evaluation",
                }
            )
        )
    return (
        pd.DataFrame(comparison_rows),
        pd.DataFrame(distribution_rows),
        pd.DataFrame(fold_rows),
        pd.concat(oof_parts, ignore_index=True),
    )


def paired_bootstrap(actual, candidate, reference, seed: int) -> dict:
    actual = np.asarray(actual, float)
    candidate = np.asarray(candidate, float)
    reference = np.asarray(reference, float)
    rng = np.random.default_rng(seed)
    rmse_differences = np.empty(BOOTSTRAP_SAMPLES)
    mae_differences = np.empty(BOOTSTRAP_SAMPLES)
    batch_size = 100
    for start in range(0, BOOTSTRAP_SAMPLES, batch_size):
        stop = min(start + batch_size, BOOTSTRAP_SAMPLES)
        indices = rng.integers(0, len(actual), size=(stop - start, len(actual)))
        actual_sample = actual[indices]
        candidate_error = candidate[indices] - actual_sample
        reference_error = reference[indices] - actual_sample
        rmse_differences[start:stop] = (
            np.sqrt(np.mean(np.square(candidate_error), axis=1))
            - np.sqrt(np.mean(np.square(reference_error), axis=1))
        )
        mae_differences[start:stop] = (
            np.mean(np.abs(candidate_error), axis=1)
            - np.mean(np.abs(reference_error), axis=1)
        )
    return {
        "Bootstrap_Samples": BOOTSTRAP_SAMPLES,
        "RMSE_CI95_Lower_RM": float(np.quantile(rmse_differences, 0.025)),
        "RMSE_CI95_Upper_RM": float(np.quantile(rmse_differences, 0.975)),
        "MAE_CI95_Lower_RM": float(np.quantile(mae_differences, 0.025)),
        "MAE_CI95_Upper_RM": float(np.quantile(mae_differences, 0.975)),
    }


def build_bootstrap(price, predictions, comparison):
    rows = []
    for model_index, model_name in enumerate(MODELS):
        reference = predictions[(model_name, "A")]
        baseline_metrics = comparison[
            (comparison["Model"] == model_name)
            & (comparison["Trim_Level"] == "A")
        ].iloc[0]
        for level_index, (level, removal_percent) in enumerate(TRIM_LEVELS):
            candidate = predictions[(model_name, level)]
            candidate_metrics = comparison[
                (comparison["Model"] == model_name)
                & (comparison["Trim_Level"] == level)
            ].iloc[0]
            rows.append(
                {
                    "Model": model_name,
                    "Trim_Level": level,
                    "Removal_Percent": removal_percent,
                    "Difference_Definition": "trimmed_minus_untrimmed",
                    "RMSE_Difference_RM": candidate_metrics["RMSE_RM"] - baseline_metrics["RMSE_RM"],
                    "MAE_Difference_RM": candidate_metrics["MAE_RM"] - baseline_metrics["MAE_RM"],
                    **paired_bootstrap(
                        price,
                        candidate,
                        reference,
                        seed=42 + model_index * 100 + level_index,
                    ),
                }
            )
            print(f"Bootstrap: {model_name}, trim {level}", flush=True)
    return pd.DataFrame(rows)


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run_experiment() -> dict:
    before_manifest, before_hash = protected_manifest()
    data, assignments, baseline, position, linkage = load_inputs()
    folds = fold_positions(assignments)
    y = data["price"].to_numpy(float)
    cutoffs, cutoff_lookup = build_training_cutoffs(y, folds)
    (
        training_comparison,
        training_fold_metrics,
        segment_metrics,
        training_oof,
        predictions,
    ) = run_training_only(
        data,
        assignments,
        baseline,
        position,
        cutoffs,
        cutoff_lookup,
        folds,
    )
    (
        trimmed_comparison,
        distribution_shift,
        trimmed_fold_metrics,
        trimmed_oof,
    ) = run_trimmed_population(data, assignments, baseline, position, folds)
    bootstrap = build_bootstrap(y, predictions, training_comparison)
    fold_metrics = pd.concat(
        [training_fold_metrics, trimmed_fold_metrics], ignore_index=True
    )
    oof = pd.concat([training_oof, trimmed_oof], ignore_index=True)

    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    training_comparison.to_csv(EXPERIMENT / "training_only_comparison.csv", index=False)
    trimmed_comparison.to_csv(EXPERIMENT / "trimmed_population_comparison.csv", index=False)
    fold_metrics.to_csv(EXPERIMENT / "fold_metrics.csv", index=False)
    cutoffs.to_csv(EXPERIMENT / "training_cutoffs.csv", index=False)
    oof.to_csv(EXPERIMENT / "oof_predictions.csv", index=False)
    segment_metrics.to_csv(EXPERIMENT / "segment_metrics.csv", index=False)
    distribution_shift.to_csv(EXPERIMENT / "distribution_shift.csv", index=False)
    bootstrap.to_csv(EXPERIMENT / "bootstrap_results.csv", index=False)

    after_manifest, after_hash = protected_manifest()
    if before_manifest != after_manifest:
        changed = sorted(
            set(before_manifest).symmetric_difference(after_manifest)
            | {
                name
                for name in set(before_manifest).intersection(after_manifest)
                if before_manifest[name] != after_manifest[name]
            }
        )
        raise AssertionError(f"Protected production files changed: {changed}")

    primary = training_comparison[
        training_comparison["Model"] == "LightGBM + Position Features"
    ].copy()
    primary_bootstrap = bootstrap[
        bootstrap["Model"] == "LightGBM + Position Features"
    ].set_index("Trim_Level")
    primary_segments = segment_metrics[
        segment_metrics["Model"] == "LightGBM + Position Features"
    ]
    base_top5 = primary_segments[
        (primary_segments["Trim_Level"] == "A")
        & (primary_segments["Segment"] == "Top 5%")
    ].iloc[0]
    base_top1 = primary_segments[
        (primary_segments["Trim_Level"] == "A")
        & (primary_segments["Segment"] == "99-100%")
    ].iloc[0]
    assessments = []
    for _, row in primary.iterrows():
        level = row["Trim_Level"]
        if level == "A":
            continue
        top5 = primary_segments[
            (primary_segments["Trim_Level"] == level)
            & (primary_segments["Segment"] == "Top 5%")
        ].iloc[0]
        top1 = primary_segments[
            (primary_segments["Trim_Level"] == level)
            & (primary_segments["Segment"] == "99-100%")
        ].iloc[0]
        boot = primary_bootstrap.loc[level]
        assessments.append(
            {
                "Trim_Level": level,
                "Removal_Percent": row["Removal_Percent"],
                "Improves_Overall_RMSE": bool(row["RMSE_Change_vs_0_RM"] < 0),
                "Improves_Overall_MAE": bool(row["MAE_Change_vs_0_RM"] < 0),
                "RMSE_CI_Excludes_Zero_in_Improvement_Direction": bool(
                    boot["RMSE_CI95_Upper_RM"] < 0
                ),
                "MAE_CI_Excludes_Zero_in_Improvement_Direction": bool(
                    boot["MAE_CI95_Upper_RM"] < 0
                ),
                "Top5_RMSE_Change_RM": float(top5["RMSE_RM"] - base_top5["RMSE_RM"]),
                "Top5_Underprediction_Change_Pct_Points": float(
                    top5["Underprediction_Pct"] - base_top5["Underprediction_Pct"]
                ),
                "Top1_RMSE_Change_RM": float(top1["RMSE_RM"] - base_top1["RMSE_RM"]),
                "RMSE_Fold_Wins": int(row["RMSE_Fold_Wins_vs_0"]),
                "MAE_Fold_Wins": int(row["MAE_Fold_Wins_vs_0"]),
            }
        )
    adoptable = [
        row
        for row in assessments
        if row["Improves_Overall_RMSE"]
        and row["Improves_Overall_MAE"]
        and row["RMSE_CI_Excludes_Zero_in_Improvement_Direction"]
        and row["MAE_CI_Excludes_Zero_in_Improvement_Direction"]
        and row["RMSE_Fold_Wins"] >= 3
        and row["MAE_Fold_Wins"] >= 3
        and row["Top5_RMSE_Change_RM"] <= 0
        and row["Top1_RMSE_Change_RM"] <= 0
    ]
    result = {
        "experiment": "upper_tail_trimming",
        "status": "complete",
        "canonical_dataset": {
            "path": DATA_PATH.relative_to(ROOT).as_posix(),
            "rows": EXPECTED_ROWS,
            "sha256": sha256(DATA_PATH),
        },
        "validation": {
            "scenario": "B",
            "folds": 5,
            "fold_assignment_path": FOLD_PATH.relative_to(ROOT).as_posix(),
            "fold_assignment_sha256": sha256(FOLD_PATH),
            "validation_rows_trimmed_in_training_only_experiment": 0,
            "training_only_oof_rows_per_variant": EXPECTED_ROWS,
        },
        "trimming_levels": [
            {"level": level, "upper_tail_removed_percent": percent}
            for level, percent in TRIM_LEVELS
        ],
        "methodology": {
            "primary_model": "LightGBM + Position Features",
            "secondary_model": "Random Forest",
            "target": "price / property_size_sqft",
            "prediction_reconstruction": "predicted PPSF * property_size_sqft",
            "trimming_variable": "total listing price",
            "training_only_threshold_scope": "outer training fold only",
            "trimmed_population_threshold_scope": "full retained evaluation population",
            "frozen_model_parameters": MODEL_PARAMETERS,
            "position_features": list(position.columns),
            "description_linkage": linkage,
            "other_outlier_treatment": {
                "winsorization": False,
                "log_target": False,
                "huber_loss": False,
                "sample_weighting": False,
                "missing_data_deletion": False,
                "manual_premium_deletion": False,
                "ppsf_filtering": False,
                "new_feature_engineering": False,
            },
            "baseline_source": "verified saved Scenario B OOF predictions",
        },
        "training_only_comparison": training_comparison.to_dict("records"),
        "trimmed_population_comparison": trimmed_comparison.to_dict("records"),
        "primary_assessment": assessments,
        "adoptable_levels_under_all_predefined_success_criteria": adoptable,
        "recommended_trim_level": (
            min(adoptable, key=lambda row: row["Removal_Percent"])["Trim_Level"]
            if adoptable
            else "A"
        ),
        "production_model_changed": False,
        "production_safety": {
            "protected_file_count": len(before_manifest),
            "before_manifest_sha256": before_hash,
            "after_manifest_sha256": after_hash,
            "all_protected_files_unchanged": before_manifest == after_manifest,
        },
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
    }
    (EXPERIMENT / "results.json").write_text(
        json.dumps(_json_ready(result), indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    result = run_experiment()
    print("\nTRAINING-ONLY PERFORMANCE")
    print(
        pd.DataFrame(result["training_only_comparison"])[
            [
                "Model",
                "Trim_Level",
                "Removal_Percent",
                "RMSE_RM",
                "RMSE_Change_vs_0_RM",
                "MAE_RM",
                "MAE_Change_vs_0_RM",
                "R2",
                "Top5_RMSE_RM",
            ]
        ].to_string(index=False)
    )
    print("\nTRIMMED-POPULATION PERFORMANCE")
    print(pd.DataFrame(result["trimmed_population_comparison"]).to_string(index=False))


if __name__ == "__main__":
    main()
