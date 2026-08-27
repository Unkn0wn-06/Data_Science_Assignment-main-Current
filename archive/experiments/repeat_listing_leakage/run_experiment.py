"""Audit repeat-like listings and rerun established models with group-safe CV."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
import time
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.advanced_real_estate_models.feature_engineering import engineered_feature_lists
from experiments.description_text_features.model_builders import fit_lightgbm_fold
from experiments.description_text_features.regex_features import REGEX_GROUPS, extract_regex_features
from experiments.description_text_features.text_cleaning import link_descriptions
from experiments.noncoordinate_target_encoding.feature_engineering import NoncoordinatePPSFRegressor
from src.models.common.features import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERICAL_FEATURES
from src.models.enhanced_city import build_ppsf_estimator


EXPERIMENT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "enhanced_city_dataset.csv"
RAW_PATH = ROOT / "data" / "raw" / "houses.csv"
HISTORICAL_OOF_PATH = ROOT / "experiments" / "description_text_features" / "oof_predictions.csv"
VALIDITY_RESULTS_PATH = ROOT / "experiments" / "data_validity_audit" / "results.json"
EXPECTED_ROWS = 3_791
POSITION_FEATURES = list(REGEX_GROUPS["position"])
MAJOR_COLUMNS = [
    "price", "property_size_sqft", "bedroom", "bathroom", "property_type",
    "building_name", "developer", "city", "state",
]
MODEL_SPECS = {
    "random_forest": {"historical_variant": "random_forest_reference", "predictors": 32},
    "lightgbm_interaction": {"historical_variant": "lightgbm_structured_reference", "predictors": 42},
    "building_name_te": {"historical_variant": "building_name_te_reference", "predictors": 42},
    "position_regex_lightgbm": {"historical_variant": "regex_group_position", "predictors": 47},
}
INVALID_RULES = {
    103609830: "MULTI_UNIT_BUNDLE_NOT_SINGLE_OBSERVATION",
    102236931: "BEDROOM_COUNT_DESCRIPTION_CONTRADICTION",
    103207012: "BATHROOM_COUNT_CONFIGURATION_CONTRADICTION",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_snapshot() -> dict[str, str]:
    roots = [ROOT / name for name in ("data", "results", "prototype", "src", "configs", "scripts")]
    files = [ROOT / name for name in ("app.py", "README.md", "requirements.txt")]
    for directory in roots:
        if directory.exists():
            files.extend(path for path in directory.rglob("*") if path.is_file())
    experiments_root = ROOT / "experiments"
    for directory in experiments_root.iterdir():
        if directory.is_dir() and directory.resolve() != EXPERIMENT.resolve():
            files.extend(
                path for path in directory.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts
            )
    return {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in sorted(set(files)) if path.exists()
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
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if value is pd.NA:
        return None
    return value


def group_members(frame: pd.DataFrame, columns: list[str]) -> list[np.ndarray]:
    """Return deterministic row-position arrays for keys occurring at least twice."""
    groups = []
    for _, index in frame.groupby(columns, dropna=False, sort=False).groups.items():
        positions = np.sort(np.asarray(index, dtype=int))
        if len(positions) > 1:
            groups.append(positions)
    return sorted(groups, key=lambda values: (int(values[0]), len(values), tuple(values.tolist())))


def assign_subgroup_ids(groups: list[np.ndarray], prefix: str, row_count: int) -> np.ndarray:
    result = np.full(row_count, "", dtype=object)
    for number, positions in enumerate(groups, 1):
        result[positions] = f"{prefix}{number:04d}"
    return result


def build_repeat_audit(
    frame: pd.DataFrame, descriptions: pd.Series, old_fold: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame, dict, np.ndarray]:
    comparison = frame.copy()
    comparison["_normalized_description"] = descriptions.to_numpy()
    exact_columns = [column for column in frame.columns if column != "listing_id"]
    level1_groups = group_members(comparison, exact_columns)
    level2_groups = group_members(comparison, MAJOR_COLUMNS + ["_normalized_description"])
    level3_groups = group_members(comparison, MAJOR_COLUMNS)

    level1_id = assign_subgroup_ids(level1_groups, "L1_", len(frame))
    level2_id = assign_subgroup_ids(level2_groups, "L2_", len(frame))
    repeat_id = assign_subgroup_ids(level3_groups, "RG_", len(frame))
    repeat_mask = repeat_id != ""
    strongest = np.full(len(frame), "", dtype=object)
    strongest[repeat_mask] = "Level 3 - possible repeat"
    strongest[level2_id != ""] = "Level 2 - strong repeat"
    strongest[level1_id != ""] = "Level 1 - exact duplicate"

    size_map = {f"RG_{number:04d}": len(rows) for number, rows in enumerate(level3_groups, 1)}
    repeat_rows = pd.DataFrame(
        {
            "row_index": np.flatnonzero(repeat_mask),
            "listing_id": frame.loc[repeat_mask, "listing_id"].astype(int).to_numpy(),
            "repeat_group_id": repeat_id[repeat_mask],
            "repeat_level": strongest[repeat_mask],
            "group_size": [size_map[value] for value in repeat_id[repeat_mask]],
            "level1_match_group_id": level1_id[repeat_mask],
            "level2_match_group_id": level2_id[repeat_mask],
        }
    )

    summaries = []
    for number, positions in enumerate(level3_groups, 1):
        group_id = f"RG_{number:04d}"
        folds = sorted(np.unique(old_fold[positions]).astype(int).tolist())
        levels = strongest[positions]
        summaries.append(
            {
                "repeat_group_id": group_id,
                "group_size": int(len(positions)),
                "strongest_repeat_level": (
                    "Level 1 - exact duplicate" if np.any(levels == "Level 1 - exact duplicate")
                    else "Level 2 - strong repeat" if np.any(levels == "Level 2 - strong repeat")
                    else "Level 3 - possible repeat"
                ),
                "level1_rows": int(np.sum(levels == "Level 1 - exact duplicate")),
                "level2_rows": int(np.sum(levels == "Level 2 - strong repeat")),
                "level3_rows": int(np.sum(levels == "Level 3 - possible repeat")),
                "level1_match_groups": int(len({value for value in level1_id[positions] if value})),
                "level2_match_groups": int(len({value for value in level2_id[positions] if value})),
                "listing_ids": "|".join(frame.iloc[positions]["listing_id"].astype(int).astype(str)),
                "historical_folds": "|".join(map(str, folds)),
                "historical_fold_count": len(folds),
                "crossed_historical_folds": len(folds) > 1,
            }
        )
    summary = pd.DataFrame(summaries)
    crossing = summary["crossed_historical_folds"]
    counts = {
        "level_definitions_are_inclusive": True,
        "level1_exact_duplicate_groups": len(level1_groups),
        "level1_exact_duplicate_rows": int(np.sum(level1_id != "")),
        "level1_duplicate_rows_beyond_one_per_group": int(np.sum(level1_id != "") - len(level1_groups)),
        "level2_strong_repeat_groups": len(level2_groups),
        "level2_strong_repeat_rows": int(np.sum(level2_id != "")),
        "level3_possible_repeat_groups": len(level3_groups),
        "level3_possible_repeat_rows": int(repeat_mask.sum()),
        "total_repeat_groups_used_for_cv": len(level3_groups),
        "total_repeat_like_rows": int(repeat_mask.sum()),
        "groups_crossing_historical_folds": int(crossing.sum()),
        "rows_in_cross_fold_repeat_groups": int(summary.loc[crossing, "group_size"].sum()),
        "fuzzy_matching_used": False,
        "rows_deleted_as_repeats": 0,
    }
    return repeat_rows, summary, counts, repeat_id


def historical_predictions(frame: pd.DataFrame) -> tuple[dict[str, np.ndarray], np.ndarray]:
    source = pd.read_csv(HISTORICAL_OOF_PATH)
    predictions = {}
    old_fold = None
    actual = frame["price"].to_numpy(float)
    listing_ids = frame["listing_id"].astype(int).to_numpy()
    for model, spec in MODEL_SPECS.items():
        selected = source[source["variant"] == spec["historical_variant"]].sort_values("row_index")
        if len(selected) != len(frame):
            raise AssertionError(f"Historical OOF rows missing for {model}.")
        if not np.array_equal(selected["row_id"].astype(int).to_numpy(), listing_ids):
            raise AssertionError(f"Historical listing order mismatch for {model}.")
        if not np.array_equal(selected["actual_price_RM"].to_numpy(float), actual):
            raise AssertionError(f"Historical targets mismatch for {model}.")
        fold = selected["fold"].to_numpy(int)
        if old_fold is None:
            old_fold = fold
        elif not np.array_equal(old_fold, fold):
            raise AssertionError("Historical model folds are not identical.")
        predictions[model] = selected["predicted_price_RM"].to_numpy(float)
    return predictions, old_fold


def create_group_safe_folds(
    frame: pd.DataFrame, repeat_id: np.ndarray
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    groups = np.asarray(
        [repeat_id[i] if repeat_id[i] else f"UNIQUE_{int(frame.iloc[i]['listing_id'])}" for i in range(len(frame))],
        dtype=object,
    )
    splitter = GroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(splitter.split(frame, frame["price"], groups=groups))
    fold_id = np.zeros(len(frame), dtype=int)
    for fold, (train_index, validation_index) in enumerate(folds, 1):
        fold_id[validation_index] = fold
        if set(groups[train_index]).intersection(groups[validation_index]):
            raise AssertionError(f"Group leakage detected in fold {fold}.")
    if np.any(fold_id == 0):
        raise AssertionError("Every row must receive one group-safe fold.")
    return folds, fold_id, groups


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
    actual = np.asarray(actual, float)
    predicted = np.asarray(predicted, float)
    result = basic_metrics(actual, predicted)
    result["Adjusted_R2"] = float(
        1.0 - (1.0 - result["R2"]) * (len(actual) - 1) / (len(actual) - predictors - 1)
    )
    masks = {
        "Top5": actual >= thresholds["p95"],
        "P95_P99": (actual >= thresholds["p95"]) & (actual < thresholds["p99"]),
        "P99_P100": actual >= thresholds["p99"],
    }
    for label, mask in masks.items():
        subset = basic_metrics(actual[mask], predicted[mask])
        result[f"{label}_count"] = subset["count"]
        result[f"{label}_RMSE_RM"] = subset["RMSE_RM"]
        result[f"{label}_MAE_RM"] = subset["MAE_RM"]
    return result


def fit_model_fold(model: str, X, y, regex, train_index, validation_index):
    X_train, X_validation = X.iloc[train_index], X.iloc[validation_index]
    y_train = y[train_index]
    if model == "random_forest":
        fitted = clone(build_ppsf_estimator("Random Forest")).fit(X_train, y_train)
        return np.asarray(fitted.predict(X_validation), float), {
            "target_encoding_outer_training_only": True,
            "validation_target_used": False,
        }
    if model == "building_name_te":
        fitted = clone(NoncoordinatePPSFRegressor(te_columns=("building_name",))).fit(X_train, y_train)
        return np.asarray(fitted.predict(X_validation), float), {
            "target_encoding_outer_training_only": True,
            "target_encoding_inner_oof": True,
            "validation_target_used": False,
            "selected_m": float(fitted.selected_m_),
        }
    train_dense = None
    validation_dense = None
    if model == "position_regex_lightgbm":
        train_dense = regex.loc[train_index, POSITION_FEATURES]
        validation_dense = regex.loc[validation_index, POSITION_FEATURES]
    output = fit_lightgbm_fold(
        X_train, y_train, X_validation, train_dense, validation_dense
    )
    return output["validation_prediction"], {
        "target_encoding_outer_training_only": True,
        "validation_target_used": False,
        "regex_target_free": model == "position_regex_lightgbm",
    }


def evaluate_models(frame, descriptions, folds, fold_id, groups, thresholds):
    X = frame[MODEL_FEATURES]
    y = frame["price"].to_numpy(float)
    regex = extract_regex_features(descriptions)
    predictions = {model: np.empty(len(frame), float) for model in MODEL_SPECS}
    fold_rows = []
    fit_audit = []
    for model, spec in MODEL_SPECS.items():
        for fold, (train_index, validation_index) in enumerate(folds, 1):
            predicted, audit = fit_model_fold(model, X, y, regex, train_index, validation_index)
            predictions[model][validation_index] = predicted
            metrics = basic_metrics(y[validation_index], predicted)
            fold_rows.append(
                {
                    "cv_scheme": "group_safe",
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
                    "model": model,
                    "fold": fold,
                    "training_rows": len(train_index),
                    "validation_rows": len(validation_index),
                    "group_overlap": len(set(groups[train_index]).intersection(groups[validation_index])),
                    **audit,
                }
            )
            print(f"Completed group-safe {model} fold {fold}/5.", flush=True)
    metrics = {
        model: complete_metrics(y, predicted, spec["predictors"], thresholds)
        for (model, spec), predicted in zip(MODEL_SPECS.items(), predictions.values())
    }
    return predictions, metrics, fold_rows, fit_audit, regex


def bootstrap_difference(actual, candidate, reference, draws=5_000, seed=42) -> dict:
    actual = np.asarray(actual, float)
    candidate_sq = np.square(np.asarray(candidate, float) - actual)
    reference_sq = np.square(np.asarray(reference, float) - actual)
    absolute_delta = np.abs(np.asarray(candidate, float) - actual) - np.abs(np.asarray(reference, float) - actual)
    rng = np.random.default_rng(seed)
    rmse_values = np.empty(draws, float)
    mae_values = np.empty(draws, float)
    batch = 100
    for start in range(0, draws, batch):
        count = min(batch, draws - start)
        sampled = rng.integers(0, len(actual), size=(count, len(actual)))
        rmse_values[start:start + count] = (
            np.sqrt(candidate_sq[sampled].mean(axis=1))
            - np.sqrt(reference_sq[sampled].mean(axis=1))
        )
        mae_values[start:start + count] = absolute_delta[sampled].mean(axis=1)
    return {
        "bootstrap_draws": draws,
        "difference_definition": "Position-regex minus reference; negative is better",
        "RMSE_difference_RM": float(np.sqrt(candidate_sq.mean()) - np.sqrt(reference_sq.mean())),
        "RMSE_bootstrap_mean_RM": float(rmse_values.mean()),
        "RMSE_CI95_lower_RM": float(np.quantile(rmse_values, 0.025)),
        "RMSE_CI95_upper_RM": float(np.quantile(rmse_values, 0.975)),
        "MAE_difference_RM": float(absolute_delta.mean()),
        "MAE_bootstrap_mean_RM": float(mae_values.mean()),
        "MAE_CI95_lower_RM": float(np.quantile(mae_values, 0.025)),
        "MAE_CI95_upper_RM": float(np.quantile(mae_values, 0.975)),
        "RMSE_statistically_reliable": bool(np.quantile(rmse_values, 0.975) < 0 or np.quantile(rmse_values, 0.025) > 0),
        "MAE_statistically_reliable": bool(np.quantile(mae_values, 0.975) < 0 or np.quantile(mae_values, 0.025) > 0),
        "fixed_oof_limitation": "Paired row bootstrap does not include model refitting, grouped resampling, or selection uncertainty.",
    }


def verify_invalid_rows(frame: pd.DataFrame, descriptions: pd.Series) -> dict:
    validity = json.loads(VALIDITY_RESULTS_PATH.read_text(encoding="utf-8"))
    prior_ids = {
        int(row["row_id"])
        for row in validity["removed_row_diagnostics"]["rows"]
    }
    if prior_ids != set(INVALID_RULES):
        raise AssertionError(f"Prior invalid IDs differ from expected evidence audit: {prior_ids}")
    indexed = frame.set_index("listing_id")
    desc_by_id = pd.Series(descriptions.to_numpy(), index=frame["listing_id"].astype(int))
    evidence = {}
    bundle = desc_by_id.loc[103609830]
    sizes = re.findall(r"(?<!\d)(\d{3,5}(?:\.\d+)?)\s*(?:sq\.?\s*ft|sqft|sft|sf)\b", bundle.replace(",", ""))
    prices = re.findall(r"\brm\s*([0-9]+(?:\.\d+)?)\s*([km]?)\b", bundle.replace(",", ""))
    if len(set(sizes)) < 2 or len(set(prices)) < 2 or not re.search(r"\b(?:two|2)\s+(?:units?|apartments?)\b", bundle):
        raise AssertionError("Multi-unit bundle evidence was not independently reproduced.")
    evidence["103609830"] = {"rule": INVALID_RULES[103609830], "distinct_sizes": len(set(sizes)), "distinct_prices": len(set(prices))}
    bedroom_row = indexed.loc[102236931]
    bedroom_text = desc_by_id.loc[102236931]
    if bedroom_row["bedroom"] < 8 or not re.search(r"\b(?:3|three)[-\s]*(?:bed|bedroom)s?\b", bedroom_text) or re.search(r"\b(?:8|eight)[-\s]*(?:bed|bedroom)s?\b", bedroom_text):
        raise AssertionError("Bedroom contradiction evidence was not independently reproduced.")
    evidence["102236931"] = {"rule": INVALID_RULES[102236931], "structured_bedroom": float(bedroom_row["bedroom"])}
    bathroom_row = indexed.loc[103207012]
    bathroom_text = desc_by_id.loc[103207012]
    if not (bathroom_row["bathroom"] >= 6 and bathroom_row["property_size_sqft"] < 600 and bathroom_row["bathroom"] - bathroom_row["bedroom"] > 3) or not re.search(r"\b(?:2|two)\s*(?:bath|bathroom)s?\b", bathroom_text):
        raise AssertionError("Bathroom contradiction evidence was not independently reproduced.")
    evidence["103207012"] = {"rule": INVALID_RULES[103207012], "structured_bathroom": float(bathroom_row["bathroom"])}
    return evidence


def invalid_fixed_fold_subtest(frame, descriptions, regex, old_fold, historical_position, thresholds):
    evidence = verify_invalid_rows(frame, descriptions)
    invalid_mask = frame["listing_id"].astype(int).isin(INVALID_RULES).to_numpy()
    retained = ~invalid_mask
    X = frame[MODEL_FEATURES]
    y = frame["price"].to_numpy(float)
    retrained = np.full(len(frame), np.nan)
    fold_rows = []
    for fold in range(1, 6):
        train_index = np.flatnonzero(retained & (old_fold != fold))
        validation_index = np.flatnonzero(retained & (old_fold == fold))
        predicted, _ = fit_model_fold(
            "position_regex_lightgbm", X, y, regex, train_index, validation_index
        )
        retrained[validation_index] = predicted
        new_metrics = basic_metrics(y[validation_index], predicted)
        old_metrics = basic_metrics(y[validation_index], historical_position[validation_index])
        for model, metric, training_rows in (
            ("original_position_retained", old_metrics, int(np.sum(old_fold != fold))),
            ("retrained_invalid_removed", new_metrics, int(len(train_index))),
        ):
            fold_rows.append(
                {
                    "cv_scheme": "invalid_fixed_original_folds", "model": model, "fold": fold,
                    "training_rows": training_rows, "validation_rows": int(len(validation_index)),
                    "RMSE_RM": metric["RMSE_RM"], "MAE_RM": metric["MAE_RM"], "R2": metric["R2"],
                }
            )
        print(f"Completed fixed-original-fold invalid-row subtest fold {fold}/5.", flush=True)
    if np.isnan(retrained[retained]).any() or np.isfinite(retrained[invalid_mask]).any():
        raise AssertionError("Fixed-fold retained prediction coverage is invalid.")
    original_metrics = complete_metrics(y[retained], historical_position[retained], 47, thresholds)
    retrained_metrics = complete_metrics(y[retained], retrained[retained], 47, thresholds)
    gain_rmse = original_metrics["RMSE_RM"] - retrained_metrics["RMSE_RM"]
    gain_mae = original_metrics["MAE_RM"] - retrained_metrics["MAE_RM"]
    rows = []
    for model, metrics in (("original_position_retained", original_metrics), ("retrained_invalid_removed", retrained_metrics)):
        rows.append({
            "model": model, **metrics,
            "invalid_rows_removed": 3, "retained_rows": int(retained.sum()),
            "original_folds_preserved": True,
            "RMSE_gain_original_minus_retrained_RM": gain_rmse,
            "MAE_gain_original_minus_retrained_RM": gain_mae,
        })
    return pd.DataFrame(rows), retrained, retained, fold_rows, evidence


def metric_columns(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def main() -> None:
    started = time.perf_counter()
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    protected_before = protected_snapshot()
    frame = pd.read_csv(DATA_PATH).reset_index(drop=True)
    if len(frame) != EXPECTED_ROWS or frame["listing_id"].nunique() != EXPECTED_ROWS:
        raise AssertionError("Canonical row count or listing-ID uniqueness changed.")
    descriptions, linkage = link_descriptions(RAW_PATH, frame["listing_id"])
    historical, old_fold = historical_predictions(frame)
    repeat_rows, repeat_summary, repeat_counts, repeat_id = build_repeat_audit(frame, descriptions, old_fold)
    folds, group_fold, groups = create_group_safe_folds(frame, repeat_id)
    repeat_fold_check = repeat_rows.assign(
        fold=group_fold[repeat_rows["row_index"].to_numpy(int)]
    )
    if repeat_fold_check.groupby("repeat_group_id")["fold"].nunique().max() != 1:
        raise AssertionError("A repeat group crosses group-safe folds.")

    assignments = pd.DataFrame({
        "row_index": np.arange(len(frame)),
        "listing_id": frame["listing_id"].astype(int),
        "group_safe_group_id": groups,
        "repeat_group_id": repeat_id,
        "is_repeat_like": repeat_id != "",
        "group_safe_fold": group_fold,
    })
    group_sizes = assignments.groupby("group_safe_group_id")["listing_id"].transform("size")
    assignments["group_size"] = group_sizes.astype(int)
    assignments = assignments.merge(
        repeat_rows[["listing_id", "repeat_level"]], on="listing_id", how="left"
    )
    assignments["repeat_level"] = assignments["repeat_level"].fillna("non-repeat")
    repeat_summary = repeat_summary.merge(
        assignments.loc[assignments["is_repeat_like"], ["repeat_group_id", "group_safe_fold"]]
        .drop_duplicates("repeat_group_id"),
        on="repeat_group_id",
        how="left",
        validate="one_to_one",
    )

    y = frame["price"].to_numpy(float)
    thresholds = {"p95": float(np.quantile(y, 0.95)), "p99": float(np.quantile(y, 0.99))}
    new_predictions, new_metrics, fold_rows, fit_audit, regex = evaluate_models(
        frame, descriptions, folds, group_fold, groups, thresholds
    )
    old_metrics = {
        model: complete_metrics(y, prediction, MODEL_SPECS[model]["predictors"], thresholds)
        for model, prediction in historical.items()
    }

    comparison_rows = []
    for model in MODEL_SPECS:
        old = old_metrics[model]
        new = new_metrics[model]
        comparison_rows.append({
            "model": model,
            **metric_columns("old", old),
            **metric_columns("group_safe", new),
            "rmse_change_RM": new["RMSE_RM"] - old["RMSE_RM"],
            "mae_change_RM": new["MAE_RM"] - old["MAE_RM"],
        })
    comparison = pd.DataFrame(comparison_rows)
    comparison["group_safe_rmse_rank"] = comparison["group_safe_RMSE_RM"].rank(method="min").astype(int)
    comparison["group_safe_mae_rank"] = comparison["group_safe_MAE_RM"].rank(method="min").astype(int)

    fold_frame = pd.DataFrame(fold_rows)
    position = new_predictions["position_regex_lightgbm"]
    direct = {}
    bootstrap = {}
    for reference in ("random_forest", "lightgbm_interaction", "building_name_te"):
        selected = fold_frame[fold_frame["model"].isin(["position_regex_lightgbm", reference])]
        rmse_pivot = selected.pivot(index="fold", columns="model", values="RMSE_RM")
        mae_pivot = selected.pivot(index="fold", columns="model", values="MAE_RM")
        direct[reference] = {
            "RMSE_difference_RM": new_metrics["position_regex_lightgbm"]["RMSE_RM"] - new_metrics[reference]["RMSE_RM"],
            "MAE_difference_RM": new_metrics["position_regex_lightgbm"]["MAE_RM"] - new_metrics[reference]["MAE_RM"],
            "RMSE_folds_won": int((rmse_pivot["position_regex_lightgbm"] < rmse_pivot[reference]).sum()),
            "MAE_folds_won": int((mae_pivot["position_regex_lightgbm"] < mae_pivot[reference]).sum()),
        }
        bootstrap[reference] = bootstrap_difference(y, position, new_predictions[reference])

    repeat_mask = repeat_id != ""
    historical_repeat_errors = {
        "repeat_like_rows": basic_metrics(y[repeat_mask], historical["position_regex_lightgbm"][repeat_mask]),
        "non_repeat_rows": basic_metrics(y[~repeat_mask], historical["position_regex_lightgbm"][~repeat_mask]),
    }
    group_safe_repeat_errors = {
        "repeat_like_rows": basic_metrics(y[repeat_mask], position[repeat_mask]),
        "non_repeat_rows": basic_metrics(y[~repeat_mask], position[~repeat_mask]),
    }
    segmented_changes = {
        segment: {
            "RMSE_change_RM": group_safe_repeat_errors[segment]["RMSE_RM"] - historical_repeat_errors[segment]["RMSE_RM"],
            "MAE_change_RM": group_safe_repeat_errors[segment]["MAE_RM"] - historical_repeat_errors[segment]["MAE_RM"],
            "Median_AE_change_RM": group_safe_repeat_errors[segment]["Median_AE_RM"] - historical_repeat_errors[segment]["Median_AE_RM"],
        }
        for segment in ("repeat_like_rows", "non_repeat_rows")
    }

    invalid_table, invalid_prediction, retained, invalid_fold_rows, invalid_evidence = invalid_fixed_fold_subtest(
        frame, descriptions, regex, old_fold, historical["position_regex_lightgbm"], thresholds
    )
    fold_frame = pd.concat([fold_frame, pd.DataFrame(invalid_fold_rows)], ignore_index=True)

    oof_parts = []
    for model, prediction in new_predictions.items():
        oof_parts.append(pd.DataFrame({
            "cv_scheme": "group_safe", "model": model,
            "row_index": np.arange(len(frame)), "listing_id": frame["listing_id"].astype(int),
            "fold": group_fold, "group_id": groups,
            "actual_price_RM": y, "predicted_price_RM": prediction,
            "residual_RM": prediction - y, "absolute_error_RM": np.abs(prediction - y),
        }))
    for model, prediction in (
        ("original_position_retained", historical["position_regex_lightgbm"]),
        ("retrained_invalid_removed", invalid_prediction),
    ):
        oof_parts.append(pd.DataFrame({
            "cv_scheme": "invalid_fixed_original_folds", "model": model,
            "row_index": np.flatnonzero(retained),
            "listing_id": frame.loc[retained, "listing_id"].astype(int).to_numpy(),
            "fold": old_fold[retained], "group_id": "original_fold_membership",
            "actual_price_RM": y[retained], "predicted_price_RM": prediction[retained],
            "residual_RM": prediction[retained] - y[retained],
            "absolute_error_RM": np.abs(prediction[retained] - y[retained]),
        }))
    oof = pd.concat(oof_parts, ignore_index=True)

    repeat_rows.to_csv(EXPERIMENT / "repeat_groups.csv", index=False)
    repeat_summary.to_csv(EXPERIMENT / "repeat_group_summary.csv", index=False)
    assignments.to_csv(EXPERIMENT / "group_safe_fold_assignments.csv", index=False)
    comparison.to_csv(EXPERIMENT / "model_comparison.csv", index=False)
    fold_frame.to_csv(EXPERIMENT / "fold_metrics.csv", index=False)
    oof.to_csv(EXPERIMENT / "oof_predictions.csv", index=False)
    invalid_table.to_csv(EXPERIMENT / "invalid_row_fixed_fold_comparison.csv", index=False)

    protected_after = protected_snapshot()
    if protected_before != protected_after:
        changed = sorted(set(protected_before) | set(protected_after))
        changed = [name for name in changed if protected_before.get(name) != protected_after.get(name)]
        raise AssertionError(f"Protected files changed: {changed}")

    best_rmse = comparison.sort_values("group_safe_RMSE_RM").iloc[0]["model"]
    best_mae = comparison.sort_values("group_safe_MAE_RM").iloc[0]["model"]
    invalid_original = invalid_table.iloc[0]
    invalid_retrained = invalid_table.iloc[1]
    position_change = comparison.set_index("model").loc["position_regex_lightgbm"]
    repeated_optimism_evidence = bool(
        position_change["rmse_change_RM"] > 0
        and position_change["mae_change_RM"] > 0
        and repeat_counts["groups_crossing_historical_folds"] > 0
        and segmented_changes["repeat_like_rows"]["RMSE_change_RM"] > segmented_changes["non_repeat_rows"]["RMSE_change_RM"]
        and segmented_changes["repeat_like_rows"]["MAE_change_RM"] > segmented_changes["non_repeat_rows"]["MAE_change_RM"]
    )
    results = {
        "question": "Are repeat-like listings leaking across historical CV folds, and does Position-regex remain strongest under group-safe CV?",
        "dataset": {
            "path": DATA_PATH.relative_to(ROOT).as_posix(), "rows": len(frame),
            "unique_listing_ids": int(frame["listing_id"].nunique()), "sha256": sha256(DATA_PATH),
            "rows_removed_as_repeats": 0,
        },
        "description_linkage": linkage,
        "repeat_audit": repeat_counts,
        "historical_position_repeat_vs_non_repeat": historical_repeat_errors,
        "group_safe_position_repeat_vs_non_repeat": group_safe_repeat_errors,
        "position_segmented_old_to_group_safe_changes": segmented_changes,
        "group_safe_cv": {
            "splitter": "sklearn.model_selection.GroupKFold(n_splits=5, shuffle=True, random_state=42)",
            "fold_sizes": {str(fold): int(np.sum(group_fold == fold)) for fold in range(1, 6)},
            "repeat_groups_crossing_folds": int(assignments[assignments["is_repeat_like"]].groupby("repeat_group_id")["group_safe_fold"].nunique().gt(1).sum()),
            "all_models_use_identical_folds": True,
        },
        "models": {
            model: {"old_cv": old_metrics[model], "group_safe_cv": new_metrics[model]}
            for model in MODEL_SPECS
        },
        "old_vs_group_safe": comparison.to_dict("records"),
        "ranking": {
            "best_by_RMSE": best_rmse, "best_by_MAE": best_mae,
            "position_best_by_both": best_rmse == "position_regex_lightgbm" and best_mae == "position_regex_lightgbm",
            "position_direct_comparisons": direct,
        },
        "bootstrap": {
            "samples": 5_000, "seed": 42,
            "comparisons": bootstrap,
            "position_advantage_reliable_on_both_metrics_vs_all_references": bool(
                all(value["RMSE_CI95_upper_RM"] < 0 and value["MAE_CI95_upper_RM"] < 0 for value in bootstrap.values())
            ),
        },
        "invalid_row_fixed_fold_subtest": {
            "invalid_listing_ids": sorted(INVALID_RULES), "independent_evidence": invalid_evidence,
            "retained_rows": int(retained.sum()), "original_fold_membership_preserved": True,
            "original_retained_RMSE_RM": float(invalid_original["RMSE_RM"]),
            "original_retained_MAE_RM": float(invalid_original["MAE_RM"]),
            "retrained_retained_RMSE_RM": float(invalid_retrained["RMSE_RM"]),
            "retrained_retained_MAE_RM": float(invalid_retrained["MAE_RM"]),
            "RMSE_gain_original_minus_retrained_RM": float(invalid_original["RMSE_RM"] - invalid_retrained["RMSE_RM"]),
            "MAE_gain_original_minus_retrained_RM": float(invalid_original["MAE_RM"] - invalid_retrained["MAE_RM"]),
            "removal_improves_both": bool(invalid_retrained["RMSE_RM"] < invalid_original["RMSE_RM"] and invalid_retrained["MAE_RM"] < invalid_original["MAE_RM"]),
            "kept_separate_from_group_safe_cv": True,
        },
        "answers": {
            "repeat_listings_cause_optimistic_cv": repeated_optimism_evidence,
            "repeat_leakage_interpretation": "Evidence is consistent with repeat leakage causing optimistic historical CV: repeat groups crossed folds, both overall errors worsened, and repeat-row RMSE/MAE increased much more than non-repeat-row errors. The exact overall delta is not purely causal because group-safe CV reallocates folds.",
            "position_regex_remains_strongest": best_rmse == "position_regex_lightgbm" and best_mae == "position_regex_lightgbm",
        },
        "leakage_controls": {
            "preprocessing_outer_fold_local": True, "building_te_outer_training_only": True,
            "building_te_inner_oof": True, "regex_target_free": True,
            "validation_target_used_for_features_or_tuning": False,
            "fit_audit": fit_audit,
        },
        "reproducibility": {
            "python": platform.python_version(), "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__, "lightgbm": lightgbm.__version__,
            "random_seed": 42, "bootstrap_samples": 5_000,
        },
        "production_safety": {
            "protected_file_count": len(protected_before),
            "before_manifest_sha256": manifest_digest(protected_before),
            "after_manifest_sha256": manifest_digest(protected_after),
            "all_protected_files_unchanged": protected_before == protected_after,
        },
        "metric_thresholds": thresholds,
        "artifacts": [
            f"experiments/repeat_listing_leakage/{name}" for name in (
                "results.json", "repeat_groups.csv", "repeat_group_summary.csv",
                "group_safe_fold_assignments.csv", "model_comparison.csv", "fold_metrics.csv",
                "oof_predictions.csv", "invalid_row_fixed_fold_comparison.csv",
                "run_experiment.py", "test_invariants.py",
            )
        ],
        "runtime_seconds": time.perf_counter() - started,
    }
    (EXPERIMENT / "results.json").write_text(
        json.dumps(json_clean(results), indent=2), encoding="utf-8"
    )
    print(f"Completed repeat-listing leakage experiment in {results['runtime_seconds']:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
