"""Test whether incomplete-row deletion improves the Position-regex LightGBM.

All comparisons use original-Ringgit out-of-fold predictions.  The historical
canonical OOF predictions are never regenerated or overwritten; they are read
as the matched-row reference for every prescribed retained subset.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.description_text_features.model_builders import fit_lightgbm_fold
from experiments.description_text_features.regex_features import (
    REGEX_GROUPS,
    extract_regex_features,
)
from experiments.description_text_features.text_cleaning import link_descriptions
from src.cleaning.missing_values import MISSING_MARKERS
from src.models.common.features import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERICAL_FEATURES,
)


EXPERIMENT = ROOT / "experiments" / "missing_data_quality"
DATA_PATH = ROOT / "data" / "processed" / "enhanced_city_dataset.csv"
RAW_PATH = ROOT / "data" / "raw" / "houses.csv"
REFERENCE_OOF_PATH = ROOT / "experiments" / "description_text_features" / "oof_predictions.csv"
REFERENCE_VARIANT = "regex_group_position"
EXPECTED_ROWS = 3_791
EXPECTED_REFERENCE = {
    "rmse_rm": 118_750.19350875785,
    "mae_rm": 60_967.10968555279,
    "r2": 0.8702674858843791,
    "top5_rmse_rm": 412_319.68890075386,
}

IMPORTANT_FEATURES = [
    "price",
    "property_size_sqft",
    "bedroom",
    "bathroom",
    "parking_lot",
    "completion_year",
    "number_of_floors",
    "total_units",
    "property_type",
    "tenure_type",
    "land_title",
    "floor_range",
    "state",
    "city",
    "building_name",
    "developer",
]
CORE_FEATURES = [
    "property_size_sqft",
    "bedroom",
    "bathroom",
    "property_type",
    "city",
]
POSITION_FEATURES = list(REGEX_GROUPS["position"])
PRICE_BANDS = (
    ("P00_P50", 0.00, 0.50),
    ("P50_P80", 0.50, 0.80),
    ("P80_P90", 0.80, 0.90),
    ("P90_P95", 0.90, 0.95),
    ("P95_P99", 0.95, 0.99),
    ("P99_P100", 0.99, 1.00),
)
VARIANT_LABELS = {
    "A_current": "A. Current canonical dataset",
    "B_valid_critical": "B. Valid/non-missing price and property size",
    "C_complete_core": "C. Complete core fields",
    "D_missing_lt3": "D. Remove rows missing >=3 important fields",
    "E_missing_lt5": "E. Remove rows missing >=5 important fields",
    "F_completeness_ge80": "F. Completeness >=80%",
    "G_completeness_ge90": "G. Completeness >=90%",
    "H_missing_indicators": "H. Canonical rows plus missing indicators",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protected_snapshot() -> dict[str, str]:
    """Hash production, data, source, and historical-result files only."""
    roots = [
        ROOT / "data",
        ROOT / "results",
        ROOT / "prototype",
        ROOT / "src",
        ROOT / "configs",
        ROOT / "scripts",
    ]
    files = [ROOT / "app.py", ROOT / "README.md", ROOT / "requirements.txt"]
    for directory in roots:
        if directory.exists():
            files.extend(path for path in directory.rglob("*") if path.is_file())
    for directory in (ROOT / "experiments").iterdir():
        if directory.is_dir() and directory.name not in {"missing_data_quality", "__pycache__"}:
            files.extend(path for path in directory.rglob("*") if path.is_file())
    return {
        path.relative_to(ROOT).as_posix(): _sha256(path)
        for path in sorted(set(files))
        if path.exists()
    }


def _manifest_digest(snapshot: dict[str, str]) -> str:
    payload = "\n".join(f"{path}:{digest}" for path, digest in sorted(snapshot.items()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    if pd.isna(value):
        return None
    raise TypeError(type(value).__name__)


def _json_clean(value):
    """Recursively replace non-finite native/Pandas scalars before strict JSON."""
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_clean(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if value is pd.NA:
        return None
    return value


def semantic_missing(series: pd.Series) -> pd.Series:
    """Recognize repository missing sentinels without classifying zero as missing."""
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.isna() | ~np.isfinite(numeric.astype(float))
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.isna() | normalized.isin(MISSING_MARKERS)


def important_missing_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    missing = pd.DataFrame(
        {column: semantic_missing(frame[column]) for column in IMPORTANT_FEATURES},
        index=frame.index,
    )
    # Invalid critical values make PPSF undefined and therefore count as incomplete.
    for column in ("price", "property_size_sqft"):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        missing[column] |= numeric.le(0)
    return missing


def metric_bundle(actual, predicted, premium_mask) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    premium_mask = np.asarray(premium_mask, dtype=bool)
    if len(actual) == 0:
        return {
            "count": 0,
            "rmse_rm": None,
            "mae_rm": None,
            "r2": None,
            "top5_count": 0,
            "top5_rmse_rm": None,
            "top5_mae_rm": None,
        }
    top_actual = actual[premium_mask]
    top_prediction = predicted[premium_mask]
    return {
        "count": int(len(actual)),
        "rmse_rm": float(mean_squared_error(actual, predicted) ** 0.5),
        "mae_rm": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else None,
        "top5_count": int(premium_mask.sum()),
        "top5_rmse_rm": (
            float(mean_squared_error(top_actual, top_prediction) ** 0.5)
            if len(top_actual)
            else None
        ),
        "top5_mae_rm": (
            float(mean_absolute_error(top_actual, top_prediction))
            if len(top_actual)
            else None
        ),
    }


def price_band_masks(price: pd.Series) -> tuple[dict[str, np.ndarray], dict[float, float]]:
    values = price.to_numpy(float)
    quantiles = sorted({q for _, low, high in PRICE_BANDS for q in (low, high)})
    boundaries = {q: float(np.quantile(values, q)) for q in quantiles}
    masks = {}
    for name, low, high in PRICE_BANDS:
        mask = values >= boundaries[low]
        mask &= values <= boundaries[high] if high == 1.0 else values < boundaries[high]
        masks[name] = mask
    return masks, boundaries


def missingness_summary(frame: pd.DataFrame, regex: pd.DataFrame) -> pd.DataFrame:
    rows = []
    relevant = ["price", *MODEL_FEATURES]
    for order, column in enumerate(relevant):
        mask = semantic_missing(frame[column])
        rows.append(
            {
                "feature": column,
                "source": "canonical",
                "feature_order": order,
                "important_feature": column in IMPORTANT_FEATURES,
                "missing_count": int(mask.sum()),
                "missing_pct": float(mask.mean() * 100.0),
                "non_missing_count": int((~mask).sum()),
                "zero_count": (
                    int(pd.to_numeric(frame[column], errors="coerce").eq(0).sum())
                    if pd.api.types.is_numeric_dtype(frame[column])
                    else None
                ),
                "zero_treated_as_missing": False,
            }
        )
    for offset, column in enumerate(POSITION_FEATURES, start=len(rows)):
        rows.append(
            {
                "feature": column,
                "source": "derived_position_regex",
                "feature_order": offset,
                "important_feature": False,
                "missing_count": 0,
                "missing_pct": 0.0,
                "non_missing_count": len(regex),
                "zero_count": int(regex[column].eq(0).sum()),
                "zero_treated_as_missing": False,
            }
        )
    return pd.DataFrame(rows)


def missingness_by_price_band(
    frame: pd.DataFrame, important_missing: pd.DataFrame, masks: dict[str, np.ndarray]
) -> pd.DataFrame:
    rows = []
    for band, mask in masks.items():
        count = int(mask.sum())
        subset = important_missing.loc[mask]
        missing_count_per_row = subset.sum(axis=1)
        for feature in IMPORTANT_FEATURES:
            missing_count = int(subset[feature].sum())
            rows.append(
                {
                    "price_band": band,
                    "row_count": count,
                    "feature": feature,
                    "missing_count": missing_count,
                    "missing_pct": float(missing_count / count * 100.0) if count else None,
                    "non_missing_count": count - missing_count,
                    "mean_missing_important_count": float(missing_count_per_row.mean()),
                    "complete_rows": int(missing_count_per_row.eq(0).sum()),
                    "complete_rows_pct": float(missing_count_per_row.eq(0).mean() * 100.0),
                }
            )
    return pd.DataFrame(rows)


def grouped_missingness(
    frame: pd.DataFrame, important_missing: pd.DataFrame, group: pd.Series, name: str
) -> list[dict]:
    labels = group.astype("string").fillna("<NA>")
    rows = []
    for label in sorted(labels.unique().tolist()):
        mask = labels.eq(label).to_numpy()
        subset = important_missing.loc[mask]
        row_missing = subset.sum(axis=1)
        rows.append(
            {
                "dimension": name,
                "value": str(label),
                "rows": int(mask.sum()),
                "mean_missing_important_count": float(row_missing.mean()),
                "complete_rows_pct": float(row_missing.eq(0).mean() * 100.0),
                "feature_missing_pct": {
                    column: float(subset[column].mean() * 100.0)
                    for column in IMPORTANT_FEATURES
                },
            }
        )
    return rows


def categorical_distribution(series: pd.Series) -> dict[str, float]:
    labels = series.astype("string").fillna("<NA>")
    return {str(k): float(v) for k, v in labels.value_counts(normalize=True).sort_index().items()}


def total_variation(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    return float(0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys))


def distribution_summary(
    frame: pd.DataFrame,
    base: pd.DataFrame,
    premium_mask: np.ndarray,
    top1_mask: np.ndarray,
    retained_indices: np.ndarray,
) -> dict:
    price = frame["price"].astype(float)
    base_price = base["price"].astype(float)
    property_distribution = categorical_distribution(frame["property_type"])
    state_distribution = categorical_distribution(frame["state"])
    base_property_distribution = categorical_distribution(base["property_type"])
    base_state_distribution = categorical_distribution(base["state"])
    stats = {
        "mean_price_rm": float(price.mean()),
        "median_price_rm": float(price.median()),
        "price_std_rm": float(price.std(ddof=1)),
        "price_p95_rm": float(price.quantile(0.95)),
        "price_p99_rm": float(price.quantile(0.99)),
        "maximum_price_rm": float(price.max()),
        "price_skewness": float(price.skew()),
        "property_type_distribution": property_distribution,
        "state_distribution": state_distribution,
        "property_type_total_variation": total_variation(
            property_distribution, base_property_distribution
        ),
        "state_total_variation": total_variation(state_distribution, base_state_distribution),
        "premium_share_pct": float(premium_mask.mean() * 100.0),
        "premium_retention_pct": float(premium_mask.sum() / max(1, (base_price >= base_price.quantile(0.95)).sum()) * 100.0),
        "top1_retention_pct": float(top1_mask.sum() / max(1, (base_price >= base_price.quantile(0.99)).sum()) * 100.0),
    }
    base_stats = {
        "mean_price_rm": float(base_price.mean()),
        "median_price_rm": float(base_price.median()),
        "price_std_rm": float(base_price.std(ddof=1)),
        "price_p95_rm": float(base_price.quantile(0.95)),
        "price_p99_rm": float(base_price.quantile(0.99)),
        "maximum_price_rm": float(base_price.max()),
        "price_skewness": float(base_price.skew()),
    }
    relative_changes = {
        name: float((stats[name] - value) / value * 100.0) if value else 0.0
        for name, value in base_stats.items()
    }
    reasons = []
    retention_pct = len(frame) / len(base) * 100.0
    if retention_pct < 90.0:
        reasons.append("row retention below 90%")
    for name in ("mean_price_rm", "median_price_rm", "price_std_rm", "price_p95_rm", "price_p99_rm", "maximum_price_rm"):
        if abs(relative_changes[name]) >= 10.0:
            reasons.append(f"{name} changed by at least 10%")
    if stats["premium_retention_pct"] < 90.0:
        reasons.append("premium retention below 90%")
    if stats["top1_retention_pct"] < 90.0:
        reasons.append("top-1% retention below 90%")
    if stats["property_type_total_variation"] >= 0.10:
        reasons.append("property-type total variation at least 0.10")
    if stats["state_total_variation"] >= 0.10:
        reasons.append("state total variation at least 0.10")
    stats["relative_change_pct"] = relative_changes
    stats["substantial_population_shift"] = bool(reasons)
    stats["population_shift_reasons"] = reasons
    stats["retained_canonical_row_indices"] = retained_indices.tolist()
    return stats


def removed_diagnostics(
    frame: pd.DataFrame,
    original_prediction: np.ndarray,
    retained_mask: np.ndarray,
    premium_threshold: float,
    top1_threshold: float,
) -> dict:
    removed = ~retained_mask
    count = int(removed.sum())
    if count == 0:
        return {
            "count": 0,
            "rmse_original_oof_rm": None,
            "mae_original_oof_rm": None,
            "mean_price_rm": None,
            "median_price_rm": None,
            "premium_pct": None,
            "top5_removed_count": 0,
            "top1_removed_count": 0,
        }
    actual = frame.loc[removed, "price"].to_numpy(float)
    prediction = original_prediction[removed]
    return {
        "count": count,
        "rmse_original_oof_rm": float(mean_squared_error(actual, prediction) ** 0.5),
        "mae_original_oof_rm": float(mean_absolute_error(actual, prediction)),
        "mean_price_rm": float(actual.mean()),
        "median_price_rm": float(np.median(actual)),
        "premium_pct": float((actual >= premium_threshold).mean() * 100.0),
        "top5_removed_count": int((actual >= premium_threshold).sum()),
        "top1_removed_count": int((actual >= top1_threshold).sum()),
    }


def fit_variant(
    variant: str,
    frame: pd.DataFrame,
    regex: pd.DataFrame,
    original_prediction: np.ndarray,
    premium_threshold: float,
    indicator_frame: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict], pd.DataFrame]:
    X = frame.drop(columns=["price"]).reset_index(drop=True)
    y = frame["price"].to_numpy(float)
    position = regex[POSITION_FEATURES].reset_index(drop=True)
    if indicator_frame is not None:
        dense = position.join(indicator_frame.reset_index(drop=True))
    else:
        dense = position
    folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(X))
    predictions = np.empty(len(frame), dtype=float)
    fold_assignment = np.empty(len(frame), dtype=int)
    fold_rows = []
    for fold, (train_index, validation_index) in enumerate(folds, 1):
        output = fit_lightgbm_fold(
            X.iloc[train_index],
            y[train_index],
            X.iloc[validation_index],
            dense.loc[train_index],
            dense.loc[validation_index],
        )
        predicted = output["validation_prediction"]
        predictions[validation_index] = predicted
        fold_assignment[validation_index] = fold
        premium = y[validation_index] >= premium_threshold
        retrained_metrics = metric_bundle(y[validation_index], predicted, premium)
        matched_metrics = metric_bundle(
            y[validation_index], original_prediction[validation_index], premium
        )
        for source, values in (
            ("retrained", retrained_metrics),
            ("original_model_matched", matched_metrics),
        ):
            fold_rows.append(
                {
                    "variant": variant,
                    "fold": fold,
                    "metric_source": source,
                    "train_rows": int(len(train_index)),
                    "validation_rows": int(len(validation_index)),
                    **values,
                }
            )
    oof = pd.DataFrame(
        {
            "variant": variant,
            "retained_row_position": np.arange(len(frame)),
            "fold": fold_assignment,
            "actual_price_rm": y,
            "original_model_oof_prediction_rm": original_prediction,
            "retrained_model_oof_prediction_rm": predictions,
            "original_absolute_error_rm": np.abs(original_prediction - y),
            "retrained_absolute_error_rm": np.abs(predictions - y),
            "premium_flag": y >= premium_threshold,
        }
    )
    return predictions, fold_assignment, fold_rows, oof


def paired_bootstrap(
    actual: np.ndarray,
    retrained: np.ndarray,
    original: np.ndarray,
    draws: int = 5_000,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    count = len(actual)
    rmse_differences = np.empty(draws, dtype=float)
    mae_differences = np.empty(draws, dtype=float)
    retrained_error = retrained - actual
    original_error = original - actual
    for draw in range(draws):
        selected = rng.integers(0, count, size=count)
        rmse_differences[draw] = (
            np.sqrt(np.mean(np.square(retrained_error[selected])))
            - np.sqrt(np.mean(np.square(original_error[selected])))
        )
        mae_differences[draw] = (
            np.mean(np.abs(retrained_error[selected]))
            - np.mean(np.abs(original_error[selected]))
        )
    return {
        "draws": draws,
        "seed": seed,
        "difference_definition": "retrained minus original on the same retained rows; negative is better",
        "rmse_difference_rm": {
            "observed": float(
                np.sqrt(np.mean(np.square(retrained_error)))
                - np.sqrt(np.mean(np.square(original_error)))
            ),
            "ci95_lower": float(np.quantile(rmse_differences, 0.025)),
            "ci95_upper": float(np.quantile(rmse_differences, 0.975)),
            "ci_crosses_zero": bool(
                np.quantile(rmse_differences, 0.025) <= 0 <= np.quantile(rmse_differences, 0.975)
            ),
        },
        "mae_difference_rm": {
            "observed": float(np.mean(np.abs(retrained_error)) - np.mean(np.abs(original_error))),
            "ci95_lower": float(np.quantile(mae_differences, 0.025)),
            "ci95_upper": float(np.quantile(mae_differences, 0.975)),
            "ci_crosses_zero": bool(
                np.quantile(mae_differences, 0.025) <= 0 <= np.quantile(mae_differences, 0.975)
            ),
        },
    }


def load_reference(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    reference = pd.read_csv(REFERENCE_OOF_PATH)
    reference = reference.loc[reference["variant"].eq(REFERENCE_VARIANT)].copy()
    if len(reference) != len(frame) or reference["row_id"].nunique() != len(frame):
        raise AssertionError("Historical Position-regex OOF reference is not one row per listing.")
    reference = reference.set_index("row_id").loc[frame["listing_id"].astype(int)].reset_index()
    if not np.array_equal(reference["actual_price_RM"].to_numpy(float), frame["price"].to_numpy(float)):
        raise AssertionError("Historical OOF actual prices do not align to the canonical dataset.")
    return (
        reference["predicted_price_RM"].to_numpy(float),
        reference["fold"].to_numpy(int),
    )


def main() -> None:
    started = time.perf_counter()
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    before = _protected_snapshot()
    frame = pd.read_csv(DATA_PATH)
    if len(frame) != EXPECTED_ROWS or frame["listing_id"].nunique() != EXPECTED_ROWS:
        raise AssertionError("Canonical row grain changed.")
    descriptions, linkage = link_descriptions(RAW_PATH, frame["listing_id"])
    all_regex = extract_regex_features(descriptions)
    original_prediction, original_fold = load_reference(frame)

    important_missing = important_missing_matrix(frame)
    missing_count = important_missing.sum(axis=1).astype(int)
    completeness = 1.0 - missing_count / len(IMPORTANT_FEATURES)
    frame_with_quality = frame.copy()
    frame_with_quality["missing_important_count"] = missing_count
    frame_with_quality["completeness_ratio"] = completeness

    price_band_mask, price_boundaries = price_band_masks(frame["price"])
    premium_threshold = price_boundaries[0.95]
    top1_threshold = price_boundaries[0.99]
    canonical_premium = frame["price"].to_numpy(float) >= premium_threshold
    canonical_top1 = frame["price"].to_numpy(float) >= top1_threshold

    summary_missing = missingness_summary(frame, all_regex)
    by_price = missingness_by_price_band(frame, important_missing, price_band_mask)
    grouped = {
        "price_band": [
            item
            for band, mask in price_band_mask.items()
            for item in grouped_missingness(
                frame.loc[mask], important_missing.loc[mask],
                pd.Series([band] * int(mask.sum()), index=frame.index[mask]), "price_band"
            )
        ],
        "property_type": grouped_missingness(
            frame, important_missing, frame["property_type"], "property_type"
        ),
        "state": grouped_missingness(frame, important_missing, frame["state"], "state"),
        "premium_status": grouped_missingness(
            frame,
            important_missing,
            pd.Series(np.where(canonical_premium, "premium", "non_premium"), index=frame.index),
            "premium_status",
        ),
    }

    critical_valid = ~(important_missing["price"] | important_missing["property_size_sqft"])
    variant_masks = {
        "A_current": np.ones(len(frame), dtype=bool),
        "B_valid_critical": critical_valid.to_numpy(),
        "C_complete_core": (critical_valid & ~important_missing[CORE_FEATURES].any(axis=1)).to_numpy(),
        "D_missing_lt3": (critical_valid & missing_count.lt(3)).to_numpy(),
        "E_missing_lt5": (critical_valid & missing_count.lt(5)).to_numpy(),
        "F_completeness_ge80": (critical_valid & completeness.ge(0.80)).to_numpy(),
        "G_completeness_ge90": (critical_valid & completeness.ge(0.90)).to_numpy(),
        "H_missing_indicators": np.ones(len(frame), dtype=bool),
    }

    meaningful_indicator_features = [
        column
        for column in MODEL_FEATURES
        if semantic_missing(frame[column]).any()
    ]
    missing_indicators = pd.DataFrame(
        {
            f"{column}_missing": semantic_missing(frame[column]).astype(np.int8)
            for column in meaningful_indicator_features
        },
        index=frame.index,
    )

    variant_results = {}
    fold_rows = []
    oof_parts = []
    summary_rows = []
    for variant, retained_mask in variant_masks.items():
        retained_indices = np.flatnonzero(retained_mask)
        subset = frame.iloc[retained_indices].reset_index(drop=True)
        subset_regex = all_regex.iloc[retained_indices].reset_index(drop=True)
        subset_original = original_prediction[retained_indices]
        subset_indicators = (
            missing_indicators.iloc[retained_indices].reset_index(drop=True)
            if variant == "H_missing_indicators"
            else None
        )
        retrained_prediction, fold_assignment, variant_folds, variant_oof = fit_variant(
            variant,
            subset,
            subset_regex,
            subset_original,
            premium_threshold,
            indicator_frame=subset_indicators,
        )
        variant_oof.insert(2, "canonical_row_index", retained_indices)
        variant_oof.insert(3, "listing_id", subset["listing_id"].to_numpy(int))
        variant_oof["missing_important_count"] = missing_count.iloc[retained_indices].to_numpy(int)
        variant_oof["completeness_ratio"] = completeness.iloc[retained_indices].to_numpy(float)
        variant_oof["top1_flag"] = subset["price"].to_numpy(float) >= top1_threshold
        oof_parts.append(variant_oof)
        fold_rows.extend(variant_folds)

        premium = subset["price"].to_numpy(float) >= premium_threshold
        top1 = subset["price"].to_numpy(float) >= top1_threshold
        retrained_metrics = metric_bundle(subset["price"], retrained_prediction, premium)
        matched_metrics = metric_bundle(subset["price"], subset_original, premium)
        gains = {
            "retraining_rmse_gain_rm": matched_metrics["rmse_rm"] - retrained_metrics["rmse_rm"],
            "retraining_mae_gain_rm": matched_metrics["mae_rm"] - retrained_metrics["mae_rm"],
            "gain_definition": "original matched metric minus retrained metric; positive is better",
        }
        removed = removed_diagnostics(
            frame, original_prediction, retained_mask, premium_threshold, top1_threshold
        )
        distribution = distribution_summary(
            subset, frame, premium, top1, retained_indices
        )
        retained_top5 = int(premium.sum())
        retained_top1 = int(top1.sum())
        result = {
            "label": VARIANT_LABELS[variant],
            "rows_retained": int(retained_mask.sum()),
            "rows_removed": int((~retained_mask).sum()),
            "retention_pct": float(retained_mask.mean() * 100.0),
            "predicate": {
                "A_current": "all canonical rows",
                "B_valid_critical": "finite positive price and property_size_sqft",
                "C_complete_core": "complete property_size_sqft, bedroom, bathroom, property_type, and city",
                "D_missing_lt3": "missing_important_count < 3",
                "E_missing_lt5": "missing_important_count < 5",
                "F_completeness_ge80": "completeness_ratio >= 0.80",
                "G_completeness_ge90": "completeness_ratio >= 0.90",
                "H_missing_indicators": "all canonical rows plus meaningful missing flags",
            }[variant],
            "retrained_model": retrained_metrics,
            "original_model_on_retained_rows": matched_metrics,
            "retraining_gains": gains,
            "premium_and_tail_retention": {
                "premium_threshold_rm": premium_threshold,
                "top1_threshold_rm": top1_threshold,
                "premium_rows_retained": retained_top5,
                "premium_rows_removed": int(canonical_premium.sum() - retained_top5),
                "premium_retention_pct": float(retained_top5 / canonical_premium.sum() * 100.0),
                "top1_rows_retained": retained_top1,
                "top1_rows_removed": int(canonical_top1.sum() - retained_top1),
                "top1_retention_pct": float(retained_top1 / canonical_top1.sum() * 100.0),
            },
            "removed_rows": removed,
            "distribution_shift": distribution,
            "missing_indicator_features": (
                list(missing_indicators.columns) if variant == "H_missing_indicators" else []
            ),
        }
        variant_results[variant] = result
        summary_rows.append(
            {
                "variant": variant,
                "label": result["label"],
                "rows_retained": result["rows_retained"],
                "rows_removed": result["rows_removed"],
                "retention_pct": result["retention_pct"],
                "rmse_rm": retrained_metrics["rmse_rm"],
                "mae_rm": retrained_metrics["mae_rm"],
                "r2": retrained_metrics["r2"],
                "top5_rmse_rm": retrained_metrics["top5_rmse_rm"],
                "top5_mae_rm": retrained_metrics["top5_mae_rm"],
                "matched_original_rmse_rm": matched_metrics["rmse_rm"],
                "matched_original_mae_rm": matched_metrics["mae_rm"],
                **gains,
                "removed_original_rmse_rm": removed["rmse_original_oof_rm"],
                "removed_original_mae_rm": removed["mae_original_oof_rm"],
                "removed_mean_price_rm": removed["mean_price_rm"],
                "removed_median_price_rm": removed["median_price_rm"],
                "removed_premium_pct": removed["premium_pct"],
                "top5_removed_count": removed["top5_removed_count"],
                "top1_removed_count": removed["top1_removed_count"],
                "premium_retention_pct": distribution["premium_retention_pct"],
                "top1_retention_pct": distribution["top1_retention_pct"],
                "mean_price_rm": distribution["mean_price_rm"],
                "median_price_rm": distribution["median_price_rm"],
                "price_std_rm": distribution["price_std_rm"],
                "price_p95_rm": distribution["price_p95_rm"],
                "price_p99_rm": distribution["price_p99_rm"],
                "maximum_price_rm": distribution["maximum_price_rm"],
                "price_skewness": distribution["price_skewness"],
                "property_type_total_variation": distribution["property_type_total_variation"],
                "state_total_variation": distribution["state_total_variation"],
                "substantial_population_shift": distribution["substantial_population_shift"],
                "population_shift_reasons": json.dumps(distribution["population_shift_reasons"]),
                "property_type_distribution": json.dumps(distribution["property_type_distribution"], sort_keys=True),
                "state_distribution": json.dumps(distribution["state_distribution"], sort_keys=True),
            }
        )
        print(
            f"Completed {variant}: n={len(subset):,}, "
            f"RMSE={retrained_metrics['rmse_rm']:,.2f}, "
            f"matched gain={gains['retraining_rmse_gain_rm']:,.2f}",
            flush=True,
        )

    bootstraps = {}
    for variant in ("B_valid_critical", "C_complete_core", "D_missing_lt3", "E_missing_lt5", "F_completeness_ge80", "G_completeness_ge90"):
        result = variant_results[variant]
        gains = result["retraining_gains"]
        if (
            result["rows_removed"] > 0
            and gains["retraining_rmse_gain_rm"] > 0
            and gains["retraining_mae_gain_rm"] > 0
        ):
            retained = variant_masks[variant]
            rows = pd.concat(oof_parts, ignore_index=True)
            selected = rows.loc[rows["variant"].eq(variant)]
            bootstraps[variant] = paired_bootstrap(
                selected["actual_price_rm"].to_numpy(float),
                selected["retrained_model_oof_prediction_rm"].to_numpy(float),
                selected["original_model_oof_prediction_rm"].to_numpy(float),
            )

    summary_frame = pd.DataFrame(summary_rows)
    fold_frame = pd.DataFrame(fold_rows)
    oof_frame = pd.concat(oof_parts, ignore_index=True)
    summary_missing.to_csv(EXPERIMENT / "missingness_summary.csv", index=False)
    by_price.to_csv(EXPERIMENT / "missingness_by_price_band.csv", index=False)
    summary_frame.to_csv(EXPERIMENT / "dataset_variant_summary.csv", index=False)
    fold_frame.to_csv(EXPERIMENT / "fold_metrics.csv", index=False)
    oof_frame.to_csv(EXPERIMENT / "oof_predictions.csv", index=False)

    reference_metrics = metric_bundle(
        frame["price"], original_prediction, canonical_premium
    )
    reference_checks = {
        name: bool(math.isclose(reference_metrics[name], expected, rel_tol=0.0, abs_tol=1e-6))
        for name, expected in EXPECTED_REFERENCE.items()
    }
    promising = [
        variant
        for variant, result in variant_results.items()
        if result["rows_removed"] > 0
        and result["retraining_gains"]["retraining_rmse_gain_rm"] > 0
        and result["retraining_gains"]["retraining_mae_gain_rm"] > 0
    ]
    significant_both = [
        variant
        for variant, interval in bootstraps.items()
        if interval["rmse_difference_rm"]["ci95_upper"] < 0
        and interval["mae_difference_rm"]["ci95_upper"] < 0
    ]
    best_filter = min(
        (name for name in variant_results if name[0] in "BCDEFG"),
        key=lambda name: variant_results[name]["retrained_model"]["rmse_rm"],
    )

    after = _protected_snapshot()
    changed = sorted(
        path for path in set(before) | set(after) if before.get(path) != after.get(path)
    )
    results = {
        "question": "Does deleting incomplete listings genuinely improve the model, or only restrict evaluation to an easier subset?",
        "methodology": {
            "source_dataset": DATA_PATH.relative_to(ROOT).as_posix(),
            "canonical_rows": len(frame),
            "unit_of_analysis": "one unique canonical listing",
            "missing_definition": "null/non-finite values plus repository missing text markers including Unknown; numeric and binary zero remains valid",
            "important_features": IMPORTANT_FEATURES,
            "completeness_formula": "1 - missing_important_count / 16",
            "model": "exact existing Position-regex LightGBM PPSF configuration",
            "position_regex_features": POSITION_FEATURES,
            "cv": {"class": "KFold", "n_splits": 5, "shuffle": True, "random_state": 42},
            "headline_evaluation": "OOF original total-price RM",
            "matched_comparison": "historical canonical Position-regex OOF predictions evaluated only on each retained row set",
            "premium_definition": "canonical actual-price top-5% threshold, fixed across variants",
            "top1_definition": "canonical actual-price top-1% threshold, fixed across variants",
            "no_price_threshold_deletion": True,
            "no_hyperparameter_retuning": True,
        },
        "inputs": {
            "canonical_dataset_sha256": _sha256(DATA_PATH),
            "raw_description_source_sha256": _sha256(RAW_PATH),
            "historical_oof_sha256": _sha256(REFERENCE_OOF_PATH),
            "description_linkage": linkage,
        },
        "canonical_reference": {
            "variant": REFERENCE_VARIANT,
            "metrics": reference_metrics,
            "expected_metrics": EXPECTED_REFERENCE,
            "expected_metrics_reproduced": all(reference_checks.values()),
            "individual_checks": reference_checks,
            "premium_threshold_rm": premium_threshold,
            "top1_threshold_rm": top1_threshold,
            "historical_fold_alignment_retained": bool(len(original_fold) == len(frame)),
        },
        "missingness": {
            "highest_missingness_features": summary_missing.sort_values("missing_pct", ascending=False).head(10).to_dict("records"),
            "important_feature_summary": summary_missing.loc[summary_missing["important_feature"]].to_dict("records"),
            "row_missing_important_distribution": {
                "0": int(missing_count.eq(0).sum()),
                "1": int(missing_count.eq(1).sum()),
                "2": int(missing_count.eq(2).sum()),
                "3": int(missing_count.eq(3).sum()),
                "4": int(missing_count.eq(4).sum()),
                "5+": int(missing_count.ge(5).sum()),
            },
            "completeness_ratio": {
                "mean": float(completeness.mean()),
                "median": float(completeness.median()),
                "minimum": float(completeness.min()),
                "maximum": float(completeness.max()),
            },
            "by_dimensions": grouped,
        },
        "missing_indicator_alternative": {
            "variant": "H_missing_indicators",
            "features_added": list(missing_indicators.columns),
            "feature_count": len(missing_indicators.columns),
            "result": variant_results["H_missing_indicators"],
        },
        "variants": variant_results,
        "bootstrap": {
            "eligibility": "filtered variant removed rows and improved both matched-row RMSE and MAE",
            "eligible_variants": promising,
            "comparisons": bootstraps,
            "significant_both_metrics": significant_both,
            "fixed_oof_limitation": "Intervals condition on these OOF predictions and do not include model-selection or repeated-CV uncertainty.",
        },
        "decision": {
            "lowest_headline_rmse_filter": best_filter,
            "filters_improving_both_matched_metrics": promising,
            "filters_significantly_improving_both_matched_metrics": significant_both,
            "any_deletion_rule_genuinely_improves_both": bool(promising),
            "any_deletion_rule_significantly_improves_both": bool(significant_both),
            "interpretation_rule": "A lower filtered headline metric is an easier-subset effect unless retraining beats the original model on the identical retained rows.",
        },
        "production_safety": {
            "protected_file_count": len(before),
            "all_protected_files_unchanged": before == after,
            "changed_protected_files": changed,
            "before_manifest_sha256": _manifest_digest(before),
            "after_manifest_sha256": _manifest_digest(after),
        },
        "internal_tests": {
            "canonical_grain": len(frame) == EXPECTED_ROWS and frame["listing_id"].nunique() == EXPECTED_ROWS,
            "reference_metrics_reproduced": all(reference_checks.values()),
            "every_variant_has_five_folds": bool(fold_frame.groupby(["variant", "metric_source"])["fold"].nunique().eq(5).all()),
            "every_retained_row_has_one_oof_prediction": bool(oof_frame.groupby("variant")["canonical_row_index"].nunique().eq(summary_frame.set_index("variant")["rows_retained"]).all()),
            "required_variants_only": set(variant_results) == set(VARIANT_LABELS),
            "zeros_not_missing": not bool(semantic_missing(pd.Series([0, 1], dtype=float)).any()),
            "protected_files_unchanged": before == after,
            "all_passed": False,
        },
        "runtime_seconds": time.perf_counter() - started,
        "artifacts": [
            "experiments/missing_data_quality/results.json",
            "experiments/missing_data_quality/dataset_variant_summary.csv",
            "experiments/missing_data_quality/missingness_summary.csv",
            "experiments/missing_data_quality/missingness_by_price_band.csv",
            "experiments/missing_data_quality/fold_metrics.csv",
            "experiments/missing_data_quality/oof_predictions.csv",
        ],
    }
    results["internal_tests"]["all_passed"] = all(
        value for key, value in results["internal_tests"].items() if key != "all_passed"
    )
    with (EXPERIMENT / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_clean(results), handle, indent=2, default=_json_default, allow_nan=False)
    print(
        f"Completed missing-data quality experiment in {results['runtime_seconds']:.1f}s; "
        f"protected files unchanged={before == after}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
