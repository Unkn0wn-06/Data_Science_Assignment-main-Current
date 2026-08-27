"""Evaluate spatial readiness and explicit leakage-safe PPSF target encoding."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from experiments.advanced_real_estate_models.feature_engineering import (
    FeatureEngineeringPPSFRegressor,
    engineered_feature_lists,
)
from experiments.advanced_real_estate_models.model_builders import (
    build_base_regressor,
    candidate_parameters,
)
from experiments.spatial_target_encoding.evaluation import (
    attach_reference_comparisons,
    evaluate_variant,
    paired_bootstrap,
)
from experiments.spatial_target_encoding.spatial_features import (
    SpatialGeometryFeatures,
    SpatialPPSFNeighborEncoder,
    find_coordinate_columns,
)
from experiments.spatial_target_encoding.target_encoding import (
    AdvancedTargetEncodingPPSFRegressor,
    DEFAULT_M_VALUES,
)
from src.cleaning.enhanced_city import ENHANCED_CITY_DATA_PATH
from src.models.common.features import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
)
from src.models.enhanced_city import shared_folds


EXPERIMENT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = EXPERIMENT_DIR / "figures"
ADVANCED_RESULTS_PATH = PROJECT_ROOT / "experiments/advanced_real_estate_models/results.json"
RAW_DATA_PATH = PROJECT_ROOT / "data/raw/houses.csv"
RESULTS_PATH = EXPERIMENT_DIR / "results.json"
COMPARISON_PATH = EXPERIMENT_DIR / "model_comparison.csv"
FOLD_METRICS_PATH = EXPERIMENT_DIR / "fold_metrics.csv"
OOF_PATH = EXPERIMENT_DIR / "oof_predictions.csv"
FEATURE_SUMMARY_PATH = EXPERIMENT_DIR / "feature_summary.json"

PROTECTED_PATHS = (
    PROJECT_ROOT / "data/raw/houses.csv",
    ENHANCED_CITY_DATA_PATH,
    PROJECT_ROOT / "results/enhanced_city/model_comparison.json",
    PROJECT_ROOT / "results/best_model/best_model_summary.json",
    ADVANCED_RESULTS_PATH,
    PROJECT_ROOT / "prototype/app.py",
    PROJECT_ROOT / "app.py",
)

VARIANT_SPECS = {
    "developer_te": {
        "columns": ("developer",), "retain_raw": False, "frequency": False,
    },
    "building_name_te": {
        "columns": ("building_name",), "retain_raw": False, "frequency": False,
    },
    "city_te": {
        "columns": ("city",), "retain_raw": False, "frequency": False,
    },
    "developer_building_te": {
        "columns": ("developer", "building_name"), "retain_raw": False,
        "frequency": False,
    },
    "all_te_replace": {
        "columns": ("developer", "building_name", "city"), "retain_raw": False,
        "frequency": False,
    },
    "all_te_plus_raw": {
        "columns": ("developer", "building_name", "city"), "retain_raw": True,
        "frequency": False,
    },
    "all_te_plus_frequency": {
        "columns": ("developer", "building_name", "city"), "retain_raw": False,
        "frequency": True,
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coordinate_inventory(data: pd.DataFrame) -> dict:
    raw_columns = pd.read_csv(RAW_DATA_PATH, nrows=0).columns.tolist()
    processed_lat, processed_lon = find_coordinate_columns(data.columns)
    raw_lat, raw_lon = find_coordinate_columns(raw_columns)
    available = processed_lat is not None and processed_lon is not None
    return {
        "coordinates_available": available,
        "processed_coordinate_columns": {
            "latitude": processed_lat,
            "longitude": processed_lon,
        },
        "raw_coordinate_columns": {"latitude": raw_lat, "longitude": raw_lon},
        "valid_coordinates": 0,
        "missing_coordinates": int(len(data)),
        "invalid_coordinates": 0,
        "spatial_coverage_percent": 0.0,
        "status": (
            "available" if available else "not_run_missing_coordinates"
        ),
        "note": (
            "No latitude/longitude aliases exist in processed or raw schemas; no coordinates were fabricated or approximated."
            if not available
            else "Coordinates found and require validation before spatial modelling."
        ),
    }


def build_lightgbm_interaction_baseline(params):
    numerical, categorical = engineered_feature_lists(
        NUMERICAL_FEATURES,
        CATEGORICAL_FEATURES,
        include_micro=False,
        include_interactions=True,
    )
    base = build_base_regressor(
        "lightgbm", params, numerical, categorical
    )
    return (
        FeatureEngineeringPPSFRegressor(
            base,
            include_micro=False,
            include_interactions=True,
        ),
        len(numerical) + len(categorical),
        numerical,
        categorical,
    )


def reference_payload(source: dict) -> dict:
    """Keep only the measured reference fields used by this experiment."""
    return {
        "name": source["name"],
        "metrics": source["metrics"],
        "generalization_gap": source.get("generalization_gap", {}),
        "source": ADVANCED_RESULTS_PATH.relative_to(PROJECT_ROOT).as_posix(),
    }


def comparison_row(key: str, result: dict) -> dict:
    metrics = result["metrics"]
    gap = result["generalization_gap"]
    rf = result["comparisons"]["random_forest"]
    lgb = result["comparisons"]["lightgbm_interaction"]
    return {
        "variant": key,
        "RMSE_RM": metrics["RMSE_RM"],
        "MAE_RM": metrics["MAE_RM"],
        "R2": metrics["R2"],
        "Adjusted_R2": metrics["Adjusted_R2"],
        "Median_AE_RM": metrics["Median_AE_RM"],
        "Mean_Error_RM": metrics["Mean_Error_RM"],
        "Median_Error_RM": metrics["Median_Error_RM"],
        "Top5_RMSE_RM": metrics["top_5_percent"]["RMSE_RM"],
        "Top5_MAE_RM": metrics["top_5_percent"]["MAE_RM"],
        "Remaining95_RMSE_RM": metrics["remaining_95_percent"]["RMSE_RM"],
        "Remaining95_MAE_RM": metrics["remaining_95_percent"]["MAE_RM"],
        "Training_RMSE_RM": gap["Training_RMSE_RM"],
        "Validation_RMSE_RM": gap["Validation_RMSE_RM"],
        "RMSE_gap_RM": gap["RMSE_gap_RM"],
        "Training_MAE_RM": gap["Training_MAE_RM"],
        "Validation_MAE_RM": gap["Validation_MAE_RM"],
        "MAE_gap_RM": gap["MAE_gap_RM"],
        "Training_R2": gap["Training_R2"],
        "Validation_R2": gap["Validation_R2"],
        "R2_gap": gap["R2_gap"],
        "RMSE_change_vs_RF_RM": rf["RMSE_difference_RM"],
        "RMSE_change_vs_RF_percent": rf["RMSE_percentage_change"],
        "MAE_change_vs_RF_RM": rf["MAE_difference_RM"],
        "MAE_change_vs_RF_percent": rf["MAE_percentage_change"],
        "R2_change_vs_RF": rf["R2_difference"],
        "Top5_RMSE_change_vs_RF_RM": rf["Top5_RMSE_difference_RM"],
        "Top5_MAE_change_vs_RF_RM": rf["Top5_MAE_difference_RM"],
        "RMSE_change_vs_LightGBM_RM": lgb["RMSE_difference_RM"],
        "RMSE_change_vs_LightGBM_percent": lgb["RMSE_percentage_change"],
        "MAE_change_vs_LightGBM_RM": lgb["MAE_difference_RM"],
        "MAE_change_vs_LightGBM_percent": lgb["MAE_percentage_change"],
        "R2_change_vs_LightGBM": lgb["R2_difference"],
        "Top5_RMSE_change_vs_LightGBM_RM": lgb["Top5_RMSE_difference_RM"],
        "Top5_MAE_change_vs_LightGBM_RM": lgb["Top5_MAE_difference_RM"],
    }


def _display(value: str) -> str:
    labels = {
        "baseline_lightgbm": "LightGBM interaction baseline",
        "developer_te": "Developer TE",
        "building_name_te": "Building name TE",
        "city_te": "City TE",
        "developer_building_te": "Developer + building TE",
        "all_te_replace": "All TE, replace raw",
        "all_te_plus_raw": "All TE + raw",
        "all_te_plus_frequency": "All TE + frequency",
    }
    return labels.get(value, value.replace("_", " "))


def _style(ax, title, subtitle):
    ax.set_title(title, loc="left", fontsize=13, color="#1f2937", pad=17)
    ax.text(0, 1.01, subtitle, transform=ax.transAxes, fontsize=9, color="#6b7280")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)


def create_figures(comparison, fold_table, oof, best_variant):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("RMSE_RM", "01_variant_rmse.png", "Variant RMSE", "RMSE (RM)", "#2563eb"),
        ("MAE_RM", "02_variant_mae.png", "Variant MAE", "MAE (RM)", "#f59e0b"),
        ("Top5_RMSE_RM", "03_top5_rmse.png", "Premium-property RMSE", "Top-5% RMSE (RM)", "#d97706"),
    ]
    for column, filename, title, xlabel, color in metrics:
        table = comparison.sort_values(column).copy()
        positions = np.arange(len(table))
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(positions, table[column], color=color)
        ax.set_yticks(positions, [_display(value) for value in table["variant"]])
        ax.invert_yaxis()
        ax.bar_label(
            bars,
            labels=[f"{value / 1000:.1f}k" for value in table[column]],
            padding=4,
            fontsize=8,
        )
        ax.margins(x=0.13)
        ax.set_xlabel(xlabel)
        _style(ax, title, "Shared five-fold OOF predictions; lower is better")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / filename, dpi=160)
        plt.close(fig)

    top_variants = comparison.head(5)["variant"].tolist()
    palette = ["#2563eb", "#f59e0b", "#6b7280", "#8b5cf6", "#d97706"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for variant, color in zip(top_variants, palette):
        rows = fold_table[fold_table["variant"] == variant].sort_values("fold")
        ax.plot(rows["fold"], rows["RMSE_RM"], marker="o", label=_display(variant), color=color)
    ax.set_xlabel("Fold")
    ax.set_ylabel("RMSE (RM)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _style(ax, "Fold RMSE comparison", "Same five outer folds for displayed variants")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "04_fold_rmse_comparison.png", dpi=160)
    plt.close(fig)

    best_rows = oof[oof["model_variant"] == best_variant].sort_values("row_index")
    actual = best_rows["actual_price_RM"].to_numpy(float)
    predicted = best_rows["predicted_price_RM"].to_numpy(float)
    positive = (actual > 0) & (predicted > 0)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(actual[positive], predicted[positive], s=12, alpha=0.35, color="#2563eb")
    limits = [min(actual[positive].min(), predicted[positive].min()), max(actual.max(), predicted[positive].max())]
    ax.plot(limits, limits, "--", color="#374151", linewidth=1.2, label="Ideal")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Actual price (RM, log scale)")
    ax.set_ylabel("Predicted price (RM, log scale)")
    ax.legend(frameon=False)
    _style(ax, "Actual vs predicted price", f"Best measured variant: {_display(best_variant)}")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "05_actual_vs_predicted_best.png", dpi=160)
    plt.close(fig)

    residual = predicted - actual
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(actual, residual, s=12, alpha=0.35, color="#2563eb")
    ax.axhline(0, linestyle="--", color="#374151", linewidth=1.2)
    ax.set_xscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1_000_000:.1f}M"))
    ax.set_xlabel("Actual price (RM, log scale)")
    ax.set_ylabel("Prediction minus actual (RM)")
    _style(ax, "Residuals vs actual price", f"Best measured variant: {_display(best_variant)}")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "06_residual_vs_actual_best.png", dpi=160)
    plt.close(fig)


def json_default(value):
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)}")


def main():
    protected_before = {str(path): sha256(path) for path in PROTECTED_PATHS}
    data = pd.read_csv(ENHANCED_CITY_DATA_PATH).reset_index(drop=True)
    required = {TARGET_COLUMN, *MODEL_FEATURES, "listing_id"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    coordinates = coordinate_inventory(data)
    advanced = json.loads(ADVANCED_RESULTS_PATH.read_text(encoding="utf-8"))
    random_forest_reference = reference_payload(advanced["baseline"])
    lightgbm_reference = reference_payload(
        advanced["feature_engineering"]["minus_micro_market"]
    )

    X = data[MODEL_FEATURES]
    y = data[TARGET_COLUMN].to_numpy(float)
    folds = shared_folds(len(data))
    params = candidate_parameters()["lightgbm"][1]
    baseline_estimator, baseline_predictors, baseline_num, baseline_cat = (
        build_lightgbm_interaction_baseline(params)
    )
    evaluations = {}
    predictions = {}
    all_oof_rows = []
    all_fold_rows = []
    baseline = evaluate_variant(
        "baseline_lightgbm",
        baseline_estimator,
        X,
        y,
        folds,
        baseline_predictors,
        data["listing_id"],
    )
    evaluations["baseline_lightgbm"] = baseline["result"]
    predictions["baseline_lightgbm"] = baseline["prediction"]
    all_oof_rows.extend(baseline["oof_rows"])
    all_fold_rows.extend(baseline["fold_rows"])

    expected = lightgbm_reference["metrics"]
    baseline_differences = {
        key: baseline["result"]["metrics"][key] - expected[key]
        for key in ("RMSE_RM", "MAE_RM", "R2")
    }
    if any(abs(value) > 1e-6 for value in baseline_differences.values()):
        raise AssertionError(
            f"LightGBM interaction baseline differs materially: {baseline_differences}"
        )
    print("Reproduced LightGBM interaction baseline exactly.", flush=True)

    for key, spec in VARIANT_SPECS.items():
        estimator = AdvancedTargetEncodingPPSFRegressor(
            lightgbm_params=params,
            te_columns=spec["columns"],
            m_values=DEFAULT_M_VALUES,
            retain_raw=spec["retain_raw"],
            add_frequency=spec["frequency"],
        )
        numerical, categorical = estimator._feature_schema()
        evaluated = evaluate_variant(
            key,
            estimator,
            X,
            y,
            folds,
            len(numerical) + len(categorical),
            data["listing_id"],
        )
        evaluations[key] = evaluated["result"]
        predictions[key] = evaluated["prediction"]
        all_oof_rows.extend(evaluated["oof_rows"])
        all_fold_rows.extend(evaluated["fold_rows"])
        print(f"Finished {key}.", flush=True)

    for result in evaluations.values():
        attach_reference_comparisons(
            result, random_forest_reference, lightgbm_reference
        )
    comparison = pd.DataFrame(
        [comparison_row(key, result) for key, result in evaluations.items()]
    ).sort_values("RMSE_RM", ignore_index=True)
    oof = pd.DataFrame(all_oof_rows).sort_values(
        ["model_variant", "row_index"], ignore_index=True
    )
    fold_table = pd.DataFrame(all_fold_rows).sort_values(
        ["variant", "fold"], ignore_index=True
    )
    comparison.to_csv(COMPARISON_PATH, index=False)
    oof.to_csv(OOF_PATH, index=False)
    fold_table.to_csv(FOLD_METRICS_PATH, index=False)

    new_keys = list(VARIANT_SPECS)
    best_new_key = min(new_keys, key=lambda key: evaluations[key]["metrics"]["RMSE_RM"])
    best_overall_key = comparison.iloc[0]["variant"]
    best_mae_key = comparison.sort_values("MAE_RM").iloc[0]["variant"]
    bootstrap = paired_bootstrap(
        y,
        predictions[best_new_key],
        predictions["baseline_lightgbm"],
    )
    baseline_fold_rmse = {
        row["fold"]: row["RMSE_RM"]
        for row in evaluations["baseline_lightgbm"]["folds"]
    }
    best_fold_rmse = {
        row["fold"]: row["RMSE_RM"] for row in evaluations[best_new_key]["folds"]
    }
    folds_won = sum(
        best_fold_rmse[fold] < baseline_fold_rmse[fold]
        for fold in best_fold_rmse
    )
    strong_candidate = (
        evaluations[best_new_key]["metrics"]["RMSE_RM"] < 120372.0
        and evaluations[best_new_key]["metrics"]["MAE_RM"] < 61217.0
        and evaluations[best_new_key]["metrics"]["top_5_percent"]["RMSE_RM"] < 420312.0
    )
    both_improved = (
        evaluations[best_new_key]["metrics"]["RMSE_RM"]
        < lightgbm_reference["metrics"]["RMSE_RM"]
        and evaluations[best_new_key]["metrics"]["MAE_RM"]
        < random_forest_reference["metrics"]["MAE_RM"]
    )
    recommendation = (
        f"Advance {best_new_key} to repeated nested CV or fresh-holdout validation; it met all point-estimate success thresholds."
        if strong_candidate
        else "Do not promote a new model. Retain the current Random Forest production candidate and the existing LightGBM interaction model as the experimental RMSE reference."
    )

    cardinality = {
        column: {
            "unique_count": int(data[column].nunique(dropna=False)),
            "missing_count": int(data[column].isna().sum()),
        }
        for column in CATEGORICAL_FEATURES
    }
    feature_summary = {
        "high_cardinality_columns": ["building_name", "developer", "city"],
        "categorical_cardinality": cardinality,
        "target_encoding_target": "price / property_size_sqft (PPSF)",
        "smoothing_values_tested": list(DEFAULT_M_VALUES),
        "smoothing_selection": "inside each outer training fold using inner-OOF PPSF proxy RMSE",
        "baseline_numerical_features": baseline_num,
        "baseline_categorical_features": baseline_cat,
        "target_encoding_variants": VARIANT_SPECS,
        "spatial_geometry_feature_names": SpatialGeometryFeatures.feature_names(),
        "spatial_ppsf_feature_names": SpatialPPSFNeighborEncoder.feature_names(),
        "spatial_status": "not_run_missing_coordinates",
        "poi_status": "not_run_no_verified_poi_coordinates",
    }
    FEATURE_SUMMARY_PATH.write_text(
        json.dumps(feature_summary, indent=2, default=json_default), encoding="utf-8"
    )
    create_figures(comparison, fold_table, oof, best_overall_key)

    protected_after = {str(path): sha256(path) for path in PROTECTED_PATHS}
    if protected_before != protected_after:
        changed = [
            path for path in protected_before
            if protected_before[path] != protected_after[path]
        ]
        raise AssertionError(f"Protected files changed: {changed}")

    spatial_status = {
        "status": "not_run_missing_coordinates",
        "measured": False,
        "reason": "No reliable property latitude/longitude columns exist.",
        "implemented_utility": "spatial_features.py",
    }
    payload = {
        "dataset": {
            "path": ENHANCED_CITY_DATA_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256(ENHANCED_CITY_DATA_PATH),
            "rows": int(len(data)),
            "rows_removed": 0,
            "premium_threshold_RM": float(np.quantile(y, 0.95)),
        },
        "coordinate_availability": coordinates,
        "baseline_random_forest": random_forest_reference,
        "baseline_lightgbm": {
            **evaluations["baseline_lightgbm"],
            "verification": {
                "matched": True,
                "absolute_tolerance": 1e-6,
                "metric_differences": baseline_differences,
            },
        },
        "spatial_geometry": spatial_status,
        "spatial_ppsf": spatial_status,
        "poi_features": {
            "status": "not_run_no_verified_poi_coordinates",
            "measured": False,
        },
        "target_encoding": {
            key: evaluations[key] for key in new_keys
        },
        "combined_variants": {
            "spatial_geometry_plus_te": spatial_status,
            "spatial_geometry_plus_spatial_ppsf_plus_te": spatial_status,
        },
        "fold_metrics": {
            "path": FOLD_METRICS_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "best_new_candidate_folds_won_vs_lightgbm": int(folds_won),
        },
        "generalization": {
            key: result["generalization_gap"]
            for key, result in evaluations.items()
        },
        "bootstrap": {"best_new_candidate": best_new_key, **bootstrap},
        "best_rmse_variant": best_overall_key,
        "best_mae_variant": best_mae_key,
        "best_balanced_variant": best_new_key if strong_candidate else None,
        "both_rmse_and_mae_improved": bool(both_improved),
        "strong_candidate_thresholds_met": bool(strong_candidate),
        "recommendation": recommendation,
        "leakage_audit": {
            "outer_validation_target_used_for_target_encoding": False,
            "outer_validation_target_used_for_smoothing_selection": False,
            "training_row_uses_own_target_encoding": False,
            "target_encoding_training_rows_inner_oof": True,
            "unseen_category_fallback": "outer-training PPSF mean",
            "missing_category_token": "__MISSING__",
            "spatial_neighbor_target_features_cross_fold_only": True,
            "spatial_features_not_evaluated_without_coordinates": True,
            "feature_selection_uses_outer_validation_targets": False,
            "protected_files_unchanged": True,
        },
        "artifacts": [
            path.relative_to(PROJECT_ROOT).as_posix()
            for path in (
                RESULTS_PATH,
                COMPARISON_PATH,
                FOLD_METRICS_PATH,
                OOF_PATH,
                FEATURE_SUMMARY_PATH,
            )
        ],
    }
    RESULTS_PATH.write_text(
        json.dumps(payload, indent=2, default=json_default), encoding="utf-8"
    )
    print("\n" + comparison[["variant", "RMSE_RM", "MAE_RM", "R2", "Top5_RMSE_RM"]].to_string(index=False))
    print(f"\nBest new candidate: {best_new_key}")
    print(f"Recommendation: {recommendation}")
    print(f"Saved: {EXPERIMENT_DIR}")


if __name__ == "__main__":
    main()
