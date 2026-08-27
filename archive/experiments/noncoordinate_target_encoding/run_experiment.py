"""Run the isolated non-coordinate target-encoding and context experiment."""

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

from experiments.noncoordinate_target_encoding.evaluation import (
    evaluate_variant,
    paired_bootstrap,
    reference_deltas,
)
from experiments.noncoordinate_target_encoding.feature_engineering import (
    CITY_AGGREGATE_FEATURES,
    NoncoordinatePPSFRegressor,
)
from experiments.noncoordinate_target_encoding.model_builders import NATIVE_OBJECTIVES
from experiments.noncoordinate_target_encoding.target_encoding import DEFAULT_M_VALUES
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
RESULTS_PATH = EXPERIMENT_DIR / "results.json"
COMPARISON_PATH = EXPERIMENT_DIR / "model_comparison.csv"
FOLD_METRICS_PATH = EXPERIMENT_DIR / "fold_metrics.csv"
OOF_PATH = EXPERIMENT_DIR / "oof_predictions.csv"
FEATURE_SUMMARY_PATH = EXPERIMENT_DIR / "feature_summary.json"
ADVANCED_DIR = PROJECT_ROOT / "experiments/advanced_real_estate_models"
ADVANCED_RESULTS_PATH = ADVANCED_DIR / "results.json"
ADVANCED_OOF_PATH = ADVANCED_DIR / "oof_predictions.csv"
PREVIOUS_TE_RESULTS_PATH = PROJECT_ROOT / "experiments/spatial_target_encoding/results.json"

PROTECTED_PATHS = (
    PROJECT_ROOT / "data/raw/houses.csv",
    ENHANCED_CITY_DATA_PATH,
    PROJECT_ROOT / "results/enhanced_city/model_comparison.json",
    PROJECT_ROOT / "results/best_model/best_model_summary.json",
    ADVANCED_RESULTS_PATH,
    ADVANCED_OOF_PATH,
    PREVIOUS_TE_RESULTS_PATH,
    PROJECT_ROOT / "prototype/app.py",
    PROJECT_ROOT / "app.py",
)

FEATURE_VARIANTS = {
    "baseline_lightgbm": {
        "te_columns": (), "add_counts": False, "add_city_aggregates": False,
    },
    "building_name_te": {
        "te_columns": ("building_name",), "add_counts": False,
        "add_city_aggregates": False,
    },
    "developer_te": {
        "te_columns": ("developer",), "add_counts": False,
        "add_city_aggregates": False,
    },
    "city_te": {
        "te_columns": ("city",), "add_counts": False,
        "add_city_aggregates": False,
    },
    "building_developer_te": {
        "te_columns": ("building_name", "developer"), "add_counts": False,
        "add_city_aggregates": False,
    },
    "combined_te": {
        "te_columns": ("building_name", "developer", "city"),
        "add_counts": False, "add_city_aggregates": False,
    },
    "combined_te_frequency": {
        "te_columns": ("building_name", "developer", "city"),
        "add_counts": True, "add_city_aggregates": False,
    },
    "combined_te_frequency_city_context": {
        "te_columns": ("building_name", "developer", "city"),
        "add_counts": True, "add_city_aggregates": True,
    },
}

# Pre-specified from the protected preceding TE experiment, where it had the
# lowest OOF RMSE. This avoids selecting the loss-test feature set on the current
# experiment's outer validation targets.
LOSS_FEATURE_SPEC = FEATURE_VARIANTS["building_name_te"]
LOSS_VARIANTS = {
    "building_name_te_l1": "regression_l1",
    "building_name_te_huber": "huber",
    "building_name_te_fair": "fair",
}
WEIGHT_VARIANTS = {
    "building_name_te_weight_005": 0.05,
    "building_name_te_weight_010": 0.10,
    "building_name_te_weight_020": 0.20,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_payload(source: dict) -> dict:
    return {
        "name": source["name"],
        "metrics": source["metrics"],
        "generalization_gap": source.get("generalization_gap", {}),
        "source": ADVANCED_RESULTS_PATH.relative_to(PROJECT_ROOT).as_posix(),
    }


def display_name(value: str) -> str:
    names = {
        "baseline_lightgbm": "LightGBM baseline",
        "building_name_te": "Building name TE",
        "developer_te": "Developer TE",
        "city_te": "City TE",
        "building_developer_te": "Building + developer TE",
        "combined_te": "Combined TE",
        "combined_te_frequency": "Combined TE + frequency",
        "combined_te_frequency_city_context": "Combined TE + frequency + city context",
        "building_name_te_l1": "Building TE + L1",
        "building_name_te_huber": "Building TE + Huber",
        "building_name_te_fair": "Building TE + Fair",
        "building_name_te_weight_005": "Building TE + weight 0.05",
        "building_name_te_weight_010": "Building TE + weight 0.10",
        "building_name_te_weight_020": "Building TE + weight 0.20",
    }
    return names.get(value, value.replace("_", " "))


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
        "OOF_RMSE_RM": gap["OOF_RMSE_RM"],
        "RMSE_gap_RM": gap["RMSE_gap_RM"],
        "Training_MAE_RM": gap["Training_MAE_RM"],
        "OOF_MAE_RM": gap["OOF_MAE_RM"],
        "MAE_gap_RM": gap["MAE_gap_RM"],
        "Training_R2": gap["Training_R2"],
        "OOF_R2": gap["OOF_R2"],
        "R2_gap": gap["R2_gap"],
        **{f"vs_RF_{name}": value for name, value in rf.items()},
        **{f"vs_LightGBM_{name}": value for name, value in lgb.items()},
    }


def style_axis(ax, title: str, subtitle: str):
    ax.set_title(title, loc="left", fontsize=13, color="#1f2937", pad=17)
    ax.text(0, 1.01, subtitle, transform=ax.transAxes, fontsize=9, color="#6b7280")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)


def create_figures(comparison, fold_table, oof, best_key):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    specs = [
        ("RMSE_RM", "01_rmse_comparison.png", "OOF RMSE comparison", "RMSE (RM)", "#2563eb"),
        ("MAE_RM", "02_mae_comparison.png", "OOF MAE comparison", "MAE (RM)", "#f59e0b"),
        ("Top5_RMSE_RM", "03_top5_rmse_comparison.png", "Premium-property RMSE comparison", "Top-5% RMSE (RM)", "#d97706"),
    ]
    for column, filename, title, xlabel, color in specs:
        table = comparison.sort_values(column).copy()
        positions = np.arange(len(table))
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(positions, table[column], color=color)
        ax.set_yticks(positions, [display_name(value) for value in table["variant"]])
        ax.invert_yaxis()
        ax.bar_label(
            bars,
            labels=[f"{value / 1000:.1f}k" for value in table[column]],
            padding=4,
            fontsize=8,
        )
        ax.margins(x=0.13)
        ax.set_xlabel(xlabel)
        style_axis(ax, title, "3,791 listings; shared five-fold OOF predictions; lower is better")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / filename, dpi=160)
        plt.close(fig)

    top = comparison.head(5)["variant"].tolist()
    colors = ["#2563eb", "#f59e0b", "#6b7280", "#8b5cf6", "#d97706"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, color in zip(top, colors):
        rows = fold_table[fold_table["variant"] == key].sort_values("fold")
        ax.plot(rows["fold"], rows["RMSE_RM"], marker="o", color=color, label=display_name(key))
    ax.set_xlabel("Outer fold")
    ax.set_ylabel("RMSE (RM)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    style_axis(ax, "Fold RMSE comparison", "Same five outer folds for the five lowest overall-RMSE variants")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "04_fold_rmse_comparison.png", dpi=160)
    plt.close(fig)

    best = oof[oof["variant"] == best_key].sort_values("row_index")
    actual = best["actual_price_RM"].to_numpy(float)
    predicted = best["predicted_price_RM"].to_numpy(float)
    positive = (actual > 0) & (predicted > 0)
    limits = [
        min(actual[positive].min(), predicted[positive].min()),
        max(actual[positive].max(), predicted[positive].max()),
    ]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(actual[positive], predicted[positive], s=12, alpha=0.32, color="#2563eb")
    ax.plot(limits, limits, "--", color="#374151", linewidth=1.2, label="Ideal")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Actual price (RM, log scale)")
    ax.set_ylabel("Predicted price (RM, log scale)")
    ax.legend(frameon=False)
    style_axis(ax, "Actual vs predicted price", f"Lowest measured OOF RMSE: {display_name(best_key)}")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "05_actual_vs_predicted_best.png", dpi=160)
    plt.close(fig)

    residual = predicted - actual
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(actual, residual, s=12, alpha=0.32, color="#2563eb")
    ax.axhline(0, linestyle="--", color="#374151", linewidth=1.2)
    ax.set_xscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1_000_000:.1f}M"))
    ax.set_xlabel("Actual price (RM, log scale)")
    ax.set_ylabel("Prediction minus actual (RM)")
    style_axis(ax, "Residuals vs actual price", f"Lowest measured OOF RMSE: {display_name(best_key)}")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "06_residuals_vs_actual_best.png", dpi=160)
    plt.close(fig)

    gaps = comparison.sort_values("RMSE_RM")
    positions = np.arange(len(gaps))
    width = 0.38
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.bar(positions - width / 2, gaps["Training_RMSE_RM"], width, color="#93c5fd", label="Training RMSE")
    ax.bar(positions + width / 2, gaps["OOF_RMSE_RM"], width, color="#2563eb", label="OOF RMSE")
    ax.set_xticks(positions, [display_name(value) for value in gaps["variant"]], rotation=55, ha="right")
    ax.set_ylabel("RMSE (RM)")
    ax.legend(frameon=False)
    style_axis(ax, "Training and OOF RMSE", "Same fitted models; smaller separation indicates less overfit")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "07_generalization_gap.png", dpi=160)
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
    protected_before = {path.as_posix(): sha256(path) for path in PROTECTED_PATHS}
    data = pd.read_csv(ENHANCED_CITY_DATA_PATH).reset_index(drop=True)
    required = {"listing_id", TARGET_COLUMN, *MODEL_FEATURES}
    if missing := sorted(required.difference(data.columns)):
        raise ValueError(f"Missing canonical columns: {missing}")
    if data["listing_id"].duplicated().any():
        raise ValueError("Canonical listing_id must be unique.")
    if data[TARGET_COLUMN].isna().any() or (data[TARGET_COLUMN] <= 0).any():
        raise ValueError("Target must be complete and positive.")
    if data["property_size_sqft"].isna().any() or (data["property_size_sqft"] <= 0).any():
        raise ValueError("Property size must be complete and positive.")

    advanced = json.loads(ADVANCED_RESULTS_PATH.read_text(encoding="utf-8"))
    random_forest = reference_payload(advanced["baseline"])
    lightgbm = reference_payload(advanced["feature_engineering"]["minus_micro_market"])
    X = data[MODEL_FEATURES]
    y = data[TARGET_COLUMN].to_numpy(float)
    folds = shared_folds(len(data))
    evaluations = {}
    predictions = {}
    all_fold_rows = []
    all_oof_rows = []

    def run_one(key, estimator):
        numerical, categorical = estimator.feature_schema()
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
        all_fold_rows.extend(evaluated["fold_rows"])
        all_oof_rows.extend(evaluated["oof_rows"])
        print(f"Finished {key}.", flush=True)

    for key, spec in FEATURE_VARIANTS.items():
        run_one(key, NoncoordinatePPSFRegressor(**spec))
        if key == "baseline_lightgbm":
            expected = lightgbm["metrics"]
            actual = evaluations[key]["metrics"]
            differences = {
                metric: actual[metric] - expected[metric]
                for metric in ("RMSE_RM", "MAE_RM", "R2")
            }
            if any(abs(value) > 1e-6 for value in differences.values()):
                raise AssertionError(f"Baseline differs materially: {differences}")
            print("Baseline reproduced exactly.", flush=True)

    for key, objective in LOSS_VARIANTS.items():
        run_one(
            key,
            NoncoordinatePPSFRegressor(**LOSS_FEATURE_SPEC, objective=objective),
        )
    for key, gamma in WEIGHT_VARIANTS.items():
        run_one(
            key,
            NoncoordinatePPSFRegressor(**LOSS_FEATURE_SPEC, weight_gamma=gamma),
        )

    for result in evaluations.values():
        result["comparisons"] = {
            "random_forest": reference_deltas(result["metrics"], random_forest["metrics"]),
            "lightgbm_interaction": reference_deltas(result["metrics"], lightgbm["metrics"]),
        }

    comparison = pd.DataFrame(
        [comparison_row(key, result) for key, result in evaluations.items()]
    ).sort_values("RMSE_RM", ignore_index=True)
    fold_table = pd.DataFrame(all_fold_rows).sort_values(["variant", "fold"], ignore_index=True)
    oof = pd.DataFrame(all_oof_rows).sort_values(["variant", "row_index"], ignore_index=True)
    comparison.to_csv(COMPARISON_PATH, index=False)
    fold_table.to_csv(FOLD_METRICS_PATH, index=False)
    oof.to_csv(OOF_PATH, index=False)

    feature_keys = list(FEATURE_VARIANTS)
    new_keys = [key for key in evaluations if key != "baseline_lightgbm"]
    best_feature_key = min(
        [key for key in feature_keys if key != "baseline_lightgbm"],
        key=lambda key: evaluations[key]["metrics"]["RMSE_RM"],
    )
    best_loss_key = min(LOSS_VARIANTS, key=lambda key: evaluations[key]["metrics"]["RMSE_RM"])
    best_weight_key = min(WEIGHT_VARIANTS, key=lambda key: evaluations[key]["metrics"]["RMSE_RM"])
    best_key = min(new_keys, key=lambda key: evaluations[key]["metrics"]["RMSE_RM"])
    best_mae_key = min(evaluations, key=lambda key: evaluations[key]["metrics"]["MAE_RM"])
    balanced = [
        key for key in new_keys
        if evaluations[key]["metrics"]["RMSE_RM"] < lightgbm["metrics"]["RMSE_RM"]
        and evaluations[key]["metrics"]["MAE_RM"] < random_forest["metrics"]["MAE_RM"]
    ]
    best_balanced = min(
        balanced,
        key=lambda key: evaluations[key]["metrics"]["RMSE_RM"],
        default=None,
    )

    advanced_oof = pd.read_csv(ADVANCED_OOF_PATH)
    if advanced_oof["listing_id"].duplicated().any():
        raise AssertionError("Protected advanced OOF listing IDs are not unique.")
    aligned = data[["listing_id", "price"]].merge(
        advanced_oof[["listing_id", "actual_price_RM", "prediction__random_forest"]],
        on="listing_id",
        how="left",
        validate="one_to_one",
    )
    if aligned.isna().any().any() or not np.allclose(aligned["price"], aligned["actual_price_RM"]):
        raise AssertionError("Random Forest OOF reference does not align to canonical rows.")
    rf_prediction = aligned["prediction__random_forest"].to_numpy(float)
    premium_threshold = float(np.quantile(y, 0.95))
    bootstrap = {
        "best_new_candidate": best_key,
        "vs_lightgbm": paired_bootstrap(
            y, predictions[best_key], predictions["baseline_lightgbm"], premium_threshold, random_state=42
        ),
        "vs_random_forest": paired_bootstrap(
            y, predictions[best_key], rf_prediction, premium_threshold, random_state=43
        ),
    }
    baseline_fold_rmse = {
        row["fold"]: row["RMSE_RM"] for row in evaluations["baseline_lightgbm"]["folds"]
    }
    folds_won = sum(
        row["RMSE_RM"] < baseline_fold_rmse[row["fold"]]
        for row in evaluations[best_key]["folds"]
    )

    feature_summary = {
        "coordinate_based_features_used": False,
        "coordinate_columns_present": [],
        "raw_metadata_excluded": ["Ad List", "Firm Type", "Firm Number", "REN Number", "Category"],
        "canonical_numerical_features": NUMERICAL_FEATURES,
        "canonical_categorical_features": CATEGORICAL_FEATURES,
        "preserved_deterministic_interactions": [
            "size_band", "state_property_type", "city_property_type", "city_tenure_type",
            "city_building_name", "city_developer", "city_size_band", "property_type_size_band",
        ],
        "explicit_te_columns": ["building_name", "developer", "city"],
        "te_target": "outer-training PPSF",
        "smoothing_values": list(DEFAULT_M_VALUES),
        "count_features": [
            "building_count", "log_building_count", "developer_count",
            "log_developer_count", "city_count", "log_city_count",
        ],
        "city_context_features": list(CITY_AGGREGATE_FEATURES),
        "city_context_target_derived": False,
        "micro_market_target_aggregates_used": False,
        "feature_variants": FEATURE_VARIANTS,
        "loss_feature_set_selection": "pre-specified building_name_te from protected preceding experiment",
        "native_objectives_tested": NATIVE_OBJECTIVES,
        "log_cosh_status": "skipped_not_needed_custom_objective",
    }
    FEATURE_SUMMARY_PATH.write_text(json.dumps(feature_summary, indent=2), encoding="utf-8")
    create_figures(comparison, fold_table, oof, best_key)

    protected_after = {path.as_posix(): sha256(path) for path in PROTECTED_PATHS}
    if protected_before != protected_after:
        changed = [path for path in protected_before if protected_before[path] != protected_after[path]]
        raise AssertionError(f"Protected files changed: {changed}")

    result_payload = {
        "dataset": {
            "path": ENHANCED_CITY_DATA_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256(ENHANCED_CITY_DATA_PATH),
            "rows": int(len(data)),
            "columns": int(len(data.columns)),
            "rows_removed_in_experiment": 0,
            "premium_threshold_RM": premium_threshold,
            "premium_rows": int(np.sum(y >= premium_threshold)),
            "coordinate_based_features_used": False,
        },
        "baseline_random_forest": random_forest,
        "baseline_lightgbm": {
            **evaluations["baseline_lightgbm"],
            "verification": {"matched_protected_reference": True, "absolute_tolerance": 1e-6},
        },
        "target_encoding_variants": {
            key: evaluations[key]
            for key in ("building_name_te", "developer_te", "city_te", "building_developer_te", "combined_te")
        },
        "frequency_features": {
            "combined_te_frequency": evaluations["combined_te_frequency"]
        },
        "city_aggregate_features": {
            "combined_te_frequency_city_context": evaluations["combined_te_frequency_city_context"]
        },
        "loss_variants": {
            **{key: evaluations[key] for key in LOSS_VARIANTS},
            "log_cosh": {"status": "skipped_not_needed_custom_objective"},
        },
        "sample_weighting": {key: evaluations[key] for key in WEIGHT_VARIANTS},
        "generalization": {
            key: result["generalization_gap"] for key, result in evaluations.items()
        },
        "bootstrap": bootstrap,
        "best_feature_variant": best_feature_key,
        "best_loss_variant": best_loss_key,
        "best_sample_weighting_variant": best_weight_key,
        "best_rmse_variant": best_key,
        "best_mae_variant": best_mae_key,
        "best_balanced_variant": best_balanced,
        "best_candidate_folds_won_vs_lightgbm": int(folds_won),
        "both_rmse_and_mae_improved": best_balanced is not None,
        "recommendation": (
            f"Advance {best_balanced} to repeated nested CV or a fresh holdout before promotion."
            if best_balanced is not None
            else "Do not promote a new model; no candidate beat both the LightGBM RMSE and Random Forest MAE references."
        ),
        "leakage_audit": {
            "outer_validation_target_used_for_te": False,
            "outer_validation_target_used_for_smoothing": False,
            "outer_validation_target_used_for_model_fit": False,
            "outer_training_te_is_inner_oof": True,
            "row_uses_own_target_for_oof_te": False,
            "count_mappings_training_partition_only": True,
            "city_context_training_partition_only": True,
            "city_context_uses_target": False,
            "target_micro_market_aggregates_used": False,
            "loss_candidates_fixed_before_evaluation": True,
            "weight_gammas_fixed_before_evaluation": True,
            "loss_feature_set_selected_from_current_outer_validation": False,
            "premium_status_used_as_predictor": False,
            "target_capped_or_clipped": False,
            "coordinate_based_features_used": False,
            "protected_files_unchanged": True,
        },
        "artifacts": [
            path.relative_to(PROJECT_ROOT).as_posix()
            for path in (RESULTS_PATH, COMPARISON_PATH, FOLD_METRICS_PATH, OOF_PATH, FEATURE_SUMMARY_PATH)
        ],
    }
    RESULTS_PATH.write_text(
        json.dumps(result_payload, indent=2, default=json_default), encoding="utf-8"
    )
    print("\n" + comparison[["variant", "RMSE_RM", "MAE_RM", "R2", "Top5_RMSE_RM"]].to_string(index=False))
    print(f"\nBest feature variant: {best_feature_key}")
    print(f"Best overall candidate: {best_key}")
    print(f"Recommendation: {result_payload['recommendation']}")
    print(f"Saved: {EXPERIMENT_DIR}")


if __name__ == "__main__":
    main()
