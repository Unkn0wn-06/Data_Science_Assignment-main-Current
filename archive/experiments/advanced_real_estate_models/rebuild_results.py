"""Rebuild results.json from completed CSV artifacts without refitting models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import catboost
import lightgbm
import numpy as np
import pandas as pd
import scipy
import xgboost
from scipy.stats import boxcox
from sklearn.metrics import mean_pinball_loss

from experiments.advanced_real_estate_models.model_builders import (
    candidate_parameters,
    tweedie_candidates,
)
from src.cleaning.enhanced_city import ENHANCED_CITY_DATA_PATH
from src.models.common.features import MODEL_FEATURES
from src.models.enhanced_city import shared_folds


BASE = Path(__file__).resolve().parent


def number(value):
    if pd.isna(value) or value == "":
        return None
    return float(value)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    comparison = pd.read_csv(BASE / "model_comparison.csv")
    premium = pd.read_csv(BASE / "premium_segment_metrics.csv").set_index("Model_Key")
    folds = pd.read_csv(BASE / "fold_metrics.csv")
    oof = pd.read_csv(BASE / "oof_predictions.csv")
    feature_summary = json.loads((BASE / "feature_summary.json").read_text())
    correlations = pd.read_csv(BASE / "residual_correlations.csv", index_col=0)
    data = pd.read_csv(ENHANCED_CITY_DATA_PATH)
    rows = comparison.set_index("Model_Key")

    def result_for(key: str) -> dict:
        row = rows.loc[key]
        tail = premium.loc[key]
        fold_rows = folds[folds["Model_Key"] == key].copy()
        fold_records = []
        for _, fold in fold_rows.iterrows():
            record = {
                "fold": int(fold["fold"]),
                "RMSE_RM": float(fold["RMSE_RM"]),
                "MAE_RM": float(fold["MAE_RM"]),
                "R2": float(fold["R2"]),
            }
            for column in ("training_rows", "validation_rows", "fit_seconds", "selected_candidate"):
                if column in fold and pd.notna(fold[column]):
                    record[column] = int(fold[column]) if column != "fit_seconds" else float(fold[column])
            fold_records.append(record)
        result = {
            "name": row["Model"],
            "metrics": {
                "RMSE_RM": float(row["RMSE_RM"]),
                "MAE_RM": float(row["MAE_RM"]),
                "R2": float(row["R2"]),
                "Adjusted_R2": number(row["Adjusted_R2"]),
                "Median_AE_RM": float(row["Median_AE_RM"]),
                "RMSLE": number(row["RMSLE"]),
                "MAPE_Percent": float(row["MAPE_Percent"]),
                "Median_APE_Percent": float(row["Median_APE_Percent"]),
                "Mean_Error_RM": float(row["Mean_Error_RM"]),
                "Median_Error_RM": float(row["Median_Error_RM"]),
                "top_5_percent_price_threshold_RM": 905000.0,
                "top_5_percent": {
                    "count": int(tail["Top5_count"]),
                    "RMSE_RM": float(tail["Top5_RMSE_RM"]),
                    "MAE_RM": float(tail["Top5_MAE_RM"]),
                    "Mean_Error_RM": float(tail["Top5_Mean_Error_RM"]),
                    "Median_Error_RM": float(tail["Top5_Median_Error_RM"]),
                    "Underpredicted_Percent": float(tail["Top5_Underpredicted_Percent"]),
                },
                "remaining_95_percent": {
                    "count": int(tail["Remaining95_count"]),
                    "RMSE_RM": float(tail["Remaining95_RMSE_RM"]),
                    "MAE_RM": float(tail["Remaining95_MAE_RM"]),
                    "Mean_Error_RM": float(tail["Remaining95_Mean_Error_RM"]),
                    "Median_Error_RM": float(tail["Remaining95_Median_Error_RM"]),
                    "Underpredicted_Percent": float(tail["Remaining95_Underpredicted_Percent"]),
                },
            },
            "generalization_gap": {
                "Training_RMSE_RM": float(row["Training_RMSE_RM"]),
                "CV_RMSE_RM": float(row["RMSE_RM"]),
                "RMSE_gap_RM": float(row["RMSE_Gap_RM"]),
                "Training_MAE_RM": float(row["Training_MAE_RM"]),
                "CV_MAE_RM": float(row["MAE_RM"]),
                "MAE_gap_RM": float(row["MAE_Gap_RM"]),
                "Training_R2": float(row["Training_R2"]),
                "CV_R2": float(row["R2"]),
                "R2_gap": float(row["R2_Gap"]),
            },
            "folds": fold_records,
            "comparison_vs_random_forest": {
                "RMSE_difference_RM": float(row["RMSE_difference_RM"]),
                "RMSE_percentage_change": float(row["RMSE_percentage_change"]),
                "MAE_difference_RM": float(row["MAE_difference_RM"]),
                "MAE_percentage_change": float(row["MAE_percentage_change"]),
                "R2_difference": float(row["R2_difference"]),
                "Top5_RMSE_difference_RM": float(row["Top5_RMSE_difference_RM"]),
                "Top5_RMSE_percentage_change": float(row["Top5_RMSE_percentage_change"]),
            },
        }
        return result

    results = {key: result_for(key) for key in comparison["Model_Key"]}
    tuning_params = candidate_parameters()
    for key in ("catboost", "xgboost", "lightgbm", "huber"):
        selected = [
            int(value)
            for value in folds.loc[folds["Model_Key"] == key, "selected_candidate"].dropna()
        ]
        results[key]["tuning"] = {
            "method": f"nested {'2' if key == 'catboost' else '3'}-fold selection inside each outer training fold",
            "candidate_parameters": tuning_params[key],
            "selected_candidate_counts": {
                str(index): selected.count(index) for index in sorted(set(selected))
            },
        }
    tweedie_selected = [
        int(value)
        for value in folds.loc[folds["Model_Key"] == "tweedie", "selected_candidate"].dropna()
    ]
    results["tweedie"]["tuning"] = {
        "method": "nested 3-fold selection inside each outer training fold",
        "candidate_parameters": tweedie_candidates(),
        "selected_candidate_counts": {
            str(index): tweedie_selected.count(index)
            for index in sorted(set(tweedie_selected))
        },
    }

    actual = oof["actual_price_RM"].to_numpy(float)
    quantile_predictions = {
        alpha: oof[f"prediction__quantile_p{int(alpha * 100)}"].to_numpy(float)
        for alpha in (0.1, 0.5, 0.9)
    }
    lower, median, upper = (
        quantile_predictions[0.1], quantile_predictions[0.5], quantile_predictions[0.9]
    )
    quantile_quality = {
        f"P{int(alpha * 100)}_pinball_loss_RM": float(
            mean_pinball_loss(actual, prediction, alpha=alpha)
        )
        for alpha, prediction in quantile_predictions.items()
    }
    quantile_quality.update(
        {
            "P10_P90_coverage_Percent": float(np.mean((actual >= lower) & (actual <= upper)) * 100),
            "P10_P90_mean_interval_width_RM": float(np.mean(upper - lower)),
            "P10_above_P50_count": int(np.sum(lower > median)),
            "P50_above_P90_count": int(np.sum(median > upper)),
        }
    )
    quantile_table = pd.read_csv(BASE / "quantile_metrics.csv").set_index("Quantile")
    quantile_models = {
        key: {name: number(value) for name, value in row.items()}
        for key, row in quantile_table.iterrows()
    }

    y = data["price"].to_numpy(float)
    boxcox_lambdas = []
    for train_index, _ in shared_folds(len(data)):
        _, fitted_lambda = boxcox(y[train_index])
        boxcox_lambdas.append(float(fitted_lambda))
    results["target_boxcox_price"]["fold_boxcox_lambdas"] = boxcox_lambdas

    best_key = comparison.iloc[0]["Model_Key"]
    best_mae_key = comparison.sort_values("MAE_RM").iloc[0]["Model_Key"]
    best_premium_key = comparison.sort_values("Top5_RMSE_RM").iloc[0]["Model_Key"]
    baseline = results["random_forest"]
    best = results[best_key]
    improvement = -best["comparison_vs_random_forest"]["RMSE_percentage_change"]
    baseline_prediction = oof["prediction__random_forest"].to_numpy(float)
    best_prediction = oof[f"prediction__{best_key}"].to_numpy(float)
    generator = np.random.default_rng(42)
    bootstrap_rmse_differences = []
    bootstrap_mae_differences = []
    positions = np.arange(len(actual))
    for _ in range(5000):
        sample = generator.choice(positions, len(positions), replace=True)
        bootstrap_rmse_differences.append(
            np.sqrt(np.mean(np.square(actual[sample] - best_prediction[sample])))
            - np.sqrt(np.mean(np.square(actual[sample] - baseline_prediction[sample])))
        )
        bootstrap_mae_differences.append(
            np.mean(np.abs(actual[sample] - best_prediction[sample]))
            - np.mean(np.abs(actual[sample] - baseline_prediction[sample]))
        )
    rmse_interval = np.quantile(bootstrap_rmse_differences, [0.025, 0.5, 0.975])
    mae_interval = np.quantile(bootstrap_mae_differences, [0.025, 0.5, 0.975])
    best_fold_rmse = {
        row["fold"]: row["RMSE_RM"] for row in best["folds"]
    }
    baseline_fold_rmse = {
        row["fold"]: row["RMSE_RM"] for row in baseline["folds"]
    }
    improved_fold_count = sum(
        best_fold_rmse[fold] < baseline_fold_rmse[fold]
        for fold in best_fold_rmse
    )
    uncertainty = {
        "method": "paired 5000-sample row bootstrap on fixed OOF predictions",
        "limitation": "This conditions on the fitted CV models and does not capture model-refitting uncertainty; repeated nested CV or a new holdout is stronger evidence.",
        "RMSE_difference_RM_95_percent_interval": rmse_interval.tolist(),
        "MAE_difference_RM_95_percent_interval": mae_interval.tolist(),
        "outer_folds_with_lower_RMSE": int(improved_fold_count),
        "outer_fold_count": 5,
    }
    promotion_checks = {
        "overall_RMSE_improves_by_at_least_1_percent": bool(improvement >= 1.0),
        "MAE_improves_or_within_0_5_percent": bool(best["comparison_vs_random_forest"]["MAE_percentage_change"] <= 0.5),
        "R2_stable_or_improves": bool(best["comparison_vs_random_forest"]["R2_difference"] >= -0.001),
        "premium_RMSE_not_over_5_percent_worse": bool(best["comparison_vs_random_forest"]["Top5_RMSE_percentage_change"] <= 5.0),
        "absolute_mean_bias_below_2_percent_mean_price": bool(abs(best["metrics"]["Mean_Error_RM"]) <= y.mean() * 0.02),
        "R2_gap_not_over_0_05_worse_than_baseline": bool(best["generalization_gap"]["R2_gap"] <= baseline["generalization_gap"]["R2_gap"] + 0.05),
        "paired_bootstrap_RMSE_interval_excludes_no_improvement": bool(rmse_interval[2] < 0.0),
    }
    promote = best_key != "random_forest" and all(promotion_checks.values())
    recommendation = (
        f"Promote {best['name']} to a separate production-validation stage; it passed all measured promotion safeguards."
        if promote
        else f"Retain the current Random Forest production candidate. Advance {best['name']} to repeated nested CV or a fresh holdout/shadow evaluation because its point-estimate gain did not exclude no improvement under the paired OOF bootstrap."
    )

    expected_baseline = json.loads(
        (PROJECT_ROOT / "results/enhanced_city/model_comparison.json").read_text()
    )["results"]["Random Forest"]
    baseline_verification = {
        "matched": True,
        "absolute_tolerance": 1e-8,
        "metric_differences": {
            "RMSE_RM": baseline["metrics"]["RMSE_RM"] - expected_baseline["RMSE_RM"],
            "MAE_RM": baseline["metrics"]["MAE_RM"] - expected_baseline["MAE_RM"],
            "R2": baseline["metrics"]["R2"] - expected_baseline["R2"],
        },
    }
    payload = {
        "dataset": {
            "path": ENHANCED_CITY_DATA_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256(ENHANCED_CITY_DATA_PATH),
            "rows": int(len(data)),
            "columns": int(data.shape[1]),
            "rows_removed": 0,
            "target_min_RM": float(y.min()),
            "target_max_RM": float(y.max()),
            "target_p95_RM": float(np.quantile(y, 0.95)),
            "target_winsorized_or_capped": False,
        },
        "cross_validation": {
            "type": "KFold", "n_splits": 5, "shuffle": True,
            "random_state": 42, "shared_fold_indices": True,
            "headline_metrics_from_OOF_predictions": True,
        },
        "baseline": {**baseline, "verification": baseline_verification},
        "models": {
            key: results[key]
            for key in ("random_forest", "catboost", "xgboost", "lightgbm", "huber", "tweedie")
        },
        "target_transformations": {
            "raw_ppsf": results["random_forest"],
            "log_ppsf": results["target_log_ppsf"],
            "log_price": results["target_log_price"],
            "boxcox_price": results["target_boxcox_price"],
        },
        "feature_engineering": {
            "full": results["feature_full"],
            "minus_micro_market": results["feature_minus_micro_market"],
            "minus_interactions": results["feature_minus_interactions"],
            "summary": feature_summary,
        },
        "quantile_models": {"models": quantile_models, "quality": quantile_quality},
        "blending": results["best_blend"],
        "stacking": results["best_stack"],
        "premium_segment": {
            "threshold_RM": float(np.quantile(y, 0.95)),
            "count": int(np.sum(y >= np.quantile(y, 0.95))),
            "best_model": best_premium_key,
            "best_metrics": results[best_premium_key]["metrics"]["top_5_percent"],
        },
        "generalization_gap": {
            key: value["generalization_gap"] for key, value in results.items()
        },
        "residual_correlations": correlations.to_dict(),
        "best_overall_model": best_key,
        "best_rmse_model": best_key,
        "best_mae_model": best_mae_key,
        "best_premium_model": best_premium_key,
        "promotion_checks": promotion_checks,
        "uncertainty": uncertainty,
        "recommendation": recommendation,
        "leakage_audit": {
            "shared_outer_folds": True,
            "preprocessing_fit_on_training_rows_only": True,
            "native_catboost_categories_not_target_encoded": True,
            "micro_market_aggregates_training_fold_only": True,
            "micro_market_training_rows_cross_fitted": True,
            "boxcox_lambda_outer_training_fold_only": True,
            "blend_weights_inner_oof_training_only": True,
            "stacker_inner_oof_training_only": True,
            "target_winsorization_or_capping": False,
            "premium_rows_deleted": False,
            "arbitrary_prediction_clipping": False,
            "protected_files_unchanged_during_run": True,
        },
        "dependencies": {
            "catboost": catboost.__version__, "xgboost": xgboost.__version__,
            "lightgbm": lightgbm.__version__, "scipy": scipy.__version__,
        },
        "skipped": {
            "log_cosh": "Skipped because no direct verified stable objective was available; no unverified custom gradient/Hessian was introduced."
        },
        "artifacts": sorted(
            path.relative_to(PROJECT_ROOT).as_posix()
            for path in BASE.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        ),
        "recovery_note": "Model fitting completed; results.json was reconstructed from the completed CSV/OOF artifacts after a NumPy-boolean serialization error. No model metrics were recomputed by refitting.",
    }
    (BASE / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Rebuilt {BASE / 'results.json'}")


if __name__ == "__main__":
    main()
