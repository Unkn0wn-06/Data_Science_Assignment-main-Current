"""Winsorize training PPSF for the new fair-comparison winning model only."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.cleaning.enhanced_city import ENHANCED_CITY_DATA_PATH
from src.cleaning.pipeline import PROJECT_ROOT
from src.models.common.evaluation import regression_metrics
from src.models.common.features import MODEL_FEATURES, TARGET_COLUMN
from src.models.common.utilities import WinsorizedPricePerSquareFootRegressor
from src.models.enhanced_city import (
    adjusted_r2,
    build_base_regressor,
    shared_folds,
)


EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = EXPERIMENT_DIR / "results.json"
COMPARISON_PATH = (
    PROJECT_ROOT / "results" / "enhanced_city" / "model_comparison.json"
)

VARIANTS = {
    "baseline": (None, None),
    "winsor_0_5_99_5": (0.005, 0.995),
    "winsor_1_99": (0.01, 0.99),
    "winsor_2_5_97_5": (0.025, 0.975),
    "winsor_5_95": (0.05, 0.95),
    "upper_99": (None, 0.99),
    "upper_97_5": (None, 0.975),
}


def group_metrics(actual, predicted, mask) -> dict:
    """Return original-price metrics for one expensive-property partition."""
    metrics = regression_metrics(actual[mask], predicted[mask])
    return {
        "count": int(np.sum(mask)),
        "RMSE_RM": float(metrics["RMSE"]),
        "MAE_RM": float(metrics["MAE"]),
    }


def evaluate_variant(data, model_name, folds, lower_quantile, upper_quantile) -> dict:
    """Learn every cap and preprocessing transform inside its outer training fold."""
    X = data[MODEL_FEATURES]
    y = data[TARGET_COLUMN]
    actual = y.to_numpy(dtype=float)
    prediction = np.empty(len(data), dtype=float)
    fold_details = []

    for fold_number, (train_index, validation_index) in enumerate(folds, start=1):
        estimator = WinsorizedPricePerSquareFootRegressor(
            regressor=build_base_regressor(model_name),
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
        )
        estimator.fit(X.iloc[train_index], y.iloc[train_index])
        prediction[validation_index] = estimator.predict(X.iloc[validation_index])
        if estimator.training_rows_ != estimator.rows_after_clipping_:
            raise AssertionError("Winsorization removed training observations.")
        fold_details.append(
            {
                "fold": fold_number,
                "training_rows": estimator.training_rows_,
                "validation_rows": len(validation_index),
                "lower_ppsf_boundary": estimator.lower_bound_,
                "upper_ppsf_boundary": estimator.upper_bound_,
                "lower_clipped_count": estimator.lower_clipped_count_,
                "upper_clipped_count": estimator.upper_clipped_count_,
            }
        )

    overall = regression_metrics(actual, prediction, include_distribution=True)
    overall_adjusted = adjusted_r2(
        float(overall["R2"]), len(data), len(MODEL_FEATURES)
    )
    threshold = float(np.quantile(actual, 0.95))
    top_mask = actual >= threshold
    lower_bounds = [
        item["lower_ppsf_boundary"]
        for item in fold_details
        if item["lower_ppsf_boundary"] is not None
    ]
    upper_bounds = [
        item["upper_ppsf_boundary"]
        for item in fold_details
        if item["upper_ppsf_boundary"] is not None
    ]
    return {
        "lower_quantile": lower_quantile,
        "upper_quantile": upper_quantile,
        "average_lower_ppsf_boundary": (
            None if not lower_bounds else float(np.mean(lower_bounds))
        ),
        "average_upper_ppsf_boundary": (
            None if not upper_bounds else float(np.mean(upper_bounds))
        ),
        "overall": {
            "RMSE_RM": float(overall["RMSE"]),
            "MAE_RM": float(overall["MAE"]),
            "R2": float(overall["R2"]),
            "Adjusted_R2": overall_adjusted,
            "Median_AE_RM": float(overall["median_absolute_error"]),
        },
        "top_5_percent": group_metrics(actual, prediction, top_mask),
        "remaining_95_percent": group_metrics(actual, prediction, ~top_mask),
        "top_5_percent_price_threshold_RM": threshold,
        "validation_rows": len(prediction),
        "folds": fold_details,
    }


def compare_to_baseline(result: dict, baseline: dict) -> None:
    """Attach signed error changes; negative RMSE/MAE changes mean improvement."""
    current = result["overall"]
    reference = baseline["overall"]
    rmse_difference = current["RMSE_RM"] - reference["RMSE_RM"]
    mae_difference = current["MAE_RM"] - reference["MAE_RM"]
    result["comparison_vs_enhanced_baseline"] = {
        "RMSE_difference_RM": rmse_difference,
        "RMSE_percentage_change": rmse_difference / reference["RMSE_RM"] * 100.0,
        "MAE_difference_RM": mae_difference,
        "MAE_percentage_change": mae_difference / reference["MAE_RM"] * 100.0,
        "R2_difference": current["R2"] - reference["R2"],
        "Adjusted_R2_difference": current["Adjusted_R2"]
        - reference["Adjusted_R2"],
        "top_5_percent_RMSE_percentage_change": (
            (result["top_5_percent"]["RMSE_RM"] - baseline["top_5_percent"]["RMSE_RM"])
            / baseline["top_5_percent"]["RMSE_RM"]
            * 100.0
        ),
        "top_5_percent_MAE_percentage_change": (
            (result["top_5_percent"]["MAE_RM"] - baseline["top_5_percent"]["MAE_RM"])
            / baseline["top_5_percent"]["MAE_RM"]
            * 100.0
        ),
    }


def select_variant(variants: dict) -> tuple[str | None, str]:
    """Require aggregate improvement without severe premium-property damage."""
    eligible = []
    for name, result in variants.items():
        change = result["comparison_vs_enhanced_baseline"]
        passed = (
            change["RMSE_difference_RM"] < 0.0
            and change["MAE_percentage_change"] <= 0.5
            and change["R2_difference"] >= 0.0
            and change["Adjusted_R2_difference"] >= 0.0
            and change["top_5_percent_RMSE_percentage_change"] <= 5.0
            and change["top_5_percent_MAE_percentage_change"] <= 5.0
        )
        result["selection_rule_passed"] = passed
        if passed:
            eligible.append(
                (
                    result["overall"]["RMSE_RM"],
                    result["overall"]["MAE_RM"],
                    name,
                )
            )
    if not eligible:
        return None, (
            "Retain the non-Winsorized enhanced baseline because no target cap "
            "improved overall RMSE while satisfying the MAE, R2, adjusted-R2, "
            "and premium-property safeguards."
        )
    _, _, winner = min(eligible)
    return winner, f"{winner} is the best configuration satisfying every safeguard."


def main() -> None:
    """Run every cap with the comparison winner and write only this experiment."""
    comparison = json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))
    model_name = comparison["selection"]["best_overall_model"]
    data_hash = hashlib.sha256(ENHANCED_CITY_DATA_PATH.read_bytes()).hexdigest()
    if data_hash != comparison["dataset_sha256"]:
        raise ValueError("Enhanced City dataset changed after the model comparison.")
    data = pd.read_csv(ENHANCED_CITY_DATA_PATH)
    folds = shared_folds(len(data))

    evaluated = {}
    for name, (lower_quantile, upper_quantile) in VARIANTS.items():
        evaluated[name] = evaluate_variant(
            data,
            model_name,
            folds,
            lower_quantile,
            upper_quantile,
        )
        overall = evaluated[name]["overall"]
        print(
            f"{name}: RMSE=RM {overall['RMSE_RM']:,.2f}; "
            f"MAE=RM {overall['MAE_RM']:,.2f}; R2={overall['R2']:.6f}",
            flush=True,
        )

    baseline = evaluated.pop("baseline")
    comparison_baseline = comparison["results"][model_name]
    for metric in ["RMSE_RM", "MAE_RM", "R2", "Adjusted_R2", "Median_AE_RM"]:
        if not np.isclose(
            baseline["overall"][metric], comparison_baseline[metric], rtol=0, atol=1e-8
        ):
            raise AssertionError(f"Winsorization baseline does not match comparison: {metric}")

    for result in evaluated.values():
        compare_to_baseline(result, baseline)
    best_variant, recommendation = select_variant(evaluated)
    configurations = {"baseline": baseline, **evaluated}
    best_rmse = min(
        configurations,
        key=lambda name: configurations[name]["overall"]["RMSE_RM"],
    )
    best_mae = min(
        configurations,
        key=lambda name: configurations[name]["overall"]["MAE_RM"],
    )
    payload = {
        "dataset": str(ENHANCED_CITY_DATA_PATH.relative_to(PROJECT_ROOT)),
        "dataset_sha256": data_hash,
        "rows": len(data),
        "model": model_name,
        "model_selected_from": str(COMPARISON_PATH.relative_to(PROJECT_ROOT)),
        "target_strategy": "price_per_square_foot",
        "features": MODEL_FEATURES,
        "cross_validation": {
            "type": "KFold",
            "n_splits": 5,
            "shuffle": True,
            "random_state": 42,
            "identical_to_model_comparison": True,
        },
        "leakage_guards": {
            "preprocessing_fit_on_training_fold_only": True,
            "winsorization_bounds_fit_on_training_ppsf_only": True,
            "validation_targets_winsorized": False,
            "rows_removed": False,
        },
        "baseline": baseline,
        "variants": evaluated,
        "best_rmse_configuration": best_rmse,
        "best_mae_configuration": best_mae,
        "best_winsorization_configuration": best_variant,
        "recommendation": recommendation,
    }
    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    print(f"Best RMSE configuration: {best_rmse}")
    print(f"Best MAE configuration: {best_mae}")
    print(f"Best Winsorization configuration: {best_variant}")
    print(f"Recommendation: {recommendation}")
    print(f"Saved {RESULTS_PATH}")


if __name__ == "__main__":
    main()

