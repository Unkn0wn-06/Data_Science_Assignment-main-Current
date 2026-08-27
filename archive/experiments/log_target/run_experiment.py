"""Compare leakage-safe logarithmic targets with the enhanced City PPSF baseline."""

# Import hashing so the experiment can verify the canonical dataset is unchanged.
import hashlib
# Import JSON so benchmark metadata can be read and experiment results can be saved.
import json
# Import Path so all experiment paths remain independent of the working directory.
from pathlib import Path
# Import sys so this nested script can resolve repository packages when run directly.
import sys

# Resolve and expose the project root before importing modules from ``src``.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Convert the root once because Python's import path stores strings.
project_root_text = str(PROJECT_ROOT)
# Support ``python experiments/log_target/run_experiment.py`` from a clean shell.
if project_root_text not in sys.path:
    sys.path.insert(0, project_root_text)

# Import NumPy for log transforms, inverse transforms, masks, and diagnostics.
import numpy as np
# Import pandas for loading the canonical enhanced City table.
import pandas as pd

# Import the promoted dataset path rather than creating another prepared dataset.
from src.cleaning.enhanced_city import ENHANCED_CITY_DATA_PATH
# Import the established original-price regression metrics.
from src.models.common.evaluation import regression_metrics
# Import the exact enhanced input schema and target name.
from src.models.common.features import MODEL_FEATURES, TARGET_COLUMN
# Reuse enhanced preprocessing, PPSF wrapping, parameters, and shared outer folds.
from src.models.enhanced_city import (
    adjusted_r2,
    build_base_regressor,
    build_ppsf_estimator,
    evaluate_estimator,
    model_parameters,
    shared_folds,
)


# Keep every output inside the isolated experiment directory.
EXPERIMENT_DIR = Path(__file__).resolve().parent
# Save the requested machine-readable comparison beside this runner.
RESULTS_PATH = EXPERIMENT_DIR / "results.json"
# Verify the reproduced baseline against the promoted fair comparison.
COMPARISON_PATH = PROJECT_ROOT / "results" / "enhanced_city" / "model_comparison.json"
# Use only the verified enhanced-comparison winner without retuning it.
MODEL_NAME = "Random Forest"
# Evaluate the baseline plus all four requested transformed-target variants.
VARIANT_ORDER = [
    "baseline_ppsf",
    "log_ppsf",
    "log_ppsf_smearing",
    "log_price",
    "log_price_smearing",
]
# Provide readable labels for the console comparison table.
VARIANT_LABELS = {
    "baseline_ppsf": "Baseline PPSF",
    "log_ppsf": "Log PPSF",
    "log_ppsf_smearing": "Log PPSF + Smearing",
    "log_price": "Log Price",
    "log_price_smearing": "Log Price + Smearing",
}


def file_sha256(path: Path) -> str:
    """Return a SHA-256 fingerprint without changing the protected file."""
    # Hash the existing bytes exactly as stored on disk.
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_and_validate_data() -> tuple[pd.DataFrame, dict]:
    """Load canonical rows and reject invalid values before any log transform."""
    # Read the same enhanced City dataset used by the current fair comparison.
    data = pd.read_csv(ENHANCED_CITY_DATA_PATH)
    # Require the original price plus all enhanced model inputs.
    required_columns = {TARGET_COLUMN, *MODEL_FEATURES}
    # Calculate any missing schema fields before numerical operations begin.
    missing_columns = sorted(required_columns.difference(data.columns))
    # Refuse to run against a structurally different dataset.
    if missing_columns:
        raise ValueError(f"Enhanced dataset is missing columns: {missing_columns}")

    # Convert the two target-strategy inputs explicitly for reliable validation.
    price = pd.to_numeric(data[TARGET_COLUMN], errors="coerce").to_numpy(float)
    size = pd.to_numeric(data["property_size_sqft"], errors="coerce").to_numpy(float)
    # Calculate PPSF only where division is mathematically valid.
    ppsf = np.divide(
        price,
        size,
        out=np.full(len(data), np.nan, dtype=float),
        where=np.isfinite(size) & (size != 0.0),
    )
    # Record every invalid-target category instead of silently filtering rows.
    validation = {
        "price_non_finite_or_non_positive": int(
            np.sum(~np.isfinite(price) | (price <= 0.0))
        ),
        "property_size_non_finite_or_non_positive": int(
            np.sum(~np.isfinite(size) | (size <= 0.0))
        ),
        "ppsf_non_finite_or_non_positive": int(
            np.sum(~np.isfinite(ppsf) | (ppsf <= 0.0))
        ),
    }
    # Stop with the reported counts because arbitrary log offsets are forbidden.
    if any(validation.values()):
        raise ValueError(f"Invalid values prevent log-target evaluation: {validation}")
    # Preserve all canonical rows and store validated floats back in the table.
    data[TARGET_COLUMN] = price
    data["property_size_sqft"] = size
    # Record that no row was removed or changed for this experiment.
    validation["rows_removed"] = 0
    validation["all_rows_valid"] = True
    # Return the unchanged row set with explicit validation evidence.
    return data.reset_index(drop=True), validation


def partition_metrics(actual: np.ndarray, predicted: np.ndarray, mask: np.ndarray) -> dict:
    """Calculate price errors and signed prediction bias for one partition."""
    # Select matching original-price targets and reconstructed predictions.
    actual_group = actual[mask]
    predicted_group = predicted[mask]
    # Calculate RMSE and MAE in Ringgit on the original target scale.
    metrics = regression_metrics(actual_group, predicted_group)
    # Define signed error consistently as prediction minus actual price.
    error = predicted_group - actual_group
    # Return requested group size, errors, and bias diagnostics.
    return {
        "count": int(np.sum(mask)),
        "RMSE_RM": float(metrics["RMSE"]),
        "MAE_RM": float(metrics["MAE"]),
        "Mean_Error_RM": float(np.mean(error)),
        "Median_Error_RM": float(np.median(error)),
    }


def summarize_predictions(
    actual: np.ndarray,
    predicted: np.ndarray,
    fold_details: list[dict],
    smearing_factors: list[float] | None = None,
) -> dict:
    """Summarize full OOF predictions and premium/remaining price partitions."""
    # Reject numerical failures before calculating aggregate metrics.
    if not np.all(np.isfinite(predicted)):
        raise ValueError("A target variant produced non-finite price predictions.")
    # Calculate all requested original-price overall metrics.
    overall_metrics = regression_metrics(actual, predicted, include_distribution=True)
    # Calculate signed overall prediction errors in Ringgit.
    overall_error = predicted - actual
    # Reuse the established 95th-percentile threshold for the premium group.
    premium_threshold = float(np.quantile(actual, 0.95))
    # Mark the exact top-price observations used by historical enhanced diagnostics.
    top_mask = actual >= premium_threshold
    # Build the common result record for baseline and transformed targets.
    result = {
        "overall": {
            "RMSE_RM": float(overall_metrics["RMSE"]),
            "MAE_RM": float(overall_metrics["MAE"]),
            "R2": float(overall_metrics["R2"]),
            "Adjusted_R2": adjusted_r2(
                float(overall_metrics["R2"]), len(actual), len(MODEL_FEATURES)
            ),
            "Median_AE_RM": float(overall_metrics["median_absolute_error"]),
            "Mean_Error_RM": float(np.mean(overall_error)),
            "Median_Error_RM": float(np.median(overall_error)),
        },
        "top_5_percent": partition_metrics(actual, predicted, top_mask),
        "remaining_95_percent": partition_metrics(actual, predicted, ~top_mask),
        "top_5_percent_price_threshold_RM": premium_threshold,
        "validation_rows": int(len(predicted)),
        "folds": fold_details,
    }
    # Store fold-local smearing evidence only for corrected variants.
    if smearing_factors is not None:
        result["fold_smearing_factors"] = [float(value) for value in smearing_factors]
        result["average_smearing_factor"] = float(np.mean(smearing_factors))
    # Return a JSON-safe record with no validation targets or row-level predictions.
    return result


def evaluate_baseline(data: pd.DataFrame, folds: list[tuple]) -> dict:
    """Reproduce the existing Random Forest PPSF benchmark exactly."""
    # Build the same PPSF wrapper used by the enhanced City comparison.
    estimator = build_ppsf_estimator(MODEL_NAME)
    # Reuse its established OOF evaluator and original-price reconstruction.
    _, prediction = evaluate_estimator(estimator, data, folds)
    # Record outer-fold row counts to demonstrate identical fold usage.
    fold_details = [
        {
            "fold": fold_number,
            "training_rows": int(len(train_index)),
            "validation_rows": int(len(validation_index)),
            "smearing_factor": None,
        }
        for fold_number, (train_index, validation_index) in enumerate(folds, start=1)
    ]
    # Calculate the expanded metrics requested for this experiment.
    return summarize_predictions(
        data[TARGET_COLUMN].to_numpy(float),
        prediction,
        fold_details,
    )


def evaluate_log_variant(
    data: pd.DataFrame,
    folds: list[tuple],
    target_strategy: str,
    use_smearing: bool,
) -> dict:
    """Evaluate one log target with fold-local preprocessing and correction."""
    # Select all enhanced City inputs without price-derived leakage features.
    X = data[MODEL_FEATURES]
    # Keep original total prices for final Ringgit evaluation.
    actual = data[TARGET_COLUMN].to_numpy(float)
    # Keep property size only for the PPSF target and total-price reconstruction.
    size = data["property_size_sqft"].to_numpy(float)
    # Allocate one untouched-fold prediction for every canonical observation.
    prediction = np.empty(len(data), dtype=float)
    # Retain fold metadata and training-only correction evidence.
    fold_details = []
    smearing_factors = [] if use_smearing else None

    # Fit preprocessing, target encoding, model, and smearing inside each fold.
    for fold_number, (train_index, validation_index) in enumerate(folds, start=1):
        # Build a fresh copy of the unchanged enhanced Random Forest pipeline.
        pipeline = build_base_regressor(MODEL_NAME)
        # Create only the requested log target from this fold's training rows.
        if target_strategy == "ppsf":
            training_target = np.log1p(
                actual[train_index] / size[train_index]
            )
        elif target_strategy == "price":
            training_target = np.log1p(actual[train_index])
        else:
            raise ValueError(f"Unknown log target strategy: {target_strategy}")

        # Fit every learned transform and the forest using training rows only.
        pipeline.fit(X.iloc[train_index], training_target)
        # Use a neutral factor when smearing is not part of the variant.
        smearing_factor = None
        # Estimate retransformation correction only from training residuals.
        if use_smearing:
            training_log_prediction = pipeline.predict(X.iloc[train_index])
            training_residual = training_target - training_log_prediction
            smearing_factor = float(np.mean(np.exp(training_residual)))
            smearing_factors.append(smearing_factor)

        # Predict the transformed target for untouched validation rows.
        validation_log_prediction = pipeline.predict(X.iloc[validation_index])
        # Apply exactly the specified inverse transformation.
        if use_smearing:
            inverse_prediction = (
                np.exp(validation_log_prediction) * smearing_factor - 1.0
            )
        else:
            inverse_prediction = np.expm1(validation_log_prediction)
        # Convert PPSF back to total price; direct log-price needs no size factor.
        if target_strategy == "ppsf":
            inverse_prediction = inverse_prediction * size[validation_index]
        # Store reconstructed total prices in original row order.
        prediction[validation_index] = inverse_prediction
        # Record enough fold evidence to audit leakage boundaries.
        fold_details.append(
            {
                "fold": fold_number,
                "training_rows": int(len(train_index)),
                "validation_rows": int(len(validation_index)),
                "smearing_factor": smearing_factor,
            }
        )

    # Calculate all official metrics against untouched original prices.
    return summarize_predictions(
        actual,
        prediction,
        fold_details,
        smearing_factors,
    )


def attach_baseline_comparison(result: dict, baseline: dict) -> None:
    """Attach signed changes where negative error changes mean improvement."""
    # Read overall metrics once to keep formulas explicit.
    current = result["overall"]
    reference = baseline["overall"]
    # Calculate raw error differences in Ringgit.
    rmse_difference = current["RMSE_RM"] - reference["RMSE_RM"]
    mae_difference = current["MAE_RM"] - reference["MAE_RM"]
    # Calculate premium changes for the balanced production safeguard.
    top_rmse_difference = (
        result["top_5_percent"]["RMSE_RM"]
        - baseline["top_5_percent"]["RMSE_RM"]
    )
    top_mae_difference = (
        result["top_5_percent"]["MAE_RM"]
        - baseline["top_5_percent"]["MAE_RM"]
    )
    # Calculate remaining-95% changes to reveal typical-property trade-offs.
    remaining_rmse_difference = (
        result["remaining_95_percent"]["RMSE_RM"]
        - baseline["remaining_95_percent"]["RMSE_RM"]
    )
    remaining_mae_difference = (
        result["remaining_95_percent"]["MAE_RM"]
        - baseline["remaining_95_percent"]["MAE_RM"]
    )
    # Save requested overall differences plus partition diagnostics.
    result["comparison_vs_baseline_ppsf"] = {
        "RMSE_difference_RM": float(rmse_difference),
        "RMSE_percentage_change": float(
            rmse_difference / reference["RMSE_RM"] * 100.0
        ),
        "MAE_difference_RM": float(mae_difference),
        "MAE_percentage_change": float(
            mae_difference / reference["MAE_RM"] * 100.0
        ),
        "R2_difference": float(current["R2"] - reference["R2"]),
        "Adjusted_R2_difference": float(
            current["Adjusted_R2"] - reference["Adjusted_R2"]
        ),
        "top_5_percent_RMSE_difference_RM": float(top_rmse_difference),
        "top_5_percent_RMSE_percentage_change": float(
            top_rmse_difference / baseline["top_5_percent"]["RMSE_RM"] * 100.0
        ),
        "top_5_percent_MAE_difference_RM": float(top_mae_difference),
        "top_5_percent_MAE_percentage_change": float(
            top_mae_difference / baseline["top_5_percent"]["MAE_RM"] * 100.0
        ),
        "remaining_95_percent_RMSE_difference_RM": float(
            remaining_rmse_difference
        ),
        "remaining_95_percent_RMSE_percentage_change": float(
            remaining_rmse_difference
            / baseline["remaining_95_percent"]["RMSE_RM"]
            * 100.0
        ),
        "remaining_95_percent_MAE_difference_RM": float(remaining_mae_difference),
        "remaining_95_percent_MAE_percentage_change": float(
            remaining_mae_difference
            / baseline["remaining_95_percent"]["MAE_RM"]
            * 100.0
        ),
    }


def select_balanced_variant(
    variants: dict,
    mean_actual_price: float,
) -> tuple[str, dict, str]:
    """Apply explicit aggregate, premium, and bias safeguards to log variants."""
    # Use a half-percent MAE tolerance for "effectively unchanged."
    mae_tolerance_percent = 0.5
    # Treat at most five-percent premium deterioration as non-severe.
    premium_deterioration_limit_percent = 5.0
    # Limit reasonable mean bias to two percent of the mean original price.
    mean_bias_limit_rm = mean_actual_price * 0.02
    # Keep every rule outcome auditable in results.json.
    evaluations = {}
    # Retain transformed variants that satisfy every production safeguard.
    eligible = []

    # Evaluate only log variants; the baseline remains the fallback.
    for name in VARIANT_ORDER[1:]:
        result = variants[name]
        change = result["comparison_vs_baseline_ppsf"]
        # Evaluate the five requested balanced-selection conditions explicitly.
        checks = {
            "overall_RMSE_improves": change["RMSE_difference_RM"] < 0.0,
            "overall_MAE_improves_or_within_0_5_percent": (
                change["MAE_percentage_change"] <= mae_tolerance_percent
            ),
            "R2_stable_or_improves": change["R2_difference"] >= 0.0,
            "top_5_percent_RMSE_not_severely_worse": (
                change["top_5_percent_RMSE_percentage_change"]
                <= premium_deterioration_limit_percent
            ),
            "top_5_percent_MAE_not_severely_worse": (
                change["top_5_percent_MAE_percentage_change"]
                <= premium_deterioration_limit_percent
            ),
            "overall_mean_bias_within_2_percent_of_mean_price": (
                abs(result["overall"]["Mean_Error_RM"]) <= mean_bias_limit_rm
            ),
        }
        # A log strategy is production-eligible only if every check passes.
        passed = all(checks.values())
        evaluations[name] = {"checks": checks, "passed": passed}
        # Rank eligible strategies through equally weighted RMSE and MAE ratios.
        if passed:
            baseline = variants["baseline_ppsf"]["overall"]
            balanced_score = 0.5 * (
                result["overall"]["RMSE_RM"] / baseline["RMSE_RM"]
            ) + 0.5 * (result["overall"]["MAE_RM"] / baseline["MAE_RM"])
            eligible.append((balanced_score, name))

    # Select the strongest eligible transformed target or retain the baseline.
    if eligible:
        _, selected = min(eligible)
        recommendation = (
            f"Review {selected} for a separate production promotion because it "
            "satisfies every aggregate, premium-property, and bias safeguard."
        )
    else:
        selected = "baseline_ppsf"
        recommendation = (
            "Retain the current non-log PPSF Random Forest because no logarithmic "
            "target satisfies every aggregate, premium-property, and bias safeguard."
        )
    # Return the decision, transparent criteria, and readable recommendation.
    criteria = {
        "MAE_effectively_unchanged_tolerance_percent": mae_tolerance_percent,
        "premium_deterioration_limit_percent": premium_deterioration_limit_percent,
        "overall_mean_bias_limit_RM": float(mean_bias_limit_rm),
        "variant_evaluations": evaluations,
    }
    return selected, criteria, recommendation


def print_comparison(variants: dict) -> None:
    """Print a readable aggregate comparison and winning configurations."""
    # Print fixed-width headers for quick terminal review.
    print()
    print(
        f"{'Variant':<24} {'RMSE':>13} {'MAE':>13} {'R2':>10} "
        f"{'Median AE':>13} {'Delta RMSE':>13} {'Delta MAE':>13}"
    )
    # Print variants in the requested conceptual order.
    for name in VARIANT_ORDER:
        overall = variants[name]["overall"]
        change = variants[name]["comparison_vs_baseline_ppsf"]
        print(
            f"{VARIANT_LABELS[name]:<24} "
            f"{overall['RMSE_RM']:>13,.2f} "
            f"{overall['MAE_RM']:>13,.2f} "
            f"{overall['R2']:>10.6f} "
            f"{overall['Median_AE_RM']:>13,.2f} "
            f"{change['RMSE_difference_RM']:>+13,.2f} "
            f"{change['MAE_difference_RM']:>+13,.2f}"
        )


def main() -> None:
    """Run the isolated log-target experiment and save its full diagnostics."""
    # Load the canonical enhanced benchmark for identity and baseline checks.
    comparison = json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))
    # Refuse to run if the promoted winner is no longer the expected forest.
    if comparison["selection"]["best_overall_model"] != MODEL_NAME:
        raise ValueError("Enhanced comparison winner is no longer Random Forest.")
    # Verify the exact canonical dataset bytes before fitting any model.
    dataset_hash = file_sha256(ENHANCED_CITY_DATA_PATH)
    if dataset_hash != comparison["dataset_sha256"]:
        raise ValueError("Enhanced City dataset changed after the fair comparison.")
    # Validate every row before applying log1p to price or PPSF.
    data, target_validation = load_and_validate_data()
    # Reuse one materialized five-fold split for all five configurations.
    folds = shared_folds(len(data))

    # Reproduce the current non-log PPSF benchmark first.
    variants = {"baseline_ppsf": evaluate_baseline(data, folds)}
    # Compare it exactly with the promoted enhanced result before proceeding.
    expected_baseline = comparison["results"][MODEL_NAME]
    baseline_differences = {}
    for metric in ["RMSE_RM", "MAE_RM", "R2", "Adjusted_R2", "Median_AE_RM"]:
        difference = (
            variants["baseline_ppsf"]["overall"][metric]
            - expected_baseline[metric]
        )
        baseline_differences[metric] = float(difference)
        if not np.isclose(difference, 0.0, rtol=0.0, atol=1e-8):
            raise AssertionError(
                f"Baseline differs from enhanced comparison for {metric}: {difference}"
            )
    # Run log-PPSF without and with training-only smearing correction.
    variants["log_ppsf"] = evaluate_log_variant(data, folds, "ppsf", False)
    variants["log_ppsf_smearing"] = evaluate_log_variant(
        data, folds, "ppsf", True
    )
    # Run direct log-price without and with training-only smearing correction.
    variants["log_price"] = evaluate_log_variant(data, folds, "price", False)
    variants["log_price_smearing"] = evaluate_log_variant(
        data, folds, "price", True
    )

    # Attach baseline-relative changes to every configuration, including zero baseline.
    for result in variants.values():
        attach_baseline_comparison(result, variants["baseline_ppsf"])
    # Identify independent metric winners without implying a production decision.
    best_rmse_variant = min(
        variants, key=lambda name: variants[name]["overall"]["RMSE_RM"]
    )
    best_mae_variant = min(
        variants, key=lambda name: variants[name]["overall"]["MAE_RM"]
    )
    best_r2_variant = max(
        variants, key=lambda name: variants[name]["overall"]["R2"]
    )
    # Apply the explicit multi-metric production safeguard separately.
    best_balanced_variant, balanced_rule, recommendation = select_balanced_variant(
        variants,
        float(data[TARGET_COLUMN].mean()),
    )

    # Preserve parameters and leakage boundaries alongside the measured results.
    payload = {
        "dataset": ENHANCED_CITY_DATA_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "dataset_sha256": dataset_hash,
        "rows": int(len(data)),
        "model": MODEL_NAME,
        "model_parameters": model_parameters()[MODEL_NAME],
        "model_selected_from": COMPARISON_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "features": MODEL_FEATURES,
        "folds": 5,
        "random_state": 42,
        "cross_validation": {
            "type": "KFold",
            "n_splits": 5,
            "shuffle": True,
            "random_state": 42,
            "shared_fold_indices_for_all_variants": True,
        },
        "target_validation": target_validation,
        "leakage_guards": {
            "preprocessing_fit_on_outer_training_rows_only": True,
            "categorical_target_encoding_fit_on_training_rows_only": True,
            "smearing_estimated_from_training_residuals_only": True,
            "validation_targets_used_for_smearing": False,
            "metrics_calculated_on_original_price_scale": True,
            "winsorization_used": False,
        },
        "baseline_verification": {
            "source": "enhanced City fair comparison",
            "absolute_tolerance": 1e-8,
            "matched": True,
            "metric_differences": baseline_differences,
        },
        "baseline_ppsf": variants["baseline_ppsf"],
        "log_ppsf": variants["log_ppsf"],
        "log_ppsf_smearing": variants["log_ppsf_smearing"],
        "log_price": variants["log_price"],
        "log_price_smearing": variants["log_price_smearing"],
        "best_rmse_variant": best_rmse_variant,
        "best_mae_variant": best_mae_variant,
        "best_r2_variant": best_r2_variant,
        "best_balanced_variant": best_balanced_variant,
        "balanced_selection_rule": balanced_rule,
        "recommendation": recommendation,
    }
    # Write only the isolated experiment artifact requested by the brief.
    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    # Print the requested table and independent winners for human review.
    print_comparison(variants)
    print()
    print(f"Best RMSE: {best_rmse_variant}")
    print(f"Best MAE: {best_mae_variant}")
    print(f"Best R2: {best_r2_variant}")
    print(f"Best balanced configuration: {best_balanced_variant}")
    print(f"Recommendation: {recommendation}")
    print(f"Saved: {RESULTS_PATH}")


# Run the experiment only when this file is executed as a script.
if __name__ == "__main__":
    main()
