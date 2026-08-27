"""Evaluate training-fold-only PPSF Winsorization with the fixed best model."""

import json
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from experiments.enhanced_models.run_experiment import (
    ENHANCED_CATEGORICAL_FEATURES,
    ENHANCED_NUMERICAL_FEATURES,
    prepare_enhanced_dataset,
)
from src.cleaning.location_cleaning import extract_city
from src.models.common.evaluation import regression_metrics
from src.models.common.preprocessing import make_target_encoding_preprocessor
from src.models.random_forest.model import build_best_city_model


# Keep all output inside this experiment so production artifacts remain untouched.
EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = EXPERIMENT_DIR / "results.json"

# Preserve the historical benchmark only as a reference, never as the experiment baseline.
RECORDED_BEST = {
    "RMSE": 123231.38594227468,
    "MAE": 61189.660400189256,
    "R2": 0.860291494082739,
}

# Define every requested target-Winsorization configuration as quantile fractions.
VARIANT_SPECS = {
    "baseline": {"lower_quantile": None, "upper_quantile": None},
    "winsor_0_5_99_5": {"lower_quantile": 0.005, "upper_quantile": 0.995},
    "winsor_1_99": {"lower_quantile": 0.01, "upper_quantile": 0.99},
    "winsor_2_5_97_5": {"lower_quantile": 0.025, "upper_quantile": 0.975},
    "winsor_5_95": {"lower_quantile": 0.05, "upper_quantile": 0.95},
    "upper_99_only": {"lower_quantile": None, "upper_quantile": 0.99},
    "upper_97_5_only": {"lower_quantile": None, "upper_quantile": 0.975},
}

DISPLAY_NAMES = {
    "baseline": "Baseline",
    "winsor_0_5_99_5": "0.5-99.5%",
    "winsor_1_99": "1-99%",
    "winsor_2_5_97_5": "2.5-97.5%",
    "winsor_5_95": "5-95%",
    "upper_99_only": "Upper 99%",
    "upper_97_5_only": "Upper 97.5%",
}


class TrainingFoldWinsorizedPPSFRegressor(RegressorMixin, BaseEstimator):
    """Learn clipped training PPSF while predicting untouched validation prices."""

    def __init__(
        self,
        regressor=None,
        size_column="property_size_sqft",
        lower_quantile=None,
        upper_quantile=None,
    ):
        # Store constructor inputs unchanged so scikit-learn can clone the wrapper.
        self.regressor = regressor
        self.size_column = size_column
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y):
        """Estimate bounds from this training fold and fit on capped training PPSF."""
        self._validate_quantiles()
        size = X[self.size_column].to_numpy(dtype=float)
        total_price = np.asarray(y, dtype=float)
        training_ppsf = total_price / size

        # Quantiles are deliberately learned here, after the outer training split.
        self.lower_bound_ = (
            None
            if self.lower_quantile is None
            else float(np.quantile(training_ppsf, self.lower_quantile))
        )
        self.upper_bound_ = (
            None
            if self.upper_quantile is None
            else float(np.quantile(training_ppsf, self.upper_quantile))
        )
        self.lower_clipped_count_ = (
            0
            if self.lower_bound_ is None
            else int(np.sum(training_ppsf < self.lower_bound_))
        )
        self.upper_clipped_count_ = (
            0
            if self.upper_bound_ is None
            else int(np.sum(training_ppsf > self.upper_bound_))
        )

        # Cap values without deleting a single training observation.
        clipped_training_ppsf = np.clip(
            training_ppsf,
            self.lower_bound_,
            self.upper_bound_,
        )
        self.training_rows_ = len(training_ppsf)
        self.rows_after_clipping_ = len(clipped_training_ppsf)

        # Clone the unchanged enhanced preprocessing and fixed Random Forest pipeline.
        self.regressor_ = clone(self.regressor)
        self.regressor_.fit(X, clipped_training_ppsf)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        """Predict PPSF, then restore total price with original validation sizes."""
        predicted_ppsf = self.regressor_.predict(X)
        validation_size = X[self.size_column].to_numpy(dtype=float)
        return np.clip(predicted_ppsf * validation_size, 0.0, None)

    def _validate_quantiles(self) -> None:
        """Reject invalid or reversed clipping limits before fitting."""
        for label, quantile in [
            ("lower_quantile", self.lower_quantile),
            ("upper_quantile", self.upper_quantile),
        ]:
            if quantile is not None and not 0.0 <= quantile <= 1.0:
                raise ValueError(f"{label} must be between zero and one.")
        if (
            self.lower_quantile is not None
            and self.upper_quantile is not None
            and self.lower_quantile >= self.upper_quantile
        ):
            raise ValueError("lower_quantile must be smaller than upper_quantile.")


def prepare_city_experiment_data():
    """Build the unchanged enhanced rows and the current city-only feature set."""
    data = prepare_enhanced_dataset()
    data["city"] = data["detailed_address"].map(extract_city)
    non_location_categories = [
        feature
        for feature in ENHANCED_CATEGORICAL_FEATURES
        if feature not in {"detailed_address", "city"}
    ]
    categorical_features = [*non_location_categories, "city"]
    features = ENHANCED_NUMERICAL_FEATURES + categorical_features
    return data, features, categorical_features


def build_variant_estimator(
    categorical_features: list[str],
    lower_quantile: float | None,
    upper_quantile: float | None,
) -> TrainingFoldWinsorizedPPSFRegressor:
    """Attach the fixed model to fold-fitted enhanced preprocessing and clipping."""
    regressor = Pipeline(
        [
            (
                "preprocessor",
                make_target_encoding_preprocessor(
                    ENHANCED_NUMERICAL_FEATURES,
                    categorical_features,
                ),
            ),
            ("model", build_best_city_model()),
        ]
    )
    return TrainingFoldWinsorizedPPSFRegressor(
        regressor=regressor,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )


def price_group_metrics(actual, predicted, mask: np.ndarray) -> dict:
    """Calculate RMSE and MAE for one original-price-defined property group."""
    metrics = regression_metrics(actual[mask], predicted[mask])
    return {
        "count": int(np.sum(mask)),
        "RMSE": float(metrics["RMSE"]),
        "MAE": float(metrics["MAE"]),
    }


def evaluate_variant(
    data,
    features: list[str],
    categorical_features: list[str],
    folds,
    lower_quantile: float | None,
    upper_quantile: float | None,
) -> dict:
    """Accumulate untouched validation predictions and fold-specific boundaries."""
    X = data[features]
    y = data["price"]
    actual = y.to_numpy(dtype=float)
    predicted = np.empty(len(data), dtype=float)
    fold_details = []

    for fold_number, (train_index, validation_index) in enumerate(folds, start=1):
        estimator = build_variant_estimator(
            categorical_features,
            lower_quantile,
            upper_quantile,
        )
        estimator.fit(X.iloc[train_index], y.iloc[train_index])
        predicted[validation_index] = estimator.predict(X.iloc[validation_index])

        # Make row preservation and per-fold bounds auditable in the result artifact.
        if estimator.training_rows_ != estimator.rows_after_clipping_:
            raise AssertionError("Winsorization must cap values without removing rows.")
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

    overall = regression_metrics(actual, predicted, include_distribution=True)
    expensive_threshold = float(np.quantile(actual, 0.95))
    expensive_mask = actual >= expensive_threshold
    remaining_mask = ~expensive_mask

    lower_bounds = [
        detail["lower_ppsf_boundary"]
        for detail in fold_details
        if detail["lower_ppsf_boundary"] is not None
    ]
    upper_bounds = [
        detail["upper_ppsf_boundary"]
        for detail in fold_details
        if detail["upper_ppsf_boundary"] is not None
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
            "RMSE": float(overall["RMSE"]),
            "MAE": float(overall["MAE"]),
            "R2": float(overall["R2"]),
            "median_absolute_error": float(overall["median_absolute_error"]),
        },
        "top_5_percent": price_group_metrics(actual, predicted, expensive_mask),
        "remaining_95_percent": price_group_metrics(actual, predicted, remaining_mask),
        "expensive_price_threshold": expensive_threshold,
        "total_validation_rows": len(predicted),
        "folds": fold_details,
    }


def attach_baseline_comparison(result: dict, baseline: dict) -> None:
    """Add signed differences and positive percentage-improvement measures in place."""
    baseline_overall = baseline["overall"]
    current = result["overall"]
    result["comparison_vs_baseline"] = {
        "RMSE_difference": current["RMSE"] - baseline_overall["RMSE"],
        "RMSE_percentage_improvement": (
            (baseline_overall["RMSE"] - current["RMSE"])
            / baseline_overall["RMSE"]
            * 100.0
        ),
        "MAE_difference": current["MAE"] - baseline_overall["MAE"],
        "MAE_percentage_improvement": (
            (baseline_overall["MAE"] - current["MAE"])
            / baseline_overall["MAE"]
            * 100.0
        ),
        "R2_difference": current["R2"] - baseline_overall["R2"],
    }


def select_balanced_variant(variants: dict, baseline: dict) -> tuple[str | None, str]:
    """Apply the documented overall, R2, and expensive-tail safeguards."""
    eligible = []
    for name, result in variants.items():
        comparison = result["comparison_vs_baseline"]
        top_rmse_change = (
            (result["top_5_percent"]["RMSE"] - baseline["top_5_percent"]["RMSE"])
            / baseline["top_5_percent"]["RMSE"]
            * 100.0
        )
        top_mae_change = (
            (result["top_5_percent"]["MAE"] - baseline["top_5_percent"]["MAE"])
            / baseline["top_5_percent"]["MAE"]
            * 100.0
        )
        result["comparison_vs_baseline"].update(
            {
                "top_5_percent_RMSE_change_percent": top_rmse_change,
                "top_5_percent_MAE_change_percent": top_mae_change,
            }
        )

        # "Meaningful" MAE deterioration is defined as more than 0.5 percent.
        # "Collapse" in the expensive tail is defined as more than 5 percent.
        meets_rule = (
            comparison["RMSE_difference"] < 0.0
            and comparison["MAE_percentage_improvement"] >= -0.5
            and comparison["R2_difference"] >= 0.0
            and top_rmse_change <= 5.0
            and top_mae_change <= 5.0
        )
        result["balanced_rule_passed"] = meets_rule
        if meets_rule:
            tail_penalty = max(0.0, top_rmse_change) + max(0.0, top_mae_change)
            balance_score = (
                comparison["RMSE_percentage_improvement"]
                + comparison["MAE_percentage_improvement"]
                - tail_penalty
            )
            result["balance_score"] = balance_score
            eligible.append((balance_score, name))
        else:
            result["balance_score"] = None

    if not eligible:
        return None, (
            "Retain the baseline. No Winsorization variant reduced overall RMSE, "
            "kept MAE within 0.5%, maintained or improved R2, and limited both "
            "top-5% error increases to 5%."
        )
    _, best_name = max(eligible)
    return best_name, (
        f"{best_name} is the best balanced target-Winsorization configuration "
        "under the stated overall, R2, and expensive-property safeguards. Review "
        "the recorded tail metrics before any separate production decision."
    )


def print_comparison(results: dict) -> None:
    """Print a compact overall comparison followed by the requested winners."""
    baseline = results["baseline"]
    print()
    print(f"{'Variant':<18} {'RMSE':>12} {'MAE':>12} {'R2':>10} {'Delta RMSE':>13} {'Delta MAE':>13}")
    print("-" * 83)
    for name in VARIANT_SPECS:
        result = baseline if name == "baseline" else results["variants"][name]
        comparison = result.get("comparison_vs_baseline", {})
        print(
            f"{DISPLAY_NAMES[name]:<18} "
            f"{result['overall']['RMSE']:>12.2f} "
            f"{result['overall']['MAE']:>12.2f} "
            f"{result['overall']['R2']:>10.6f} "
            f"{comparison.get('RMSE_difference', 0.0):>13.2f} "
            f"{comparison.get('MAE_difference', 0.0):>13.2f}"
        )
    print()
    print(f"Best RMSE configuration: {results['best_rmse_configuration']}")
    print(f"Best MAE configuration: {results['best_mae_configuration']}")
    print(
        "Best balanced configuration: "
        + (results["best_variant"] or "None (baseline retained)")
    )
    print(f"Recommendation: {results['recommendation']}")


def main() -> None:
    """Run the fixed-fold comparison and write only this experiment's JSON."""
    data, features, categorical_features = prepare_city_experiment_data()
    folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(data))

    evaluated = {}
    for name, specification in VARIANT_SPECS.items():
        evaluated[name] = evaluate_variant(
            data,
            features,
            categorical_features,
            folds,
            specification["lower_quantile"],
            specification["upper_quantile"],
        )
        metrics = evaluated[name]["overall"]
        print(
            f"Completed {name}: RMSE=RM {metrics['RMSE']:,.2f}; "
            f"MAE=RM {metrics['MAE']:,.2f}; R2={metrics['R2']:.6f}",
            flush=True,
        )

    baseline = evaluated.pop("baseline")
    variants = evaluated
    for result in variants.values():
        attach_baseline_comparison(result, baseline)
    best_variant, recommendation = select_balanced_variant(variants, baseline)

    all_configurations = {"baseline": baseline, **variants}
    best_rmse = min(
        all_configurations,
        key=lambda name: all_configurations[name]["overall"]["RMSE"],
    )
    best_mae = min(
        all_configurations,
        key=lambda name: all_configurations[name]["overall"]["MAE"],
    )
    results = {
        "rows": len(data),
        "feature_count": len(features),
        "model": "Random Forest",
        "target_strategy": "price_per_square_foot",
        "location_feature": "city",
        "outer_validation": {
            "folds": 5,
            "shuffle": True,
            "random_state": 42,
            "validation_targets_winsorized": False,
            "rows_removed": False,
        },
        "recorded_best_reference": RECORDED_BEST,
        "selection_rule": {
            "RMSE": "must decrease",
            "MAE": "may not worsen by more than 0.5 percent",
            "R2": "must remain equal or improve",
            "top_5_percent_RMSE": "may not worsen by more than 5 percent",
            "top_5_percent_MAE": "may not worsen by more than 5 percent",
        },
        "baseline": baseline,
        "variants": variants,
        "best_rmse_configuration": best_rmse,
        "best_mae_configuration": best_mae,
        "best_variant": best_variant,
        "recommendation": recommendation,
    }
    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print_comparison(results)
    print(f"Saved experiment results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()

