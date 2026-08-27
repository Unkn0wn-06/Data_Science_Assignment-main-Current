"""Tune enhanced price-per-square-foot regressors against total-price errors."""

# Import JSON support for saving inspectable experimental results.
import json
# Import Path so the output file is stored beside this script.
from pathlib import Path

# Import clone so the selected configuration can be evaluated independently.
from sklearn.base import clone
# Import the three strongest candidate tree model families from the first experiment.
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
# Import cross-validation and randomized-search utilities.
from sklearn.model_selection import KFold, RandomizedSearchCV
# Import Pipeline so enhanced preprocessing remains attached to each estimator.
from sklearn.pipeline import Pipeline

# Reuse enhanced data, feature names, preprocessing, and balanced-rank selection.
from experiments.enhanced_models.run_experiment import (
    ENHANCED_FEATURES,
    make_enhanced_preprocessor,
    prepare_enhanced_dataset,
)
# Reuse the tested candidate selector that balances RMSE and MAE validation ranks.
from src.models.common.evaluation import out_of_fold_metrics
from src.models.common.utilities import PricePerSquareFootRegressor, select_balanced_candidate


# Resolve the directory containing this tuning script.
BASE_DIR = Path(__file__).resolve().parent
# Save enhanced tuning output separately from production parameters.
RESULTS_PATH = BASE_DIR / "tuning_results.json"


# Build the base enhanced pipeline and wrapper for one supplied tree estimator.
def wrap_model(model) -> PricePerSquareFootRegressor:
    # Attach leakage-safe enhanced preprocessing before the supplied estimator.
    pipeline = Pipeline([
        ("preprocessor", make_enhanced_preprocessor()),  # Encode enhanced raw features inside each fold.
        ("model", model),  # Fit the supplied tree estimator to price per square foot.
    ])
    # Return the total-price-compatible target-normalization wrapper.
    return PricePerSquareFootRegressor(regressor=pipeline)


# Construct base estimators, search spaces, and iteration budgets by model name.
def build_searches() -> dict:
    # Return one search specification for each competitive model family.
    return {
        "Extra Trees": {  # Configure highly randomized tree tuning.
            "estimator": wrap_model(ExtraTreesRegressor(random_state=42, n_jobs=1)),  # Avoid nested parallelism.
            "iterations": 30,  # Sample thirty broad configurations.
            "parameters": {  # Define preprocessing and estimator candidates.
                "regressor__preprocessor__categorical__target_encoder__smooth": ["auto", 5.0, 10.0, 20.0, 50.0, 100.0],  # Tune category shrinkage.
                "regressor__model__n_estimators": [300, 500, 700, 900],  # Tune ensemble size.
                "regressor__model__criterion": ["squared_error", "poisson"],  # Tune valid split loss.
                "regressor__model__max_depth": [None, 10, 16, 24, 32],  # Tune tree depth.
                "regressor__model__min_samples_split": [2, 4, 6, 10],  # Tune internal-node regularization.
                "regressor__model__min_samples_leaf": [1, 2, 3, 5],  # Tune leaf regularization.
                "regressor__model__max_features": [0.5, 0.7, 0.85, 1.0, "sqrt"],  # Tune split feature sampling.
                "regressor__model__bootstrap": [False, True],  # Compare full-sample and bootstrapped trees.
            },
        },
        "Random Forest": {  # Configure conventional bagged-tree tuning.
            "estimator": wrap_model(RandomForestRegressor(random_state=42, n_jobs=1)),  # Avoid nested parallelism.
            "iterations": 30,  # Sample thirty broad configurations.
            "parameters": {  # Define preprocessing and estimator candidates.
                "regressor__preprocessor__categorical__target_encoder__smooth": ["auto", 5.0, 10.0, 20.0, 50.0, 100.0],  # Tune category shrinkage.
                "regressor__model__n_estimators": [300, 500, 700, 900],  # Tune ensemble size.
                "regressor__model__criterion": ["squared_error", "poisson"],  # Tune valid split loss.
                "regressor__model__max_depth": [None, 10, 16, 24, 32],  # Tune tree depth.
                "regressor__model__min_samples_split": [2, 4, 6, 10],  # Tune internal-node regularization.
                "regressor__model__min_samples_leaf": [1, 2, 3, 5],  # Tune leaf regularization.
                "regressor__model__max_features": [0.5, 0.7, 0.85, 1.0, "sqrt"],  # Tune split feature sampling.
                "regressor__model__bootstrap": [True, False],  # Compare bootstrapped and full-sample trees.
            },
        },
        "Histogram Gradient Boosting": {  # Configure regularized histogram boosting.
            "estimator": wrap_model(HistGradientBoostingRegressor(random_state=42)),  # Build a deterministic base booster.
            "iterations": 40,  # Sample forty broad boosting configurations.
            "parameters": {  # Define preprocessing and estimator candidates.
                "regressor__preprocessor__categorical__target_encoder__smooth": ["auto", 5.0, 10.0, 20.0, 50.0, 100.0],  # Tune category shrinkage.
                "regressor__model__loss": ["squared_error", "absolute_error", "poisson"],  # Tune normalized-target loss.
                "regressor__model__max_iter": [200, 350, 500, 700],  # Tune boosting-stage count.
                "regressor__model__learning_rate": [0.02, 0.03, 0.05, 0.075, 0.1],  # Tune stage shrinkage.
                "regressor__model__max_leaf_nodes": [15, 31, 63],  # Tune component-tree complexity.
                "regressor__model__max_depth": [None, 4, 6, 10],  # Optionally limit tree depth.
                "regressor__model__min_samples_leaf": [10, 20, 30, 40],  # Tune leaf regularization.
                "regressor__model__l2_regularization": [0.0, 0.1, 1.0, 5.0, 10.0],  # Tune weight shrinkage.
            },
        },
    }


# Run randomized tuning and save the strongest honest results.
def main() -> None:
    # Build the enhanced recleaned dataset.
    data = prepare_enhanced_dataset()
    # Select only enhanced model input fields.
    X = data[ENHANCED_FEATURES]
    # Select total Ringgit price as the scoring target.
    y = data["price"]
    # Create fixed shuffled outer folds shared across searches and final metrics.
    folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(X))
    # Score every candidate on both total-price objectives requested by the user.
    scoring = {
        "rmse": "neg_root_mean_squared_error",  # Penalize large Ringgit misses.
        "mae": "neg_mean_absolute_error",  # Measure typical absolute Ringgit misses.
    }
    # Allocate a complete result document.
    results = {}
    # Tune each candidate model family independently.
    for name, specification in build_searches().items():
        # Configure randomized search around the normalized-target wrapper.
        search = RandomizedSearchCV(
            specification["estimator"],  # Tune this wrapped enhanced estimator.
            specification["parameters"],  # Sample from its broad parameter space.
            n_iter=specification["iterations"],  # Evaluate its allocated candidates.
            scoring=scoring,  # Calculate total-price RMSE and MAE.
            refit=select_balanced_candidate,  # Balance both metric ranks for selection.
            cv=folds,  # Use identical outer validation rows for all candidates.
            random_state=42,  # Make candidate sampling reproducible.
            n_jobs=-1,  # Parallelize independent candidate-fold fits.
            verbose=1,  # Report search progress.
            error_score="raise",  # Surface incompatible candidates immediately.
        )
        # Fit all sampled candidates and refit the selected configuration.
        search.fit(X, y)
        # Copy selected parameters for safe JSON serialization and production adjustment.
        best_parameters = search.best_params_.copy()
        # Restore full estimator parallelism for standalone final evaluation.
        if name in {"Extra Trees", "Random Forest"}:
            # Use every available core when one selected forest is fitted at a time.
            best_parameters["regressor__model__n_jobs"] = -1
        # Apply the selected production-compatible parameter values.
        selected_estimator = clone(specification["estimator"]).set_params(**best_parameters)
        # Calculate concatenated out-of-fold metrics for the selected configuration.
        metrics = out_of_fold_metrics(selected_estimator, X, y, folds)
        # Store selected settings and honest total-price metrics.
        results[name] = {
            "parameters": best_parameters,  # Preserve the winning nested parameters.
            "metrics": metrics,  # Preserve out-of-fold total-price performance.
        }
        # Print a compact completion summary for this model family.
        print(f"{name}: RMSE=RM {metrics['RMSE']:,.2f}; MAE=RM {metrics['MAE']:,.2f}")
    # Open the experimental output artifact for UTF-8 writing.
    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        # Write dataset dimensions and every tuning result as readable JSON.
        json.dump(
            {  # Build the complete tuning artifact.
                "rows": len(data),  # Record evaluated listing count.
                "features": len(ENHANCED_FEATURES),  # Record enhanced feature count.
                "target_strategy": "price_per_square_foot",  # Document target normalization.
                "selection": "minimum combined total-price RMSE and MAE rank",  # Document selection.
                "results": results,  # Store parameters and metrics by model family.
            },
            file,  # Write to the opened artifact.
            indent=2,  # Make the result easy to inspect.
        )
    # Report the completed output path.
    print(f"Saved enhanced tuning results to {RESULTS_PATH}")


# Execute tuning only when this file is launched directly.
if __name__ == "__main__":
    # Start the complete enhanced price-per-square-foot search.
    main()
