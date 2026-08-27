"""Test richer leakage-safe property features against the recleaned model baseline."""

# Import JSON support so experimental metrics remain inspectable after execution.
import json
# Import Path so output paths are relative to this script.
from pathlib import Path

# Import NumPy for arrays, clipping, and error calculations.
import numpy as np
# Import clone so every validation fold receives a fresh estimator pipeline.
from sklearn.base import clone
# Import ColumnTransformer to preprocess numeric and categorical fields separately.
from sklearn.compose import ColumnTransformer
# Import four strong nonlinear regressors available in the installed scikit-learn.
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
# Import median imputation for partially populated numeric and categorical source fields.
# Import shuffled folds for consistent out-of-fold predictions.
from sklearn.model_selection import KFold
# Import Pipeline to keep fold-fitted preprocessing attached to each estimator.
from sklearn.pipeline import Pipeline

# Import the original model feature groups.
from src.models.common.evaluation import regression_metrics
from src.models.common.features import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from src.models.common.preprocessing import make_target_encoding_preprocessor
from src.cleaning.enhanced_city import prepare_enhanced_dataset


# Resolve the directory containing this experiment.
BASE_DIR = Path(__file__).resolve().parent
# Store experiment results separately from deployable model settings.
RESULTS_PATH = BASE_DIR / "results.json"

# Reuse the canonical enhanced numerical order for every enhanced experiment.
ENHANCED_NUMERICAL_FEATURES = NUMERICAL_FEATURES

# The historical enhanced run predates city in the production schema. Keep city
# out here so the dedicated city experiment remains the only experiment adding it.
ENHANCED_CATEGORICAL_FEATURES = [
    *[feature for feature in CATEGORICAL_FEATURES if feature != "city"],
    "detailed_address",  # Add the normalized full listing address.
]

# Combine enhanced feature groups into one stable model input order.
ENHANCED_FEATURES = ENHANCED_NUMERICAL_FEATURES + ENHANCED_CATEGORICAL_FEATURES


# Build a leakage-safe feature transformer for enhanced mixed-type inputs.
def make_enhanced_preprocessor() -> ColumnTransformer:
    # Return separate numeric-imputation and categorical-target-encoding branches.
    return make_target_encoding_preprocessor(
        ENHANCED_NUMERICAL_FEATURES,
        ENHANCED_CATEGORICAL_FEATURES,
    )


# Construct nonlinear estimators that can exploit the compact target-encoded features.
def build_experimental_models() -> dict:
    # Return one full preprocessing-and-regression pipeline per model family.
    return {
        "Extra Trees": Pipeline([  # Build a highly randomized tree ensemble.
            ("preprocessor", make_enhanced_preprocessor()),  # Attach fold-fitted feature preparation.
            ("model", ExtraTreesRegressor(  # Fit many decorrelated randomized trees.
                n_estimators=600,  # Average six hundred component trees.
                max_features=1.0,  # Consider all compact encoded features at each split.
                min_samples_leaf=1,  # Permit detailed local fits for repeated projects.
                random_state=42,  # Make ensemble randomness reproducible.
                n_jobs=-1,  # Use all available CPU cores within this estimator.
            )),
        ]),
        "Random Forest": Pipeline([  # Build a conventional bagged tree ensemble.
            ("preprocessor", make_enhanced_preprocessor()),  # Attach fold-fitted feature preparation.
            ("model", RandomForestRegressor(  # Fit bootstrap-sampled regression trees.
                n_estimators=500,  # Average five hundred component trees.
                max_features=0.8,  # Randomly expose most encoded features per split.
                min_samples_leaf=1,  # Preserve local project-level detail.
                random_state=42,  # Make ensemble randomness reproducible.
                n_jobs=-1,  # Use all available CPU cores within this estimator.
            )),
        ]),
        "Gradient Boosting": Pipeline([  # Build a sequential residual-correction model.
            ("preprocessor", make_enhanced_preprocessor()),  # Attach fold-fitted feature preparation.
            ("model", GradientBoostingRegressor(  # Fit shallow trees sequentially.
                loss="huber",  # Limit domination by the most expensive luxury listings.
                n_estimators=400,  # Use four hundred boosting stages.
                learning_rate=0.05,  # Shrink each stage for steadier generalization.
                max_depth=3,  # Keep component trees shallow.
                random_state=42,  # Make fitting reproducible.
            )),
        ]),
        "Histogram Gradient Boosting": Pipeline([  # Build an efficient histogram-based booster.
            ("preprocessor", make_enhanced_preprocessor()),  # Attach fold-fitted feature preparation.
            ("model", HistGradientBoostingRegressor(  # Fit regularized histogram trees.
                loss="squared_error",  # Optimize the objective most aligned with RMSE.
                max_iter=500,  # Allow up to five hundred boosting stages.
                learning_rate=0.05,  # Shrink each stage for stable learning.
                max_leaf_nodes=31,  # Bound tree complexity.
                l2_regularization=1.0,  # Penalize overly extreme leaf values.
                random_state=42,  # Make fitting reproducible.
            )),
        ]),
    }


# Produce honest out-of-fold metrics for direct-price and normalized-target training.
def evaluate_experiments(data: pd.DataFrame) -> dict:
    # Create the same five shuffled outer folds for every model and target strategy.
    folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(data))
    # Create the result dictionary returned and written by this experiment.
    results = {}
    # Evaluate ordinary price learning and price-per-square-foot learning separately.
    for target_strategy in ["direct_price", "price_per_square_foot"]:
        # Allocate a nested dictionary for this target strategy.
        results[target_strategy] = {}
        # Evaluate every experimental pipeline under identical outer folds.
        for model_name, pipeline in build_experimental_models().items():
            # Allocate arrays for one prediction and actual value per listing.
            actual = np.empty(len(data), dtype=float)
            # Allocate the matching out-of-fold prediction array.
            predicted = np.empty(len(data), dtype=float)
            # Train on four folds and predict the held-out fold five times.
            for train_index, test_index in folds:
                # Select enhanced training features.
                x_train = data.iloc[train_index][ENHANCED_FEATURES]
                # Select enhanced held-out features.
                x_test = data.iloc[test_index][ENHANCED_FEATURES]
                # Select ordinary held-out prices for final metric calculation.
                actual[test_index] = data.iloc[test_index]["price"].to_numpy(dtype=float)
                # Create a fresh pipeline with no fitted state from previous folds.
                fitted = clone(pipeline)
                # Use ordinary price as the training target for direct modeling.
                if target_strategy == "direct_price":
                    # Fit preprocessing and regression directly against Ringgit prices.
                    fitted.fit(x_train, data.iloc[train_index]["price"])
                    # Predict held-out Ringgit prices.
                    fold_prediction = fitted.predict(x_test)
                # Use price per square foot as a normalized training target otherwise.
                else:
                    # Divide training price by known training square footage.
                    normalized_target = (
                        data.iloc[train_index]["price"].to_numpy(dtype=float)
                        / data.iloc[train_index]["property_size_sqft"].to_numpy(dtype=float)
                    )
                    # Fit preprocessing and regression against normalized prices.
                    fitted.fit(x_train, normalized_target)
                    # Convert predicted price per square foot back to total price.
                    fold_prediction = (
                        fitted.predict(x_test)
                        * data.iloc[test_index]["property_size_sqft"].to_numpy(dtype=float)
                    )
                # Prevent invalid negative price estimates before metric calculation.
                predicted[test_index] = np.clip(fold_prediction, 0.0, None)
            # Store unchanged metrics through the project's single metric implementation.
            results[target_strategy][model_name] = regression_metrics(
                actual,
                predicted,
                include_distribution=True,
            )
            # Print progress after completing all folds for one model and strategy.
            print(
                f"{target_strategy} | {model_name}: "
                f"RMSE=RM {results[target_strategy][model_name]['RMSE']:,.2f}; "
                f"MAE=RM {results[target_strategy][model_name]['MAE']:,.2f}"
            )
    # Return all experimental metrics.
    return results


# Execute enhanced feature construction and evaluation only when run directly.
if __name__ == "__main__":
    # Construct the richer leakage-safe dataset from raw listings.
    enhanced_data = prepare_enhanced_dataset()
    # Evaluate every model and target strategy using out-of-fold predictions.
    experiment_results = evaluate_experiments(enhanced_data)
    # Write row count, feature count, and all metrics to an inspectable JSON artifact.
    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        # Serialize plain Python-compatible results with readable indentation.
        json.dump(
            {  # Build the complete experiment document.
                "rows": len(enhanced_data),  # Record evaluated listing count.
                "features": len(ENHANCED_FEATURES),  # Record enhanced model feature count.
                "results": experiment_results,  # Store all out-of-fold metrics.
            },
            file,  # Write to the opened results file.
            indent=2,  # Make the artifact easy to inspect manually.
        )
    # Report the saved artifact path after successful completion.
    print(f"Saved enhanced experiment results to {RESULTS_PATH}")
