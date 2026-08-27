"""Compare detailed-address and extracted-city features on identical validation folds."""

# Import JSON support so comparison results remain inspectable.
import json
# Import Path so the result artifact is stored beside this script.
from pathlib import Path

# Import KFold for the same reproducible five outer validation folds.
from sklearn.model_selection import KFold
# Import ColumnTransformer to route numeric and categorical columns separately.
from sklearn.compose import ColumnTransformer

# Reuse enhanced feature construction and its established feature groups.
from experiments.enhanced_models.run_experiment import (
    ENHANCED_CATEGORICAL_FEATURES,
    ENHANCED_NUMERICAL_FEATURES,
    prepare_enhanced_dataset,
)
# Reuse the new address-to-city abstraction.
from src.cleaning.location_cleaning import extract_city
# Reuse the normalized-target wrapper and honest out-of-fold metric calculation.
from src.models.common.evaluation import out_of_fold_metrics
from src.models.common.preprocessing import make_target_encoding_preprocessor
from src.models.common.utilities import PricePerSquareFootRegressor
from src.models.random_forest.model import build_best_city_model
from sklearn.pipeline import Pipeline


# Resolve the directory containing this experiment.
BASE_DIR = Path(__file__).resolve().parent
# Store the city comparison separately from production model settings.
RESULTS_PATH = BASE_DIR / "results.json"


# Build enhanced preprocessing for a supplied categorical feature variant.
def make_preprocessor(categorical_features: list[str]) -> ColumnTransformer:
    # Return independent numeric and leakage-safe categorical branches.
    return make_target_encoding_preprocessor(
        ENHANCED_NUMERICAL_FEATURES,
        categorical_features,
    )


# Build the previously selected Random Forest around one location-feature variant.
def make_model(categorical_features: list[str]) -> PricePerSquareFootRegressor:
    # Attach enhanced preprocessing to the strongest tuned forest configuration.
    regressor = Pipeline([
        ("preprocessor", make_preprocessor(categorical_features)),  # Prepare mixed raw features.
        (  # Fit the tuned forest to price per square foot.
            "model",  # Give the estimator an inspectable pipeline name.
            build_best_city_model(),  # Recreate the selected tuning result.
        ),
    ])
    # Normalize the training target and restore total price during prediction.
    return PricePerSquareFootRegressor(regressor=regressor)


# Compare location abstraction variants under identical data and fold assignments.
def main() -> None:
    # Build the same enhanced recleaned rows used by the previous experiment.
    data = prepare_enhanced_dataset()
    # Extract one city/locality category from every normalized detailed address.
    data["city"] = data["detailed_address"].map(extract_city)
    # Keep categorical features unrelated to either tested location representation.
    non_location_categories = [
        feature  # Preserve the current non-location categorical name.
        for feature in ENHANCED_CATEGORICAL_FEATURES  # Inspect all enhanced categories.
        if feature not in {"detailed_address", "city"}  # Remove both tested location fields.
    ]
    # Define detailed-only, city-only, and combined categorical feature sets.
    variants = {
        "detailed_address_only": [*non_location_categories, "detailed_address"],  # Use granular location only.
        "city_only": [*non_location_categories, "city"],  # Use the lower-cardinality abstraction only.
        "detailed_address_and_city": [*non_location_categories, "detailed_address", "city"],  # Use both granularities.
    }
    # Create identical reproducible outer folds for every location variant.
    folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(data))
    # Allocate a dictionary for metrics and feature cardinalities.
    results = {}
    # Evaluate every location representation independently.
    for variant_name, categorical_features in variants.items():
        # Combine fixed enhanced numerics with this variant's categories.
        features = ENHANCED_NUMERICAL_FEATURES + categorical_features
        # Calculate honest price-per-square-foot out-of-fold total-price metrics.
        metrics = out_of_fold_metrics(
            make_model(categorical_features),  # Build this variant's wrapped tuned forest.
            data[features],  # Supply only features declared for this variant.
            data["price"],  # Score final total Ringgit price.
            folds,  # Reuse identical outer fold membership.
        )
        # Store metrics plus the number of supplied features.
        results[variant_name] = {
            "feature_count": len(features),  # Record model input width before encoding.
            "metrics": metrics,  # Record honest total-price validation results.
        }
        # Print a compact summary after each completed variant.
        print(f"{variant_name}: RMSE=RM {metrics['RMSE']:,.2f}; MAE=RM {metrics['MAE']:,.2f}")
    # Open the comparison artifact for UTF-8 output.
    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        # Write city cardinality and every variant's metrics as readable JSON.
        json.dump(
            {  # Build the complete comparison document.
                "rows": len(data),  # Record evaluated listing count.
                "city_categories": int(data["city"].nunique()),  # Record city/locality cardinality.
                "detailed_address_categories": int(data["detailed_address"].nunique()),  # Record address cardinality.
                "results": results,  # Store all comparable metric results.
            },
            file,  # Write to the opened artifact.
            indent=2,  # Keep the output easy to inspect.
        )
    # Report the completed artifact path.
    print(f"Saved city comparison results to {RESULTS_PATH}")


# Run the comparison only when this script is launched directly.
if __name__ == "__main__":
    # Execute the complete city-feature experiment.
    main()
