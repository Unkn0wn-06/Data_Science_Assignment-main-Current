"""Evaluate the recorded best Random Forest + price/sq.ft. + city workflow."""

if __package__:
    from scripts import _bootstrap  # noqa: F401
else:
    import _bootstrap  # noqa: F401

from sklearn.model_selection import KFold

from experiments.city_feature.run_experiment import make_model
from experiments.enhanced_models.run_experiment import (
    ENHANCED_CATEGORICAL_FEATURES,
    ENHANCED_NUMERICAL_FEATURES,
    prepare_enhanced_dataset,
)
from src.cleaning.location_cleaning import extract_city
from src.models.common.evaluation import out_of_fold_metrics


def main() -> None:
    """Run only the selected city variant on its original five shuffled folds."""
    data = prepare_enhanced_dataset()
    data["city"] = data["detailed_address"].map(extract_city)
    non_location_categories = [
        feature
        for feature in ENHANCED_CATEGORICAL_FEATURES
        if feature not in {"detailed_address", "city"}
    ]
    categorical_features = [*non_location_categories, "city"]
    features = ENHANCED_NUMERICAL_FEATURES + categorical_features
    folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(data))
    metrics = out_of_fold_metrics(
        make_model(categorical_features), data[features], data["price"], folds
    )
    print("Model: Random Forest")
    print("Target strategy: price_per_square_foot")
    print("Location feature: city")
    print(f"RMSE: RM {metrics['RMSE']:,.2f}")
    print(f"MAE: RM {metrics['MAE']:,.2f}")
    print(f"R2: {metrics['R2']:.6f}")


if __name__ == "__main__":
    main()
