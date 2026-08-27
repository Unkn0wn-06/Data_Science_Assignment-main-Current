"""Optional broad tuning runner retained separately from ordinary model execution."""

import json
from pathlib import Path
import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split

from src.models.common.features import FEATURES
from src.models.common.parameters import DEFAULT_PARAMS
from src.models.common.utilities import sanitize_model_data, select_balanced_candidate
from src.models.gradient_boosting.tuning import SEARCH_ITERATIONS as GB_ITERATIONS
from src.models.gradient_boosting.tuning import SEARCH_SPACE as GB_SPACE
from src.models.knn.tuning import SEARCH_ITERATIONS as KNN_ITERATIONS
from src.models.knn.tuning import SEARCH_SPACE as KNN_SPACE
from src.models.random_forest.tuning import SEARCH_ITERATIONS as RF_ITERATIONS
from src.models.random_forest.tuning import SEARCH_SPACE as RF_SPACE
from src.models.registry import build_pipelines
from src.models.ridge.tuning import SEARCH_ITERATIONS as RIDGE_ITERATIONS
from src.models.ridge.tuning import SEARCH_SPACE as RIDGE_SPACE


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "production_prepared_dataset.csv"
OUTPUT_PATH = PROJECT_ROOT / "configs" / "tuning_candidates.json"
SEARCH_SPACES = {
    "Ridge Regression": RIDGE_SPACE,
    "Random Forest": RF_SPACE,
    "Gradient Boosting": GB_SPACE,
    "KNN": KNN_SPACE,
}
SEARCH_ITERATIONS = {
    "Ridge Regression": RIDGE_ITERATIONS,
    "Random Forest": RF_ITERATIONS,
    "Gradient Boosting": GB_ITERATIONS,
    "KNN": KNN_ITERATIONS,
}
SCORING = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
}


def main() -> None:
    """Run the preserved expensive randomized searches only when explicitly invoked."""
    warnings.filterwarnings(
        "ignore", message="Found unknown categories.*", category=UserWarning
    )
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    data = sanitize_model_data(pd.read_csv(DATA_PATH))
    X = data[FEATURES]
    y = data["price"]
    x_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    cross_validation = KFold(n_splits=5, shuffle=True, random_state=42)
    best = {}
    scores = {}
    for name, pipeline in build_pipelines(
        DEFAULT_PARAMS, include_price_per_square_foot=False
    ).items():
        search = RandomizedSearchCV(
            pipeline,
            SEARCH_SPACES[name],
            n_iter=SEARCH_ITERATIONS[name],
            cv=cross_validation,
            scoring=SCORING,
            refit=select_balanced_candidate,
            random_state=42,
            n_jobs=-1,
            verbose=1,
            return_train_score=True,
            error_score="raise",
        )
        search.fit(x_train, y_train)
        best[name] = search.best_params_.copy()
        if name == "Random Forest":
            best[name]["model__n_jobs"] = -1
        scores[name] = {
            "RMSE": -search.cv_results_["mean_test_rmse"][search.best_index_],
            "MAE": -search.cv_results_["mean_test_mae"][search.best_index_],
        }
        print(
            f"{name}: CV RMSE=RM {scores[name]['RMSE']:,.2f}; "
            f"CV MAE=RM {scores[name]['MAE']:,.2f}; {best[name]}"
        )
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "status": "tuned",
                "search": "RandomizedSearchCV",
                "cv": 5,
                "selection": "minimum combined RMSE and MAE rank",
                "iterations": SEARCH_ITERATIONS,
                "parameters": best,
                "cv_scores": scores,
            },
            file,
            indent=2,
        )


if __name__ == "__main__":
    main()

