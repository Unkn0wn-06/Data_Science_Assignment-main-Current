"""Assemble four assignment pipelines without duplicating preprocessing."""

from sklearn.pipeline import Pipeline

from src.models.common.parameters import split_parameter_sets
from src.models.common.preprocessing import make_preprocessor
from src.models.common.utilities import PricePerSquareFootRegressor
from src.models.gradient_boosting.model import build_model as build_gradient_boosting
from src.models.knn.model import build_model as build_knn
from src.models.random_forest.model import build_model as build_random_forest
from src.models.ridge.model import build_model as build_ridge


def build_pipelines(params: dict, include_price_per_square_foot: bool = True) -> dict:
    """Build standard pipelines and the current optional production normalized model."""
    estimator_params, pipeline_params = split_parameter_sets(params)
    estimators = {
        "Ridge Regression": build_ridge(estimator_params["Ridge Regression"]),
        "Random Forest": build_random_forest(estimator_params["Random Forest"]),
        "Gradient Boosting": build_gradient_boosting(
            estimator_params["Gradient Boosting"]
        ),
        "KNN": build_knn(estimator_params["KNN"]),
    }
    pipelines = {
        name: Pipeline(
            [
                (
                    "preprocessor",
                    make_preprocessor(name in {"Ridge Regression", "KNN"}),
                ),
                ("model", estimator),
            ]
        ).set_params(**pipeline_params[name])
        for name, estimator in estimators.items()
    }
    if include_price_per_square_foot:
        normalized_pipeline = Pipeline(
            [
                ("preprocessor", make_preprocessor(False)),
                (
                    "model",
                    build_gradient_boosting(estimator_params["Gradient Boosting"]),
                ),
            ]
        ).set_params(**pipeline_params["Gradient Boosting"])
        pipelines["Gradient Boosting (Price/sq.ft.)"] = (
            PricePerSquareFootRegressor(regressor=normalized_pipeline)
        )
    return pipelines

