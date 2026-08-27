"""Fallback model settings and loading of saved parameter artifacts."""

import json
from pathlib import Path


# Preserve the established fallback estimator settings from model_utils.py.
DEFAULT_PARAMS = {
    "Ridge Regression": {"alpha": 10.0},
    "Random Forest": {
        "n_estimators": 150,
        "max_depth": 15,
        "min_samples_split": 2,
    },
    "Gradient Boosting": {
        "n_estimators": 150,
        "learning_rate": 0.05,
        "max_depth": 3,
    },
    "KNN": {
        "n_neighbors": 7,
        "weights": "distance",
    },
}


def load_params(path: Path) -> dict:
    """Load the JSON parameter section or use the documented fallbacks."""
    if not path.exists():
        return DEFAULT_PARAMS
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    return payload.get("parameters", DEFAULT_PARAMS)


def split_parameter_sets(params: dict) -> tuple[dict, dict]:
    """Separate estimator constructor settings from nested pipeline settings."""
    estimator_params = {
        name: {key: value for key, value in values.items() if "__" not in key}
        for name, values in params.items()
    }
    pipeline_params = {
        name: {key: value for key, value in values.items() if "__" in key}
        for name, values in params.items()
    }
    return estimator_params, pipeline_params

