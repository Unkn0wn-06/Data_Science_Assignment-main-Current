"""Fixed-complexity LightGBM builders for feature and loss ablations."""

from __future__ import annotations

from typing import Any

from experiments.advanced_real_estate_models.model_builders import (
    build_base_regressor,
    candidate_parameters,
)


NATIVE_OBJECTIVES = {
    "regression": "MSE / L2",
    "regression_l1": "MAE / L1",
    "huber": "Huber",
    "fair": "Fair",
}


def lightgbm_parameters(objective: str = "regression") -> dict[str, Any]:
    """Reuse the exact winning complexity and change only the objective."""
    if objective not in NATIVE_OBJECTIVES:
        raise ValueError(f"Unsupported controlled objective: {objective}")
    params = candidate_parameters()["lightgbm"][1].copy()
    params["objective"] = objective
    return params


def build_lightgbm_regressor(
    numerical_features: list[str],
    categorical_features: list[str],
    objective: str = "regression",
):
    return build_base_regressor(
        "lightgbm",
        lightgbm_parameters(objective),
        numerical_features,
        categorical_features,
    )
