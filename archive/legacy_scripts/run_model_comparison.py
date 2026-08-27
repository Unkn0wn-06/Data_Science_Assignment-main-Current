"""Fit and evaluate the standard four assignment models without retuning."""

if __package__:
    from scripts import _bootstrap  # noqa: F401
else:
    import _bootstrap  # noqa: F401

import pandas as pd
from sklearn.model_selection import train_test_split

from src.cleaning.pipeline import PROCESSED_DATA_PATH, PROJECT_ROOT
from src.models.common.evaluation import evaluate
from src.models.common.features import FEATURES
from src.models.common.parameters import load_params
from src.models.common.utilities import sanitize_model_data
from src.models.registry import build_pipelines


PARAMS_PATH = PROJECT_ROOT / "configs" / "best_params.json"


def main() -> None:
    """Reproduce the current 80/20, seed-42 comparison of four model families."""
    data = sanitize_model_data(pd.read_csv(PROCESSED_DATA_PATH))
    X = data[FEATURES]
    y = data["price"]
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    pipelines = build_pipelines(
        load_params(PARAMS_PATH), include_price_per_square_foot=False
    )
    for name, pipeline in pipelines.items():
        pipeline.fit(x_train, y_train)
        metrics, _ = evaluate(pipeline, x_test, y_test)
        print(
            f"{name}: RMSE=RM {metrics['RMSE']:,.2f}; "
            f"MAE=RM {metrics['MAE']:,.2f}; R2={metrics['R2']:.4f}"
        )


if __name__ == "__main__":
    main()
