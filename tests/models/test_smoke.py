"""Fit all four assignment pipelines on a small deterministic sample."""

import unittest

import pandas as pd

from src.cleaning.pipeline import PROCESSED_DATA_PATH, PROJECT_ROOT
from src.models.common.features import FEATURES
from src.models.common.parameters import load_params
from src.models.common.utilities import sanitize_model_data
from src.models.registry import build_pipelines


class ModelSmokeTests(unittest.TestCase):
    """Ensure each model can fit and return finite predictions."""

    def test_four_assignment_models(self) -> None:
        data = sanitize_model_data(pd.read_csv(PROCESSED_DATA_PATH)).head(240)
        x_train = data.iloc[:200][FEATURES]
        y_train = data.iloc[:200]["price"]
        x_test = data.iloc[200:][FEATURES]
        parameters = load_params(PROJECT_ROOT / "configs" / "best_params.json")
        pipelines = build_pipelines(parameters, include_price_per_square_foot=False)
        self.assertEqual(
            set(pipelines),
            {"Ridge Regression", "Random Forest", "Gradient Boosting", "KNN"},
        )
        for name, pipeline in pipelines.items():
            with self.subTest(model=name):
                prediction = pipeline.fit(x_train, y_train).predict(x_test)
                self.assertEqual(len(prediction), len(x_test))
                self.assertTrue(pd.Series(prediction).notna().all())


if __name__ == "__main__":
    unittest.main()

