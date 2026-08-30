"""Focused tests for live-prediction resilience and accessible table styling."""

from __future__ import annotations

import inspect
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from prototype.app import (
    BEST_CELL_STYLE,
    final_model_prediction,
    LIVE_NUMERIC_FIELDS,
    PREDICTOR_VIEW,
    RIDGE_UNAVAILABLE_MESSAGE,
    observed_numeric_ranges,
    outside_observed_range_fields,
    predict_scope_models,
    prediction_comparison_frame,
    render_model_evaluation,
    safe_predict_scope_model,
    style_best_metric_cells,
    validate_live_numeric_inputs,
)
from src.cleaning.pipeline import PROJECT_ROOT
from src.models.final.final_evaluation import FINAL_MODELS
from src.models.final.position_regex_lightgbm import FINAL_MODEL_NAME


class _StubModel:
    description_length_median_ = 100.0


def _available(total_price: float, property_size: float = 1_000.0) -> dict:
    return {
        "available": True,
        "prediction": {
            "total_price_RM": total_price,
            "ppsf_RM": total_price / property_size,
        },
        "message": None,
    }


def _unavailable(message: str = "Prediction unavailable") -> dict:
    return {"available": False, "prediction": None, "message": message}


class PredictorRobustnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(
            PROJECT_ROOT / "data" / "processed" / "enhanced_city_dataset.csv"
        )
        cls.ranges = observed_numeric_ranges(cls.data)
        cls.values = {
            field: float(pd.to_numeric(cls.data[field], errors="coerce").median())
            for field, _ in LIVE_NUMERIC_FIELDS
        }

    def test_normal_positive_prediction_is_available_and_ppsf_is_recalculated(self):
        raw = {"total_price_RM": 528_400.0, "ppsf_RM": -999.0}
        with patch("prototype.app.predict_scope_model", return_value=raw):
            outcome = safe_predict_scope_model(
                "Random Forest", _StubModel(), self.values, ""
            )

        self.assertTrue(outcome["available"])
        self.assertEqual(528_400.0, outcome["prediction"]["total_price_RM"])
        self.assertAlmostEqual(
            528_400.0 / self.values["property_size_sqft"],
            outcome["prediction"]["ppsf_RM"],
        )

    def test_nonpositive_and_nonfinite_ridge_outputs_are_unavailable(self):
        for invalid in (0.0, -1.0, np.nan, np.inf, -np.inf):
            with self.subTest(invalid=invalid):
                with patch(
                    "prototype.app.predict_scope_model",
                    return_value={"total_price_RM": invalid, "ppsf_RM": invalid},
                ):
                    outcome = safe_predict_scope_model(
                        "Ridge Regression", _StubModel(), self.values, ""
                    )

                self.assertFalse(outcome["available"])
                self.assertIsNone(outcome["prediction"])
                self.assertEqual(RIDGE_UNAVAILABLE_MESSAGE, outcome["message"])

    def test_one_model_failure_does_not_discard_other_predictions(self):
        def fake_prediction(model_name, *_args):
            if model_name == "Ridge Regression":
                raise RuntimeError("internal model failure")
            return {"total_price_RM": 500_000.0, "ppsf_RM": 500.0}

        models = {name: _StubModel() for name in FINAL_MODELS}
        with patch("prototype.app.predict_scope_model", side_effect=fake_prediction):
            outcomes = predict_scope_models(models, self.values, "")

        self.assertFalse(outcomes["Ridge Regression"]["available"])
        for model_name in FINAL_MODELS:
            if model_name != "Ridge Regression":
                self.assertTrue(outcomes[model_name]["available"])

    def test_selected_and_full_market_availability_are_independent(self):
        selected = {name: _available(500_000.0) for name in FINAL_MODELS}
        full = {name: _available(510_000.0) for name in FINAL_MODELS}
        full["Ridge Regression"] = _unavailable()
        selected["Gradient Boosting"] = _unavailable()

        frame = prediction_comparison_frame(selected, full).set_index("Model")
        ridge = frame.loc["Ridge Regression"]
        self.assertEqual(500_000.0, ridge["Selected-Scope Prediction"])
        self.assertTrue(pd.isna(ridge["Full-Market Prediction"]))
        self.assertTrue(pd.isna(ridge["Difference (RM)"]))
        self.assertTrue(pd.isna(ridge["Difference (%)"]))
        self.assertEqual("Full-market prediction unavailable", ridge["Status"])

        gradient = frame.loc["Gradient Boosting"]
        self.assertTrue(pd.isna(gradient["Selected-Scope Prediction"]))
        self.assertEqual(510_000.0, gradient["Full-Market Prediction"])
        self.assertEqual("Selected-scope prediction unavailable", gradient["Status"])

        self.assertEqual("Available", frame.loc[FINAL_MODELS[-1], "Status"])
        self.assertEqual(-10_000.0, frame.loc[FINAL_MODELS[-1], "Difference (RM)"])

    def test_unavailable_final_model_is_not_replaced_by_an_alternative(self):
        outcomes = {name: _available(500_000.0) for name in FINAL_MODELS}
        outcomes[FINAL_MODEL_NAME] = _unavailable()
        self.assertIsNone(final_model_prediction(outcomes))

    def test_unavailable_prediction_is_never_substituted_with_zero(self):
        selected = {name: _available(500_000.0) for name in FINAL_MODELS}
        full = {name: _available(500_000.0) for name in FINAL_MODELS}
        selected["Ridge Regression"] = _unavailable()
        full["Ridge Regression"] = _unavailable()

        frame = prediction_comparison_frame(selected, full)
        ridge = frame.set_index("Model").loc["Ridge Regression"]
        self.assertTrue(pd.isna(ridge["Selected-Scope Prediction"]))
        self.assertTrue(pd.isna(ridge["Full-Market Prediction"]))
        self.assertTrue(pd.isna(ridge["Difference (RM)"]))
        self.assertEqual("Prediction unavailable", ridge["Status"])
        html = frame.style.format(
            {
                "Selected-Scope Prediction": "RM {:,.0f}",
                "Full-Market Prediction": "RM {:,.0f}",
                "Difference (RM)": "RM {:+,.0f}",
                "Difference (%)": "{:+.2f}%",
            },
            na_rep="N/A",
        ).to_html()
        self.assertIn("N/A", html)
        self.assertNotIn("RM 0", html)

    def test_impossible_inputs_are_rejected(self):
        invalid = dict(self.values)
        invalid.update(
            {
                "property_size_sqft": 0,
                "bedroom": -1,
                "bathroom": -1,
                "parking_lot": -1,
                "facilities_count": -1,
                "number_of_floors": 0,
                "total_units": 0,
            }
        )
        errors = validate_live_numeric_inputs(invalid)
        self.assertEqual(7, len(errors))
        self.assertTrue(any("Property Size" in error for error in errors))
        self.assertTrue(any("Bedrooms" in error for error in errors))
        self.assertTrue(any("Number of Floors" in error for error in errors))

    def test_impossible_input_stops_streamlit_before_model_loading(self):
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=60)
        app.sidebar.radio[0].set_value(PREDICTOR_VIEW).run(timeout=60)
        size = next(
            item for item in app.number_input if item.label == "Property Size (sq.ft.)"
        )
        size.set_value(0.0).run(timeout=60)
        next(
            item
            for item in app.button
            if item.label == "Estimate Property Price"
        ).click().run(timeout=60)

        self.assertEqual([], list(app.exception))
        self.assertTrue(
            any(
                "Property Size must be greater than zero" in item.value
                for item in app.error
            )
        )
        self.assertEqual([], list(app.dataframe))

    def test_observed_ranges_come_from_all_eight_prepared_dataset_columns(self):
        self.assertEqual({field for field, _ in LIVE_NUMERIC_FIELDS}, set(self.ranges))
        for field, (minimum, maximum) in self.ranges.items():
            numeric = pd.to_numeric(self.data[field], errors="coerce")
            self.assertEqual(float(numeric.min()), minimum)
            self.assertEqual(float(numeric.max()), maximum)

    def test_valid_out_of_range_input_warns_but_is_not_invalid(self):
        extrapolated = dict(self.values)
        extrapolated["bedroom"] = self.ranges["bedroom"][1] + 1
        self.assertEqual([], validate_live_numeric_inputs(extrapolated))
        self.assertEqual(
            ["Bedrooms"],
            outside_observed_range_fields(extrapolated, self.ranges),
        )


class BestMetricStyleTests(unittest.TestCase):
    def test_dark_green_style_is_explicit_and_best_metric_logic_is_preserved(self):
        self.assertEqual(
            "background-color: #14532D;color: #FFFFFF;font-weight: 700;",
            BEST_CELL_STYLE,
        )
        frame = pd.DataFrame(
            {
                "Model": ["A", "B"],
                "RMSE": [200.0, 100.0],
                "MAE": [80.0, 90.0],
                "R²": [0.8, 0.9],
                "Adjusted R²": [0.7, 0.6],
            }
        )
        styled = style_best_metric_cells(
            frame,
            {"RMSE": "{:.0f}", "MAE": "{:.0f}", "R²": "{:.1f}", "Adjusted R²": "{:.1f}"},
            "R²",
            "Adjusted R²",
        )
        context = styled._compute().ctx
        expected_winners = {(1, 1), (0, 2), (1, 3), (0, 4)}
        self.assertEqual(expected_winners, set(context))
        for properties in context.values():
            self.assertIn(("background-color", "#14532D"), properties)
            self.assertIn(("color", "#FFFFFF"), properties)
            self.assertIn(("font-weight", "700"), properties)

    def test_active_performance_table_uses_shared_style_helper(self):
        source = inspect.getsource(render_model_evaluation)
        self.assertIn("style_best_metric_cells", source)
        self.assertNotIn("#d1fae5", source)


if __name__ == "__main__":
    unittest.main()
