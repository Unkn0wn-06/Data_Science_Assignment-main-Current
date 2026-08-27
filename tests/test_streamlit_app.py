"""End-to-end Streamlit page tests for the final saved-results application."""

from __future__ import annotations

import inspect
import json
import unittest

from streamlit.testing.v1 import AppTest

from src.cleaning.pipeline import PROJECT_ROOT
from src.models.final.final_evaluation import FINAL_MODELS
from src.models.final.position_regex_lightgbm import FINAL_MODEL_NAME
from prototype.app import (
    VIEWS,
    comparison_display_frame,
    render_outlier_trimming,
)


class FinalStreamlitAppTests(unittest.TestCase):
    def test_all_pages_and_live_prediction(self):
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=60)
        self.assertEqual(0, len(app.exception))
        expected_views = [
            "Model Comparison",
            "Feature Importance",
            "Actual vs Predicted",
            "Outlier & Trimming Analysis",
            "Live House Price Predictor",
        ]
        self.assertEqual(expected_views, list(VIEWS))
        self.assertEqual(expected_views, list(app.sidebar.radio[0].options))
        self.assertEqual("Model Comparison", app.sidebar.radio[0].value)
        self.assertEqual(3, len(app.get("plotly_chart")))
        self.assertEqual(1, len(app.dataframe))
        expected_columns = ["Model", "RMSE", "MAE", "R²", "Adjusted R²"]
        payload = json.loads(
            (PROJECT_ROOT / "results" / "final_models" / "model_comparison.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(expected_columns, list(comparison_display_frame(payload).columns))
        self.assertEqual(expected_columns, list(app.dataframe[0].value.columns))
        metric_values = {item.label: item.value for item in app.metric}
        self.assertEqual(FINAL_MODEL_NAME, metric_values["Selected Final Model"])
        self.assertEqual(FINAL_MODEL_NAME, metric_values["Lowest RMSE"])
        self.assertEqual("Random Forest", metric_values["Lowest MAE"])
        visible = " ".join(
            [item.value for item in app.caption]
            + [item.value for item in app.info]
            + [item.value for item in app.subheader]
            + list(metric_values.values())
        )
        for model_name in FINAL_MODELS:
            self.assertIn(model_name, visible)
        self.assertNotIn("KNN", visible)
        self.assertNotIn("Median AE", visible)
        self.assertNotIn("Top-5%", visible)
        self.assertNotIn(
            "Retained Data Training & Validation",
            " ".join(item.value for item in app.markdown),
        )

        app.sidebar.radio[0].set_value("Feature Importance").run(timeout=60)
        self.assertEqual(0, len(app.exception))
        self.assertEqual(FINAL_MODEL_NAME, app.selectbox[0].value)
        self.assertEqual(1, len(app.get("plotly_chart")))

        app.sidebar.radio[0].set_value("Actual vs Predicted").run(timeout=60)
        self.assertEqual(0, len(app.exception))
        self.assertEqual(FINAL_MODEL_NAME, app.selectbox[0].value)
        self.assertEqual(1, len(app.get("plotly_chart")))
        self.assertEqual(3, len(app.metric))

        app.sidebar.radio[0].set_value("Outlier & Trimming Analysis").run(timeout=60)
        self.assertEqual(0, len(app.exception))
        self.assertEqual("Upper-tail trimming level", app.selectbox[0].label)
        self.assertEqual(
            ["0%", "0.5%", "1%", "2.5%", "5%", "10%"],
            list(app.selectbox[0].options),
        )
        self.assertEqual("0%", app.selectbox[0].value)
        self.assertEqual(4, len(app.get("plotly_chart")))
        self.assertEqual(3, len(app.dataframe))
        self.assertEqual(
            ["Fold", "Training Listings", "Validation Listings"],
            list(app.dataframe[0].value.columns),
        )
        outlier_text = " ".join(
            item.value
            for element_type in (
                app.caption,
                app.info,
                app.markdown,
                app.subheader,
                app.success,
                app.warning,
            )
            for item in element_type
            if isinstance(item.value, str)
        )
        self.assertIn("Final Decision: 0% Trimming", outlier_text)
        self.assertIn(f"Final Production Model: {FINAL_MODEL_NAME}", outlier_text)
        self.assertIn("Retained Data Training & Validation", outlier_text)
        retained_metrics = {item.label: item.value for item in app.metric}
        self.assertEqual("3,791", retained_metrics["Original Listings"])
        self.assertEqual("3,791", retained_metrics["Listings Retained"])
        self.assertEqual("0", retained_metrics["Listings Removed"])
        app.selectbox[0].set_value("5%").run(timeout=60)
        self.assertEqual(0, len(app.exception))
        warning_text = " ".join(item.value for item in app.warning)
        self.assertIn("market scope changes substantially", warning_text)
        retained_metrics = {item.label: item.value for item in app.metric}
        self.assertEqual("3,601", retained_metrics["Listings Retained"])
        self.assertEqual("190", retained_metrics["Listings Removed"])

        app.sidebar.radio[0].set_value("Live House Price Predictor").run(timeout=60)
        self.assertEqual(0, len(app.exception))
        self.assertNotIn(
            "Upper-tail trimming level", [item.label for item in app.selectbox]
        )
        self.assertNotIn(
            "Retained Data Training & Validation",
            " ".join(item.value for item in app.markdown),
        )
        self.assertEqual("Listing Description", app.text_area[0].label)
        app.text_area[0].set_value(
            "Spacious high floor unit with a large balcony"
        )
        self.assertEqual("Building Name (Optional)", app.text_input[0].label)
        self.assertEqual("Developer (Optional)", app.text_input[1].label)
        app.text_input[0].set_value("Unseen Streamlit Tower")
        app.text_input[1].set_value("Unseen Streamlit Developer")
        app.button[0].click().run(timeout=120)
        self.assertEqual(0, len(app.exception))
        live_metrics = {item.label: item.value for item in app.metric}
        self.assertIn("Estimated Listing Price", live_metrics)
        self.assertIn("Estimated Price per sq.ft.", live_metrics)
        self.assertTrue(live_metrics["Estimated Listing Price"].startswith("RM "))
        self.assertIn(FINAL_MODEL_NAME, " ".join(item.value for item in app.success))

    def test_trimming_page_is_saved_results_only(self):
        streamlit_source = (PROJECT_ROOT / "prototype" / "app.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"results" / "outlier_trimming"', streamlit_source)
        self.assertNotIn("experiments/upper_tail_trimming", streamlit_source)
        self.assertNotIn("experiments.upper_tail_trimming", streamlit_source)
        trimming_renderer = inspect.getsource(render_outlier_trimming)
        self.assertNotIn("fit_", trimming_renderer)
        self.assertNotIn("run_experiment", trimming_renderer)
        self.assertIn("load_trimming_results", trimming_renderer)


if __name__ == "__main__":
    unittest.main()
