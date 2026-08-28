"""End-to-end Streamlit page tests for the final saved-results application."""

from __future__ import annotations

import inspect
import json
import unittest

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.cleaning.pipeline import PROJECT_ROOT
from src.models.common.features import MODEL_FEATURES
from src.models.final.final_evaluation import FINAL_MODELS
from src.models.final.position_regex_lightgbm import (
    FINAL_MODEL_NAME,
    fit_final_model,
    predict_total_price,
)
from src.models.final.trimmed_market import fit_trimmed_market_model
from prototype.app import (
    COMPARISON_METRICS,
    VIEWS,
    build_official_metric_chart,
    build_trimmed_metric_chart,
    comparison_frame,
    comparison_display_frame,
    condition_feature_values,
    load_all_models_trimming_summary,
    load_trimming_results,
    render_all_models_trimming_comparison,
    render_outlier_trimming,
    trimming_display_frame,
)
from prototype.eda_page import EDA_VISUALIZATIONS, render_eda_page


class FinalStreamlitAppTests(unittest.TestCase):
    def test_all_pages_and_live_prediction(self):
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=60)
        self.assertEqual(0, len(app.exception))
        expected_views = [
            "Model Comparison",
            "Exploratory Data Analysis",
            "Feature Importance",
            "Actual vs Predicted",
            "Outlier & Trimming Analysis",
            "Live House Price Predictor",
        ]
        self.assertEqual(expected_views, list(VIEWS))
        self.assertEqual(expected_views, list(app.sidebar.radio[0].options))
        self.assertEqual("Model Comparison", app.sidebar.radio[0].value)
        self.assertEqual(2, len(app.get("plotly_chart")))
        self.assertEqual(5, len(app.dataframe))
        self.assertEqual(2, len(app.selectbox))
        self.assertEqual(
            ["Select Evaluation Metric", "Select Trimming Metric"],
            [item.label for item in app.selectbox],
        )
        self.assertEqual(["RMSE", "MAE", "R²", "Adjusted R²"], list(app.selectbox[0].options))
        self.assertEqual(["RMSE", "MAE", "R²", "Adjusted R²"], list(app.selectbox[1].options))
        self.assertEqual("RMSE", app.selectbox[0].value)
        self.assertEqual("RMSE", app.selectbox[1].value)
        self.assertEqual(0, len(app.metric))
        self.assertEqual(0, len(app.info))
        self.assertEqual(0, len(app.warning))
        self.assertEqual(0, len(app.caption))
        self.assertEqual(0, len(app.markdown))
        expected_columns = ["Model", "RMSE", "MAE", "R²", "Adjusted R²"]
        payload = json.loads(
            (PROJECT_ROOT / "results" / "final_models" / "model_comparison.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(expected_columns, list(comparison_display_frame(payload).columns))
        self.assertEqual(expected_columns, list(app.dataframe[0].value.columns))
        self.assertEqual(set(FINAL_MODELS), set(app.dataframe[0].value["Model"]))
        self.assertNotIn("KNN", set(app.dataframe[0].value["Model"]))

        expected_trim_columns = [
            "Trim Level",
            "Retained Listings",
            "RMSE",
            "MAE",
            "R²",
            "Adjusted R²",
        ]
        expected_trim_levels = ["0%", "0.5%", "1%", "2.5%", "5%", "10%"]
        saved_trim = load_all_models_trimming_summary()
        for table_index, model_name in enumerate(FINAL_MODELS, start=1):
            with self.subTest(model=model_name):
                displayed = app.dataframe[table_index].value
                self.assertEqual(expected_trim_columns, list(displayed.columns))
                self.assertEqual(expected_trim_levels, displayed["Trim Level"].tolist())
                self.assertEqual(6, len(displayed))
                pd.testing.assert_frame_equal(
                    trimming_display_frame(saved_trim, model_name),
                    displayed,
                    check_dtype=False,
                )
        five_percent = app.dataframe[4].value.loc[
            app.dataframe[4].value["Trim Level"].eq("5%")
        ].iloc[0]
        self.assertEqual(3_601, int(five_percent["Retained Listings"]))
        self.assertAlmostEqual(72_463.70989937868, five_percent["RMSE"])
        self.assertAlmostEqual(47_919.00262780433, five_percent["MAE"])

        chart_specs = [json.loads(chart.proto.spec) for chart in app.get("plotly_chart")]
        self.assertEqual(
            ["RMSE by Model", "RMSE Across Trimming Levels"],
            [spec["layout"]["title"]["text"] for spec in chart_specs],
        )
        self.assertEqual(4, len(chart_specs[0]["data"]))
        self.assertEqual(4, len(chart_specs[1]["data"]))

        expected_titles = {
            "RMSE": "RMSE by Model",
            "MAE": "MAE by Model",
            "R²": "R² by Model",
            "Adjusted R²": "Adjusted R² by Model",
        }
        for metric, title in expected_titles.items():
            with self.subTest(official_metric=metric):
                app.selectbox[0].set_value(metric).run(timeout=60)
                self.assertEqual(0, len(app.exception))
                spec = json.loads(app.get("plotly_chart")[0].proto.spec)
                self.assertEqual(title, spec["layout"]["title"]["text"])
                self.assertEqual(4, len(spec["data"]))

        expected_trim_titles = {
            "RMSE": "RMSE Across Trimming Levels",
            "MAE": "MAE Across Trimming Levels",
            "R²": "R² Across Trimming Levels",
            "Adjusted R²": "Adjusted R² Across Trimming Levels",
        }
        for metric, title in expected_trim_titles.items():
            with self.subTest(trimming_metric=metric):
                app.selectbox[1].set_value(metric).run(timeout=60)
                self.assertEqual(0, len(app.exception))
                spec = json.loads(app.get("plotly_chart")[1].proto.spec)
                self.assertEqual(title, spec["layout"]["title"]["text"])
                self.assertEqual(4, len(spec["data"]))

        app.selectbox[0].set_value("RMSE")
        app.selectbox[1].set_value("RMSE").run(timeout=60)

        app.sidebar.radio[0].set_value("Exploratory Data Analysis").run(timeout=60)
        self.assertEqual(0, len(app.exception))
        self.assertEqual(2, len(app.dataframe))
        self.assertEqual(0, len(app.get("plotly_chart")))
        self.assertEqual(1, len(app.get("imgs")))
        self.assertEqual(["EDA Category", "Select Visualization"], [item.label for item in app.selectbox])
        self.assertEqual(list(EDA_VISUALIZATIONS), list(app.selectbox[0].options))
        self.assertEqual("Price & Location", app.selectbox[0].value)
        self.assertEqual(
            list(EDA_VISUALIZATIONS["Price & Location"]),
            list(app.selectbox[1].options),
        )
        app.selectbox[1].set_value("Mean and Median Condominium Price by State").run(
            timeout=60
        )
        self.assertEqual(0, len(app.exception))
        self.assertEqual(0, len(app.get("plotly_chart")))
        self.assertEqual(1, len(app.get("imgs")))
        app.selectbox[0].set_value("Property Characteristics").run(timeout=60)
        self.assertEqual(0, len(app.exception))
        self.assertEqual(
            list(EDA_VISUALIZATIONS["Property Characteristics"]),
            list(app.selectbox[1].options),
        )
        self.assertEqual(0, len(app.get("plotly_chart")))
        self.assertEqual(1, len(app.get("imgs")))
        app.selectbox[1].set_value("Listing Price Distribution by Land Title").run(
            timeout=60
        )
        self.assertEqual(0, len(app.exception))
        self.assertEqual(1, len(app.get("imgs")))

        app.sidebar.radio[0].set_value("Model Comparison").run(timeout=60)
        self.assertEqual(0, len(app.exception))
        self.assertEqual(2, len(app.get("plotly_chart")))
        self.assertEqual(5, len(app.dataframe))

        app.sidebar.radio[0].set_value("Exploratory Data Analysis").run(timeout=60)
        self.assertEqual(0, len(app.exception))
        self.assertEqual(0, len(app.get("plotly_chart")))
        self.assertEqual(1, len(app.get("imgs")))
        self.assertEqual(2, len(app.selectbox))

        app.sidebar.radio[0].set_value("Feature Importance").run(timeout=60)
        self.assertEqual(0, len(app.exception))
        self.assertEqual(FINAL_MODEL_NAME, app.selectbox[0].value)
        self.assertEqual(1, len(app.get("plotly_chart")))

        app.sidebar.radio[0].set_value("Exploratory Data Analysis").run(timeout=60)
        self.assertEqual(0, len(app.exception))
        self.assertEqual(0, len(app.get("plotly_chart")))
        self.assertEqual(1, len(app.get("imgs")))
        self.assertEqual(2, len(app.dataframe))

        app.sidebar.radio[0].set_value("Outlier & Trimming Analysis").run(timeout=60)
        self.assertEqual(0, len(app.exception))
        self.assertEqual(4, len(app.get("plotly_chart")))

        app.sidebar.radio[0].set_value("Exploratory Data Analysis").run(timeout=60)
        self.assertEqual(0, len(app.exception))
        self.assertEqual(0, len(app.get("plotly_chart")))
        self.assertEqual(1, len(app.get("imgs")))
        self.assertEqual(2, len(app.selectbox))

        app.sidebar.radio[0].set_value("Model Comparison").run(timeout=60)
        self.assertEqual(0, len(app.exception))
        self.assertEqual(2, len(app.get("plotly_chart")))
        self.assertEqual(5, len(app.dataframe))

        app.sidebar.radio[0].set_value("Actual vs Predicted").run(timeout=60)
        self.assertEqual(0, len(app.exception))
        self.assertEqual(FINAL_MODEL_NAME, app.selectbox[0].value)
        self.assertEqual(1, len(app.get("plotly_chart")))
        self.assertEqual(3, len(app.metric))

        app.sidebar.radio[0].set_value("Outlier & Trimming Analysis").run(timeout=60)
        self.assertEqual(0, len(app.exception))
        self.assertEqual("Retained-population trim level", app.selectbox[0].label)
        self.assertEqual(
            ["0%", "0.5%", "1%", "2.5%", "5%", "10%"],
            list(app.selectbox[0].options),
        )
        self.assertEqual("0%", app.selectbox[0].value)
        self.assertEqual(4, len(app.get("plotly_chart")))
        self.assertEqual(6, len(app.dataframe))
        self.assertEqual(
            ["Fold", "Training Listings", "Validation Listings"],
            list(app.dataframe[4].value.columns),
        )
        self.assertEqual(
            [
                "Method",
                "Stage",
                "How It Handles Outliers",
                "Deletes Rows?",
                "Caps Values?",
                "Outcome",
                "Final Decision",
            ],
            list(app.dataframe[1].value.columns),
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
        for heading in (
            "Outlier Detection",
            "Outlier Treatment Methods",
            "Market Scope After Trimming",
            "Premium Property Impact",
            "Statistical Validation",
            "Final Outlier-Treatment Decision",
        ):
            self.assertIn(heading, outlier_text)
        self.assertNotIn("Full-Market Performance", outlier_text)
        self.assertNotIn("Restricted-Market Performance", outlier_text)
        outlier_chart_titles = [
            json.loads(chart.proto.spec)["layout"]["title"]["text"]
            for chart in app.get("plotly_chart")
        ]
        self.assertEqual(
            [
                "Saved Listing-Price Distribution Landmarks",
                "Maximum Retained Price Across Trim Levels",
                "Premium Underprediction as Training Examples Are Removed",
                "Training and Validation Listings by Fold — 0% Trimming",
            ],
            outlier_chart_titles,
        )
        retained_metrics = {item.label: item.value for item in app.metric}
        self.assertEqual("3,791", retained_metrics["Original Listings"])
        self.assertEqual("3,791", retained_metrics["Listings Retained"])
        self.assertEqual("0", retained_metrics["Listings Removed"])
        app.selectbox[0].set_value("5%").run(timeout=60)
        self.assertEqual(0, len(app.exception))
        retained_metrics = {item.label: item.value for item in app.metric}
        self.assertEqual("3,601", retained_metrics["Listings Retained"])
        self.assertEqual("190", retained_metrics["Listings Removed"])

        app.sidebar.radio[0].set_value("Model Comparison").run(timeout=60)
        self.assertEqual(0, len(app.exception))
        self.assertEqual(2, len(app.get("plotly_chart")))
        self.assertEqual(5, len(app.dataframe))
        self.assertEqual(2, len(app.selectbox))
        self.assertEqual(0, len(app.metric))

        app.sidebar.radio[0].set_value("Live House Price Predictor").run(timeout=60)

        self.assertEqual(0, len(app.exception))
        prediction_modes = [item for item in app.radio if item.label == "Prediction Mode"]
        self.assertEqual(1, len(prediction_modes))
        self.assertEqual("Final Full-Market Model", prediction_modes[0].value)
        self.assertNotIn(
            "Upper-tail trimming level", [item.label for item in app.selectbox]
        )
        furnishing_selectors = [
            item for item in app.selectbox if item.label == "Furnishing Status"
        ]
        renovation_selectors = [
            item for item in app.selectbox if item.label == "Renovation Status"
        ]
        self.assertEqual(1, len(furnishing_selectors))
        self.assertEqual(1, len(renovation_selectors))
        self.assertEqual("Unfurnished", furnishing_selectors[0].value)
        self.assertEqual("Not Renovated", renovation_selectors[0].value)
        self.assertEqual(["Unfurnished", "Furnished"], list(furnishing_selectors[0].options))
        self.assertEqual(["Not Renovated", "Renovated"], list(renovation_selectors[0].options))
        self.assertNotIn("Furnished", [item.label for item in app.checkbox])
        self.assertNotIn("Renovated", [item.label for item in app.checkbox])
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
        furnishing_selectors[0].set_value("Furnished")
        renovation_selectors[0].set_value("Renovated")
        app.button[0].click().run(timeout=120)
        self.assertEqual(0, len(app.exception))
        live_metrics = {item.label: item.value for item in app.metric}
        self.assertIn("Estimated Listing Price", live_metrics)
        self.assertIn("Estimated Price per sq.ft.", live_metrics)
        self.assertTrue(live_metrics["Estimated Listing Price"].startswith("RM "))
        self.assertIn(FINAL_MODEL_NAME, " ".join(item.value for item in app.success))
        self.assertEqual(
            {"is_furnished": 1, "is_renovated": 1},
            condition_feature_values("Furnished", "Renovated"),
        )

        prediction_modes[0].set_value("Experimental Trimmed-Market Model").run(
            timeout=60
        )
        self.assertEqual(0, len(app.exception))
        trim_selectors = [
            item for item in app.selectbox if item.label == "Experimental trim level"
        ]
        self.assertEqual(1, len(trim_selectors))
        self.assertEqual(["0.5%", "1%", "2.5%", "5%", "10%"], list(trim_selectors[0].options))
        trim_selectors[0].set_value("5%").run(timeout=60)
        app.button[0].click().run(timeout=180)
        self.assertEqual(0, len(app.exception))
        experimental_metrics = {item.label: item.value for item in app.metric}
        self.assertEqual("3,601", experimental_metrics["Retained Rows"])
        self.assertIn("Official Full-Market Prediction", experimental_metrics)
        self.assertIn("5% Trimmed-Market Prediction", experimental_metrics)
        self.assertIn("Difference", experimental_metrics)

    def test_metric_chart_builders_use_saved_values(self):
        payload = json.loads(
            (PROJECT_ROOT / "results" / "final_models" / "model_comparison.json")
            .read_text(encoding="utf-8")
        )
        official = comparison_frame(payload).set_index("Model")
        trimmed = load_all_models_trimming_summary()
        for metric_name, (column, title_metric, _, _) in COMPARISON_METRICS.items():
            with self.subTest(metric=metric_name):
                official_figure = build_official_metric_chart(
                    official.reset_index(), metric_name
                )
                self.assertEqual(
                    f"{title_metric} by Model", official_figure.layout.title.text
                )
                self.assertEqual(620, official_figure.layout.height)
                self.assertEqual(4, len(official_figure.data))
                for trace in official_figure.data:
                    self.assertAlmostEqual(
                        float(official.loc[trace.name, column]), float(trace.y[0])
                    )

                trimmed_figure = build_trimmed_metric_chart(trimmed, metric_name)
                self.assertEqual(
                    f"{title_metric} Across Trimming Levels",
                    trimmed_figure.layout.title.text,
                )
                self.assertEqual(650, trimmed_figure.layout.height)
                self.assertEqual(4, len(trimmed_figure.data))
                for trace in trimmed_figure.data:
                    expected = trimmed.loc[trimmed["Model"].eq(trace.name), column]
                    np.testing.assert_allclose(
                        np.asarray(trace.y, dtype=float), expected.to_numpy(float)
                    )
                    self.assertEqual(6, len(trace.x))

    def test_all_furnishing_and_renovation_combinations_predict(self):
        data = pd.read_csv(
            PROJECT_ROOT / "data" / "processed" / "enhanced_city_dataset.csv"
        ).reset_index(drop=True)
        official_model = fit_final_model()
        trimmed_model = fit_trimmed_market_model(5.0)
        cases = {
            ("Furnished", "Not Renovated"): (1, 0),
            ("Unfurnished", "Renovated"): (0, 1),
            ("Furnished", "Renovated"): (1, 1),
            ("Unfurnished", "Not Renovated"): (0, 0),
        }
        for (furnishing, renovation), expected in cases.items():
            with self.subTest(furnishing=furnishing, renovation=renovation):
                mapped = condition_feature_values(furnishing, renovation)
                self.assertEqual(expected, (mapped["is_furnished"], mapped["is_renovated"]))
                values = data.iloc[0][MODEL_FEATURES].to_dict()
                values.update(mapped)
                for model in (official_model, trimmed_model):
                    prediction = predict_total_price(
                        model,
                        values,
                        "High floor property with a balcony",
                        model.description_length_median_,
                    )
                    self.assertTrue(np.isfinite(prediction["total_price_RM"]))
                    self.assertGreater(prediction["total_price_RM"], 0)

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
        summary_renderer = inspect.getsource(render_all_models_trimming_comparison)
        self.assertNotIn("fit_", summary_renderer)
        self.assertIn("selectbox", summary_renderer)
        self.assertIn("trimmed_comparison_metric", summary_renderer)
        self.assertNotIn("st.metric", summary_renderer)
        self.assertIn("plotly_chart", summary_renderer)
        self.assertIn("load_all_models_trimming_summary", summary_renderer)
        summary = load_all_models_trimming_summary()
        self.assertEqual(24, len(summary))
        self.assertEqual(6, len(trimming_display_frame(summary, FINAL_MODEL_NAME)))
        self.assertEqual(6, len(load_trimming_results()[1]["trimmed_population_comparison"]))


if __name__ == "__main__":
    unittest.main()
