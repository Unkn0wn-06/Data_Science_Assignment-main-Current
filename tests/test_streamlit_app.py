"""Streamlit tests for scope-aware comparison, inference, and tuning evidence."""

from __future__ import annotations

import inspect
import json
import unittest

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from prototype.app import (
    COMPARISON_METRICS,
    MARKET_SCOPE_OPTIONS,
    build_official_metric_chart,
    condition_feature_values,
    load_all_models_trimming_summary,
    load_scope_models,
    load_tuning_details,
    normalized_scope_display_frame,
    predict_scope_model,
    prediction_comparison_frame,
    recommended_model_for_scope,
    render_scope_comparison_v2,
    render_scope_predictor,
    scope_comparison_frame,
)
from src.cleaning.pipeline import PROJECT_ROOT
from src.models.common.features import MODEL_FEATURES
from src.models.final.final_evaluation import FINAL_MODELS
from src.models.final.position_regex_lightgbm import FINAL_MODEL_NAME
from src.models.final.trimmed_market import fit_market_scope_models


class ScopeAwareStreamlitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = load_all_models_trimming_summary()
        cls.data = pd.read_csv(
            PROJECT_ROOT / "data" / "processed" / "enhanced_city_dataset.csv"
        ).reset_index(drop=True)

    def test_model_comparison_defaults_and_updates_from_saved_scope_results(self):
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=60)
        self.assertEqual([], list(app.exception))
        self.assertEqual(1, len(app.get("plotly_chart")))

        selectors = {item.label: item for item in app.selectbox}
        self.assertEqual(
            ["Select Market Scope", "Select Evaluation Metric", "Select Model for Hyperparameter Details"],
            [item.label for item in app.selectbox],
        )
        scope = selectors["Select Market Scope"]
        self.assertEqual(list(MARKET_SCOPE_OPTIONS), list(scope.options))
        self.assertEqual("10%", scope.value)
        self.assertEqual(list(COMPARISON_METRICS), list(selectors["Select Evaluation Metric"].options))
        self.assertEqual(FINAL_MODEL_NAME, selectors["Select Model for Hyperparameter Details"].value)
        self.assertEqual(list(FINAL_MODELS), list(selectors["Select Model for Hyperparameter Details"].options))

        expected = normalized_scope_display_frame(self.summary, "10%")
        pd.testing.assert_frame_equal(expected, app.dataframe[0].value, check_dtype=False)
        spec = json.loads(app.get("plotly_chart")[0].proto.spec)
        self.assertEqual("RMSE by Model", spec["layout"]["title"]["text"])

        scope.set_value("5%").run(timeout=60)
        self.assertEqual([], list(app.exception))
        expected = normalized_scope_display_frame(self.summary, "5%")
        pd.testing.assert_frame_equal(expected, app.dataframe[0].value, check_dtype=False)

        metric = next(item for item in app.selectbox if item.label == "Select Evaluation Metric")
        metric.set_value("Adjusted R\u00b2").run(timeout=60)
        spec = json.loads(app.get("plotly_chart")[0].proto.spec)
        self.assertEqual("Adjusted R\u00b2 by Model", spec["layout"]["title"]["text"])
        self.assertEqual(4, len(spec["data"]))

    def test_predictor_scope_selector_is_independent_and_defaults_to_ten_percent(self):
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=60)
        app.sidebar.radio[0].set_value("Live House Price Predictor").run(timeout=60)
        self.assertEqual([], list(app.exception))
        selector = next(item for item in app.selectbox if item.label == "Select Market Scope")
        self.assertEqual(list(MARKET_SCOPE_OPTIONS), list(selector.options))
        self.assertEqual("10%", selector.value)
        self.assertEqual("predictor_market_scope", selector.key)
        self.assertFalse(any(item.label == "Prediction Mode" for item in app.radio))

        selector.set_value("0%").run(timeout=60)
        self.assertEqual([], list(app.exception))
        self.assertEqual("0%", selector.value)
        self.assertEqual(0, len(app.dataframe))
        app.button[0].click().run(timeout=120)
        self.assertEqual([], list(app.exception))
        self.assertEqual(1, len(app.dataframe))
        output = app.dataframe[0].value
        self.assertEqual(list(FINAL_MODELS), output["Model"].tolist())
        self.assertEqual(
            ["Selected-Scope Prediction", "Full-Market Prediction", "Difference (RM)", "Difference (%)"],
            list(output.columns[1:]),
        )
        np.testing.assert_array_equal(output["Difference (RM)"], np.zeros(4))
        np.testing.assert_array_equal(output["Difference (%)"], np.zeros(4))

    def test_four_model_selected_and_full_market_predictions(self):
        values = self.data.iloc[0][MODEL_FEATURES].to_dict()
        values.update(condition_feature_values("Furnished", "Renovated"))
        description = "High floor home with a large balcony"

        full_models = load_scope_models("0%")
        selected_models = load_scope_models("10%")
        self.assertEqual(list(FINAL_MODELS), list(full_models))
        self.assertEqual(list(FINAL_MODELS), list(selected_models))

        full = {
            name: predict_scope_model(
                name, full_models[name], values, description,
                full_models[name].description_length_median_,
            )
            for name in FINAL_MODELS
        }
        selected = {
            name: predict_scope_model(
                name, selected_models[name], values, description,
                selected_models[name].description_length_median_,
            )
            for name in FINAL_MODELS
        }
        output = prediction_comparison_frame(selected, full)
        self.assertEqual(list(FINAL_MODELS), output["Model"].tolist())
        self.assertEqual(
            ["Model", "Selected-Scope Prediction", "Full-Market Prediction", "Difference (RM)", "Difference (%)"],
            list(output.columns),
        )
        self.assertTrue(np.isfinite(output.iloc[:, 1:].to_numpy(float)).all())
        self.assertTrue(output["Difference (RM)"].abs().gt(0).any())

        zero = prediction_comparison_frame(full, full)
        np.testing.assert_array_equal(zero["Difference (RM)"], np.zeros(4))
        np.testing.assert_array_equal(zero["Difference (%)"], np.zeros(4))

    def test_recommendation_uses_each_scopes_saved_rmse(self):
        for scope in MARKET_SCOPE_OPTIONS:
            with self.subTest(scope=scope):
                rows = scope_comparison_frame(self.summary, scope)
                expected = rows.sort_values("RMSE_RM").iloc[0]["Model"]
                self.assertEqual(expected, recommended_model_for_scope(self.summary, scope))

    def test_tuning_tables_use_repository_search_and_builder_values(self):
        details = load_tuning_details()
        self.assertEqual(set(FINAL_MODELS), set(details))
        ridge = details["Ridge Regression"]
        alpha = ridge["search_space"].loc[
            ridge["search_space"]["Hyperparameter"].eq("model__alpha")
        ].iloc[0]
        self.assertIn("0.001", alpha["Values Tested"])
        self.assertIn("10000.0", alpha["Values Tested"])
        final_alpha = ridge["final_parameters"].loc[
            ridge["final_parameters"]["Hyperparameter"].eq("alpha"), "Final Value"
        ].iloc[0]
        self.assertEqual("10.0", final_alpha)
        self.assertIn("RandomizedSearchCV", ridge["method"])
        self.assertIn("not group-safe", ridge["validation"])

        lightgbm = details[FINAL_MODEL_NAME]
        self.assertTrue(lightgbm["search_space"].empty)
        n_estimators = lightgbm["final_parameters"].loc[
            lightgbm["final_parameters"]["Hyperparameter"].eq("n_estimators"), "Final Value"
        ].iloc[0]
        self.assertEqual("1000", n_estimators)
        self.assertTrue(lightgbm["final_parameters"]["Status"].eq("Fixed model parameter").all())

    def test_comparison_interactions_do_not_fit_models(self):
        source = inspect.getsource(render_scope_comparison_v2)
        self.assertNotIn("load_scope_models", source)
        self.assertNotIn("fit_", source)
        self.assertIn("load_all_models_trimming_summary", source)

        loader_source = inspect.getsource(load_scope_models)
        self.assertIn("@st.cache_resource", loader_source)
        self.assertEqual(["scope"], list(inspect.signature(load_scope_models).parameters))
        predictor_source = inspect.getsource(render_scope_predictor)
        self.assertIn('selected_models if selected_scope == "0%"', predictor_source)

    def test_scope_model_builder_rejects_arbitrary_percentages(self):
        with self.assertRaises(ValueError):
            fit_market_scope_models(7.5)

    def test_chart_builder_uses_selected_scope_rows(self):
        rows = scope_comparison_frame(self.summary, "2.5%")
        for metric, (column, title, _, _) in COMPARISON_METRICS.items():
            with self.subTest(metric=metric):
                figure = build_official_metric_chart(rows, metric)
                self.assertEqual(f"{title} by Model", figure.layout.title.text)
                self.assertEqual(4, len(figure.data))
                plotted = {trace.name: float(trace.y[0]) for trace in figure.data}
                saved = rows.set_index("Model")
                for model_name in FINAL_MODELS:
                    self.assertAlmostEqual(saved.loc[model_name, column], plotted[model_name])


if __name__ == "__main__":
    unittest.main()
