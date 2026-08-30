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
    DIAGNOSTICS_VIEW,
    EVALUATION_VIEW,
    MARKET_SCOPE_OPTIONS,
    OUTLIER_VIEW,
    PREDICTOR_VIEW,
    VIEWS,
    actual_vs_predicted_plot_frame,
    build_official_metric_chart,
    condition_feature_values,
    load_all_models_trimming_oof,
    load_all_models_trimming_summary,
    _load_scope_models_cached,
    load_scope_models,
    load_tuning_details,
    normalized_scope_display_frame,
    predict_scope_models,
    prediction_comparison_frame,
    recommended_model_for_scope,
    render_outlier_trimming,
    render_model_evaluation,
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

    def test_overview_orients_without_revealing_project_conclusions(self):
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=60)
        self.assertEqual([], list(app.exception))
        self.assertEqual(VIEWS[0], app.sidebar.radio[0].value)
        self.assertEqual(
            {
                "Prepared Listings": "3,791",
                "Prepared Features": "34",
                "Problem Type": "Regression",
                "Target Variable": "Listing Price (RM)",
            },
            {item.label: item.value for item in app.main.metric},
        )
        main_text = "\n".join(
            item.value
            for element_type in ("header", "markdown", "caption", "text")
            for item in app.main.get(element_type)
            if isinstance(item.value, str)
        )
        for hidden_result in (
            FINAL_MODEL_NAME,
            "RMSE",
            "MAE",
            "R²",
            "0% Trimming",
            "Final Full-Market Performance",
        ):
            self.assertNotIn(hidden_result, main_text)
        for section in (
            "### Project at a Glance",
            "### Project Objective",
            "### What Does the Dataset Contain?",
            "### Project Workflow",
            "### Explore the Project",
        ):
            self.assertIn(section, main_text)

        sidebar_text = "\n".join(item.value for item in app.sidebar.caption)
        self.assertIn("Problem\n\nRegression", sidebar_text)
        self.assertNotIn(FINAL_MODEL_NAME, sidebar_text)
        self.assertNotIn("Upper-Tail Trimming", sidebar_text)

    def test_model_comparison_defaults_and_updates_from_saved_scope_results(self):
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=60)
        self.assertEqual(VIEWS[0], app.sidebar.radio[0].value)
        app.sidebar.radio[0].set_value(EVALUATION_VIEW).run(timeout=60)
        self.assertEqual([], list(app.exception))
        self.assertEqual(1, len(app.get("plotly_chart")))

        selectors = {item.label: item for item in app.selectbox}
        scope = selectors["Market Scope"]
        self.assertEqual(list(MARKET_SCOPE_OPTIONS), list(scope.options))
        self.assertEqual("10%", scope.value)
        self.assertEqual("evaluation_market_scope", scope.key)
        self.assertEqual(list(COMPARISON_METRICS), list(selectors["Evaluation Metric"].options))
        self.assertEqual(FINAL_MODEL_NAME, selectors["Model for Hyperparameter Details"].value)
        self.assertEqual(list(FINAL_MODELS), list(selectors["Model for Hyperparameter Details"].options))

        spec = json.loads(app.get("plotly_chart")[0].proto.spec)
        self.assertEqual("RMSE by Model", spec["layout"]["title"]["text"])
        recommended = recommended_model_for_scope(self.summary, "10%")
        expected = scope_comparison_frame(self.summary, "10%").set_index("Model").loc[recommended]
        metric_values = {item.label: item.value for item in app.metric}
        self.assertEqual(f"RM {expected['RMSE_RM']:,.0f}", metric_values["RMSE"])

        scope.set_value("5%").run(timeout=60)
        self.assertEqual([], list(app.exception))
        self.assertEqual("5%", next(item for item in app.selectbox if item.label == "Market Scope").value)

        metric = next(item for item in app.selectbox if item.label == "Evaluation Metric")
        metric.set_value("Adjusted R\u00b2").run(timeout=60)
        spec = json.loads(app.get("plotly_chart")[0].proto.spec)
        self.assertEqual("Adjusted R\u00b2 by Model", spec["layout"]["title"]["text"])
        self.assertEqual(4, len(spec["data"]))

    def test_predictor_scope_selector_is_independent_and_defaults_to_ten_percent(self):
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=60)
        app.sidebar.radio[0].set_value(PREDICTOR_VIEW).run(timeout=60)
        self.assertEqual([], list(app.exception))
        selector = next(item for item in app.selectbox if item.label == "Market Scope")
        self.assertEqual(list(MARKET_SCOPE_OPTIONS), list(selector.options))
        self.assertEqual("10%", selector.value)
        self.assertEqual("predictor_market_scope", selector.key)
        self.assertFalse(any(item.label == "Prediction Mode" for item in app.radio))

        selector.set_value("0%").run(timeout=60)
        self.assertEqual([], list(app.exception))
        self.assertEqual("0%", selector.value)
        self.assertEqual(0, len(app.dataframe))
        next(item for item in app.button if item.label == "Estimate Property Price").click().run(timeout=120)
        self.assertEqual([], list(app.exception))
        self.assertEqual(1, len(app.dataframe))
        output = app.dataframe[0].value
        self.assertEqual(list(FINAL_MODELS), output["Model"].tolist())
        self.assertEqual(
            [
                "Selected-Scope Prediction",
                "Full-Market Prediction",
                "Difference (RM)",
                "Difference (%)",
                "Status",
            ],
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

        full = predict_scope_models(full_models, values, description)
        selected = predict_scope_models(selected_models, values, description)
        output = prediction_comparison_frame(selected, full)
        self.assertEqual(list(FINAL_MODELS), output["Model"].tolist())
        self.assertEqual(
            [
                "Model",
                "Selected-Scope Prediction",
                "Full-Market Prediction",
                "Difference (RM)",
                "Difference (%)",
                "Status",
            ],
            list(output.columns),
        )
        self.assertTrue(
            np.isfinite(
                output[
                    [
                        "Selected-Scope Prediction",
                        "Full-Market Prediction",
                        "Difference (RM)",
                        "Difference (%)",
                    ]
                ].to_numpy(float)
            ).all()
        )
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
            ridge["search_space"]["Hyperparameter"].eq("alpha")
        ].iloc[0]
        self.assertIn("0.001", alpha["Values Tested"])
        self.assertIn("1000.0", alpha["Values Tested"])
        final_alpha = ridge["final_parameters"].loc[
            ridge["final_parameters"]["Hyperparameter"].eq("alpha"), "Final Value"
        ].iloc[0]
        self.assertEqual("1000.0", final_alpha)
        self.assertIn("31", ridge["method"])
        self.assertIn("group-safe", ridge["validation"])
        self.assertIn("StandardScaler", ridge["scaling"])
        self.assertLess(
            ridge["before_after"].loc[
                ridge["before_after"]["Metric"].eq("RMSE"), "Change"
            ].iloc[0],
            0,
        )

        lightgbm = details[FINAL_MODEL_NAME]
        self.assertFalse(lightgbm["search_space"].empty)
        n_estimators = lightgbm["final_parameters"].loc[
            lightgbm["final_parameters"]["Hyperparameter"].eq("n_estimators"), "Final Value"
        ].iloc[0]
        self.assertEqual("1200", n_estimators)
        self.assertEqual(80, lightgbm["candidate_count"])
        self.assertTrue(
            lightgbm["final_parameters"]["Status"].eq(
                "Selected by current formal tuning"
            ).all()
        )

    def test_comparison_interactions_do_not_fit_models(self):
        source = inspect.getsource(render_model_evaluation)
        self.assertNotIn("load_scope_models", source)
        self.assertNotIn("fit_", source)
        self.assertIn("load_all_models_trimming_summary", source)

        loader_source = inspect.getsource(load_scope_models)
        self.assertIn("final_tuned_params_sha256", loader_source)
        self.assertEqual(["scope"], list(inspect.signature(load_scope_models).parameters))
        self.assertIn("@st.cache_resource", inspect.getsource(_load_scope_models_cached))
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

    def test_actual_vs_predicted_uses_saved_cutoffs_for_every_trim_level(self):
        oof = load_all_models_trimming_oof()
        expected_counts = {
            "0%": 3_791,
            "0.5%": 3_772,
            "1%": 3_758,
            "2.5%": 3_700,
            "5%": 3_601,
            "10%": 3_412,
        }
        for model_name in FINAL_MODELS:
            for scope, expected_count in expected_counts.items():
                with self.subTest(model=model_name, scope=scope):
                    plot, metadata = actual_vs_predicted_plot_frame(
                        oof, model_name, scope
                    )
                    self.assertEqual(expected_count, len(plot))
                    self.assertEqual(expected_count, metadata["retained_rows"])
                    self.assertTrue(
                        np.isfinite(
                            plot[
                                ["Actual Price (RM)", "OOF Predicted Price (RM)"]
                            ].to_numpy(float)
                        ).all()
                    )
                    if metadata["cutoff_RM"] is not None:
                        self.assertLessEqual(
                            plot["Actual Price (RM)"].max(), metadata["cutoff_RM"]
                        )

    def test_actual_vs_predicted_rejects_invalid_saved_oof_data(self):
        oof = load_all_models_trimming_oof()
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            actual_vs_predicted_plot_frame(oof, "Unknown Model", "5%")

        invalid = oof.copy()
        row = invalid.index[invalid["Model"].eq("Ridge Regression")][0]
        invalid.loc[row, "predicted_price_RM"] = np.inf
        with self.assertRaisesRegex(ValueError, "must all be finite"):
            actual_vs_predicted_plot_frame(invalid, "Ridge Regression", "0%")

    def test_actual_vs_predicted_controls_metrics_and_scope_update_together(self):
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=60)
        app.sidebar.radio[0].set_value(DIAGNOSTICS_VIEW).run(timeout=60)
        next(item for item in app.radio if item.label == "Diagnostic View").set_value(
            "Actual vs Predicted"
        ).run(timeout=60)
        self.assertEqual([], list(app.exception))
        selectors = {item.label: item for item in app.selectbox}
        trim = selectors["Market Scope"]
        self.assertEqual(list(MARKET_SCOPE_OPTIONS), list(trim.options))
        self.assertEqual("10%", trim.value)
        self.assertEqual("prediction_trim_level", trim.key)
        self.assertFalse(any(item.label == "Listing Segment" for item in app.radio))
        default_metrics = {item.label: item.value for item in app.metric}
        self.assertEqual({"RMSE", "MAE", "R²"}, set(default_metrics))
        scope_details = app.dataframe[0].value.iloc[0]
        self.assertEqual(3_412, scope_details["Retained Listings"])
        self.assertEqual(379, scope_details["Removed Listings"])
        self.assertAlmostEqual(90.0026, scope_details["Retention Percentage"], places=3)
        self.assertEqual(699_999, scope_details["Maximum Retained Actual Price"])

        trim.set_value("5%").run(timeout=60)
        self.assertEqual([], list(app.exception))
        expected = self.summary.loc[
            self.summary["Model"].eq(FINAL_MODEL_NAME)
            & self.summary["Trim_Level"].eq("5%")
        ].iloc[0]
        metric_values = {item.label: item.value for item in app.metric}
        self.assertEqual(f"RM {expected['RMSE_RM']:,.0f}", metric_values["RMSE"])
        self.assertEqual(f"RM {expected['MAE_RM']:,.0f}", metric_values["MAE"])
        scope_details = app.dataframe[0].value.iloc[0]
        self.assertEqual(3_601, scope_details["Retained Listings"])
        self.assertEqual(190, scope_details["Removed Listings"])
        self.assertEqual(900_000, scope_details["Maximum Retained Actual Price"])
        spec = json.loads(app.get("plotly_chart")[0].proto.spec)
        self.assertEqual(
            f"{FINAL_MODEL_NAME}: Actual vs OOF Predicted Price \u2014 5% Trimming",
            spec["layout"]["title"]["text"],
        )
        self.assertTrue(any("Displayed listings: 3,601" in item.value for item in app.caption))

        model = next(item for item in app.selectbox if item.key == "prediction_model")
        model.set_value("Ridge Regression").run(timeout=60)
        expected_ridge = self.summary.loc[
            self.summary["Model"].eq("Ridge Regression")
            & self.summary["Trim_Level"].eq("5%")
        ].iloc[0]
        metric_values = {item.label: item.value for item in app.metric}
        self.assertEqual(
            f"RM {expected_ridge['RMSE_RM']:,.0f}", metric_values["RMSE"]
        )

    def test_outlier_page_is_explanatory_without_redundant_scope_controls(self):
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=60)
        app.sidebar.radio[0].set_value(OUTLIER_VIEW).run(timeout=60)
        self.assertEqual([], list(app.exception))
        self.assertEqual([], list(app.selectbox))
        self.assertEqual(
            [
                "### Outlier Detection",
                "### Outlier Treatment Methods",
                "### Premium Property Impact",
                "### Statistical Validation",
            ],
            [
                item.value
                for item in app.main.markdown
                if item.value.startswith("###")
            ],
        )
        plot_titles = [
            json.loads(item.proto.spec)["layout"]["title"]["text"]
            for item in app.get("plotly_chart")
        ]
        self.assertEqual(
            [
                "Saved Listing-Price Distribution Landmarks",
                "Premium Underprediction as Training Examples Are Removed",
            ],
            plot_titles,
        )
        self.assertEqual(
            [
                "View Outlier Treatment Methods",
                "View Detailed Premium-Segment Results",
                "View Bootstrap Statistical Results",
                "Technical Details",
            ],
            [item.label for item in app.expander],
        )
        self.assertEqual(
            "Final Full-Market Decision: 0% Upper-Tail Trimming",
            app.success[0].value,
        )
        self.assertTrue(
            any(
                "10% default" in item.value
                and "official 0% full-market strategy" in item.value
                for item in app.info
            )
        )

        source = inspect.getsource(render_outlier_trimming)
        for removed_text in (
            "Market Scope After Trimming",
            "Retained-population trim level",
            "Retained Data Training & Validation",
            "Maximum Retained Price Across Trim Levels",
            "Training and Validation Listings by Fold",
        ):
            self.assertNotIn(removed_text, source)
        self.assertIn("load_trimming_results", source)


if __name__ == "__main__":
    unittest.main()
