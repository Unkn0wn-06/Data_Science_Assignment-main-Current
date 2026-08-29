"""Streamlit dashboard backed by current tuned Scenario B evaluation artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.common.features import MODEL_FEATURES
from src.models.final.final_evaluation import FINAL_MODELS, PREDICTION_COLUMNS
from src.models.final.model_builders import final_tuned_params_sha256
from src.models.final.position_regex_lightgbm import (
    FINAL_MODEL_NAME,
    POSITION_DISPLAY_NAMES,
    POSITION_FEATURES,
    extract_position_features,
    fit_final_model,
    prepare_live_features,
    predict_total_price,
)
from src.models.final.trimmed_market import (
    SUPPORTED_TRIM_LEVELS,
    fit_market_scope_models,
    fit_trimmed_market_model,
    get_trim_market_metadata,
)
from prototype.eda_page import EDA_VISUALIZATIONS, render_eda_page


DATA_PATH = PROJECT_ROOT / "data" / "processed" / "enhanced_city_dataset.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "final_models"
COMPARISON_PATH = RESULTS_DIR / "model_comparison.json"
OOF_PATH = RESULTS_DIR / "oof_predictions.csv"
IMPORTANCE_PATH = RESULTS_DIR / "feature_importance.csv"
TRIMMING_RESULTS_DIR = PROJECT_ROOT / "results" / "outlier_trimming"
TRIMMING_METADATA_PATH = TRIMMING_RESULTS_DIR / "metadata.json"
ALL_MODELS_TRIMMING_PATH = (
    TRIMMING_RESULTS_DIR / "all_models_trimmed_market_summary.csv"
)
ALL_MODELS_TRIMMING_OOF_PATH = (
    TRIMMING_RESULTS_DIR / "all_models_trimmed_market_oof.csv"
)
TUNING_DIR = PROJECT_ROOT / "results" / "tuning"
TUNING_METADATA_PATH = TUNING_DIR / "metadata.json"
TUNING_SUMMARY_PATH = TUNING_DIR / "tuning_summary.csv"
TUNING_CANDIDATES_PATH = TUNING_DIR / "tuning_candidates.csv"
TUNED_CV_RESULTS_PATH = TUNING_DIR / "tuned_cv_results.csv"
MARKET_SCOPE_OPTIONS = tuple(f"{level:g}%" for level in SUPPORTED_TRIM_LEVELS)
OVERVIEW_VIEW = "\U0001f3e0 Overview"
EDA_VIEW = "\U0001f4ca Data & EDA"
EVALUATION_VIEW = "\U0001f4c8 Model Evaluation"
DIAGNOSTICS_VIEW = "\U0001f50d Model Diagnostics"
OUTLIER_VIEW = "\U0001f9ea Outlier Study"
PREDICTOR_VIEW = "\U0001f3e1 Price Predictor"
VIEWS = (
    OVERVIEW_VIEW,
    EDA_VIEW,
    EVALUATION_VIEW,
    DIAGNOSTICS_VIEW,
    OUTLIER_VIEW,
    PREDICTOR_VIEW,
)
FURNISHING_STATUS_VALUES = {"Unfurnished": 0, "Furnished": 1}
RENOVATION_STATUS_VALUES = {"Not Renovated": 0, "Renovated": 1}
MODEL_COLORS = {
    "Ridge Regression": "#2563eb",
    "Random Forest": "#d97706",
    "Gradient Boosting": "#db2777",
    FINAL_MODEL_NAME: "#4d7c0f",
}
COMPARISON_METRICS = {
    "RMSE": ("RMSE_RM", "RMSE", "RMSE (RM)", ",.0f"),
    "MAE": ("MAE_RM", "MAE", "MAE (RM)", ",.0f"),
    "R²": ("R2", "R²", "R²", ".4f"),
    "Adjusted R²": ("Adjusted_R2", "Adjusted R²", "Adjusted R²", ".4f"),
}

# ASCII source escapes keep the labels stable on Windows checkouts regardless
# of the active console code page.
COMPARISON_METRICS = {
    "RMSE": ("RMSE_RM", "RMSE", "RMSE (RM)", ",.0f"),
    "MAE": ("MAE_RM", "MAE", "MAE (RM)", ",.0f"),
    "R\u00b2": ("R2", "R\u00b2", "R\u00b2", ".4f"),
    "Adjusted R\u00b2": (
        "Adjusted_R2",
        "Adjusted R\u00b2",
        "Adjusted R\u00b2",
        ".4f",
    ),
}

# Re-declare with presentation-safe Unicode labels; the historical file passed
# through a legacy Windows encoding and its two R-squared labels were damaged.
COMPARISON_METRICS = {
    "RMSE": ("RMSE_RM", "RMSE", "RMSE (RM)", ",.0f"),
    "MAE": ("MAE_RM", "MAE", "MAE (RM)", ",.0f"),
    "R²": ("R2", "R²", "R²", ".4f"),
    "Adjusted R²": ("Adjusted_R2", "Adjusted R²", "Adjusted R²", ".4f"),
}

# This final assignment intentionally follows the legacy declarations above.
COMPARISON_METRICS = {
    "RMSE": ("RMSE_RM", "RMSE", "RMSE (RM)", ",.0f"),
    "MAE": ("MAE_RM", "MAE", "MAE (RM)", ",.0f"),
    "R\u00b2": ("R2", "R\u00b2", "R\u00b2", ".4f"),
    "Adjusted R\u00b2": ("Adjusted_R2", "Adjusted R\u00b2", "Adjusted R\u00b2", ".4f"),
}


@st.cache_data
def load_dataset() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH).reset_index(drop=True)
    required = {"listing_id", "price", *MODEL_FEATURES}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Canonical dataset is missing required columns: {missing}")
    if len(data) != 3_791 or data["listing_id"].nunique() != 3_791:
        raise ValueError("Canonical dataset must contain 3,791 unique listings.")
    return data


@st.cache_data
def load_comparison() -> dict:
    payload = json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))
    names = tuple(row["Model"] for row in payload["models"])
    if set(names) != set(FINAL_MODELS) or len(names) != len(FINAL_MODELS):
        raise ValueError("Saved comparison does not contain exactly the four final models.")
    if payload["selected_final_model"] != FINAL_MODEL_NAME:
        raise ValueError("Saved comparison has an unexpected selected final model.")
    return payload


@st.cache_data
def load_oof_predictions() -> pd.DataFrame:
    oof = pd.read_csv(OOF_PATH)
    required = {
        "listing_id",
        "actual_price",
        "scenario_b_fold",
        *PREDICTION_COLUMNS.values(),
    }
    missing = sorted(required.difference(oof.columns))
    if missing:
        raise ValueError(f"Saved OOF predictions are missing columns: {missing}")
    if len(oof) != 3_791 or oof["listing_id"].nunique() != 3_791:
        raise ValueError("Saved OOF predictions must cover all 3,791 listings exactly once.")
    return oof


@st.cache_data
def load_feature_importance() -> pd.DataFrame:
    importance = pd.read_csv(IMPORTANCE_PATH)
    required = {"Model", "Feature", "Raw_Feature", "Importance", "Importance_Type"}
    missing = sorted(required.difference(importance.columns))
    if missing:
        raise ValueError(f"Saved feature importance is missing columns: {missing}")
    if set(importance["Model"]) != set(FINAL_MODELS):
        raise ValueError("Saved feature importance does not cover all final models.")
    return importance


@st.cache_data
def load_trimming_results() -> tuple[dict, dict[str, pd.DataFrame]]:
    """Load presentation-only copies of the completed trimming experiment."""
    metadata = json.loads(TRIMMING_METADATA_PATH.read_text(encoding="utf-8"))
    filenames = (
        "training_only_comparison.csv",
        "trimmed_population_comparison.csv",
        "segment_metrics.csv",
        "distribution_shift.csv",
        "bootstrap_results.csv",
        "retained_cv_summary.csv",
    )
    tables = {
        filename.removesuffix(".csv"): pd.read_csv(TRIMMING_RESULTS_DIR / filename)
        for filename in filenames
    }
    expected_levels = [0.0, 0.5, 1.0, 2.5, 5.0, 10.0]
    levels = sorted(tables["training_only_comparison"]["Removal_Percent"].unique())
    if levels != expected_levels:
        raise ValueError("Saved trimming results do not contain the six expected levels.")
    if metadata.get("recommended_trimming") not in MARKET_SCOPE_OPTIONS:
        raise ValueError("Saved trimming recommendation is not a supported level.")
    if metadata.get("production_model") != FINAL_MODEL_NAME:
        raise ValueError("Saved trimming results reference an unexpected production model.")
    retained_cv = tables["retained_cv_summary"]
    retained_required = {
        "trim_level",
        "fold",
        "original_rows",
        "retained_rows",
        "removed_rows",
        "training_rows",
        "validation_rows",
        "retention_percentage",
    }
    if not retained_required.issubset(retained_cv.columns):
        raise ValueError("Saved retained-CV summary has an unexpected schema.")
    if len(retained_cv) != 30 or not (
        retained_cv.groupby("trim_level")["fold"].nunique() == 5
    ).all():
        raise ValueError("Saved retained-CV summary must contain five folds per level.")
    return metadata, tables


@st.cache_data
def load_all_models_trimming_summary() -> pd.DataFrame:
    """Load the validated all-model restricted-market comparison artifact."""
    summary = pd.read_csv(ALL_MODELS_TRIMMING_PATH)
    required = [
        "Model",
        "Trim_Level",
        "Removal_Percent",
        "Original_Rows",
        "Retained_Rows",
        "Removed_Rows",
        "Retention_Percentage",
        "RMSE_RM",
        "MAE_RM",
        "R2",
        "Adjusted_R2",
    ]
    if list(summary.columns) != required:
        raise ValueError("All-model trimming summary has an unexpected schema.")
    expected_levels = ["0%", "0.5%", "1%", "2.5%", "5%", "10%"]
    if len(summary) != 24 or set(summary["Model"]) != set(FINAL_MODELS):
        raise ValueError("All-model trimming summary must contain exactly 24 rows.")
    if summary.duplicated(["Model", "Trim_Level"]).any():
        raise ValueError("All-model trimming summary contains duplicate model/trim pairs.")
    for model_name in FINAL_MODELS:
        levels = summary.loc[
            summary["Model"].eq(model_name), "Trim_Level"
        ].tolist()
        if levels != expected_levels:
            raise ValueError(f"All-model trimming levels are incomplete for {model_name}.")
    if not (
        summary.groupby("Trim_Level", sort=False)["Retained_Rows"].nunique() == 1
    ).all():
        raise ValueError("All models must use identical retained rows per trim level.")
    if not np.isfinite(
        summary[["RMSE_RM", "MAE_RM", "R2", "Adjusted_R2"]].to_numpy(float)
    ).all():
        raise ValueError("All-model trimming summary contains non-finite metrics.")
    return summary


@st.cache_data
def load_all_models_trimming_oof() -> pd.DataFrame:
    """Load and verify retrained OOF predictions for every model/scope pair."""
    oof = pd.read_csv(ALL_MODELS_TRIMMING_OOF_PATH)
    required = [
        "Model",
        "Trim_Level",
        "Removal_Percent",
        "listing_id",
        "scenario_b_fold",
        "actual_price_RM",
        "predicted_price_RM",
    ]
    if list(oof.columns) != required:
        raise ValueError("All-model trimming OOF data has an unexpected schema.")
    if set(oof["Model"]) != set(FINAL_MODELS):
        raise ValueError("All-model trimming OOF data does not cover all models.")
    if oof.duplicated(["Model", "Trim_Level", "listing_id"]).any():
        raise ValueError("All-model trimming OOF data contains duplicate listing predictions.")
    if not np.isfinite(
        oof[["actual_price_RM", "predicted_price_RM"]].to_numpy(float)
    ).all():
        raise ValueError("All-model trimming OOF prices must be finite.")

    summary = load_all_models_trimming_summary().set_index(["Model", "Trim_Level"])
    for key, rows in oof.groupby(["Model", "Trim_Level"], sort=False):
        if key not in summary.index:
            raise ValueError(f"OOF data has an unexpected model/scope pair: {key}.")
        saved = summary.loc[key]
        actual = rows["actual_price_RM"].to_numpy(float)
        predicted = rows["predicted_price_RM"].to_numpy(float)
        residual = predicted - actual
        rmse = float(np.sqrt(np.mean(np.square(residual))))
        mae = float(np.mean(np.abs(residual)))
        r2 = float(1.0 - np.square(residual).sum() / np.square(actual - actual.mean()).sum())
        if len(rows) != int(saved["Retained_Rows"]):
            raise ValueError(f"OOF row count does not match saved metrics for {key}.")
        if not np.allclose(
            [rmse, mae, r2],
            [saved["RMSE_RM"], saved["MAE_RM"], saved["R2"]],
            rtol=1e-10,
            atol=[1e-6, 1e-6, 1e-12],
        ):
            raise ValueError(f"OOF predictions do not reproduce saved metrics for {key}.")
    if len(summary) != oof.groupby(["Model", "Trim_Level"]).ngroups:
        raise ValueError("All-model trimming OOF data is missing model/scope pairs.")
    return oof


@st.cache_resource
def _load_deployment_model_cached(model_config_hash: str):
    """Fit only the selected final model, once per Streamlit process."""
    if model_config_hash != final_tuned_params_sha256():
        raise ValueError("Deployment cache fingerprint does not match tuned parameters.")
    model = fit_final_model()
    if model.training_rows_ != 3_791:
        raise AssertionError("Deployment model did not train on all canonical rows.")
    return model


def load_deployment_model():
    return _load_deployment_model_cached(final_tuned_params_sha256())


@st.cache_resource
def _load_trimmed_deployment_model_cached(
    trim_level: float,
    model_config_hash: str,
):
    """Fit and cache one experimental deployment model per saved trim level."""
    if model_config_hash != final_tuned_params_sha256():
        raise ValueError("Trimmed-model cache fingerprint does not match tuned parameters.")
    model = fit_trimmed_market_model(trim_level)
    metadata = get_trim_market_metadata(trim_level)
    if model.training_rows_ != metadata["retained_rows"]:
        raise AssertionError("Trimmed deployment model used an unexpected row count.")
    return model


def load_trimmed_deployment_model(trim_level: float):
    return _load_trimmed_deployment_model_cached(
        trim_level, final_tuned_params_sha256()
    )


@st.cache_resource
def _load_scope_models_cached(
    scope: str,
    model_config_hash: str,
) -> dict[str, object]:
    """Fit and cache the four deployment families for one saved scope."""
    if model_config_hash != final_tuned_params_sha256():
        raise ValueError("Scope-model cache fingerprint does not match tuned parameters.")
    if scope not in MARKET_SCOPE_OPTIONS:
        raise ValueError(f"Market scope must be one of: {', '.join(MARKET_SCOPE_OPTIONS)}.")
    models = fit_market_scope_models(float(scope.removesuffix("%")))
    if tuple(models) != FINAL_MODELS:
        raise AssertionError("Scope model loader did not return the four final families.")
    return models


def load_scope_models(scope: str) -> dict[str, object]:
    """Load one cached scope using the current tuned-configuration fingerprint."""
    return _load_scope_models_cached(scope, final_tuned_params_sha256())


def scope_comparison_frame(summary: pd.DataFrame, scope: str) -> pd.DataFrame:
    """Return the four saved Scenario B metric rows for one selected scope."""
    if scope not in MARKET_SCOPE_OPTIONS:
        raise ValueError(f"Unknown market scope: {scope}")
    rows = summary.loc[summary["Trim_Level"].eq(scope)].copy()
    if len(rows) != len(FINAL_MODELS) or set(rows["Model"]) != set(FINAL_MODELS):
        raise ValueError(f"Saved validation results are incomplete for {scope} scope.")
    return rows.set_index("Model").loc[list(FINAL_MODELS)].reset_index()


def recommended_model_for_scope(summary: pd.DataFrame, scope: str) -> str:
    """Select the saved lowest-RMSE model for the current scope."""
    rows = scope_comparison_frame(summary, scope)
    ranked = rows.sort_values(
        ["RMSE_RM", "MAE_RM", "R2", "Adjusted_R2"],
        ascending=[True, True, False, False],
        kind="stable",
    )
    return str(ranked.iloc[0]["Model"])


def scope_display_frame(summary: pd.DataFrame, scope: str) -> pd.DataFrame:
    rows = scope_comparison_frame(summary, scope)
    return rows.rename(
        columns={
            "RMSE_RM": "RMSE",
            "MAE_RM": "MAE",
            "R2": "R²",
            "Adjusted_R2": "Adjusted R²",
        }
    ).loc[:, ["Model", "RMSE", "MAE", "R²", "Adjusted R²"]]


def normalized_scope_display_frame(summary: pd.DataFrame, scope: str) -> pd.DataFrame:
    """Presentation-safe metric table with stable R-squared labels."""
    rows = scope_comparison_frame(summary, scope)
    r2_label = "R\u00b2"
    adjusted_label = "Adjusted R\u00b2"
    return rows.rename(
        columns={
            "RMSE_RM": "RMSE",
            "MAE_RM": "MAE",
            "R2": r2_label,
            "Adjusted_R2": adjusted_label,
        }
    ).loc[:, ["Model", "RMSE", "MAE", r2_label, adjusted_label]]


def predict_scope_model(
    model_name: str,
    model: object,
    values: dict,
    description: str,
    description_length_fallback: float,
) -> dict[str, float | dict[str, bool]]:
    """Generate a common total-price response for any final model family."""
    if model_name == FINAL_MODEL_NAME:
        return predict_total_price(
            model,
            values,
            description,
            description_length_fallback,
        )
    structured, _, detected = prepare_live_features(
        values,
        description,
        description_length_fallback,
    )
    total_price = float(model.predict(structured)[0])
    size = float(structured.iloc[0]["property_size_sqft"])
    if not np.isfinite(total_price) or total_price <= 0:
        raise ValueError(f"{model_name} returned a non-positive or non-finite estimate.")
    return {
        "total_price_RM": total_price,
        "ppsf_RM": total_price / size,
        "detected_position_features": detected,
        "description_length": float(structured.iloc[0]["description_length"]),
    }


def prediction_comparison_frame(
    selected_predictions: dict[str, dict],
    full_market_predictions: dict[str, dict],
) -> pd.DataFrame:
    """Build the required four-model selected/full-market prediction table."""
    rows = []
    for model_name in FINAL_MODELS:
        selected = float(selected_predictions[model_name]["total_price_RM"])
        full = float(full_market_predictions[model_name]["total_price_RM"])
        difference = selected - full
        rows.append(
            {
                "Model": model_name,
                "Selected-Scope Prediction": selected,
                "Full-Market Prediction": full,
                "Difference (RM)": difference,
                "Difference (%)": 100.0 * difference / full,
            }
        )
    return pd.DataFrame(rows)


def _display_value(value) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join("None" if item is None else str(item) for item in value)
    if value is None:
        return "None"
    return str(value)


@st.cache_data
def load_tuning_details() -> dict[str, dict]:
    """Load current formal Scenario B tuning evidence for all four models."""
    metadata = json.loads(TUNING_METADATA_PATH.read_text(encoding="utf-8"))
    summary = pd.read_csv(TUNING_SUMMARY_PATH).set_index("Model")
    candidates = pd.read_csv(TUNING_CANDIDATES_PATH)
    selected = pd.read_csv(TUNED_CV_RESULTS_PATH).set_index("Model")
    if metadata.get("status") != "complete" or metadata.get("folds") != 5:
        raise ValueError("Current tuning metadata is incomplete or has an unexpected fold count.")
    if metadata.get("frozen_configuration_sha256") != final_tuned_params_sha256():
        raise ValueError("Current tuning evidence does not match the frozen model configuration.")
    if set(summary.index) != set(FINAL_MODELS) or set(selected.index) != set(FINAL_MODELS):
        raise ValueError("Current tuning artifacts do not cover the four submitted models.")
    if set(candidates["Model"]) != set(FINAL_MODELS):
        raise ValueError("Current tuning candidates do not cover the four submitted models.")
    details: dict[str, dict] = {}
    for model_name in FINAL_MODELS:
        tuned_parameters = metadata["selected_parameters"][model_name]
        search_space = metadata["search_spaces"][model_name]
        summary_row = summary.loc[model_name]
        selected_row = selected.loc[model_name]
        search_rows = [
            {
                "Hyperparameter": parameter,
                "Values Tested": _display_value(values),
                "Selected Value": _display_value(tuned_parameters[parameter]),
            }
            for parameter, values in search_space.items()
        ]
        final_rows = [
            {
                "Hyperparameter": parameter,
                "Final Value": _display_value(value),
                "Status": "Selected by current formal tuning",
            }
            for parameter, value in tuned_parameters.items()
        ]
        before_after = pd.DataFrame(
            [
                {
                    "Metric": "RMSE",
                    "Before": summary_row["Pre_Tuning_RMSE_RM"],
                    "After": summary_row["Tuned_RMSE_RM"],
                    "Change": summary_row["Tuned_RMSE_RM"] - summary_row["Pre_Tuning_RMSE_RM"],
                },
                {
                    "Metric": "MAE",
                    "Before": summary_row["Pre_Tuning_MAE_RM"],
                    "After": summary_row["Tuned_MAE_RM"],
                    "Change": summary_row["Tuned_MAE_RM"] - summary_row["Pre_Tuning_MAE_RM"],
                },
                {
                    "Metric": "R²",
                    "Before": summary_row["Pre_Tuning_R2"],
                    "After": summary_row["Tuned_R2"],
                    "Change": summary_row["R2_Change"],
                },
                {
                    "Metric": "Adjusted R²",
                    "Before": summary_row["Pre_Tuning_Adjusted_R2"],
                    "After": summary_row["Tuned_Adjusted_R2"],
                    "Change": summary_row["Adjusted_R2_Change"],
                },
            ]
        )
        details[model_name] = {
            "search_space": pd.DataFrame(search_rows),
            "final_parameters": pd.DataFrame(final_rows),
            "before_after": before_after,
            "method": (
                f"{metadata['search_method'][model_name]} with "
                f"{int(metadata['candidate_counts'][model_name])} candidate configurations."
            ),
            "validation": metadata["validation"],
            "scaling": metadata["scaling_policy"][model_name],
            "candidate_count": int(metadata["candidate_counts"][model_name]),
            "tuned_metrics": {
                "RMSE_RM": float(selected_row["CV_RMSE_RM"]),
                "MAE_RM": float(selected_row["CV_MAE_RM"]),
                "R2": float(selected_row["CV_R2"]),
                "Adjusted_R2": float(selected_row["CV_Adjusted_R2"]),
            },
        }
    return details


def comparison_frame(payload: dict) -> pd.DataFrame:
    frame = pd.DataFrame(payload["models"]).loc[
        :, [
            "Model",
            "RMSE_RM",
            "MAE_RM",
            "R2",
            "Adjusted_R2",
        ]
    ]
    return frame.set_index("Model").loc[list(FINAL_MODELS)].reset_index()


def render_overview(data: pd.DataFrame) -> None:
    """Orient users to the project without disclosing later analytical conclusions."""
    st.header("Malaysian Residential Property Price Prediction")
    st.write(
        "A machine-learning project for estimating residential property listing prices "
        "across Malaysia using structural, location, classification and amenity information."
    )

    st.write("### Project at a Glance")
    headline = st.columns(4)
    headline[0].metric("Prepared Listings", f"{len(data):,}")
    headline[1].metric("Prepared Features", f"{data.shape[1]:,}")
    headline[2].metric("Problem Type", "Regression")
    headline[3].metric("Target Variable", "Listing Price (RM)")

    st.write("### Project Objective")
    st.write(
        "Develop and evaluate multiple machine-learning models to estimate Malaysian "
        "residential property listing prices using available property characteristics."
    )
    st.write(
        "The prototype allows users to explore the dataset, compare models, investigate "
        "prediction behaviour and generate an estimated listing price for a property."
    )

    st.write("### What Does the Dataset Contain?")
    feature_groups = st.columns(4)
    for column, title, features in zip(
        feature_groups,
        ("Property Details", "Location", "Classification", "Amenities"),
        (
            ("Property Size", "Bedrooms", "Bathrooms", "Parking", "Completion Year"),
            ("State", "City / Locality", "Building", "Developer"),
            ("Property Type", "Tenure", "Land Title", "Floor Range"),
            ("Schools", "Malls", "Hospitals", "Transport", "Property Facilities"),
        ),
    ):
        column.markdown(f"#### {title}")
        column.markdown("\n".join(f"- {feature}" for feature in features))

    st.write("### Project Workflow")
    workflow = st.columns(5)
    for column, number, title, description in zip(
        workflow,
        ("01", "02", "03", "04", "05"),
        ("Data Understanding", "Data Preparation", "Modelling", "Evaluation", "Deployment"),
        (
            "Inspect property listings and identify useful variables.",
            "Clean, transform and prepare modelling features.",
            "Train and tune multiple regression algorithms.",
            "Compare predictive performance and model behaviour.",
            "Integrate the trained approach into an interactive Streamlit prototype.",
        ),
    ):
        column.caption(number)
        column.markdown(f"**{title}**")
        column.write(description)

    st.write("### Explore the Project")
    destinations = (
        ("📊 Data & EDA", "Explore patterns, distributions and relationships in the prepared dataset."),
        ("📈 Model Evaluation", "Compare the predictive performance of the regression models."),
        ("🔍 Model Diagnostics", "Inspect feature importance and actual-versus-predicted behaviour."),
        ("🧪 Outlier Study", "Investigate the effect of premium listings and trimming experiments."),
        ("🏡 Price Predictor", "Enter property information and generate an estimated listing price."),
    )
    first_row = st.columns(3)
    second_row = st.columns(2)
    for column, (title, description) in zip((*first_row, *second_row), destinations):
        column.markdown(f"#### {title}")
        column.write(description)


def comparison_display_frame(payload: dict) -> pd.DataFrame:
    return comparison_frame(payload).rename(
        columns={
            "RMSE_RM": "RMSE",
            "MAE_RM": "MAE",
            "R2": "R²",
            "Adjusted_R2": "Adjusted R²",
        }
    )


def category_values(data: pd.DataFrame, column: str) -> list[str]:
    values = data[column].dropna().astype(str).str.strip()
    choices = sorted(value for value in values.unique() if value)
    return choices or ["Unknown"]


def condition_feature_values(
    furnishing_status: str,
    renovation_status: str,
) -> dict[str, int]:
    """Map the two independent structured status controls to model features."""
    if furnishing_status not in FURNISHING_STATUS_VALUES:
        raise ValueError(f"Unknown furnishing status: {furnishing_status}")
    if renovation_status not in RENOVATION_STATUS_VALUES:
        raise ValueError(f"Unknown renovation status: {renovation_status}")
    return {
        "is_furnished": FURNISHING_STATUS_VALUES[furnishing_status],
        "is_renovated": RENOVATION_STATUS_VALUES[renovation_status],
    }


def build_official_metric_chart(table: pd.DataFrame, metric_name: str) -> go.Figure:
    """Build one full-width official comparison chart from saved metrics."""
    metric, title_metric, y_label, text_format = COMPARISON_METRICS[metric_name]
    figure = px.bar(
        table,
        x="Model",
        y=metric,
        text_auto=text_format,
        color="Model",
        color_discrete_map=MODEL_COLORS,
        category_orders={"Model": list(FINAL_MODELS)},
        labels={metric: y_label},
        title=f"{title_metric} by Model",
    )
    figure.update_layout(showlegend=False, height=620)
    figure.update_yaxes(rangemode="tozero", tickformat=text_format)
    return figure


def render_comparison(payload: dict) -> None:
    st.subheader("Final Four-Model Comparison")
    table = comparison_frame(payload)
    selected_metric = st.selectbox(
        "Select Evaluation Metric",
        list(COMPARISON_METRICS),
        key="model_comparison_metric",
    )
    st.plotly_chart(
        build_official_metric_chart(table, selected_metric),
        width="stretch",
    )

    st.subheader("Detailed Comparison Table")
    display = comparison_display_frame(payload)
    styled = (
        display.style.format(
            {
                "RMSE": "RM {:,.0f}",
                "MAE": "RM {:,.0f}",
                "R²": "{:.4f}",
                "Adjusted R²": "{:.4f}",
            }
        )
        .highlight_min(subset=["RMSE", "MAE"], color="#d1fae5")
        .highlight_max(subset=["R²", "Adjusted R²"], color="#d1fae5")
    )
    st.dataframe(styled, width="stretch", hide_index=True)


def render_scope_comparison() -> None:
    """Render one scope-selected four-model chart, table, and tuning panel."""
    st.subheader("Model Comparison")
    summary = load_all_models_trimming_summary()
    selected_scope = st.selectbox(
        "Select Market Scope",
        list(MARKET_SCOPE_OPTIONS),
        index=list(MARKET_SCOPE_OPTIONS).index("10%"),
        key="comparison_market_scope",
    )
    selected_metric = st.selectbox(
        "Select Evaluation Metric",
        list(COMPARISON_METRICS),
        key="model_comparison_metric",
    )
    table = scope_comparison_frame(summary, selected_scope)
    st.plotly_chart(
        build_official_metric_chart(table, selected_metric),
        width="stretch",
    )

    st.subheader("Final Model Performance")
    display = scope_display_frame(summary, selected_scope)
    styled = (
        display.style.format(
            {
                "RMSE": "RM {:,.0f}",
                "MAE": "RM {:,.0f}",
                "R²": "{:.4f}",
                "Adjusted R²": "{:.4f}",
            }
        )
        .highlight_min(subset=["RMSE", "MAE"], color="#d1fae5")
        .highlight_max(subset=["R²", "Adjusted R²"], color="#d1fae5")
    )
    st.dataframe(styled, width="stretch", hide_index=True)
    recommended = recommended_model_for_scope(summary, selected_scope)
    st.caption(
        f"Recommended for {selected_scope} scope: {recommended} "
        "(lowest saved validation RMSE; MAE, R², and adjusted R² support tie-breaking)."
    )

    st.divider()
    st.subheader("Hyperparameter Tuning")
    selected_model = st.selectbox(
        "Select Model for Hyperparameter Details",
        list(FINAL_MODELS),
        index=list(FINAL_MODELS).index(FINAL_MODEL_NAME),
        key="tuning_model",
    )
    tuning = load_tuning_details()[selected_model]
    st.write("#### Hyperparameter Search Space")
    if tuning["search_space"].empty:
        st.info(
            "No formal search-space artifact is saved for this model. "
            "The fixed final configuration is shown below without claiming a tuning search."
        )
    else:
        st.dataframe(tuning["search_space"], width="stretch", hide_index=True)
        st.caption(
            "The selected values come from the current formal Scenario B tuning artifacts "
            "and are used by the final model builders."
        )
    st.write("#### Selected / Final Hyperparameters")
    st.dataframe(tuning["final_parameters"], width="stretch", hide_index=True)
    st.write("#### Tuning Method")
    st.write(tuning["method"])
    st.write("#### Validation Method")
    st.write(tuning["validation"])
    st.write("#### Before vs After Hyperparameter Tuning")
    st.dataframe(tuning["before_after"], width="stretch", hide_index=True)


def render_scope_comparison_v2() -> None:
    """Encoding-stable implementation of the scope comparison page."""
    st.subheader("Model Comparison")
    summary = load_all_models_trimming_summary()
    selected_scope = st.selectbox(
        "Select Market Scope",
        list(MARKET_SCOPE_OPTIONS),
        index=list(MARKET_SCOPE_OPTIONS).index("10%"),
        key="comparison_market_scope",
    )
    selected_metric = st.selectbox(
        "Select Evaluation Metric",
        list(COMPARISON_METRICS),
        key="model_comparison_metric",
    )
    table = scope_comparison_frame(summary, selected_scope)
    st.plotly_chart(build_official_metric_chart(table, selected_metric), width="stretch")

    st.subheader("Final Model Performance")
    display = normalized_scope_display_frame(summary, selected_scope)
    r2_label = "R\u00b2"
    adjusted_label = "Adjusted R\u00b2"
    styled = (
        display.style.format(
            {
                "RMSE": "RM {:,.0f}",
                "MAE": "RM {:,.0f}",
                r2_label: "{:.4f}",
                adjusted_label: "{:.4f}",
            }
        )
        .highlight_min(subset=["RMSE", "MAE"], color="#d1fae5")
        .highlight_max(subset=[r2_label, adjusted_label], color="#d1fae5")
    )
    st.dataframe(styled, width="stretch", hide_index=True)
    recommended = recommended_model_for_scope(summary, selected_scope)
    st.caption(
        f"Recommended for {selected_scope} scope: {recommended} "
        "(lowest saved validation RMSE; MAE and R-squared metrics support tie-breaking)."
    )

    st.divider()
    st.subheader("Hyperparameter Tuning")
    selected_model = st.selectbox(
        "Select Model for Hyperparameter Details",
        list(FINAL_MODELS),
        index=list(FINAL_MODELS).index(FINAL_MODEL_NAME),
        key="tuning_model",
    )
    tuning = load_tuning_details()[selected_model]
    st.write("#### Hyperparameter Search Space")
    if tuning["search_space"].empty:
        st.info(
            "No formal search-space artifact is saved for this model. "
            "The fixed final configuration is shown below without claiming a tuning search."
        )
    else:
        st.dataframe(tuning["search_space"], width="stretch", hide_index=True)
        st.caption(
            "Selected values come from the current formal Scenario B tuning artifacts "
            "and are used by the final model builders."
        )
    st.write("#### Selected / Final Hyperparameters")
    st.dataframe(tuning["final_parameters"], width="stretch", hide_index=True)
    st.write("#### Tuning Method")
    st.write(tuning["method"])
    st.write("#### Validation Method")
    st.write(tuning["validation"])
    st.write("#### Before vs After Hyperparameter Tuning")
    st.dataframe(tuning["before_after"], width="stretch", hide_index=True)


def render_model_evaluation(selected_scope: str, selected_metric: str) -> None:
    """Render selected-scope metrics, comparison, and tabbed supporting evidence."""
    st.header("Model Evaluation")
    st.caption("Which model performs best for the selected experimental market scope?")
    summary = load_all_models_trimming_summary()
    rows = scope_comparison_frame(summary, selected_scope)
    recommended = recommended_model_for_scope(summary, selected_scope)
    recommended_row = rows.set_index("Model").loc[recommended]

    st.write("### Recommended Model for Selected Scope")
    st.success(f"{recommended}  |  {selected_scope} market scope")
    cards = st.columns(3)
    cards[0].metric("RMSE", f"RM {recommended_row['RMSE_RM']:,.0f}")
    cards[1].metric("MAE", f"RM {recommended_row['MAE_RM']:,.0f}")
    cards[2].metric("R\u00b2", f"{recommended_row['R2']:.4f}")
    st.plotly_chart(
        build_official_metric_chart(rows, selected_metric),
        width="stretch",
    )

    performance_tab, tuning_tab = st.tabs(
        ["Performance Details", "Hyperparameter Tuning"]
    )
    with performance_tab:
        r2_label = "R\u00b2"
        adjusted_label = "Adjusted R\u00b2"
        display = rows.rename(
            columns={
                "Retained_Rows": "Retained Listings",
                "RMSE_RM": "RMSE",
                "MAE_RM": "MAE",
                "R2": r2_label,
                "Adjusted_R2": adjusted_label,
            }
        ).loc[
            :, ["Model", "Retained Listings", "RMSE", "MAE", r2_label, adjusted_label]
        ]
        styled = (
            display.style.format(
                {
                    "Retained Listings": "{:,}",
                    "RMSE": "RM {:,.0f}",
                    "MAE": "RM {:,.0f}",
                    r2_label: "{:.4f}",
                    adjusted_label: "{:.4f}",
                }
            )
            .highlight_min(subset=["RMSE", "MAE"], color="#d1fae5")
            .highlight_max(subset=[r2_label, adjusted_label], color="#d1fae5")
        )
        st.dataframe(styled, width="stretch", hide_index=True)

    with tuning_tab:
        selected_model = st.selectbox(
            "Model for Hyperparameter Details",
            list(FINAL_MODELS),
            index=list(FINAL_MODELS).index(FINAL_MODEL_NAME),
            key="tuning_model",
        )
        tuning = load_tuning_details()[selected_model]
        st.info(f"Scaling: {tuning['scaling']}")
        st.caption(
            f"Candidate configurations evaluated: {tuning['candidate_count']}. "
            "Primary selection metric: reconstructed total-price RMSE."
        )
        st.write("#### Hyperparameter Search Space")
        st.dataframe(tuning["search_space"], width="stretch", hide_index=True)
        st.write("#### Selected / Final Hyperparameters")
        st.dataframe(tuning["final_parameters"], width="stretch", hide_index=True)
        st.write("#### Before vs After Hyperparameter Tuning")
        before_after = tuning["before_after"].copy()
        before_after[["Before", "After", "Change"]] = before_after[
            ["Before", "After", "Change"]
        ].astype(object)
        for index, row in before_after.iterrows():
            if row["Metric"] in {"RMSE", "MAE"}:
                before_after.loc[index, "Before"] = f"RM {row['Before']:,.0f}"
                before_after.loc[index, "After"] = f"RM {row['After']:,.0f}"
                before_after.loc[index, "Change"] = f"RM {row['Change']:+,.0f}"
            else:
                before_after.loc[index, "Before"] = f"{row['Before']:.4f}"
                before_after.loc[index, "After"] = f"{row['After']:.4f}"
                before_after.loc[index, "Change"] = f"{row['Change']:+.4f}"
        st.dataframe(before_after, width="stretch", hide_index=True)
        st.caption(
            "For RMSE and MAE, a negative change is better. For R² and adjusted R², "
            "a positive change is better."
        )
        st.write("#### Tuning Method")
        st.write(tuning["method"])
        st.write("#### Validation Method")
        st.write(tuning["validation"])


def render_feature_importance(
    selected: str = FINAL_MODEL_NAME,
    top_n: int = 10,
) -> None:
    st.write("### Feature Importance")
    importance = load_feature_importance()
    if selected not in FINAL_MODELS:
        raise ValueError(f"Unknown feature-importance model: {selected}")
    if top_n not in {10, 15, 20}:
        raise ValueError("Top Features must be one of: 10, 15, 20.")
    rows = importance[importance["Model"] == selected]
    importance_type = rows["Importance_Type"].iloc[0]
    top = rows.nlargest(top_n, "Importance").sort_values("Importance")
    title = (
        f"Top {top_n} Features — {selected}"
        if selected != "Ridge Regression"
        else f"Top {top_n} Absolute Coefficient Magnitudes — Ridge Regression"
    )
    figure = px.bar(
        top,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Viridis",
        title=title,
    )
    figure.update_layout(height=max(380, 28 * top_n + 120), coloraxis_showscale=False)
    st.plotly_chart(figure, width="stretch")
    st.caption(f"Measure shown: {importance_type}.")
    if selected == FINAL_MODEL_NAME:
        position_rows = rows[rows["Raw_Feature"].isin(POSITION_FEATURES)].copy()
        if len(position_rows) != len(POSITION_FEATURES):
            raise AssertionError("Saved final-model importance is missing position features.")
        with st.expander("Position Features Used by the Final Model"):
            st.dataframe(
                position_rows[["Feature", "Importance"]].sort_values(
                    "Importance", ascending=False
                ),
                width="stretch",
                hide_index=True,
            )


def render_actual_vs_predicted(
    payload: dict,
    selected: str = FINAL_MODEL_NAME,
    selected_trim: str = "10%",
) -> None:
    st.write(f"### Actual vs Predicted \u2014 {selected}")
    st.caption("Scenario B saved out-of-fold predictions")
    try:
        summary = load_all_models_trimming_summary()
        scope_rows = scope_comparison_frame(summary, selected_trim).set_index("Model")
        if selected not in scope_rows.index:
            raise ValueError(
                f"Saved trimming results do not contain {selected} at {selected_trim}."
            )
        metrics = scope_rows.loc[selected]
        plot, metadata = actual_vs_predicted_plot_frame(
            load_all_models_trimming_oof(),
            selected,
            selected_trim,
        )
    except (AssertionError, KeyError, ValueError) as error:
        st.error(str(error))
        return
    cards = st.columns(3)
    cards[0].metric("RMSE", f"RM {metrics['RMSE_RM']:,.0f}")
    cards[1].metric("MAE", f"RM {metrics['MAE_RM']:,.0f}")
    cards[2].metric("R²", f"{metrics['R2']:.4f}")

    figure = px.scatter(
        plot,
        x="Actual Price (RM)",
        y="OOF Predicted Price (RM)",
        hover_data=["listing_id", "scenario_b_fold"],
        opacity=0.55,
        title=(
            f"{selected}: Actual vs OOF Predicted Price \u2014 "
            f"{selected_trim} Trimming"
        ),
    )
    lower = float(
        min(plot["Actual Price (RM)"].min(), plot["OOF Predicted Price (RM)"].min())
    )
    upper = float(
        max(plot["Actual Price (RM)"].max(), plot["OOF Predicted Price (RM)"].max())
    )
    figure.add_trace(
        go.Scatter(
            x=[lower, upper],
            y=[lower, upper],
            mode="lines",
            name="Ideal Fit (y=x)",
            line={"dash": "dash", "color": "red"},
        )
    )
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "Every point is a saved Scenario B out-of-fold prediction from the matching "
        f"{selected_trim} retrained restricted-market experiment. "
        f"Displayed listings: {len(plot):,}."
    )
    with st.expander("Market Scope Details"):
        scope_details = pd.DataFrame(
            {
                "Original Listings": [metadata["original_rows"]],
                "Retained Listings": [metadata["retained_rows"]],
                "Removed Listings": [metadata["removed_rows"]],
                "Retention Percentage": [metadata["retention_percentage"]],
                "Maximum Retained Actual Price": [
                    metadata["maximum_retained_price_RM"]
                ],
            }
        )
        st.dataframe(
            scope_details.style.format(
                {
                    "Original Listings": "{:,}",
                    "Retained Listings": "{:,}",
                    "Removed Listings": "{:,}",
                    "Retention Percentage": "{:.2f}%",
                    "Maximum Retained Actual Price": "RM {:,.0f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )


def actual_vs_predicted_plot_frame(
    oof: pd.DataFrame,
    model_name: str,
    trim_level: str,
) -> tuple[pd.DataFrame, dict[str, float | int | str | None]]:
    """Return matching retrained restricted-market OOF points for one scope."""
    if model_name not in FINAL_MODELS:
        raise ValueError(f"Unknown model for saved OOF predictions: {model_name}")
    if trim_level not in MARKET_SCOPE_OPTIONS:
        raise ValueError(f"Unknown trimming level for saved OOF predictions: {trim_level}")
    required = {
        "Model",
        "Trim_Level",
        "listing_id",
        "scenario_b_fold",
        "actual_price_RM",
        "predicted_price_RM",
    }
    missing = sorted(required.difference(oof.columns))
    if missing:
        raise ValueError(f"Saved scope-specific OOF data is missing columns: {missing}")

    metadata = get_trim_market_metadata(float(trim_level.removesuffix("%")))
    retained = oof.loc[
        oof["Model"].eq(model_name) & oof["Trim_Level"].eq(trim_level)
    ].copy()
    if retained.empty:
        raise ValueError(f"No saved OOF predictions exist for {model_name}, {trim_level}.")
    if retained["listing_id"].duplicated().any():
        raise ValueError(f"Saved OOF predictions contain duplicate listings for {model_name}, {trim_level}.")
    actual = pd.to_numeric(retained["actual_price_RM"], errors="coerce")
    predicted = pd.to_numeric(retained["predicted_price_RM"], errors="coerce")
    if not (
        np.isfinite(actual.to_numpy(float)).all()
        and np.isfinite(predicted.to_numpy(float)).all()
    ):
        raise ValueError("Saved actual and OOF predicted prices must all be finite.")
    retained["actual_price_RM"] = actual
    retained["predicted_price_RM"] = predicted
    if len(retained) != metadata["retained_rows"]:
        raise ValueError(
            f"{metadata['trim_label']} trimming retained {len(retained):,} OOF rows; "
            f"the saved experiment requires {metadata['retained_rows']:,}."
        )

    plot = retained.rename(
        columns={
            "actual_price_RM": "Actual Price (RM)",
            "predicted_price_RM": "OOF Predicted Price (RM)",
        }
    )
    return plot, metadata


def render_model_diagnostics(payload: dict) -> None:
    """Combine feature importance and saved OOF diagnostics under one page."""
    st.header("Model Diagnostics")
    st.caption("How is the model behaving?")
    diagnostic_view = st.radio(
        "Diagnostic View",
        ["Feature Importance", "Actual vs Predicted"],
        horizontal=True,
        label_visibility="collapsed",
        key="diagnostic_view",
    )
    if diagnostic_view == "Feature Importance":
        st.sidebar.write("### FEATURE IMPORTANCE")
        model_name = st.sidebar.selectbox(
            "Model",
            list(FINAL_MODELS),
            index=list(FINAL_MODELS).index(FINAL_MODEL_NAME),
            key="importance_model",
        )
        top_n = st.sidebar.selectbox(
            "Top Features",
            [10, 15, 20],
            index=0,
            key="importance_top_n",
        )
        render_feature_importance(model_name, int(top_n))
    else:
        st.sidebar.write("### ACTUAL VS PREDICTED")
        model_name = st.sidebar.selectbox(
            "Model",
            list(FINAL_MODELS),
            index=list(FINAL_MODELS).index(FINAL_MODEL_NAME),
            key="prediction_model",
        )
        selected_scope = st.sidebar.selectbox(
            "Market Scope",
            list(MARKET_SCOPE_OPTIONS),
            index=list(MARKET_SCOPE_OPTIONS).index("10%"),
            key="prediction_trim_level",
        )
        render_actual_vs_predicted(payload, model_name, selected_scope)


def trim_label(value: float) -> str:
    return f"{value:g}%"


def trimming_display_frame(summary: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Create one six-row model table from saved restricted-market metrics."""
    rows = summary.loc[
        summary["Model"].eq(model_name),
        ["Trim_Level", "Retained_Rows", "RMSE_RM", "MAE_RM", "R2", "Adjusted_R2"],
    ].copy()
    rows.columns = [
        "Trim Level",
        "Retained Listings",
        "RMSE",
        "MAE",
        "R²",
        "Adjusted R²",
    ]
    if len(rows) != 6:
        raise ValueError(f"Expected six saved trimming rows for {model_name}.")
    rows["Retained Listings"] = rows["Retained Listings"].astype(int)
    return rows


def render_all_models_trimming_comparison() -> None:
    """Render all four models using the validated saved 24-row artifact only."""
    st.subheader("Trimmed-Data Model Comparison")
    summary = load_all_models_trimming_summary()
    selected_metric = st.selectbox(
        "Select Trimming Metric",
        list(COMPARISON_METRICS),
        key="trimmed_comparison_metric",
    )
    st.plotly_chart(
        build_trimmed_metric_chart(summary, selected_metric),
        width="stretch",
    )

    for model_name in FINAL_MODELS:
        st.subheader(model_name)
        display = trimming_display_frame(summary, model_name)
        st.dataframe(
            display.style.format(
                {
                    "Retained Listings": "{:,}",
                    "RMSE": "RM {:,.0f}",
                    "MAE": "RM {:,.0f}",
                    "R²": "{:.4f}",
                    "Adjusted R²": "{:.4f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )


def build_trimmed_metric_chart(summary: pd.DataFrame, metric_name: str) -> go.Figure:
    """Build one full-width four-model trimming chart from saved metrics."""
    metric, title_metric, y_label, text_format = COMPARISON_METRICS[metric_name]
    trim_order = ["0%", "0.5%", "1%", "2.5%", "5%", "10%"]
    figure = px.line(
        summary,
        x="Trim_Level",
        y=metric,
        color="Model",
        line_dash="Model",
        markers=True,
        category_orders={
            "Trim_Level": trim_order,
            "Model": list(FINAL_MODELS),
        },
        color_discrete_map=MODEL_COLORS,
        labels={"Trim_Level": "Trim Level", metric: y_label},
        title=f"{title_metric} Across Trimming Levels",
    )
    figure.update_layout(height=650, legend_title_text="")
    figure.update_yaxes(tickformat=text_format)
    return figure


def render_outlier_trimming() -> None:
    metadata, tables = load_trimming_results()
    recommended_trimming = str(metadata["recommended_trimming"])
    st.header("Outlier & Trimming Study")
    st.success(
        f"Final Full-Market Decision: {recommended_trimming} Upper-Tail Trimming"
    )
    if recommended_trimming == "0%":
        recommendation_evidence = (
            "The regenerated training-only experiment did not support removing "
            "valid upper-tail listings."
        )
        production_population = "Final full-market modelling retains valid premium observations."
    else:
        recommendation_evidence = (
            f"The regenerated training-only experiment supports {recommended_trimming} "
            "upper-tail trimming."
        )
        production_population = (
            f"The saved production recommendation uses the {recommended_trimming} population."
        )
    st.markdown(
        "- Invalid and impossible observations are removed.\n"
        "- Duplicate records are removed.\n"
        "- Valid premium listings remain distinct from invalid observations.\n"
        f"- {recommendation_evidence}\n"
        "- Restricted-market trimming remains a separate experiment.\n"
        f"- {production_population}"
    )
    st.info(
        "The 10% default used by interactive tools is an experimental restricted-market "
        f"scope. It does not replace the official {recommended_trimming} full-market strategy."
    )

    training = tables["training_only_comparison"]
    distribution = tables["distribution_shift"]
    bootstrap = tables["bootstrap_results"]
    production_training = training[training["Model"] == FINAL_MODEL_NAME].copy()
    baseline_distribution = distribution.loc[
        distribution["Removal_Percent"].eq(0.0)
    ].iloc[0]

    detection_tab, experiment_tab, statistics_tab = st.tabs(
        ["Outlier Detection", "Trimming Experiment", "Statistical Evidence"]
    )

    detection_tab.write("### Outlier Detection")
    detection_tab.info(
        "An extreme listing price is not automatically invalid. Canonical cleaning "
        "removes impossible values and listings outside the established RM50–RM5,000 "
        "PPSF plausibility range; valid premium observations remain eligible."
    )
    detection = pd.DataFrame(
        {
            "Market Statistic": [
                "Original Listings",
                "Median Price",
                "90th Percentile",
                "95th Percentile / Premium Threshold",
                "99th Percentile",
                "Maximum Price",
                "Top-5% Premium Listings",
                "PPSF Plausibility Rule",
            ],
            "Saved Value": [
                f"{int(baseline_distribution['Before_Row_Count']):,}",
                f"RM {baseline_distribution['Before_Median_Price_RM']:,.0f}",
                f"RM {baseline_distribution['Before_P90_Price_RM']:,.0f}",
                f"RM {baseline_distribution['Premium_Threshold_RM']:,.0f}",
                f"RM {baseline_distribution['Before_P99_Price_RM']:,.0f}",
                f"RM {baseline_distribution['Before_Maximum_Price_RM']:,.0f}",
                f"{int(baseline_distribution['Premium_Rows_Retained']):,}",
                "RM 50–RM 5,000 / sqft",
            ],
        }
    )
    detection_tab.dataframe(detection, width="stretch", hide_index=True)
    price_landmarks = pd.DataFrame(
        {
            "Price Landmark": ["Median", "P90", "P95", "P99", "Maximum"],
            "Price_RM": [
                baseline_distribution["Before_Median_Price_RM"],
                baseline_distribution["Before_P90_Price_RM"],
                baseline_distribution["Before_P95_Price_RM"],
                baseline_distribution["Before_P99_Price_RM"],
                baseline_distribution["Before_Maximum_Price_RM"],
            ],
        }
    )
    detection_chart = px.bar(
        price_landmarks,
        x="Price Landmark",
        y="Price_RM",
        text_auto=",.0f",
        labels={"Price_RM": "Listing Price (RM)"},
        title="Saved Listing-Price Distribution Landmarks",
    )
    detection_chart.update_traces(marker_color="#64748b")
    detection_chart.update_layout(height=390, showlegend=False)
    detection_chart.update_yaxes(tickformat=",")
    detection_tab.plotly_chart(detection_chart, width="stretch")

    detection_tab.write("### Outlier Treatment Methods")
    methods = pd.DataFrame(
        [
            {
                "Method": "Validity and PPSF plausibility checks",
                "Stage": "Canonical cleaning",
                "How It Handles Outliers": "Drops impossible target/size rows and PPSF outside RM50–RM5,000",
                "Deletes Rows?": "Yes, invalid only",
                "Caps Values?": "No",
                "Outcome": "Defines the eligible canonical market",
                "Final Decision": "Keep",
            },
            {
                "Method": "Exact and listing-ID duplicate removal",
                "Stage": "Canonical cleaning",
                "How It Handles Outliers": "Keeps one record for exact duplicates and repeated Ad List IDs",
                "Deletes Rows?": "Yes, duplicates",
                "Caps Values?": "No",
                "Outcome": "Prevents duplicated listings from entering evaluation",
                "Final Decision": "Keep",
            },
            {
                "Method": "Upper-tail training-only trimming",
                "Stage": "Saved experiment",
                "How It Handles Outliers": "Removes expensive rows only from each outer training fold",
                "Deletes Rows?": "Training only",
                "Caps Values?": "No",
                "Outcome": "Every nonzero level worsened final-model full-market RMSE and MAE",
                "Final Decision": "Reject",
            },
            {
                "Method": "Restricted-market trimming",
                "Stage": "Saved experiment",
                "How It Handles Outliers": "Removes the upper tail before both training and validation",
                "Deletes Rows?": "Yes",
                "Caps Values?": "No",
                "Outcome": "Lower error on a narrower, easier retained market",
                "Final Decision": "Not a full-market replacement",
            },
            {
                "Method": "Winsorization, log target, Huber loss, sample weighting",
                "Stage": "Saved experiment controls",
                "How It Handles Outliers": "Alternative robust treatments",
                "Deletes Rows?": "No",
                "Caps Values?": "Not applied",
                "Outcome": "Explicitly recorded as not applied in this trimming experiment",
                "Final Decision": "No result claimed",
            },
            {
                "Method": "Retain valid premium examples with frozen feature engineering",
                "Stage": "Final deployment",
                "How It Handles Outliers": "Keeps legitimate premium listings and models PPSF with existing features",
                "Deletes Rows?": "No",
                "Caps Values?": "No",
                "Outcome": "Preserves the complete 3,791-listing market",
                "Final Decision": "Selected: 0% trimming",
            },
        ]
    )
    with detection_tab.expander("View Outlier Treatment Methods"):
        st.dataframe(methods, width="stretch", hide_index=True)

    experiment_tab.write("### Premium Property Impact")
    premium_impact = production_training[
        [
            "Removal_Percent",
            "Top5_RMSE_RM",
            "P95_99_RMSE_RM",
            "P99_100_RMSE_RM",
            "Top5_Underprediction_Pct",
        ]
    ].rename(
        columns={
            "Removal_Percent": "Trim Level",
            "Top5_RMSE_RM": "Top-5% RMSE",
            "P95_99_RMSE_RM": "95–99% RMSE",
            "P99_100_RMSE_RM": "99–100% RMSE",
            "Top5_Underprediction_Pct": "Premium Underprediction %",
        }
    )
    premium_impact["Trim Level"] = premium_impact["Trim Level"].map(trim_label)
    premium_chart = px.line(
        premium_impact,
        x="Trim Level",
        y="Premium Underprediction %",
        markers=True,
        title="Premium Underprediction as Training Examples Are Removed",
    )
    premium_chart.update_traces(line_color="#dc2626", marker_size=8)
    premium_chart.update_layout(height=390, showlegend=False)
    experiment_tab.plotly_chart(premium_chart, width="stretch")
    experiment_tab.warning(
        "As progressively more high-priced training examples are removed, the model "
        "becomes less capable of predicting premium properties and premium "
        "underprediction increases."
    )
    with experiment_tab.expander("View Detailed Premium-Segment Results"):
        st.dataframe(
            premium_impact.style.format(
                {
                    "Top-5% RMSE": "RM {:,.0f}",
                    "95–99% RMSE": "RM {:,.0f}",
                    "99–100% RMSE": "RM {:,.0f}",
                    "Premium Underprediction %": "{:.1f}%",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    statistics_tab.write("### Statistical Validation")
    statistical = bootstrap.loc[
        bootstrap["Model"].eq(FINAL_MODEL_NAME),
        [
            "Removal_Percent",
            "RMSE_Difference_RM",
            "RMSE_CI95_Lower_RM",
            "RMSE_CI95_Upper_RM",
            "MAE_Difference_RM",
            "MAE_CI95_Lower_RM",
            "MAE_CI95_Upper_RM",
        ],
    ].copy()
    statistical.insert(
        0,
        "Trim Level",
        statistical["Removal_Percent"].map(trim_label),
    )
    statistical["Interpretation"] = np.where(
        statistical["Removal_Percent"].eq(0.0),
        "Baseline",
        np.where(
            statistical["RMSE_CI95_Lower_RM"].gt(0),
            "Reliably worse RMSE",
            "RMSE difference inconclusive",
        ),
    )
    statistical = statistical.rename(
        columns={
            "RMSE_Difference_RM": "RMSE Change",
            "RMSE_CI95_Lower_RM": "RMSE CI95 Lower",
            "RMSE_CI95_Upper_RM": "RMSE CI95 Upper",
            "MAE_Difference_RM": "MAE Change",
            "MAE_CI95_Lower_RM": "MAE CI95 Lower",
            "MAE_CI95_Upper_RM": "MAE CI95 Upper",
        }
    ).drop(columns="Removal_Percent")
    statistics_tab.write(
        "The saved bootstrap comparison uses 0% trimming as its baseline. "
        "Positive changes mean trimming performed worse than the 0% baseline. "
        "A confidence interval entirely above zero indicates reliable deterioration."
    )
    with statistics_tab.expander("View Bootstrap Statistical Results"):
        st.dataframe(
            statistical.style.format(
                {
                    "RMSE Change": "RM {:,.0f}",
                    "RMSE CI95 Lower": "RM {:,.0f}",
                    "RMSE CI95 Upper": "RM {:,.0f}",
                    "MAE Change": "RM {:,.0f}",
                    "MAE CI95 Lower": "RM {:,.0f}",
                    "MAE CI95 Upper": "RM {:,.0f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    with statistics_tab.expander("Technical Details", expanded=False):
        st.write(f"Canonical rows: {metadata['canonical_rows']:,}")
        st.write(f"Validation: {metadata['validation_method']}")
        st.write(f"Target: {metadata['target']}")
        st.write(f"Evaluation unit: {metadata['evaluation_unit']}")
        st.write(f"Bootstrap samples: {metadata['bootstrap_samples']:,}")
        st.write(
            "Validation rows removed during training-only trimming: "
            f"{metadata['training_only_validation_rows_removed']}"
        )
        st.write(f"Recommended trim: {metadata['recommended_trimming']}")


def render_live_predictor(data: pd.DataFrame) -> None:
    st.subheader("Live House Price Predictor")
    prediction_mode = st.radio(
        "Prediction Mode",
        ("Final Full-Market Model", "Experimental Trimmed-Market Model"),
        index=0,
        horizontal=True,
        key="prediction_mode",
    )
    experimental_mode = prediction_mode == "Experimental Trimmed-Market Model"
    selected_trim_level: float | None = None
    selected_trim_metadata: dict | None = None

    if experimental_mode:
        st.caption("EXPERIMENTAL — the official final model remains the production model.")
        selected_trim_label = st.selectbox(
            "Experimental trim level",
            [trim_label(level) for level in EXPERIMENTAL_TRIM_LEVELS],
            index=0,
            key="predictor_trim_level",
        )
        selected_trim_level = float(selected_trim_label.removesuffix("%"))
        selected_trim_metadata = get_trim_market_metadata(selected_trim_level)
        scope = st.columns(6)
        scope[0].metric("Trim Level", selected_trim_metadata["trim_label"])
        scope[1].metric(
            "Original Rows", f"{selected_trim_metadata['original_rows']:,}"
        )
        scope[2].metric("Rows Removed", f"{selected_trim_metadata['removed_rows']:,}")
        scope[3].metric("Retained Rows", f"{selected_trim_metadata['retained_rows']:,}")
        scope[4].metric(
            "Retention", f"{selected_trim_metadata['retention_percentage']:.2f}%"
        )
        scope[5].metric(
            "Maximum Retained Price",
            f"RM {selected_trim_metadata['maximum_retained_price_RM']:,.0f}",
        )
        st.caption(
            "Mean retained training price: "
            f"RM {selected_trim_metadata['mean_retained_price_RM']:,.0f}."
        )
        st.warning(
            "Experimental restricted-market model. It was trained after excluding "
            f"the highest-priced {selected_trim_metadata['trim_label']} of listings. "
            "Predictions for properties outside the retained market, especially premium "
            "properties, may be unreliable."
        )
        st.info(
            "Lower restricted-market RMSE partly occurs because the prediction scope "
            "becomes narrower and the most difficult premium listings are excluded."
        )
    else:
        st.info(
            "Status: FINAL MODEL  |  Trim: 0%  |  Training population: 3,791 listings  |  "
            "Validation methodology: Scenario B group-safe 5-fold CV  |  "
            "Deployment fitting: all eligible rows after evaluation"
        )

    st.write(
        f"Both modes use {FINAL_MODEL_NAME} and the same structured, target-encoding, "
        "PPSF, and position-feature pipeline. Description inputs remain supplementary."
    )
    description = st.text_area(
        "Listing Description",
        placeholder="Example: Spacious high floor unit with a large balcony and city views.",
        key="listing_description",
    )
    detected = extract_position_features([description]).iloc[0]
    st.write("#### Detected position features")
    status_columns = st.columns(5)
    for column, feature in zip(status_columns, POSITION_FEATURES):
        mark = "✓" if bool(detected[feature]) else "—"
        column.caption(f"{mark} {POSITION_DISPLAY_NAMES[feature]}")

    with st.form("final_prediction_form"):
        numeric, category = st.columns(2)
        with numeric:
            property_size = st.number_input(
                "Property Size (sq.ft.)",
                min_value=1.0,
                max_value=20_000.0,
                value=float(data["property_size_sqft"].median()),
                step=50.0,
            )
            bedrooms = st.number_input(
                "Bedrooms", min_value=0.0, max_value=20.0,
                value=float(data["bedroom"].median()), step=1.0,
            )
            bathrooms = st.number_input(
                "Bathrooms", min_value=0.0, max_value=20.0,
                value=float(data["bathroom"].median()), step=1.0,
            )
            parking_lots = st.number_input(
                "Parking Lots", min_value=0.0, max_value=20.0,
                value=float(data["parking_lot"].median()), step=1.0,
            )
            facilities_count = st.number_input(
                "Facilities Count", min_value=0, max_value=50,
                value=int(data["facilities_count"].median()), step=1,
            )
            completion_year = st.number_input(
                "Completion Year", min_value=1800, max_value=2030,
                value=int(data["completion_year"].median()), step=1,
            )
            number_of_floors = st.number_input(
                "Number of Floors", min_value=1, max_value=200,
                value=max(1, int(data["number_of_floors"].median())), step=1,
            )
            total_units = st.number_input(
                "Total Units", min_value=1, max_value=20_000,
                value=max(1, int(data["total_units"].median())), step=1,
            )
        with category:
            property_type = st.selectbox("Property Type", category_values(data, "property_type"))
            tenure_type = st.selectbox("Tenure Type", category_values(data, "tenure_type"))
            land_title = st.selectbox("Land Title", category_values(data, "land_title"))
            floor_range = st.selectbox("Floor Range", category_values(data, "floor_range"))
            state = st.selectbox("State", category_values(data, "state"))
            city = st.selectbox("City / Locality", category_values(data, "city"))
            building_name = st.text_input("Building Name (Optional)", value="")
            developer = st.text_input("Developer (Optional)", value="")

        st.write("#### Nearby Amenities")
        nearby = st.columns(4)
        has_school = int(nearby[0].checkbox("School"))
        has_mall = int(nearby[1].checkbox("Mall"))
        has_hospital = int(nearby[2].checkbox("Hospital"))
        has_railway = int(nearby[3].checkbox("Railway Station"))
        has_bus_stop = int(nearby[0].checkbox("Bus Stop"))
        has_park = int(nearby[1].checkbox("Park"))
        has_highway = int(nearby[2].checkbox("Highway"))

        st.write("#### Property Facilities")
        facilities = st.columns(5)
        has_swimming_pool = int(facilities[0].checkbox("Swimming Pool"))
        has_security = int(facilities[1].checkbox("Security"))
        has_lift = int(facilities[2].checkbox("Lift"))
        has_gym = int(facilities[3].checkbox("Gym"))
        has_playground = int(facilities[4].checkbox("Playground"))

        status_columns = st.columns(2)
        furnishing_status = status_columns[0].selectbox(
            "Furnishing Status",
            list(FURNISHING_STATUS_VALUES),
            index=0,
        )
        renovation_status = status_columns[1].selectbox(
            "Renovation Status",
            list(RENOVATION_STATUS_VALUES),
            index=0,
        )
        submitted = st.form_submit_button("Estimate Price", type="primary")

    if submitted:
        condition_values = condition_feature_values(
            furnishing_status,
            renovation_status,
        )
        values = {
            "property_size_sqft": property_size,
            "bedroom": bedrooms,
            "bathroom": bathrooms,
            "parking_lot": parking_lots,
            "facilities_count": facilities_count,
            "has_school": has_school,
            "has_mall": has_mall,
            "has_hospital": has_hospital,
            "has_railway": has_railway,
            "has_bus_stop": has_bus_stop,
            "has_park": has_park,
            "has_highway": has_highway,
            "completion_year": completion_year,
            "number_of_floors": number_of_floors,
            "total_units": total_units,
            "has_swimming_pool": has_swimming_pool,
            "has_security": has_security,
            "has_lift": has_lift,
            "has_gym": has_gym,
            "has_playground": has_playground,
            "is_furnished": condition_values["is_furnished"],
            "is_renovated": condition_values["is_renovated"],
            "property_type": property_type,
            "tenure_type": tenure_type,
            "land_title": land_title,
            "floor_range": floor_range,
            "state": state,
            "building_name": building_name,
            "developer": developer,
            "city": city,
        }
        try:
            official_model = load_deployment_model()
            official_prediction = predict_total_price(
                official_model,
                values,
                description,
                float(data["description_length"].median()),
            )
            trimmed_model = None
            trimmed_prediction = None
            if experimental_mode:
                if selected_trim_level is None:
                    raise ValueError("Select an experimental trim level.")
                trimmed_model = load_trimmed_deployment_model(selected_trim_level)
                trimmed_prediction = predict_total_price(
                    trimmed_model,
                    values,
                    description,
                    trimmed_model.description_length_median_,
                )
        except ValueError as error:
            st.error(str(error))
        else:
            if experimental_mode:
                if trimmed_prediction is None or selected_trim_metadata is None:
                    raise AssertionError("Experimental prediction was not calculated.")
                difference = (
                    trimmed_prediction["total_price_RM"]
                    - official_prediction["total_price_RM"]
                )
                difference_percent = (
                    100.0 * difference / official_prediction["total_price_RM"]
                )
                output = st.columns(3)
                output[0].metric(
                    "Official Full-Market Prediction",
                    f"RM {official_prediction['total_price_RM']:,.0f}",
                )
                output[1].metric(
                    f"{selected_trim_metadata['trim_label']} Trimmed-Market Prediction",
                    f"RM {trimmed_prediction['total_price_RM']:,.0f}",
                )
                output[2].metric(
                    "Difference",
                    f"RM {difference:+,.0f}",
                    f"{difference_percent:+.1f}%",
                    delta_color="off",
                )
                st.caption(
                    "Estimated PPSF — official: "
                    f"RM {official_prediction['ppsf_RM']:,.0f} / sqft; "
                    f"{selected_trim_metadata['trim_label']} trimmed: "
                    f"RM {trimmed_prediction['ppsf_RM']:,.0f} / sqft."
                )
                st.success(
                    "Comparison shown: official 0% deployment model versus the cached "
                    f"{selected_trim_metadata['trim_label']} experimental retained-market model."
                )
            else:
                output = st.columns(2)
                output[0].metric(
                    "Estimated Listing Price",
                    f"RM {official_prediction['total_price_RM']:,.0f}",
                )
                output[1].metric(
                    "Estimated Price per sq.ft.",
                    f"RM {official_prediction['ppsf_RM']:,.0f} / sqft",
                )
                st.success(f"Model: {FINAL_MODEL_NAME}")
            if description.strip():
                st.caption(
                    "Description length used: "
                    f"{official_prediction['description_length']:,.0f} characters."
                )
            else:
                st.caption(
                    "Description was blank, so each model's training-population median "
                    "description length was used."
                )
            st.warning(
                "This is a statistical listing-price estimate and not a certified property valuation."
            )


def render_scope_predictor(data: pd.DataFrame, selected_scope: str = "10%") -> None:
    """Render four-model live inference for a selected saved market scope."""
    st.header("Price Predictor")
    st.caption(
        "Estimate a listing price with the existing saved models and frozen feature pipeline."
    )
    if selected_scope not in MARKET_SCOPE_OPTIONS:
        raise ValueError(f"Unsupported market scope: {selected_scope}")
    scope_level = float(selected_scope.removesuffix("%"))
    if scope_level > 0:
        st.warning(
            f"{selected_scope} scope excludes the saved upper price tail from model training. "
            "It is experimental; the official full-market strategy remains 0%."
        )

    with st.form("scope_prediction_form"):
        st.write("### 1. Property Details")
        numeric_columns = st.columns(4)
        with numeric_columns[0]:
            property_size = st.number_input(
                "Property Size (sq.ft.)",
                min_value=1.0,
                max_value=20_000.0,
                value=float(data["property_size_sqft"].median()),
                step=50.0,
            )
            bedrooms = st.number_input(
                "Bedrooms", min_value=0.0, max_value=20.0,
                value=float(data["bedroom"].median()), step=1.0,
            )
        with numeric_columns[1]:
            bathrooms = st.number_input(
                "Bathrooms", min_value=0.0, max_value=20.0,
                value=float(data["bathroom"].median()), step=1.0,
            )
            parking_lots = st.number_input(
                "Parking Lots", min_value=0.0, max_value=20.0,
                value=float(data["parking_lot"].median()), step=1.0,
            )
        with numeric_columns[2]:
            facilities_count = st.number_input(
                "Facilities Count", min_value=0, max_value=50,
                value=int(data["facilities_count"].median()), step=1,
            )
            completion_year = st.number_input(
                "Completion Year", min_value=1800, max_value=2030,
                value=int(data["completion_year"].median()), step=1,
            )
        with numeric_columns[3]:
            number_of_floors = st.number_input(
                "Number of Floors", min_value=1, max_value=200,
                value=max(1, int(data["number_of_floors"].median())), step=1,
            )
            total_units = st.number_input(
                "Total Units", min_value=1, max_value=20_000,
                value=max(1, int(data["total_units"].median())), step=1,
            )

        st.write("### 2. Location & Classification")
        classification_columns = st.columns(2)
        with classification_columns[0]:
            property_type = st.selectbox("Property Type", category_values(data, "property_type"))
            tenure_type = st.selectbox("Tenure Type", category_values(data, "tenure_type"))
            land_title = st.selectbox("Land Title", category_values(data, "land_title"))
            floor_range = st.selectbox("Floor Range", category_values(data, "floor_range"))
        with classification_columns[1]:
            state = st.selectbox("State", category_values(data, "state"))
            city = st.selectbox("City / Locality", category_values(data, "city"))
            building_name = st.text_input("Building Name (Optional)", value="")
            developer = st.text_input("Developer (Optional)", value="")

        st.write("### 3. Amenities & Property Condition")
        st.write("#### Nearby Amenities")
        nearby = st.columns(4)
        has_school = int(nearby[0].checkbox("School"))
        has_mall = int(nearby[1].checkbox("Mall"))
        has_hospital = int(nearby[2].checkbox("Hospital"))
        has_railway = int(nearby[3].checkbox("Railway Station"))
        has_bus_stop = int(nearby[0].checkbox("Bus Stop"))
        has_park = int(nearby[1].checkbox("Park"))
        has_highway = int(nearby[2].checkbox("Highway"))

        st.write("#### Property Facilities")
        facilities = st.columns(5)
        has_swimming_pool = int(facilities[0].checkbox("Swimming Pool"))
        has_security = int(facilities[1].checkbox("Security"))
        has_lift = int(facilities[2].checkbox("Lift"))
        has_gym = int(facilities[3].checkbox("Gym"))
        has_playground = int(facilities[4].checkbox("Playground"))

        conditions = st.columns(2)
        furnishing_status = conditions[0].selectbox(
            "Furnishing Status", list(FURNISHING_STATUS_VALUES), index=0
        )
        renovation_status = conditions[1].selectbox(
            "Renovation Status", list(RENOVATION_STATUS_VALUES), index=0
        )

        st.write("### 4. Listing Description / Position Signals")
        description = st.text_area(
            "Listing Description",
            placeholder="Example: Spacious high floor unit with a large balcony and city views.",
            key="listing_description",
        )
        submitted = st.form_submit_button("Estimate Property Price", type="primary")

    detected = extract_position_features([description]).iloc[0]
    st.caption("Detected position signals from the listing description")
    status_columns = st.columns(5)
    for column, feature in zip(status_columns, POSITION_FEATURES):
        mark = "✓" if bool(detected[feature]) else "—"
        column.caption(f"{mark} {POSITION_DISPLAY_NAMES[feature]}")

    if not submitted:
        return

    condition_values = condition_feature_values(furnishing_status, renovation_status)
    values = {
        "property_size_sqft": property_size,
        "bedroom": bedrooms,
        "bathroom": bathrooms,
        "parking_lot": parking_lots,
        "facilities_count": facilities_count,
        "has_school": has_school,
        "has_mall": has_mall,
        "has_hospital": has_hospital,
        "has_railway": has_railway,
        "has_bus_stop": has_bus_stop,
        "has_park": has_park,
        "has_highway": has_highway,
        "completion_year": completion_year,
        "number_of_floors": number_of_floors,
        "total_units": total_units,
        "has_swimming_pool": has_swimming_pool,
        "has_security": has_security,
        "has_lift": has_lift,
        "has_gym": has_gym,
        "has_playground": has_playground,
        "is_furnished": condition_values["is_furnished"],
        "is_renovated": condition_values["is_renovated"],
        "property_type": property_type,
        "tenure_type": tenure_type,
        "land_title": land_title,
        "floor_range": floor_range,
        "state": state,
        "building_name": building_name,
        "developer": developer,
        "city": city,
    }
    try:
        selected_models = load_scope_models(selected_scope)
        full_market_models = (
            selected_models if selected_scope == "0%" else load_scope_models("0%")
        )
        selected_predictions = {
            name: predict_scope_model(
                name,
                selected_models[name],
                values,
                description,
                selected_models[name].description_length_median_,
            )
            for name in FINAL_MODELS
        }
        full_market_predictions = (
            selected_predictions
            if selected_scope == "0%"
            else {
                name: predict_scope_model(
                    name,
                    full_market_models[name],
                    values,
                    description,
                    full_market_models[name].description_length_median_,
                )
                for name in FINAL_MODELS
            }
        )
        output = prediction_comparison_frame(
            selected_predictions,
            full_market_predictions,
        )
    except ValueError as error:
        st.error(str(error))
        return

    final_prediction = selected_predictions[FINAL_MODEL_NAME]
    st.write("### Estimated Listing Price")
    result_columns = st.columns(3)
    result_columns[0].metric(
        "Estimated Price", f"RM {final_prediction['total_price_RM']:,.0f}"
    )
    result_columns[1].metric(
        "Estimated PPSF", f"RM {final_prediction['ppsf_RM']:,.0f}"
    )
    result_columns[2].metric("Market Scope", selected_scope)
    st.caption(f"Model: {FINAL_MODEL_NAME} · Saved deployment artifact")

    recommendation = recommended_model_for_scope(
        load_all_models_trimming_summary(), selected_scope
    )
    st.info(
        f"Recommended Model for {selected_scope} scope: {recommendation} "
        "(lowest saved validation RMSE)."
    )
    with st.expander("Compare Predictions from All Models"):
        styled = output.style.format(
            {
                "Selected-Scope Prediction": "RM {:,.0f}",
                "Full-Market Prediction": "RM {:,.0f}",
                "Difference (RM)": "RM {:+,.0f}",
                "Difference (%)": "{:+.2f}%",
            }
        )
        st.dataframe(styled, width="stretch", hide_index=True)
    if selected_scope == "0%":
        st.caption(
            "The selected scope is the full market, so both prediction columns use the "
            "same cached deployment models and all differences are exactly zero."
        )
    st.warning(
        "These are statistical listing-price estimates and not certified property valuations."
    )


def main() -> None:
    st.set_page_config(
        page_title="Malaysian Property Price ML",
        page_icon="\U0001f3e0",
        layout="wide",
    )
    st.sidebar.title("Property Price ML")
    st.sidebar.caption("Malaysian Residential Property Price Prediction")
    st.sidebar.write("### NAVIGATION")
    view = st.sidebar.radio(
        "Navigation",
        list(VIEWS),
        key="navigation",
        label_visibility="collapsed",
    )

    if view == OVERVIEW_VIEW:
        render_overview(load_dataset())
    elif view == EDA_VIEW:
        st.sidebar.write("### DATA EXPLORATION")
        category = st.sidebar.selectbox(
            "EDA Category", list(EDA_VISUALIZATIONS), key="eda_category"
        )
        chart_name = st.sidebar.selectbox(
            "Visualisation",
            list(EDA_VISUALIZATIONS[category]),
            key="eda_visualization",
        )
        render_eda_page(category, chart_name)
    elif view == EVALUATION_VIEW:
        st.sidebar.write("### MODEL EVALUATION")
        selected_scope = st.sidebar.selectbox(
            "Market Scope",
            list(MARKET_SCOPE_OPTIONS),
            index=list(MARKET_SCOPE_OPTIONS).index("10%"),
            key="evaluation_market_scope",
        )
        selected_metric = st.sidebar.selectbox(
            "Evaluation Metric",
            list(COMPARISON_METRICS),
            key="model_comparison_metric",
        )
        st.sidebar.caption("Official full-market reporting remains at 0% trimming.")
        render_model_evaluation(selected_scope, selected_metric)
    elif view == DIAGNOSTICS_VIEW:
        render_model_diagnostics(load_comparison())
    elif view == OUTLIER_VIEW:
        outlier_metadata, _ = load_trimming_results()
        recommended_trimming = str(outlier_metadata["recommended_trimming"])
        st.sidebar.write("### OUTLIER STUDY")
        st.sidebar.info(
            f"Final strategy: {recommended_trimming} upper-tail trimming\n\n"
            "Canonical market: 3,791 listings\n\n"
            "PPSF plausibility: RM 50–RM 5,000"
        )
        render_outlier_trimming()
    elif view == PREDICTOR_VIEW:
        st.sidebar.write("### PREDICTION SETTINGS")
        selected_scope = st.sidebar.selectbox(
            "Market Scope",
            list(MARKET_SCOPE_OPTIONS),
            index=list(MARKET_SCOPE_OPTIONS).index("10%"),
            key="predictor_market_scope",
        )
        metadata = get_trim_market_metadata(float(selected_scope.removesuffix("%")))
        st.sidebar.metric("Retained Market", f"{metadata['retention_percentage']:.2f}%")
        st.sidebar.metric("Training Listings", f"{metadata['retained_rows']:,}")
        st.sidebar.metric("Excluded", f"{metadata['removed_rows']:,}")
        st.sidebar.caption("Official full-market reporting remains at 0% trimming.")
        render_scope_predictor(load_dataset(), selected_scope)

    st.sidebar.divider()
    st.sidebar.write("### PROJECT")
    st.sidebar.caption("Problem\n\nRegression")
    st.sidebar.caption("Target\n\nListing Price")
    st.sidebar.caption("Dataset\n\nMalaysian Residential Properties")
    st.sidebar.caption("Validation\n\nScenario B Group-Safe CV")


if __name__ == "__main__":
    main()
