"""Final Streamlit dashboard backed by saved Scenario B evaluation artifacts."""

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
from src.models.final.position_regex_lightgbm import (
    FINAL_MODEL_NAME,
    POSITION_DISPLAY_NAMES,
    POSITION_FEATURES,
    extract_position_features,
    fit_final_model,
    predict_total_price,
)


DATA_PATH = PROJECT_ROOT / "data" / "processed" / "enhanced_city_dataset.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "final_models"
COMPARISON_PATH = RESULTS_DIR / "model_comparison.json"
OOF_PATH = RESULTS_DIR / "oof_predictions.csv"
IMPORTANCE_PATH = RESULTS_DIR / "feature_importance.csv"
TRIMMING_RESULTS_DIR = PROJECT_ROOT / "results" / "outlier_trimming"
TRIMMING_METADATA_PATH = TRIMMING_RESULTS_DIR / "metadata.json"
VIEWS = (
    "Model Comparison",
    "Feature Importance",
    "Actual vs Predicted",
    "Outlier & Trimming Analysis",
    "Live House Price Predictor",
)


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
    if metadata.get("recommended_trimming") != "0%":
        raise ValueError("Saved trimming recommendation must remain 0%.")
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


@st.cache_resource
def load_deployment_model():
    """Fit only the selected final model, once per Streamlit process."""
    model = fit_final_model()
    if model.training_rows_ != 3_791:
        raise AssertionError("Deployment model did not train on all canonical rows.")
    return model


def comparison_frame(payload: dict) -> pd.DataFrame:
    return pd.DataFrame(payload["models"]).loc[
        :, [
            "Model",
            "RMSE_RM",
            "MAE_RM",
            "R2",
            "Adjusted_R2",
        ]
    ]


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


def render_comparison(payload: dict) -> None:
    st.subheader("Final Four-Model Comparison")
    table = comparison_frame(payload)
    metric_by_model = table.set_index("Model")
    cards = st.columns(3)
    cards[0].metric("Selected Final Model", FINAL_MODEL_NAME)
    cards[1].metric(
        "Lowest RMSE",
        payload["lowest_rmse_model"],
        f"RM {metric_by_model.loc[payload['lowest_rmse_model'], 'RMSE_RM']:,.0f}",
        delta_color="off",
    )
    cards[2].metric(
        "Lowest MAE",
        payload["lowest_mae_model"],
        f"RM {metric_by_model.loc[payload['lowest_mae_model'], 'MAE_RM']:,.0f}",
        delta_color="off",
    )
    st.info(
        "All reported metrics are based on the same Scenario B group-safe "
        "5-fold validation assignments."
    )
    st.caption("Models: " + " | ".join(FINAL_MODELS))

    for metric, title, color in (
        ("RMSE_RM", "Root Mean Squared Error (RM)", "Oranges_r"),
        ("MAE_RM", "Mean Absolute Error (RM)", "Reds_r"),
        ("R2", "R² Score", "Blues"),
    ):
        figure = px.bar(
            table,
            x="Model",
            y=metric,
            text_auto=",.0f" if metric != "R2" else ".4f",
            color=metric,
            color_continuous_scale=color,
            title=title,
        )
        figure.update_layout(showlegend=False, height=390)
        st.plotly_chart(figure, width="stretch")

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
    st.write(payload["selection_rationale"])


def render_feature_importance() -> None:
    st.subheader("Feature Importance")
    importance = load_feature_importance()
    selected = st.selectbox(
        "Select Model",
        list(FINAL_MODELS),
        index=list(FINAL_MODELS).index(FINAL_MODEL_NAME),
        key="importance_model",
    )
    rows = importance[importance["Model"] == selected]
    importance_type = rows["Importance_Type"].iloc[0]
    top = rows.nlargest(20, "Importance").sort_values("Importance")
    title = (
        f"Top 20 Features — {selected}"
        if selected != "Ridge Regression"
        else "Top 20 Absolute Coefficient Magnitudes — Ridge Regression"
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
    figure.update_layout(height=650, coloraxis_showscale=False)
    st.plotly_chart(figure, width="stretch")
    st.caption(f"Measure shown: {importance_type}.")
    if selected == FINAL_MODEL_NAME:
        position_rows = rows[rows["Raw_Feature"].isin(POSITION_FEATURES)].copy()
        if len(position_rows) != len(POSITION_FEATURES):
            raise AssertionError("Saved final-model importance is missing position features.")
        st.write("#### Position features used by the final model")
        st.dataframe(
            position_rows[["Feature", "Importance"]].sort_values(
                "Importance", ascending=False
            ),
            width="stretch",
            hide_index=True,
        )


def render_actual_vs_predicted(payload: dict) -> None:
    st.subheader("Actual vs Scenario B OOF Predicted Price")
    selected = st.selectbox(
        "Select Model to Visualize",
        list(FINAL_MODELS),
        index=list(FINAL_MODELS).index(FINAL_MODEL_NAME),
        key="prediction_model",
    )
    segment = st.radio(
        "Listing Segment", ["All Listings", "Top 5%"], horizontal=True
    )
    metrics = comparison_frame(payload).set_index("Model").loc[selected]
    cards = st.columns(3)
    cards[0].metric("RMSE", f"RM {metrics['RMSE_RM']:,.0f}")
    cards[1].metric("MAE", f"RM {metrics['MAE_RM']:,.0f}")
    cards[2].metric("R²", f"{metrics['R2']:.4f}")

    oof = load_oof_predictions()
    if segment == "Top 5%":
        threshold = float(np.quantile(oof["actual_price"], 0.95))
        oof = oof[oof["actual_price"] >= threshold].copy()
    prediction_column = PREDICTION_COLUMNS[selected]
    plot = oof.rename(
        columns={
            "actual_price": "Actual Price (RM)",
            prediction_column: "OOF Predicted Price (RM)",
        }
    )
    figure = px.scatter(
        plot,
        x="Actual Price (RM)",
        y="OOF Predicted Price (RM)",
        hover_data=["listing_id", "scenario_b_fold"],
        opacity=0.55,
        title=f"{selected}: Actual vs OOF Predicted Price",
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
        f"Every plotted row is a saved Scenario B out-of-fold prediction. "
        f"Displayed listings: {len(plot):,}."
    )


def trim_label(value: float) -> str:
    return f"{value:g}%"


def render_outlier_trimming() -> None:
    st.subheader("Outlier & Trimming Analysis")
    st.caption(
        "Evaluating whether removing high-priced properties improves model generalization"
    )
    st.write(
        "Upper-tail trimming was evaluated at 0%, 0.5%, 1%, 2.5%, 5%, and "
        "10%. The experiment compared training-only trimming against "
        "trimmed-population evaluation. The final model retains 0% trimming "
        "because removing premium training observations worsened generalization "
        "to the complete housing market."
    )

    metadata, tables = load_trimming_results()
    training = tables["training_only_comparison"]
    restricted = tables["trimmed_population_comparison"]
    distribution = tables["distribution_shift"]
    retained_cv = tables["retained_cv_summary"]
    levels = metadata["trim_levels_percent"]
    labels = [trim_label(float(value)) for value in levels]
    selected_label = st.selectbox(
        "Upper-tail trimming level",
        labels,
        index=0,
        key="trim_level",
    )
    selected_percent = float(selected_label.removesuffix("%"))

    production_training = training[training["Model"] == FINAL_MODEL_NAME].copy()
    production_restricted = restricted[restricted["Model"] == FINAL_MODEL_NAME].copy()
    full_row = production_training.loc[
        production_training["Removal_Percent"] == selected_percent
    ].iloc[0]
    restricted_row = production_restricted.loc[
        production_restricted["Removal_Percent"] == selected_percent
    ].iloc[0]

    st.write("### Full-Market Performance")
    st.write(
        "Only training-fold observations above the selected price threshold were "
        "removed. All 3,791 listings remained in validation."
    )
    st.caption(
        "Training data: selected upper tail removed | Validation/test data: full and "
        "untouched | OOF evaluated rows: 3,791"
    )
    full_cards = st.columns(3)
    full_cards[0].metric("RMSE", f"RM {full_row['RMSE_RM']:,.0f}")
    full_cards[1].metric("MAE", f"RM {full_row['MAE_RM']:,.0f}")
    full_cards[2].metric("R²", f"{full_row['R2']:.4f}")
    premium_cards = st.columns(3)
    premium_cards[0].metric("Top-5% RMSE", f"RM {full_row['Top5_RMSE_RM']:,.0f}")
    premium_cards[1].metric(
        "99–100% RMSE", f"RM {full_row['P99_100_RMSE_RM']:,.0f}"
    )
    premium_cards[2].metric(
        "Premium Underprediction", f"{full_row['Top5_Underprediction_Pct']:.1f}%"
    )

    st.write("### Restricted-Market Performance")
    st.write(
        "In this evaluation, upper-tail properties are removed from both modelling "
        "and evaluation. Lower errors therefore represent performance on a narrower "
        "and easier housing market."
    )
    st.caption(
        "Dataset first trimmed | Retained listings used for both training and "
        "validation through 5-fold CV | OOF evaluated rows: retained listings"
    )
    scope_cards = st.columns(2)
    scope_cards[0].metric("Rows Retained", f"{int(restricted_row['Retained_OOF_Rows']):,}")
    scope_cards[1].metric(
        "Rows Removed", f"{int(restricted_row['Removed_Evaluation_Rows']):,}"
    )
    rmse_cards = st.columns(3)
    rmse_cards[0].metric(
        "Original RMSE", f"RM {restricted_row['Matched_Original_RMSE_RM']:,.0f}"
    )
    rmse_cards[1].metric(
        "Retrained RMSE", f"RM {restricted_row['Matched_Retrained_RMSE_RM']:,.0f}"
    )
    rmse_cards[2].metric(
        "RMSE Gain", f"RM {restricted_row['Matched_RMSE_Gain_RM']:,.0f}"
    )
    mae_cards = st.columns(3)
    mae_cards[0].metric(
        "Original MAE", f"RM {restricted_row['Matched_Original_MAE_RM']:,.0f}"
    )
    mae_cards[1].metric(
        "Retrained MAE", f"RM {restricted_row['Matched_Retrained_MAE_RM']:,.0f}"
    )
    mae_cards[2].metric(
        "MAE Gain", f"RM {restricted_row['Matched_MAE_Gain_RM']:,.0f}"
    )
    st.caption("A positive gain means retraining helped on the retained population.")

    if selected_percent > 0:
        st.warning(
            "The lower restricted-market error should not be interpreted as improved "
            "full-market performance. Training-only trimming worsens prediction when "
            "all original listings remain in validation."
        )
    if selected_percent >= 5:
        st.warning(
            "At this trimming level, the market scope changes substantially because "
            "many premium properties are excluded from evaluation."
        )

    full_chart = px.line(
        production_training,
        x="Removal_Percent",
        y="RMSE_RM",
        markers=True,
        labels={"Removal_Percent": "Trim (%)", "RMSE_RM": "RMSE (RM)"},
        title="Training-Only Trimming: Full-Market RMSE",
    )
    full_chart.update_traces(line_color="#2563eb", marker_size=9)
    full_chart.update_layout(height=420, showlegend=False)
    full_chart.update_xaxes(tickvals=levels, ticksuffix="%")
    full_chart.update_yaxes(tickformat=",")
    st.plotly_chart(full_chart, width="stretch")
    st.caption(
        "All points use the same 3,791-listing validation population; only training "
        "rows change."
    )

    restricted_chart_data = production_restricted.melt(
        id_vars=["Removal_Percent", "Retained_OOF_Rows"],
        value_vars=["Matched_Original_RMSE_RM", "Matched_Retrained_RMSE_RM"],
        var_name="Series",
        value_name="RMSE_RM",
    )
    restricted_chart_data["Series"] = restricted_chart_data["Series"].map(
        {
            "Matched_Original_RMSE_RM": "Original model",
            "Matched_Retrained_RMSE_RM": "Retrained model",
        }
    )
    restricted_chart = px.line(
        restricted_chart_data,
        x="Removal_Percent",
        y="RMSE_RM",
        color="Series",
        markers=True,
        labels={"Removal_Percent": "Trim (%)", "RMSE_RM": "RMSE (RM)"},
        title="Restricted-Market RMSE on Matched Retained Populations",
        color_discrete_map={
            "Original model": "#64748b",
            "Retrained model": "#d97706",
        },
    )
    restricted_chart.update_traces(marker_size=9)
    restricted_chart.update_layout(height=420, legend_title_text="")
    restricted_chart.update_xaxes(tickvals=levels, ticksuffix="%")
    restricted_chart.update_yaxes(tickformat=",")
    st.plotly_chart(restricted_chart, width="stretch")
    st.caption(
        "Each trim level has a different retained population. Falling errors mainly "
        "show that excluding premium listings creates an easier market; the gap "
        "between lines is the additional retraining benefit."
    )

    st.write("### Retained Data Training & Validation")
    st.write(
        "In the restricted-market experiment, properties above the selected trimming "
        "threshold are excluded first. The remaining listings are then evaluated "
        "using the same group-safe five-fold cross-validation approach. Therefore, "
        "each retained listing is used for training in four folds and validation once."
    )
    st.info(
        "The retained listings are evaluated using five-fold group-safe "
        "cross-validation. In each fold, approximately 80% of the retained listings "
        "are used for training and 20% for validation. Every retained listing is "
        "validated exactly once."
    )
    selected_cv = retained_cv[retained_cv["trim_level"].eq(selected_label)].copy()
    if len(selected_cv) != 5:
        raise ValueError(f"Retained-CV summary is incomplete for {selected_label}.")
    original_rows = int(selected_cv["original_rows"].iloc[0])
    retained_rows = int(selected_cv["retained_rows"].iloc[0])
    removed_rows = int(selected_cv["removed_rows"].iloc[0])
    retention = float(selected_cv["retention_percentage"].iloc[0])
    average_training = float(selected_cv["training_rows"].mean())
    average_validation = float(selected_cv["validation_rows"].mean())

    retained_cards = st.columns(4)
    retained_cards[0].metric("Original Listings", f"{original_rows:,}")
    retained_cards[1].metric("Listings Removed", f"{removed_rows:,}")
    retained_cards[2].metric("Listings Retained", f"{retained_rows:,}")
    retained_cards[3].metric("Retention", f"{retention:.2f}%")
    fold_share_cards = st.columns(2)
    fold_share_cards[0].metric(
        "Avg. Training per Fold",
        f"{average_training:,.1f}",
        f"{100 * average_training / retained_rows:.1f}% of retained listings",
        delta_color="off",
    )
    fold_share_cards[1].metric(
        "Avg. Validation per Fold",
        f"{average_validation:,.1f}",
        f"{100 * average_validation / retained_rows:.1f}% of retained listings",
        delta_color="off",
    )
    st.caption(
        f"3,791 original listings → apply {selected_label} trimming → "
        f"{retained_rows:,} retained listings → Scenario B group-safe 5-fold CV → "
        "each retained listing validated once → restricted-market RMSE and MAE."
    )

    fold_table = selected_cv[["fold", "training_rows", "validation_rows"]].rename(
        columns={
            "fold": "Fold",
            "training_rows": "Training Listings",
            "validation_rows": "Validation Listings",
        }
    )
    st.dataframe(
        fold_table.style.format(
            {"Training Listings": "{:,}", "Validation Listings": "{:,}"}
        ),
        width="stretch",
        hide_index=True,
    )

    fold_chart_data = fold_table.melt(
        id_vars="Fold",
        value_vars=["Training Listings", "Validation Listings"],
        var_name="Partition",
        value_name="Listings",
    )
    fold_chart = px.bar(
        fold_chart_data,
        x="Fold",
        y="Listings",
        color="Partition",
        barmode="stack",
        text_auto=",",
        title=f"Training and Validation Listings by Fold — {selected_label} Trimming",
        color_discrete_map={
            "Training Listings": "#2563eb",
            "Validation Listings": "#d97706",
        },
    )
    fold_chart.update_layout(height=420, legend_title_text="")
    fold_chart.update_xaxes(tickvals=list(range(1, 6)), title="Scenario B Fold")
    fold_chart.update_yaxes(rangemode="tozero", tickformat=",")
    st.plotly_chart(fold_chart, width="stretch")
    st.caption(
        "Each stacked bar is one CV run: that fold is validation/test data and the "
        "other four retained folds are training data."
    )

    retained_trend = (
        retained_cv.groupby("trim_level", sort=False, as_index=False)
        .agg(retained_rows=("retained_rows", "first"))
    )
    retained_trend["trim_level"] = pd.Categorical(
        retained_trend["trim_level"], categories=labels, ordered=True
    )
    retained_trend = retained_trend.sort_values("trim_level")
    retained_chart = px.bar(
        retained_trend,
        x="trim_level",
        y="retained_rows",
        text_auto=",",
        labels={"trim_level": "Trim Level", "retained_rows": "Retained Listings"},
        title="Retained Modelling Population by Trim Level",
    )
    retained_chart.update_traces(marker_color="#2563eb")
    retained_chart.update_layout(height=390, showlegend=False)
    retained_chart.update_yaxes(rangemode="tozero", tickformat=",")
    st.plotly_chart(retained_chart, width="stretch")
    st.caption(
        "Increasing the trim level progressively reduces the population available "
        "for both training and validation."
    )
    st.warning(
        "This restricted-market result measures prediction accuracy after premium "
        "properties are excluded. It does not represent performance on the original "
        "complete 3,791-listing market."
    )

    st.write("### Market Scope")
    market_scope = pd.DataFrame(
        {
            "Trim Level": distribution["Removal_Percent"].map(trim_label),
            "Rows Retained": distribution["After_Row_Count"].astype(int),
            "Rows Removed": (
                distribution["Before_Row_Count"] - distribution["After_Row_Count"]
            ).astype(int),
            "Retention %": (
                100 * distribution["After_Row_Count"] / distribution["Before_Row_Count"]
            ),
            "Maximum Retained Price": distribution["After_Maximum_Price_RM"],
            "Mean Price": distribution["After_Mean_Price_RM"],
            "Price Skewness": distribution["After_Skewness"],
        }
    )
    st.dataframe(
        market_scope.style.format(
            {
                "Rows Retained": "{:,}",
                "Rows Removed": "{:,}",
                "Retention %": "{:.1f}%",
                "Maximum Retained Price": "RM {:,.0f}",
                "Mean Price": "RM {:,.0f}",
                "Price Skewness": "{:.2f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.write("### Premium Impact")
    premium_impact = production_training[
        [
            "Removal_Percent",
            "Top5_RMSE_RM",
            "P99_100_RMSE_RM",
            "Top5_Underprediction_Pct",
        ]
    ].rename(
        columns={
            "Removal_Percent": "Trim Level",
            "Top5_RMSE_RM": "Top-5% RMSE",
            "P99_100_RMSE_RM": "99–100% RMSE",
            "Top5_Underprediction_Pct": "Premium Underprediction %",
        }
    )
    premium_impact["Trim Level"] = premium_impact["Trim Level"].map(trim_label)
    st.dataframe(
        premium_impact.style.format(
            {
                "Top-5% RMSE": "RM {:,.0f}",
                "99–100% RMSE": "RM {:,.0f}",
                "Premium Underprediction %": "{:.1f}%",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.write(
        "Premium underprediction increases as more upper-tail examples are removed "
        "from training."
    )

    st.success("Final Decision: 0% Trimming")
    st.write(
        "Upper-tail trimming was not adopted. Every nonzero training-only trimming "
        "level worsened both RMSE and MAE on the complete 3,791-listing validation "
        "population. Premium-tail prediction also deteriorated substantially as more "
        "high-priced training observations were removed."
    )
    st.info(f"Final Production Model: {FINAL_MODEL_NAME}")

    with st.expander("Technical Details", expanded=False):
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
    st.info(
        f"Model used: {FINAL_MODEL_NAME}  |  Validation: Scenario B group-safe 5-fold CV"
    )
    st.write(
        "The final model uses selected property-position phrases from the listing description."
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

        st.write("#### Property Condition")
        condition = st.columns(2)
        is_furnished = int(condition[0].checkbox("Furnished"))
        is_renovated = int(condition[1].checkbox("Renovated"))
        submitted = st.form_submit_button("Estimate Price", type="primary")

    if submitted:
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
            "is_furnished": is_furnished,
            "is_renovated": is_renovated,
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
            model = load_deployment_model()
            prediction = predict_total_price(
                model,
                values,
                description,
                float(data["description_length"].median()),
            )
        except ValueError as error:
            st.error(str(error))
        else:
            output = st.columns(2)
            output[0].metric(
                "Estimated Listing Price",
                f"RM {prediction['total_price_RM']:,.0f}",
            )
            output[1].metric(
                "Estimated Price per sq.ft.",
                f"RM {prediction['ppsf_RM']:,.0f} / sqft",
            )
            st.success(f"Model: {FINAL_MODEL_NAME}")
            if description.strip():
                st.caption(
                    f"Description length used: {prediction['description_length']:,.0f} characters."
                )
            else:
                st.caption(
                    "Description was blank, so the canonical training median description "
                    f"length ({model.description_length_median_:,.0f}) was used."
                )
            st.warning(
                "This is a statistical listing-price estimate and not a certified property valuation."
            )


def main() -> None:
    st.set_page_config(
        page_title="Real Estate Price Prediction Dashboard",
        page_icon="🏠",
        layout="wide",
    )
    st.title("Real Estate Price Prediction Dashboard")
    st.caption("Scenario B leakage-safe group cross-validation")
    payload = load_comparison()
    data = load_dataset()
    st.sidebar.title("Navigation")
    view = st.sidebar.radio("Select View", list(VIEWS), key="navigation")
    if view == "Model Comparison":
        render_comparison(payload)
    elif view == "Feature Importance":
        render_feature_importance()
    elif view == "Actual vs Predicted":
        render_actual_vs_predicted(payload)
    elif view == "Outlier & Trimming Analysis":
        render_outlier_trimming()
    elif view == "Live House Price Predictor":
        render_live_predictor(data)


if __name__ == "__main__":
    main()
