"""Dropdown-driven exploratory analysis backed by repository datasets."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "houses.csv"
PREPARED_PATH = PROJECT_ROOT / "data" / "processed" / "production_prepared_dataset.csv"
CANONICAL_PATH = PROJECT_ROOT / "data" / "processed" / "enhanced_city_dataset.csv"
EXPECTED_SHAPES = {
    "Raw source": (4_000, 32),
    "Production prepared": (3_791, 22),
    "Enhanced canonical": (3_791, 34),
}
BLUE = "#2563eb"
ORANGE = "#d97706"
PINK = "#db2777"
OLIVE = "#4d7c0f"
NEUTRAL = "#64748b"
PROPERTY_TYPE_COLORS = {
    "Condominium": BLUE,
    "Apartment": ORANGE,
    "Service Residence": PINK,
    "Flat": OLIVE,
    "Studio": "#7c3aed",
    "Duplex": "#0891b2",
    "Townhouse Condo": "#92400e",
    "Others": NEUTRAL,
}
SIZE_LABELS = ["<500", "500–749", "750–999", "1,000–1,499", "1,500–1,999", "2,000+"]
SIZE_BINS = [0, 500, 750, 1_000, 1_500, 2_000, np.inf]
COMPLETION_LABELS = ["Before 2000", "2000–2009", "2010–2014", "2015–2019", "2020+"]
COMPLETION_BINS = [0, 2_000, 2_010, 2_015, 2_020, np.inf]


@st.cache_data
def load_eda_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read and validate the three immutable datasets used by the EDA page."""
    raw = pd.read_csv(RAW_PATH)
    prepared = pd.read_csv(PREPARED_PATH)
    canonical = pd.read_csv(CANONICAL_PATH)
    frames = {
        "Raw source": raw,
        "Production prepared": prepared,
        "Enhanced canonical": canonical,
    }
    for name, frame in frames.items():
        if frame.shape != EXPECTED_SHAPES[name]:
            raise ValueError(
                f"{name} has shape {frame.shape}; expected {EXPECTED_SHAPES[name]}."
            )
    required = {
        "price",
        "property_size_sqft",
        "bedroom",
        "bathroom",
        "parking_lot",
        "completion_year",
        "property_type",
        "tenure_type",
        "floor_range",
        "state",
    }
    missing = sorted(required.difference(canonical.columns))
    if missing:
        raise ValueError(f"Canonical EDA data is missing fields: {missing}")
    return raw, prepared, canonical


def _raw_with_standard_missing(raw: pd.DataFrame) -> pd.DataFrame:
    return raw.replace(r"^\s*(?:-|N/A|NA)?\s*$", np.nan, regex=True)


def _parse_raw_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.replace(",", "", regex=False)
    extracted = cleaned.str.extract(r"([-+]?\d*\.?\d+)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def _condominiums(canonical: pd.DataFrame) -> pd.DataFrame:
    return canonical[canonical["property_type"].eq("Condominium")].copy()


def _with_ppsf(canonical: pd.DataFrame) -> pd.DataFrame:
    frame = canonical.copy()
    frame["PPSF"] = frame["price"] / frame["property_size_sqft"]
    return frame[np.isfinite(frame["PPSF"]) & frame["PPSF"].gt(0)]


def _with_size_group(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["Size Group (sq.ft.)"] = pd.cut(
        result["property_size_sqft"],
        bins=SIZE_BINS,
        labels=SIZE_LABELS,
        right=False,
    )
    return result


def _with_completion_period(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.dropna(subset=["completion_year"]).copy()
    result = result[result["completion_year"].between(1900, 2026)]
    result["Completion Period"] = pd.cut(
        result["completion_year"],
        bins=COMPLETION_BINS,
        labels=COMPLETION_LABELS,
        right=False,
    )
    return result


def _finish(figure: go.Figure, *, height: int = 620) -> go.Figure:
    figure.update_layout(
        height=height,
        template="plotly_white",
        margin={"l": 40, "r": 30, "t": 80, "b": 80},
        legend_title_text="",
    )
    return figure


def dataset_overview_frame(
    raw: pd.DataFrame,
    prepared: pd.DataFrame,
    canonical: pd.DataFrame,
) -> pd.DataFrame:
    normalized_raw = _raw_with_standard_missing(raw)
    rows = []
    for stage, source, frame, missing_cells in (
        ("Raw source", "data/raw/houses.csv", raw, int(normalized_raw.isna().sum().sum())),
        (
            "Production prepared",
            "data/processed/production_prepared_dataset.csv",
            prepared,
            int(prepared.isna().sum().sum()),
        ),
        (
            "Enhanced canonical",
            "data/processed/enhanced_city_dataset.csv",
            canonical,
            int(canonical.isna().sum().sum()),
        ),
    ):
        rows.append(
            {
                "Dataset Stage": stage,
                "Source": source,
                "Rows": len(frame),
                "Columns": len(frame.columns),
                "Missing Cells": missing_cells,
                "Duplicate Rows": int(frame.duplicated().sum()),
            }
        )
    return pd.DataFrame(rows)


def descriptive_statistics_frame(canonical: pd.DataFrame) -> pd.DataFrame:
    labels = {
        "price": "Listing Price (RM)",
        "property_size_sqft": "Property Size (sq.ft.)",
        "bedroom": "Bedrooms",
        "bathroom": "Bathrooms",
        "parking_lot": "Parking Lots",
        "completion_year": "Completion Year",
    }
    statistics = (
        canonical[list(labels)]
        .describe(percentiles=[0.25, 0.5, 0.75])
        .T.rename(index=labels)
        .reset_index(names="Feature")
        .rename(
            columns={
                "count": "Count",
                "mean": "Mean",
                "std": "Std. Dev.",
                "min": "Minimum",
                "25%": "25th Percentile",
                "50%": "Median",
                "75%": "75th Percentile",
                "max": "Maximum",
            }
        )
    )
    return statistics


def build_average_price_by_state(
    raw: pd.DataFrame, prepared: pd.DataFrame, canonical: pd.DataFrame
) -> go.Figure:
    grouped = (
        canonical.groupby("state", as_index=False, observed=True)
        .agg(Average_Price_RM=("price", "mean"), Listings=("listing_id", "size"))
        .sort_values("Average_Price_RM")
    )
    figure = px.bar(
        grouped,
        x="Average_Price_RM",
        y="state",
        orientation="h",
        hover_data={"Listings": ":,", "Average_Price_RM": ":,.0f"},
        labels={"Average_Price_RM": "Average Listing Price (RM)", "state": "State"},
        title="Average Property Listing Price by State",
    )
    figure.update_traces(marker_color=BLUE)
    figure.update_xaxes(rangemode="tozero", tickformat=",")
    return _finish(figure, height=650)


def build_condo_mean_median_by_state(
    raw: pd.DataFrame, prepared: pd.DataFrame, canonical: pd.DataFrame
) -> go.Figure:
    grouped = (
        _condominiums(canonical)
        .groupby("state", as_index=False, observed=True)
        .agg(Mean=("price", "mean"), Median=("price", "median"), Listings=("listing_id", "size"))
    )
    long = grouped.melt(
        id_vars=["state", "Listings"],
        value_vars=["Mean", "Median"],
        var_name="Statistic",
        value_name="Price_RM",
    )
    order = grouped.sort_values("Median")["state"].tolist()
    figure = px.bar(
        long,
        x="Price_RM",
        y="state",
        color="Statistic",
        barmode="group",
        orientation="h",
        category_orders={"state": order},
        color_discrete_map={"Mean": BLUE, "Median": ORANGE},
        hover_data={"Listings": ":,"},
        labels={"Price_RM": "Condominium Price (RM)", "state": "State"},
        title="Mean and Median Condominium Price by State",
    )
    figure.update_xaxes(rangemode="tozero", tickformat=",")
    return _finish(figure, height=650)


def build_property_type_price_distribution(
    raw: pd.DataFrame, prepared: pd.DataFrame, canonical: pd.DataFrame
) -> go.Figure:
    order = (
        canonical.groupby("property_type", observed=True)["price"]
        .median()
        .sort_values()
        .index.tolist()
    )
    figure = px.box(
        canonical,
        x="property_type",
        y="price",
        color="property_type",
        category_orders={"property_type": order},
        color_discrete_map=PROPERTY_TYPE_COLORS,
        points=False,
        labels={"property_type": "Property Type", "price": "Listing Price (RM)"},
        title="Listing Price Distribution by Property Type",
    )
    figure.update_layout(showlegend=False)
    figure.update_yaxes(rangemode="tozero", tickformat=",")
    return _finish(figure)


def build_condo_price_by_bedrooms(
    raw: pd.DataFrame, prepared: pd.DataFrame, canonical: pd.DataFrame
) -> go.Figure:
    grouped = (
        _condominiums(canonical)
        .groupby("bedroom", as_index=False, observed=True)
        .agg(Median_Price_RM=("price", "median"), Listings=("listing_id", "size"))
        .sort_values("bedroom")
    )
    grouped["Bedrooms"] = grouped["bedroom"].astype(int).astype(str)
    figure = px.bar(
        grouped,
        x="Bedrooms",
        y="Median_Price_RM",
        hover_data={"Listings": ":,"},
        labels={"Median_Price_RM": "Median Condominium Price (RM)"},
        title="Median Condominium Price by Bedroom Count",
    )
    figure.update_traces(marker_color=BLUE)
    figure.update_yaxes(rangemode="tozero", tickformat=",")
    return _finish(figure)


def build_bedroom_size_heatmap(
    raw: pd.DataFrame, prepared: pd.DataFrame, canonical: pd.DataFrame
) -> go.Figure:
    frame = _with_size_group(_condominiums(canonical))
    frame = frame[frame["bedroom"].between(1, 6)]
    pivot = frame.pivot_table(
        index="bedroom",
        columns="Size Group (sq.ft.)",
        values="price",
        aggfunc="median",
        observed=False,
    ).reindex(columns=SIZE_LABELS)
    figure = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(float),
            x=pivot.columns.astype(str),
            y=pivot.index.astype(int).astype(str),
            colorscale="Blues",
            colorbar={"title": "Median RM"},
            hovertemplate="Bedrooms: %{y}<br>Size: %{x} sq.ft.<br>Median: RM %{z:,.0f}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Median Condominium Price by Bedroom Count and Property Size",
        xaxis_title="Property Size Group (sq.ft.)",
        yaxis_title="Bedrooms",
    )
    return _finish(figure, height=650)


def build_condo_price_by_bathrooms(
    raw: pd.DataFrame, prepared: pd.DataFrame, canonical: pd.DataFrame
) -> go.Figure:
    frame = _condominiums(canonical)
    frame = frame[frame["bathroom"].between(1, 6)].copy()
    frame["Bathrooms"] = frame["bathroom"].astype(int).astype(str)
    figure = px.box(
        frame,
        x="Bathrooms",
        y="price",
        points=False,
        labels={"price": "Condominium Price (RM)"},
        title="Condominium Price Distribution by Bathroom Count",
    )
    figure.update_traces(marker_color=BLUE, line_color=BLUE)
    figure.update_yaxes(rangemode="tozero", tickformat=",")
    return _finish(figure)


def build_size_vs_price(
    raw: pd.DataFrame, prepared: pd.DataFrame, canonical: pd.DataFrame
) -> go.Figure:
    figure = px.scatter(
        canonical,
        x="property_size_sqft",
        y="price",
        color="property_type",
        color_discrete_map=PROPERTY_TYPE_COLORS,
        opacity=0.55,
        hover_data=["state", "bedroom", "bathroom"],
        labels={
            "property_size_sqft": "Property Size (sq.ft.)",
            "price": "Listing Price (RM)",
            "property_type": "Property Type",
        },
        title="Property Size vs Listing Price",
    )
    figure.update_yaxes(rangemode="tozero", tickformat=",")
    figure.update_xaxes(rangemode="tozero", tickformat=",")
    return _finish(figure, height=680)


def build_condo_price_by_size_group(
    raw: pd.DataFrame, prepared: pd.DataFrame, canonical: pd.DataFrame
) -> go.Figure:
    grouped = (
        _with_size_group(_condominiums(canonical))
        .groupby("Size Group (sq.ft.)", as_index=False, observed=False)
        .agg(Median_Price_RM=("price", "median"), Listings=("listing_id", "size"))
    )
    figure = px.bar(
        grouped,
        x="Size Group (sq.ft.)",
        y="Median_Price_RM",
        category_orders={"Size Group (sq.ft.)": SIZE_LABELS},
        hover_data={"Listings": ":,"},
        labels={"Median_Price_RM": "Median Condominium Price (RM)"},
        title="Median Condominium Price Across Property Size Groups",
    )
    figure.update_traces(marker_color=BLUE)
    figure.update_yaxes(rangemode="tozero", tickformat=",")
    return _finish(figure)


def build_price_cdf_by_tenure(
    raw: pd.DataFrame, prepared: pd.DataFrame, canonical: pd.DataFrame
) -> go.Figure:
    figure = go.Figure()
    for tenure, color in (("Freehold", BLUE), ("Leasehold", ORANGE)):
        values = np.sort(canonical.loc[canonical["tenure_type"].eq(tenure), "price"].to_numpy(float))
        cumulative = np.arange(1, len(values) + 1) / len(values)
        figure.add_trace(
            go.Scatter(
                x=values,
                y=cumulative,
                mode="lines",
                name=tenure,
                line={"color": color, "width": 3},
            )
        )
    figure.update_layout(
        title="Cumulative Distribution of Listing Prices by Tenure",
        xaxis_title="Listing Price (RM)",
        yaxis_title="Cumulative Share of Listings",
    )
    figure.update_xaxes(rangemode="tozero", tickformat=",")
    figure.update_yaxes(tickformat=".0%", range=[0, 1])
    return _finish(figure)


def build_condo_price_state_tenure(
    raw: pd.DataFrame, prepared: pd.DataFrame, canonical: pd.DataFrame
) -> go.Figure:
    grouped = (
        _condominiums(canonical)
        .groupby(["state", "tenure_type"], as_index=False, observed=True)
        .agg(Median_Price_RM=("price", "median"), Listings=("listing_id", "size"))
    )
    order = (
        grouped.groupby("state", observed=True)["Median_Price_RM"]
        .median()
        .sort_values()
        .index.tolist()
    )
    figure = px.bar(
        grouped,
        x="Median_Price_RM",
        y="state",
        color="tenure_type",
        barmode="group",
        orientation="h",
        category_orders={"state": order},
        color_discrete_map={"Freehold": BLUE, "Leasehold": ORANGE},
        hover_data={"Listings": ":,"},
        labels={
            "Median_Price_RM": "Median Condominium Price (RM)",
            "state": "State",
            "tenure_type": "Tenure Type",
        },
        title="Median Condo Price by State and Tenure Type",
    )
    figure.update_xaxes(rangemode="tozero", tickformat=",")
    return _finish(figure, height=650)


def build_condo_price_by_completion_year(
    raw: pd.DataFrame, prepared: pd.DataFrame, canonical: pd.DataFrame
) -> go.Figure:
    frame = _condominiums(canonical)
    frame = frame[frame["completion_year"].between(1900, 2026)]
    grouped = (
        frame.groupby("completion_year", as_index=False, observed=True)
        .agg(Median_Price_RM=("price", "median"), Listings=("listing_id", "size"))
        .sort_values("completion_year")
    )
    figure = px.line(
        grouped,
        x="completion_year",
        y="Median_Price_RM",
        markers=True,
        hover_data={"Listings": ":,"},
        labels={
            "completion_year": "Completion Year",
            "Median_Price_RM": "Median Condominium Price (RM)",
        },
        title="Median Condominium Price by Completion Year",
    )
    figure.update_traces(line_color=BLUE, marker_size=7)
    figure.update_yaxes(rangemode="tozero", tickformat=",")
    return _finish(figure)


def build_condo_price_state_completion_period(
    raw: pd.DataFrame, prepared: pd.DataFrame, canonical: pd.DataFrame
) -> go.Figure:
    frame = _with_completion_period(_condominiums(canonical))
    pivot = frame.pivot_table(
        index="state",
        columns="Completion Period",
        values="price",
        aggfunc="median",
        observed=False,
    ).reindex(columns=COMPLETION_LABELS)
    figure = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(float),
            x=pivot.columns.astype(str),
            y=pivot.index.astype(str),
            colorscale="Blues",
            colorbar={"title": "Median RM"},
            hovertemplate="State: %{y}<br>Period: %{x}<br>Median: RM %{z:,.0f}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Median Condominium Price by State and Completion Period",
        xaxis_title="Completion Period",
        yaxis_title="State",
    )
    return _finish(figure, height=680)


def build_condo_price_by_floor_range(
    raw: pd.DataFrame, prepared: pd.DataFrame, canonical: pd.DataFrame
) -> go.Figure:
    order = ["Low", "Medium", "High", "Unknown"]
    figure = px.box(
        _condominiums(canonical),
        x="floor_range",
        y="price",
        category_orders={"floor_range": order},
        points=False,
        labels={"floor_range": "Floor Range", "price": "Condominium Price (RM)"},
        title="Condominium Price by Floor Range",
    )
    figure.update_traces(marker_color=BLUE, line_color=BLUE)
    figure.update_yaxes(rangemode="tozero", tickformat=",")
    return _finish(figure)


def build_condo_price_density_by_parking(
    raw: pd.DataFrame, prepared: pd.DataFrame, canonical: pd.DataFrame
) -> go.Figure:
    frame = _condominiums(canonical).copy()
    frame["Parking Allocation"] = np.where(
        frame["parking_lot"].le(5),
        frame["parking_lot"].astype(int).astype(str),
        "6+",
    )
    order = ["1", "2", "3", "4", "5", "6+"]
    figure = px.violin(
        frame,
        x="Parking Allocation",
        y="price",
        category_orders={"Parking Allocation": order},
        box=True,
        points=False,
        labels={"price": "Condominium Price (RM)"},
        title="Condominium Price Density by Parking Allocation",
    )
    figure.update_traces(fillcolor="#bfdbfe", line_color=BLUE)
    figure.update_yaxes(rangemode="tozero", tickformat=",")
    return _finish(figure)


def build_median_ppsf_state_property_type(
    raw: pd.DataFrame, prepared: pd.DataFrame, canonical: pd.DataFrame
) -> go.Figure:
    frame = _with_ppsf(canonical)
    pivot = frame.pivot_table(
        index="state",
        columns="property_type",
        values="PPSF",
        aggfunc="median",
        observed=True,
    )
    figure = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(float),
            x=pivot.columns.astype(str),
            y=pivot.index.astype(str),
            colorscale="Blues",
            colorbar={"title": "Median RM/sq.ft."},
            hovertemplate="State: %{y}<br>Type: %{x}<br>Median PPSF: RM %{z:,.0f}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Median Price per Square Foot by State and Property Type",
        xaxis_title="Property Type",
        yaxis_title="State",
    )
    return _finish(figure, height=680)


def build_ppsf_distribution_property_type(
    raw: pd.DataFrame, prepared: pd.DataFrame, canonical: pd.DataFrame
) -> go.Figure:
    frame = _with_ppsf(canonical)
    order = (
        frame.groupby("property_type", observed=True)["PPSF"]
        .median()
        .sort_values()
        .index.tolist()
    )
    figure = px.box(
        frame,
        x="property_type",
        y="PPSF",
        color="property_type",
        category_orders={"property_type": order},
        color_discrete_map=PROPERTY_TYPE_COLORS,
        points=False,
        labels={"property_type": "Property Type", "PPSF": "Price per sq.ft. (RM)"},
        title="Price per Square Foot Distribution by Property Type",
    )
    figure.update_layout(showlegend=False)
    figure.update_yaxes(rangemode="tozero", tickformat=",")
    return _finish(figure)


def build_missing_values_before_preparation(
    raw: pd.DataFrame, prepared: pd.DataFrame, canonical: pd.DataFrame
) -> go.Figure:
    missing = _raw_with_standard_missing(raw).isna().sum()
    missing = missing[missing.gt(0)].sort_values().rename_axis("Feature").reset_index(name="Missing Values")
    figure = px.bar(
        missing,
        x="Missing Values",
        y="Feature",
        orientation="h",
        title="Missing Values by Feature Before Data Preparation",
    )
    figure.update_traces(marker_color=ORANGE)
    figure.update_xaxes(rangemode="tozero", tickformat=",")
    return _finish(figure, height=680)


def build_property_size_before_preparation(
    raw: pd.DataFrame, prepared: pd.DataFrame, canonical: pd.DataFrame
) -> go.Figure:
    sizes = _parse_raw_numeric(raw["Property Size"]).dropna()
    figure = px.histogram(
        x=sizes,
        nbins=50,
        labels={"x": "Property Size (sq.ft.)", "count": "Listings"},
        title="Property Size Distribution Before Data Preparation",
    )
    figure.update_traces(marker_color=ORANGE)
    figure.update_xaxes(rangemode="tozero", tickformat=",")
    return _finish(figure)


def build_property_size_after_preparation(
    raw: pd.DataFrame, prepared: pd.DataFrame, canonical: pd.DataFrame
) -> go.Figure:
    figure = px.histogram(
        canonical,
        x="property_size_sqft",
        nbins=50,
        labels={"property_size_sqft": "Property Size (sq.ft.)", "count": "Listings"},
        title="Property Size Distribution After Data Preparation",
    )
    figure.update_traces(marker_color=BLUE)
    figure.update_xaxes(rangemode="tozero", tickformat=",")
    return _finish(figure)


def build_dataset_size_stages(
    raw: pd.DataFrame, prepared: pd.DataFrame, canonical: pd.DataFrame
) -> go.Figure:
    stages = pd.DataFrame(
        {
            "Dataset Stage": ["Raw Source", "Production Prepared", "Enhanced Canonical"],
            "Rows": [len(raw), len(prepared), len(canonical)],
        }
    )
    figure = px.bar(
        stages,
        x="Dataset Stage",
        y="Rows",
        text_auto=",",
        title="Dataset Size Across Data Preparation Stages",
    )
    figure.update_traces(marker_color=[ORANGE, BLUE, OLIVE])
    figure.update_yaxes(rangemode="tozero", tickformat=",")
    return _finish(figure)


ChartBuilder = Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame], go.Figure]
EDA_VISUALIZATIONS: dict[str, dict[str, ChartBuilder]] = {
    "Price & Location": {
        "Average Property Listing Price by State": build_average_price_by_state,
        "Mean and Median Condominium Price by State": build_condo_mean_median_by_state,
        "Listing Price Distribution by Property Type": build_property_type_price_distribution,
    },
    "Property Characteristics": {
        "Median Condominium Price by Bedroom Count": build_condo_price_by_bedrooms,
        "Median Condominium Price by Bedroom Count and Property Size": build_bedroom_size_heatmap,
        "Condominium Price Distribution by Bathroom Count": build_condo_price_by_bathrooms,
        "Condominium Price by Floor Range": build_condo_price_by_floor_range,
        "Condominium Price Density by Parking Allocation": build_condo_price_density_by_parking,
    },
    "Size & Price": {
        "Property Size vs Listing Price": build_size_vs_price,
        "Median Condominium Price Across Property Size Groups": build_condo_price_by_size_group,
    },
    "Tenure & Development": {
        "Cumulative Distribution of Listing Prices by Tenure": build_price_cdf_by_tenure,
        "Median Condo Price by State and Tenure Type": build_condo_price_state_tenure,
        "Median Condominium Price by Completion Year": build_condo_price_by_completion_year,
        "Median Condominium Price by State and Completion Period": build_condo_price_state_completion_period,
    },
    "Price per Square Foot": {
        "Median Price per Square Foot by State and Property Type": build_median_ppsf_state_property_type,
        "Price per Square Foot Distribution by Property Type": build_ppsf_distribution_property_type,
    },
    "Data Quality & Preparation": {
        "Missing Values by Feature Before Data Preparation": build_missing_values_before_preparation,
        "Property Size Distribution Before Data Preparation": build_property_size_before_preparation,
        "Property Size Distribution After Data Preparation": build_property_size_after_preparation,
        "Dataset Size Across Data Preparation Stages": build_dataset_size_stages,
    },
}


def render_eda_page() -> None:
    """Render overview tables and exactly one selected EDA visualization."""
    st.subheader("Exploratory Data Analysis")
    raw, prepared, canonical = load_eda_sources()

    st.write("### Dataset Overview")
    overview = dataset_overview_frame(raw, prepared, canonical)
    st.dataframe(
        overview.style.format(
            {
                "Rows": "{:,}",
                "Columns": "{:,}",
                "Missing Cells": "{:,}",
                "Duplicate Rows": "{:,}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.write("### Descriptive Statistics")
    statistics = descriptive_statistics_frame(canonical)
    st.dataframe(
        statistics.style.format(
            {column: "{:,.2f}" for column in statistics.columns if column != "Feature"}
        ),
        width="stretch",
        hide_index=True,
    )

    st.write("### Visual Analysis")
    category = st.selectbox(
        "EDA Category",
        list(EDA_VISUALIZATIONS),
        key="eda_category",
    )
    chart_name = st.selectbox(
        "Select Visualization",
        list(EDA_VISUALIZATIONS[category]),
        key="eda_visualization",
    )
    figure = EDA_VISUALIZATIONS[category][chart_name](raw, prepared, canonical)
    st.plotly_chart(figure, width="stretch")
